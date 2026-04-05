"""Replay-based policy improvement orchestration."""

from __future__ import annotations

from pathlib import Path

from openbias.config.settings import Settings
from openbias.improve.schema import (
    ImprovementAggregate,
    ImprovementResult,
    ImprovementTraceRun,
    ImprovementVariantResult,
    PolicyVariant,
    VariantProvenance,
)
from openbias.policy.build import build_engine_for_policy
from openbias.policy.engines.llm.llm_client import LLMClient
from openbias.policy.rules import POLICY_FILENAME, resolve_project_rules_path
from openbias.replay import ReplayRunner
from openbias.traces import load_trace_dataset

_SYSTEM_PROMPT = """You rewrite Open Bias RULES.md policy files.
Return strict JSON with this shape:
{"variants":[{"policy_markdown":"..."}]}

Requirements:
- Produce exactly the requested number of variants.
- Each variant must be a complete standalone replacement for RULES.md.
- Preserve Markdown format and policy clarity.
- Apply the instruction in meaningfully different ways across variants.
- Do not include commentary outside the JSON response."""


async def generate_variants(
    *,
    settings: Settings,
    config_path: Path | None,
    instruction: str,
    variant_count: int,
    output_dir: Path,
) -> list[PolicyVariant]:
    """Generate baseline plus candidate policy variants and persist them."""

    base_dir = (config_path.parent if config_path else Path.cwd()).resolve()
    baseline_policy_path = resolve_project_rules_path(base_dir)
    if not baseline_policy_path.is_file():
        raise ValueError(f"Baseline {POLICY_FILENAME} not found at {baseline_policy_path}")

    baseline_text = baseline_policy_path.read_text(encoding="utf-8")
    variants_dir = output_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        _persist_variant(
            baseline_policy_path=baseline_policy_path,
            instruction=instruction,
            variant_id="baseline",
            policy_text=baseline_text,
            variants_dir=variants_dir,
        )
    ]
    generated_documents = await _request_variant_documents(
        model=settings.proxy.default_model,
        baseline_text=baseline_text,
        instruction=instruction,
        variant_count=variant_count,
    )
    for index, policy_text in enumerate(generated_documents, start=1):
        variants.append(
            _persist_variant(
                baseline_policy_path=baseline_policy_path,
                instruction=instruction,
                variant_id=f"candidate-{index}",
                policy_text=policy_text,
                variants_dir=variants_dir,
            )
        )
    return variants


async def run_improvement(
    *,
    settings: Settings,
    config_path: Path | None,
    trace_paths: tuple[Path, ...],
    instruction: str,
    variant_count: int,
    output_dir: Path,
) -> ImprovementResult:
    """Run the replay-based improvement flow."""

    variants = await generate_variants(
        settings=settings,
        config_path=config_path,
        instruction=instruction,
        variant_count=variant_count,
        output_dir=output_dir,
    )
    replay_runner = ReplayRunner(boundary=settings.replay.boundary)
    variant_results = [
        await _run_variant_replay(
            settings=settings,
            config_path=config_path,
            replay_runner=replay_runner,
            variant=variant,
            trace_paths=trace_paths,
        )
        for variant in variants
    ]
    ranked_results = sorted(
        variant_results,
        key=lambda result: (
            -result.aggregate.matched_rate,
            result.aggregate.mismatched_cases,
            -result.aggregate.detection_rate,
            result.aggregate.failures,
            result.variant.variant_id,
        ),
    )

    baseline_policy_path = variants[0].provenance.baseline_policy_path
    total_labeled_cases = sum(result.aggregate.labeled_cases for result in ranked_results)
    if total_labeled_cases == 0:
        return ImprovementResult(
            status="review",
            boundary=settings.replay.boundary,
            baseline_policy_path=baseline_policy_path,
            instruction=instruction,
            variants=ranked_results,
            ranked_variant_ids=[result.variant.variant_id for result in ranked_results],
            review_reason="No labeled replay coverage was available across the provided traces.",
        )

    winner = ranked_results[0]
    return ImprovementResult(
        status="pass",
        boundary=settings.replay.boundary,
        baseline_policy_path=baseline_policy_path,
        instruction=instruction,
        variants=ranked_results,
        winner_variant_id=winner.variant.variant_id,
        winner_policy_path=winner.variant.policy_path,
        ranked_variant_ids=[result.variant.variant_id for result in ranked_results],
    )


def _persist_variant(
    *,
    baseline_policy_path: Path,
    instruction: str,
    variant_id: str,
    policy_text: str,
    variants_dir: Path,
) -> PolicyVariant:
    target = variants_dir / f"{variant_id}.md"
    content = policy_text.rstrip() + "\n"
    target.write_text(content, encoding="utf-8")
    return PolicyVariant(
        variant_id=variant_id,
        policy_path=str(target),
        provenance=VariantProvenance(
            baseline_policy_path=str(baseline_policy_path),
            instruction=instruction,
            variant_id=variant_id,
            generated_policy_path=str(target),
        ),
    )


async def _request_variant_documents(
    *,
    model: str | None,
    baseline_text: str,
    instruction: str,
    variant_count: int,
) -> list[str]:
    if not model:
        raise ValueError("Policy improvement requires proxy.default_model for variant generation.")

    client = LLMClient(
        model=model,
        temperature=0.8,
        max_tokens=8192,
        timeout=120.0,
    )
    payload = await client.complete_json(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=(
            f"Instruction:\n{instruction}\n\n"
            f"Variant count: {variant_count}\n\n"
            f"Baseline {POLICY_FILENAME}:\n"
            "```markdown\n"
            f"{baseline_text}\n"
            "```"
        ),
        session_id="openbias:improve:variant-generation",
    )

    variants = payload.get("variants")
    if not isinstance(variants, list) or len(variants) != variant_count:
        count = len(variants) if isinstance(variants, list) else "invalid"
        raise ValueError(
            f"Variant generation returned {count} variants; expected exactly {variant_count}."
        )

    documents: list[str] = []
    for item in variants:
        if not isinstance(item, dict):
            raise ValueError("Variant generation returned a non-object entry.")
        policy_markdown = item.get("policy_markdown")
        if not isinstance(policy_markdown, str) or not policy_markdown.strip():
            raise ValueError("Variant generation returned an empty policy_markdown value.")
        documents.append(policy_markdown)
    return documents


async def _run_variant_replay(
    *,
    settings: Settings,
    config_path: Path | None,
    replay_runner: ReplayRunner,
    variant: PolicyVariant,
    trace_paths: tuple[Path, ...],
) -> ImprovementVariantResult:
    trace_results: list[ImprovementTraceRun] = []
    matched_cases = 0
    mismatched_cases = 0
    labeled_cases = 0
    detected_cases = 0
    supported_cases = 0
    failure_count = 0

    try:
        engine = await build_engine_for_policy(
            settings=settings,
            config_path=config_path,
            rules_path=Path(variant.policy_path),
        )
    except Exception as exc:
        return ImprovementVariantResult(
            variant=variant,
            aggregate=ImprovementAggregate(
                labeled_cases=0,
                matched_cases=0,
                mismatched_cases=0,
                matched_rate=0.0,
                detection_rate=0.0,
                failures=len(trace_paths) or 1,
            ),
            traces=[
                ImprovementTraceRun(
                    trace_path=str(path),
                    dataset_name=path.stem,
                    summary={},
                    failures=[{"case_id": "__engine__", "error": str(exc)}],
                )
                for path in trace_paths
            ],
        )

    try:
        for trace_path in trace_paths:
            try:
                dataset = load_trace_dataset(trace_path)
                replay_result = await replay_runner.run(engine, dataset)
                trace_results.append(
                    ImprovementTraceRun(
                        trace_path=str(trace_path),
                        dataset_name=dataset.name,
                        summary=replay_result.summary.__dict__,
                        failures=list(replay_result.failures),
                    )
                )
                matched_cases += replay_result.summary.matched_cases
                mismatched_cases += replay_result.summary.mismatched_cases
                labeled_cases += (
                    replay_result.summary.matched_cases
                    + replay_result.summary.mismatched_cases
                )
                supported_cases += replay_result.summary.supported_cases
                detected_cases += sum(
                    1 for outcome in replay_result.outcomes if outcome.observed_detection
                )
                failure_count += len(replay_result.failures)
            except Exception as exc:
                trace_results.append(
                    ImprovementTraceRun(
                        trace_path=str(trace_path),
                        dataset_name=trace_path.stem,
                        summary={},
                        failures=[{"case_id": "__trace__", "error": str(exc)}],
                    )
                )
                failure_count += 1
    finally:
        await engine.shutdown()

    return ImprovementVariantResult(
        variant=variant,
        aggregate=ImprovementAggregate(
            labeled_cases=labeled_cases,
            matched_cases=matched_cases,
            mismatched_cases=mismatched_cases,
            matched_rate=(matched_cases / labeled_cases) if labeled_cases else 0.0,
            detection_rate=(detected_cases / supported_cases) if supported_cases else 0.0,
            failures=failure_count,
        ),
        traces=trace_results,
    )
