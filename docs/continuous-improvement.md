# Continuous Improvement

Open Bias is not just a response-time enforcement layer. It also gives you a reviewable improvement loop for business rules that change over time.

The OSS boundary is deliberate:

- `rules.md` stays the only human-edited source of truth.
- `openbias improve` can generate a small set of policy variants plus a replay-backed recommendation.
- Open Bias does not auto-apply policy changes.

## The Loop

1. Author your current business logic in project-local `rules.md`.
2. Enable replayable tracing so production traffic is captured to local JSONL datasets.
3. Run repo-owned eval suites plus trace replay against the baseline policy.
4. Run `openbias improve` with one or more trace datasets plus an instruction describing how the policy should vary.
5. Review the generated variants and replay scores in `.openbias/reports/latest/`.
6. Have a human reviewer approve the change before copying or merging any variant into `rules.md`.

## Recommended Trace Config

```yaml
tracing:
  type: jsonl
  path: .openbias/traces/%Y-%m-%d.jsonl
```

This creates replayable datasets that feed `openbias replay` and `openbias improve`.

## Local Walkthrough

```bash
# 1. Capture replayable traces while serving traffic
openbias serve

# 2. Replay one or more trace datasets against the current rules.md
openbias replay --trace .openbias/traces/2026-04-05.jsonl

# 3. Generate variants, replay them, and write a review artifact
openbias improve \
  --trace .openbias/traces/2026-04-05.jsonl \
  --instruction "Tighten the policy around refund abuse while preserving benign support workflows."
```

The improvement surfaces are intentionally review-oriented:

- `improvement.json` is the machine-readable artifact with variant provenance, per-trace summaries, and aggregate ranking.
- `improvement.md` is the quick reviewer-facing summary with the recommended winner, if any.
- `variants/` contains the copied baseline plus generated policy variants evaluated during the run.

Replay results are runtime-aware for the configured replay boundary, so trace analysis stays aligned with the offline detection contract defined in `replay.boundary`.

## Nightly GitHub Actions Example

An example nightly workflow lives at [`examples/github-actions/nightly-improvement.yml`](../examples/github-actions/nightly-improvement.yml).

It is stored under `examples/` instead of `.github/workflows/` so this repository can ship the pattern without enabling a failing scheduled job for itself.

The example workflow:

- installs Open Bias in CI
- runs repo-owned eval suites through `openbias eval`
- replays any captured JSONL trace datasets
- runs `openbias improve` when traces are present
- generates `improvement.json`, `improvement.md`, and variant files under `.openbias/reports/nightly/`
- uploads `.openbias/reports/nightly/` as a build artifact

## Human Approval Boundary

The OSS flow stops at evidence generation.

Use a human step to decide whether the candidate policy should replace the current one. In practice that usually means:

- open a PR that includes the reviewed winner from `variants/` plus the nightly artifacts
- review the recommended winner, changed cases, and replay failures
- merge only after the reviewer agrees the behavior change matches the intended business rule
