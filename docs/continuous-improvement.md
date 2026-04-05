# Continuous Improvement

Open Bias is not just a response-time enforcement layer. It also gives you a reviewable improvement loop for business rules that change over time.

The OSS boundary is deliberate:

- `rules.md` stays the only human-edited source of truth.
- Candidate policies are manual or externally generated in OSS.
- Open Bias can replay, compare, and package evidence, but it does not auto-apply policy changes.

## The Loop

1. Author your current business logic in project-local `rules.md`.
2. Enable replayable tracing so production traffic is captured to local JSONL datasets.
3. Run repo-owned eval suites plus trace replay against the baseline policy.
4. Produce a candidate policy file such as `rules.candidate.md` outside the OSS runtime.
5. Compare baseline vs candidate behavior and generate a review pack.
6. Have a human reviewer approve the change before copying or merging the candidate into `rules.md`.

## Recommended Trace Config

```yaml
tracing:
  type: jsonl
  path: .openbias/traces/%Y-%m-%d.jsonl
```

This creates replayable datasets that feed `openbias replay` and `openbias compare`.

## Local Walkthrough

```bash
# 1. Capture replayable traces while serving traffic
openbias serve

# 2. Replay one or more trace datasets against the current rules.md
openbias replay --trace .openbias/traces/2026-04-05.jsonl

# 3. Compare current rules.md against a candidate policy file
openbias compare \
  --candidate rules.candidate.md \
  --trace .openbias/traces/2026-04-05.jsonl

# 4. Turn the comparison output into a reviewer-facing artifact
openbias review-pack --comparison .openbias/reports/latest/comparison.json
```

The comparison and review pack surfaces are intentionally review-oriented:

- `comparison.json` is the machine-readable artifact for automation and trend tracking.
- `comparison.md` is a quick human summary of gates and metric deltas.
- `review-pack.md` is the approval artifact with provenance, wins, regressions, and reproduction steps.

Replay results are runtime-aware: they honor the configured evaluator phase, `mode`, `fail_action`, and intervention strategy so trace analysis stays aligned with live enforcement behavior.

## Nightly GitHub Actions Example

An example nightly workflow lives at [`examples/github-actions/nightly-improvement.yml`](../examples/github-actions/nightly-improvement.yml).

It is stored under `examples/` instead of `.github/workflows/` so this repository can ship the pattern without enabling a failing scheduled job for itself.

The example workflow:

- installs Open Bias in CI
- runs repo-owned eval suites through `openbias eval`
- replays any captured JSONL trace datasets
- compares `rules.md` against `rules.candidate.md` when a candidate file is present
- generates `comparison.json`, `comparison.md`, and `review-pack.md`
- uploads `.openbias/reports/nightly/` as a build artifact

## Human Approval Boundary

The OSS flow stops at evidence generation.

Use a human step to decide whether the candidate policy should replace the current one. In practice that usually means:

- open a PR that includes `rules.candidate.md` and the nightly artifacts
- review changed cases and trace regressions
- merge only after the reviewer agrees the behavior change matches the intended business rule
