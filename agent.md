# Public project work log

This repository was prepared after the measured experiment. It contains a
generic refactor and sanitized aggregate results, rather than the original
private task workspace.

## Decisions

- Preserve the measured baseline, RSFT and GRPO counts and uncertainty.
- Disclose task-specific collection guidance and the small accepted training set.
- Keep the synthetic arithmetic demo separate from measured task performance.
- Publish reusable source, examples, references, figures and report sources.
- Exclude private task assets, expected outputs, raw generations and adapters.
- Make CPU installation, tests, result verification and package building safe
  defaults; keep GPU model loading and training explicit.
- Retain the original implementation's limitations in the public methods.

## Work completed for the public refactor

- Established environment/model interfaces, rollout records and statistical tools.
- Added generic own-success dataset construction, RSFT and GRPO components.
- Documented the workflow, original method and reproducibility boundaries.
- Added pinned development/GPU dependency groups and CPU CI.
- Prepared a public report and sanitized aggregate evidence.

Validation results and publication status are recorded only after the final
checks complete; this log does not substitute for those checks.
