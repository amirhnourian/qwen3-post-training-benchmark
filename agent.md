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

## Release validation

- Source version: `048f6443185712c5179b09c01cd12c1a33b7e725`.
- CPU ML regression: **85 tests and 8 subtests passed**, with CUDA hidden and
  no pretrained model download. Minimal environment: **65 passed, 2 optional
  ML modules skipped**.
- Ruff, synthetic demo, aggregate-result checks and source/wheel build passed.
- The installed wheel demo also passed outside the source checkout.
- The two-page IEEEtran report was rendered and both pages visually reviewed;
  chart labels, table, formula, references and hardware details are readable.
- Public source/report identifier scans passed. The repository contains only
  the reviewed generic implementation and sanitized artifacts.
- Ready for publication at
  https://github.com/amirhnourian/qwen3-post-training-benchmark.
  The live CI badge records remote workflow status independently.

See `results/public_validation.json` for the local release checks. This log
records the public extraction; it does not replace private execution evidence.
