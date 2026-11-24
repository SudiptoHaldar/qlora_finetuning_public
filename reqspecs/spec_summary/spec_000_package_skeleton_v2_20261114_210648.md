# Spec summary: Package skeleton, settings, CI (req_000 v0.02)

**PRP**: [prp_000_package_skeleton_v2.md](../PRPs/prp_000_package_skeleton_v2.md)  
**Origin**: `reqspecs/reqs/req_000_project_scaffold/req_000_02_spec.md`  
**Created**: 2026-11-14  

## Todo list

1. Create the package: `__init__.py` with `__version__`, `settings.py`, `logging_setup.py`,
   and eight sub-package stubs.
2. Write `tests/test_package.py`, `tests/test_settings.py`, `tests/test_logging_setup.py`, and
   an empty test package per sub-package.
3. Move `scripts/aws_check.py` onto `settings.load_config()`; give a one-line hint when it is
   started by file path; update its tests.
4. Register the `gpu` and `paid` markers; add `ruff.toml`; fix the over-long lines.
5. Add `.github/workflows/ci.yml`; parse it and run its three commands locally.
6. Switch the runbook, `coding.md`, commands, templates, and references to the module form.
7. Add guard tests for the markers and for by-path script commands.
8. Validate, run the live AWS check, update the roadmap, plan, and summaries.

## Summary

This version gives every later version one package to import from, one config loader, and the
same lint and test gate locally and on each pull request. `load_config("<name>")` reads
`config/<name>.yaml`, requires an integer `version` key, and reports every missing required
key in one `ConfigError`. `__version__` is read from `release_number.md`. Scripts are run as
modules (`python -m scripts.<name>`), so no packaging step is needed. CI is one GitHub Actions
job on Ubuntu with Python 3.13 that installs `requirements.txt`, lints, and tests, with no
secrets.

The CI run itself needs a push and a pull request, which only the owner can make.

## Confidence score

8/10. The code is small and follows patterns already in `aws_check`. The workflow cannot be run
from the development machine, and many documents change at once.

## Implementation Time Estimate

2–3 hours.

| Task | Time |
|---|---|
| Package modules and stubs | 30 min |
| Package tests | 35 min |
| `aws_check` loader swap and tests | 20 min |
| Markers, `ruff.toml`, lint fixes | 20 min |
| CI workflow | 10 min |
| Documents | 25 min |
| Guard tests | 10 min |
| Validation, live run, status documents | 20–40 min |

## Time Taken for PRP Generation

~9 minutes.

| Phase | Time |
|-------|------|
| Reading requirements & project guidelines | 1 min |
| Codebase exploration | 2 min |
| Research (action versions, ruleset state) | 1 min |
| Writing PRP document | 5 min |
