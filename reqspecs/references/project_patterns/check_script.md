# Pattern: operator check script

**Last verified**: 2026-11-15  
**Shown in**: `scripts/aws_check.py` (req_000 v0.05), `scripts/env_check.py` (req_000 v0.03),
and their tests in `tests/scripts/`.

A script the operator runs to learn whether something outside the code is in the state the
project needs: an AWS account, this machine. It only reads, prints one line per check, and
tells the truth through its exit code.

## Shape

- **One function per check.** It takes what it inspects as arguments (a boto3 client, the
  `torch` module, a probe function) plus the config, and returns a `CheckResult`. It never
  reaches for a global and never prints.
- **`CheckResult`** is a frozen dataclass: check id (`C1`, `E3`), name, outcome, one-line
  detail. `aws_check` uses pass or fail; `env_check` adds SKIP for a check whose prerequisite
  failed or that was not requested.
- **Expected values come from `config/<name>.yaml`** through `load_config()` with a list of
  required keys. No number or name is written in the script.
- **A guard turns service and library errors into FAIL lines.** The operator sees the error
  type and its first line, never a traceback. Problems the operator must fix first (expired
  sign-in, broken config) end the run with exit code 2.
- **Exit codes**: 0 nothing failed, 1 at least one check failed, 2 the run could not start.
- **Output**: `STATUS  ID  Name: detail` on standard output, ASCII only, then a summary line.
  Log lines go to standard error.
- **Run as a module**: `venv\Scripts\python -m scripts.<name>`. A run by file path prints a
  one-line hint.
- **Heavy or optional imports happen inside functions**, so the script and its tests load
  without them.

## Tests

- Each check gets one passing and at least one failing case, using stand-ins
  (`MagicMock` clients, `SimpleNamespace` modules, small probe functions).
- `main()` is tested for each exit code with `capsys`; the real session or environment is
  replaced (`monkeypatch` of `boto3.Session`, or an `env=` argument).
- Two subprocess tests: the module form works, the file-path form gives the hint.
- One real run sits behind a marker (`network` or `gpu`) and is deselected by default.

## Reuse checklist

- [ ] Checks listed in the spec as a table: id, what it checks, when it passes
- [ ] `config/<name>.yaml` with `version: 1`; required keys listed in the script
- [ ] Check functions take their dependencies as arguments
- [ ] Guard for expected errors; exit codes 0, 1, 2
- [ ] By-path hint at the import of `qlora_finetuning`
- [ ] Stand-in tests per check, `main()` tests, two subprocess tests, one marked real test
- [ ] Runbook section: how to run, what each check means, troubleshooting
- [ ] Anything that creates, changes, deletes, or is billed is not in a check script
