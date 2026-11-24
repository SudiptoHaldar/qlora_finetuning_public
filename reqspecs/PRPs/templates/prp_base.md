name: "Base PRP Template v2 - Context-Rich with Validation Loops"
description: |

## Purpose
Template optimized for AI agents to implement features with sufficient context and self-validation capabilities to achieve working code through iterative refinement.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
[What needs to be built - be specific about the end state and desires]

## Why
- [Business value and user impact]
- [Integration with existing features]
- [Problems this solves and for whom]

## What
[User-visible behavior and technical requirements]

### Success Criteria
- [ ] [Specific measurable outcomes]

## All Needed Context

### Documentation & References (list all context needed to implement the feature)
```yaml
# MUST READ - Include these in your context window
- url: [Official API docs URL]
  why: [Specific sections/methods you'll need]
  
- file: [path/to/example.py]
  why: [Pattern to follow, gotchas to avoid]
  
- doc: [Library documentation URL] 
  section: [Specific section about common pitfalls]
  critical: [Key insight that prevents common errors]

- docfile: [PRPs/ai_docs/file.md]
  why: [docs that the user has pasted in to the project]

```

### Current Codebase tree (run `tree` in the root of the project) to get an overview of the codebase
```bash

```

### Desired Codebase tree with files to be added and responsibility of file
```bash

```

### Known Gotchas of our codebase & Library Quirks
```python
# CRITICAL: [Library name] requires [specific setup]
# Example: Qwen3 chat templates need enable_thinking=False for parseable output
# Example: boto3 reports a bucket in us-east-1 with a null region
# Example: tunable numbers live in config/*.yaml, never in code
```

## Implementation Blueprint

### Data models and structure

Create the core data models, we ensure type safety and consistency.
```python
Examples:
 - dataclasses for results and records
 - YAML keys read from config/*.yaml
 - JSONL record shapes (one object per line)

```

### list of tasks to be completed to fullfill the PRP in the order they should be completed

```yaml
Task 1:
MODIFY qlora_finetuning/<module>/existing_module.py:
  - FIND pattern: "class OldImplementation"
  - INJECT after line containing "def __init__"
  - PRESERVE existing method signatures

CREATE qlora_finetuning/<module>/new_feature.py:
  - MIRROR pattern from: qlora_finetuning/<module>/similar_feature.py
  - MODIFY class name and core logic
  - KEEP error handling pattern identical

...(...)

Task N:
...

```


### Per task pseudocode as needed added to each task
```python

# Task 1
# Pseudocode with CRITICAL details; do not write the entire code
def new_feature(param: str, client, cfg: dict) -> Result:
    # PATTERN: validate input first and fail with a clear message
    validated = validate_input(param)  # raises ValueError

    # PATTERN: read tunable numbers from config, never hard-code them
    limit = cfg["feature"]["limit"]

    # GOTCHA: the external client is passed in, so a test can hand over a stub
    # CRITICAL: under unit test nothing here may need a GPU, the network,
    # credentials, or a paid call
    raw = client.fetch(validated, limit)

    # PATTERN: return a small typed result (see CheckResult in scripts/aws_check.py)
    return Result(ok=True, detail=summarize(raw))
```

### Integration Points
```yaml
CONFIG:
  - add to: config/<name>.yaml
  - pattern: "tunable numbers and resource names; no secrets"

DEPENDENCIES:
  - add to: requirements.txt (pinned)

TESTS:
  - markers: pytest.ini (network, gpu, paid; deselected by default)

RUNBOOK:
  - add to: docs/runbook.md
  - pattern: "how to run the new script, its exit codes, troubleshooting"

AWS:
  - log every command run in: administration/aws_commands_run.md
```

## Validation Loop

### Level 1: Syntax & Style
```powershell
# Run these FIRST - fix any errors before proceeding
venv\Scripts\python -m ruff check qlora_finetuning/<module>/new_feature.py --fix  # Auto-fix what's possible

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests each new feature/file/function use existing test patterns
```python
# CREATE test_new_feature.py with these test cases:
def test_happy_path():
    """Basic functionality works"""
    result = new_feature("valid_input")
    assert result.status == "success"

def test_validation_error():
    """Invalid input raises ValidationError"""
    with pytest.raises(ValidationError):
        new_feature("")

def test_external_api_timeout():
    """Handles timeouts gracefully"""
    with mock.patch('external_api.call', side_effect=TimeoutError):
        result = new_feature("valid")
        assert result.status == "error"
        assert "timeout" in result.message
```

```powershell
# Run and iterate until passing:
venv\Scripts\python -m pytest tests/<module>/test_new_feature.py -q
# If failing: Read error, understand root cause, fix code, re-run (never mock to pass)
```

### Level 3: Integration Test
```powershell
# Run the new or changed script end to end and show its output
venv\Scripts\python -m scripts.<script>

# Expected: [exit code and the lines that prove success]

# Checks that need the network or AWS credentials are opt-in:
venv\Scripts\python -m pytest -m network -q

# Anything billed is started by the owner, not by the agent
```

## Final validation Checklist
- [ ] All tests pass: `venv\Scripts\python -m pytest -q`
- [ ] No linting errors: `venv\Scripts\python -m ruff check .`
- [ ] Manual run successful: [specific script and expected output]
- [ ] Error cases handled gracefully
- [ ] Logs are informative but not verbose
- [ ] Documentation updated if needed

---

## Time Estimate

**Total Estimated Time**: [X-Y hours]

### Breakdown by Task

#### Task 1: [Task Name] ([X-Y hours])
- **[Subtask A]**: [Time estimate]
  - [Detail 1]: [X min]
  - [Detail 2]: [X min]
  - [Detail 3]: [X min]
- **[Subtask B]**: [Time estimate]
- **[Subtask C]**: [Time estimate]

#### Task 2: [Task Name] ([X-Y hours])
- **[Subtask A]**: [Time estimate]
- **[Subtask B]**: [Time estimate]

[Continue for all tasks...]

### Assumptions
- ✅ [Assumption 1 - e.g., Developer familiar with the boto3 client API]
- ✅ [Assumption 2 - e.g., AWS sign-in valid and `aws_check` passing]
- ✅ [Assumption 3 - e.g., Reference patterns well understood]

### Risk Factors & Additional Time

**If [Risk Condition 1] (+X-Y hours)**
- +[X min]: [Specific risk detail]
- +[X min]: [Specific risk detail]

**If [Risk Condition 2] (+X-Y hours)**
- +[X min]: [Specific risk detail]

### Time Scenarios

**Optimistic Scenario ([X hours])**
- [Condition leading to faster completion]
- [Another favorable condition]

**Realistic Scenario ([Y hours])**
- [Expected normal conditions]
- [Minor expected issues]

**Pessimistic Scenario ([Z hours])**
- [Challenging conditions]
- [Complex issues expected]

### Time Comparison with Similar Features
- **[Similar Feature 1]**: ~[X hours] ([complexity comparison])
- **[Similar Feature 2]**: ~[Y hours] ([complexity comparison])
- **Expected for this feature**: [X-Y hours] ([complexity level])

### Recommended Approach for Time Management
1. **Start with [Task]** - [Rationale], ~[X hours]
2. **[Next task]** - [Rationale], ~[X min]
3. **[Continue...]**

### Success Metrics
- [Metric 1 - e.g., All N tests passing]
- [Metric 2 - e.g., 100% code coverage]
- [Metric 3 - e.g., Integration test successful]
- [Metric 4 - e.g., No type/lint errors]

---

## Anti-Patterns to Avoid
- ❌ Don't create new patterns when existing ones work
- ❌ Don't skip validation because "it should work"  
- ❌ Don't ignore failing tests - fix them
- ❌ Don't let a unit test need a GPU, the network, credentials, or a paid call
- ❌ Don't hardcode values that should be config
- ❌ Don't catch all exceptions - be specific