name: "Lean PRP - [Feature Name]"
description: Streamlined PRP for simple/medium complexity features

---

## Goal
[1-2 sentences: what gets built and why]

## Why & What
- **User Value**: [Primary benefit in one line]
- **Technical**: [What gets built in one line]

## Success Criteria
- [ ] [Outcome 1]
- [ ] [Outcome 2]
- [ ] [Outcome 3]

---

## Context

### References
```yaml
# Use local references instead of web fetches
refs:
  - reqspecs/references/tech/[tech].md
  - reqspecs/references/patterns/[pattern].md
  - reqspecs/references/common/gotchas.md

code:  # Existing files to mirror/extend
  - file: [existing_file_to_mirror]
    why: [Pattern to follow]
```

### Critical Gotchas
- [Gotcha 1 in one line]
- [Gotcha 2 in one line]

---

## Implementation

### Files
```yaml
CREATE:
  - [file_path]: [purpose in one line]

MODIFY:
  - [file_path]: [change in one line]
```

### Tasks (~[X] min total)
1. **[Task 1]** (~[X]min): [Action] following pattern from [reference]
2. **[Task 2]** (~[X]min): [Action]
3. **[Task 3]** (~[X]min): [Action]

### Pseudocode (key logic only)
```python
# Only include non-obvious logic, 10-20 lines max
```

---

## Validation

```powershell
# Lint
venv\Scripts\python -m ruff check [files]

# Tests (no GPU, network, credentials, or paid calls)
venv\Scripts\python -m pytest tests/[path]/test_[feature].py -q

# Manual check (if a script's command line or output changed)
venv\Scripts\python -m scripts.[script]
```

**Expected**: All pass, no errors

---

## Checklist
- [ ] Files created/modified as specified
- [ ] Tests pass (expected + edge + failure cases)
- [ ] No lint errors
- [ ] Implementation summary created

---

**Target Time**: [X-Y] minutes
**Complexity**: Simple/Medium
