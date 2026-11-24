# PRP Guidelines: Quality First, Then Efficiency

**Core Principle**: Quality is the primary constraint; size is secondary. A PRP must contain
everything the agent needs to implement correctly on the first pass. Never sacrifice
implementation detail to hit a line count.

---

## Quality Checklist (MUST pass before considering size)

Before finalizing any PRP, verify:

- [ ] **Success criteria** are specific, measurable, and testable
- [ ] **File paths** are exact (CREATE/MODIFY lists every file that will be touched)
- [ ] **Feature-specific gotchas** are listed (things unique to THIS feature, not just generic)
- [ ] **Integration points** are explicit (config keys added, dependencies pinned, markers registered, runbook section named)
- [ ] **Pseudocode** covers any non-obvious logic (tricky algorithms, state machines, data transforms)
- [ ] **Edge cases** are identified (null handling, empty lists, concurrent access, error paths)
- [ ] **Validation steps** are runnable (actual commands, not "test manually")

If any of these are missing, add them regardless of line count.

---

## Template Selection

| Complexity | Template | Typical Lines | When to Use |
|------------|----------|---------------|-------------|
| **Simple** | `prp_lean.md` | 100-200 | Small script or module following existing patterns, well-known tech |
| **Medium** | `prp_lean.md` (expanded) | 200-350 | New module or script, moderate complexity |
| **Complex** | `prp_base.md` | 350-500 | New integration, architectural changes |
| **Very Complex** | `prp_base.md` | 500-700 | Multi-system, new patterns, significant unknowns |

**Guidance on size**:
- Under 500 lines: ideal; no action needed
- 500-700 lines: acceptable if the quality checklist demands it; consider whether any
  inline content could become a reference file instead
- Over 700 lines: split into phased PRPs, each independently valuable

---

## Efficiency Checklist (apply AFTER quality is satisfied)

- [ ] References used instead of inline docs (link to `reqspecs/references/`)
- [ ] No external URLs that already have a reference file
- [ ] Examples show pattern only, not full code
- [ ] Time estimates are bulleted, not paragraphs
- [ ] Generic gotchas point to `reqspecs/references/common/gotchas.md` rather than repeating them

---

## What Goes in the PRP vs. References

| Content Type | In PRP | In Tech/Pattern Reference | In Project Pattern |
|-------------|--------|--------------------------|-------------------|
| Feature-specific gotchas | Yes | No | No |
| Generic tech gotchas | Link only | Yes | No |
| Pseudocode for this feature's logic | Yes | No | No |
| Standard tech patterns (dataclass, config loader) | Link only | Yes | No |
| QLoRA Finetuning component patterns (split manifest reader, DSL executor, cost ledger record) | Link only | No | Yes |
| How to add a new read-only check script | Link only | No | Yes |
| Integration points for this feature | Yes | No | No |
| API/library documentation | Link only | Yes | No |
| Edge cases specific to this feature | Yes | No | No |
| Project conventions (naming, file structure) | Link only | Yes | No |

**Rules**:
- If the content is reusable across features, it belongs in a reference
- If it's specific to this feature's implementation, it stays in the PRP
- **Project patterns** (`reqspecs/references/project_patterns/`): QLoRA Finetuning-specific
  solutions that evolved through iteration — check these first for data loading,
  evaluation, and cost-tracking patterns

---

## Token Budget Targets

| Phase | Budget | What Counts |
|-------|--------|-------------|
| **PRP Reading** | 800-3,000 tokens | The PRP file itself |
| **Reference Reading** | 500-1,500 tokens | Local reference files |
| **Web Fetches** | 0-3,000 tokens | Only for missing references |
| **Implementation** | 15,000-25,000 tokens | Code generation, tests |
| **Total** | 15,000-30,000 tokens | Full execution |

**Previous baseline**: ~40,000 tokens per PRP execution
**Target**: ~20,000 tokens per PRP execution (50% reduction)

---

## Reference-First Rule

Before adding any external URL to a PRP:

1. Check if a reference exists: `reqspecs/references/tech/[tech].md`
2. If yes: use the reference, not the URL
3. If no: add the URL, and after PRP execution, create the reference for next time

---

## Anti-Patterns

### Size Anti-Patterns (bloat without value)
- Pasting entire documentation sections into the PRP
- Including multiple full code examples when one shows the pattern
- Writing prose where a bullet list works
- Duplicating gotchas already in reference files
- Elaborate time estimate scenarios (optimistic/pessimistic/realistic)

### Quality Anti-Patterns (brevity at the cost of correctness)
- Omitting feature-specific gotchas to keep the PRP short
- Saying "follow existing pattern" without specifying WHICH file to mirror
- Leaving out integration points (config keys, dependencies, pytest markers, env vars)
- Skipping pseudocode for non-trivial logic because "the agent will figure it out"
- Vague success criteria like "feature works correctly"
- Splitting a feature into PRPs that aren't independently valuable
