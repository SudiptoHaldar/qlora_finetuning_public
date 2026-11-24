# References

Local notes that PRPs link to instead of repeating documentation. `/generate-prp` and
`/execute-prp` read from here before fetching anything from the web.

| Folder | Holds | Example |
|---|---|---|
| `common/` | Pitfalls and fixes that apply across the project | `gotchas.md`, `troubleshooting.md` |
| `tech/` | One file per library or service: the calls we use, response shapes, quirks | `boto3.md`, `bedrock_nova.md`, `finqa_dataset.md` |
| `patterns/` | General coding patterns worth reusing, not tied to this project | (empty) |
| `project_patterns/` | Solutions specific to this project that settled through iteration | see its `README.md` |

## Reference-first rule

1. Before adding an external URL to a PRP, check whether a file here already covers it. If it
   does, link to the file.
2. Fetch from the web only when no reference exists or its "Last verified" date is more than
   six months old.
3. After a PRP that needed a web fetch is executed, add or update the reference so the next
   PRP does not fetch again.

## Entry format

Each file starts with a title, a **Last verified** date, and its **Sources**. Keep entries
short: what goes wrong or what the call returns, what to do, and where it was learned. Mark
anything taken from a guide but not yet confirmed against the installed libraries as
unverified.
