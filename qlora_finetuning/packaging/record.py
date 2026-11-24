"""The package record and checks P1 to P10 (req_006 v6.02, spec section 4.3).

One function, ``gather``, reads everything there is to know about the two packages on this
machine and returns the record: what each folder holds, the dev runs, how their replies
compare, what running the cards' code gave, the licence entries, and ten checks. It writes
nothing. ``write_record`` writes the record's two committed files.

Usage::

    from qlora_finetuning.packaging.record import gather, write_record

    state = gather(cfg)
    state.record["checks"]            # P1 to P10, each with passed and detail
    state.record["adapter"]["status"] # "ready" or "not ready"
    state.record["merged"]["status"]  # "ready", "withheld", "not ready", or "not built"
    state.cards                       # {"adapter": text or None, "merged": text or None}

The adapter package is ready when every check passes for it. The merged copy is withheld,
and gets no card, when a dev run of it has too few examples right; that does not hold the
adapter back. The record holds counts, ids, and hashes: no reply and no dataset text.

This module imports no part of the training stack; the tokenizer check reaches it through
``backend.py`` and reports a missing stack as a failed check.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from qlora_finetuning.comparison.report_tables import table
from qlora_finetuning.data.split import load_split_config, read_manifest
from qlora_finetuning.data.token_budget import tokenizer_fingerprint
from qlora_finetuning.evaluation.metrics import METRICS, EvaluationError
from qlora_finetuning.evaluation.runner import RunnerContext, run_predictions
from qlora_finetuning.local_qlora.training_config import load_training_config
from qlora_finetuning.packaging.adapter_package import (
    CARD_NAME,
    adapter_bytes_problems,
    contents_problems,
    package_names,
    sample_problems,
    tokenizer_problems,
)
from qlora_finetuning.packaging.cards import ADAPTER, MERGED, build_card, usage_code
from qlora_finetuning.packaging.config import PackagingConfig
from qlora_finetuning.packaging.files import (
    listing,
    read_json,
    read_text_lf,
    sha256_file,
    sha256_text,
    write_json_lf,
    write_text_lf,
)
from qlora_finetuning.packaging.merged_package import (
    merged_names,
    merged_weights,
    read_merge_metadata,
)
from qlora_finetuning.packaging.predictors import MERGED as MERGED_PREDICTOR
from qlora_finetuning.packaging.predictors import (
    MERGED_FULL,
    PACKAGE,
    merged_settings,
    package_settings,
)
from qlora_finetuning.packaging.scan import scan_folder, scan_text
from qlora_finetuning.packaging.sources import PUBLISHED_ARM, records_problems
from qlora_finetuning.packaging.verify import compare, find_run
from qlora_finetuning.settings import ConfigError

SCHEMA = 1
COMMAND = "python -m scripts.hf_package"
TUNED = "qwen3-4b-tuned"
READY, NOT_READY, WITHHELD, NOT_BUILT = "ready", "not ready", "withheld", "not built"
CARD_RUNS = (ADAPTER, "merged_4bit", "merged_full")
RUN_NAMES = {
    "R1": "the selected adapter, by name",
    "R2": "the adapter package, from its folder",
    "R3": "the merged copy, in 4-bit",
    "R4": "the merged copy, at full precision",
}


@dataclass
class Facts:
    """What the cards and the checks are computed from."""

    cfg: PackagingConfig
    prompt: Any
    budget: Any
    local_cfg: Any
    source: Any
    records: Any
    manifest: dict[str, Any]
    allowance: int
    noise: int
    runs: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    comparisons: dict[str, dict[str, Any]] = field(default_factory=dict)
    card_runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    merged_meta: dict[str, Any] | None = None
    merged_weights: dict[str, Any] | None = None
    merged_status: str = NOT_BUILT


@dataclass
class State:
    """The record, and the two cards as they would be written now."""

    record: dict[str, Any]
    cards: dict[str, str | None]
    facts: Facts


def check_result(check_id: str, name: str, problems: list[str], ok_detail: str) -> dict[str, Any]:
    return {
        "id": check_id,
        "name": name,
        "passed": not problems,
        "detail": "; ".join(problems) if problems else ok_detail,
    }


def _run_entry(run_id: str, summary: dict[str, Any], predictor: str) -> dict[str, Any]:
    block = summary["all"]
    return {
        "run_id": run_id,
        "predictor": predictor,
        "examples": block["examples"],
        "counts": {metric: block[metric]["count"] for metric in METRICS},
        # A share is the runner's own rate, shown with two decimals; none is computed here.
        "percent": {metric: f"{100 * block[metric]['rate']:.2f}" for metric in METRICS},
    }


def read_card_runs(cfg: PackagingConfig) -> dict[str, dict[str, Any]]:
    """What running the cards' code gave; empty before ``hf_package verify``."""
    if not cfg.card_runs_file.is_file():
        return {}
    data = read_json(cfg.card_runs_file, "the record of the cards' code runs")
    runs = data.get("runs") if isinstance(data, dict) else None
    return runs if isinstance(runs, dict) else {}


def _card_run_problems(label: str, code: str, card_runs: dict[str, Any]) -> list[str]:
    run = card_runs.get(label)
    if run is None:
        return [f"the card's code was not run yet ({label}): {COMMAND} verify"]
    if run.get("code_sha256") != sha256_text(code):
        return [f"the card's code changed since it was run ({label}): {COMMAND} verify"]
    return [f"the card's code ({label}): {problem}" for problem in run.get("problems") or []]


def _tokenizercheck_result(
    folder: Path, cfg: PackagingConfig, budget: Any, base: dict[str, Path] | None,
    fingerprint: Callable[[Path], str],
) -> list[str]:
    if not folder.is_dir():
        return []
    problems = [] if base is None else tokenizer_problems(cfg.tokenizer_files, base, folder)
    if any(not (folder / name).is_file() for name in cfg.tokenizer_files):
        return problems or ["a tokenizer file is missing"]
    try:
        if fingerprint(folder) != budget.fingerprint:
            problems.append("the tokenizer is not the one pinned in config/token_budget.yaml")
    except ConfigError as exc:
        problems.append(f"the tokenizer could not be checked ({exc})")
    return problems


def _default_fingerprint(folder: Path) -> str:
    from qlora_finetuning.local_qlora.backend import load_tokenizer_from

    return tokenizer_fingerprint(load_tokenizer_from(folder))


def _named(problems: list[str], what: str) -> list[str]:
    return [f"{what}: {problem}" for problem in problems]


def gather(
    cfg: PackagingConfig,
    inputs: Any,
    *,
    ctx: RunnerContext | None = None,
    today: date | None = None,
    base: dict[str, Path] | None = None,
    fingerprint: Callable[[Path], str] | None = None,
) -> State:
    """Everything about the two packages as they are now. Reads only.

    ``inputs`` is ``workflow.load_inputs(cfg)``: the checked checkpoint and the records.
    ``base`` maps the tokenizer file names to the base model's files; without it only the
    fingerprint is checked. ``fingerprint`` and ``ctx`` are stand-ins for tests.
    """
    ctx = (ctx or RunnerContext()).resolved()
    today = today or date.today()
    fingerprint = fingerprint or _default_fingerprint
    source, records = inputs.source, inputs.records
    split_cfg = inputs.context.split_cfg or load_split_config()
    training_cfg = load_training_config(root=inputs.context.tracking_cfg.root)
    allowance = cfg.allowance_examples or training_cfg.noise_examples
    facts = Facts(
        cfg=cfg,
        prompt=inputs.prompt,
        budget=inputs.budget,
        local_cfg=inputs.local_cfg,
        source=source,
        records=records,
        manifest=read_manifest(split_cfg),
        allowance=allowance,
        noise=training_cfg.noise_examples,
        card_runs=read_card_runs(cfg),
    )
    merged_built = (cfg.merged_dir / "config.json").is_file()

    # ---- the runs
    wanted = {
        "R1": (TUNED, package_settings(facts.local_cfg, facts.budget, source.sha256)),
        "R2": (PACKAGE, package_settings(facts.local_cfg, facts.budget, source.sha256)),
    }
    merged_problems: list[str] = []
    if merged_built:
        try:
            facts.merged_meta = read_merge_metadata(cfg.merged_dir)
            facts.merged_weights = merged_weights(cfg.merged_dir)
        except ConfigError as exc:
            merged_problems.append(str(exc))
    if facts.merged_weights is not None:
        for label, name, quantized in (
            ("R3", MERGED_PREDICTOR, True),
            ("R4", MERGED_FULL, False),
        ):
            wanted[label] = (
                name,
                merged_settings(
                    facts.local_cfg,
                    facts.budget,
                    weights_sha256=facts.merged_weights["sha256"],
                    adapter=source.sha256,
                    quantized=quantized,
                ),
            )
    lines: dict[str, Any] = {}
    for label, (name, settings) in wanted.items():
        found = find_run(ctx, name, settings, cfg.split)
        facts.runs[label] = None if found is None else _run_entry(found[0], found[1], name)
        if found is not None:
            try:
                lines[label] = run_predictions(found[0], ctx)
            except (ConfigError, EvaluationError):
                facts.runs[label] = None
    for label in ("R2", "R3", "R4"):
        if label in lines and "R1" in lines:
            facts.comparisons[label] = compare(lines["R1"], lines[label])

    # ---- P1 to P3: the records and the source (the source was checked to get here)
    checks = [
        check_result(
            "P1",
            "Records in place",
            records_problems(records),
            "the six-arm record passed C1 to C8, the cost record K1 to K8, the comparison's "
            "record R1 to R8; none is provisional",
        ),
        check_result(
            "P2",
            "The source is arm A1",
            [],
            f"{source.name}, weights {source.sha256[:12]}..., as the selection record says",
        ),
        check_result(
            "P3",
            "Identity",
            [],
            "the adapter's model commit, prompt, and split are today's",
        ),
    ]

    # ---- P4 to P6: what the folders hold
    adapter_p4 = contents_problems(cfg.adapter_dir, package_names(cfg))
    adapter_p4 += adapter_bytes_problems(cfg.adapter_dir, source)
    merged_p4 = contents_problems(cfg.merged_dir, merged_names(cfg)) if merged_built else []
    if facts.merged_meta is not None:
        merged_from = facts.merged_meta["adapter"].get("weights_sha256")
        if merged_from != source.sha256:
            merged_p4.append("it was merged from another adapter than the selected checkpoint")
    adapter_p5 = scan_folder(cfg.adapter_dir, cfg.forbidden, skip=cfg.tokenizer_files)
    adapter_p5 += sample_problems(cfg.adapter_dir)
    merged_p5 = scan_folder(cfg.merged_dir, cfg.forbidden, skip=cfg.tokenizer_files)
    merged_p5 += sample_problems(cfg.merged_dir)
    adapter_p6 = _tokenizercheck_result(cfg.adapter_dir, cfg, facts.budget, base, fingerprint)
    merged_p6 = _tokenizercheck_result(cfg.merged_dir, cfg, facts.budget, base, fingerprint)

    # ---- P7: licences
    p7 = cfg.licence_problems(today)

    # ---- P8: the adapter package answers
    adapter_code, code_problems = usage_code(ADAPTER, cfg, facts.budget, facts.local_cfg)
    p8 = list(code_problems)
    recorded = records.statistics["arms"][PUBLISHED_ARM]["dev"]
    r1, r2 = facts.runs.get("R1"), facts.runs.get("R2")
    if r1 is None:
        p8.append(f"no complete local run of {TUNED} --adapter selected on {cfg.split} (R1)")
    else:
        gap = abs(r1["counts"]["execution_accuracy"] - recorded["execution_accuracy"])
        if gap >= facts.noise:
            p8.append(
                f"R1 has {r1['counts']['execution_accuracy']} right and the recorded run "
                f"{recorded['execution_accuracy']}: {gap} apart, not under {facts.noise}"
            )
    if r2 is None:
        p8.append(f"no complete local run of {PACKAGE} on {cfg.split} (R2)")
    elif "R2" in facts.comparisons and facts.comparisons["R2"]["replies_that_differ"]:
        difference = facts.comparisons["R2"]
        p8.append(
            f"R2 differs from R1 on {difference['replies_that_differ']} replies "
            f"(first: {', '.join(difference['differing_ids'][:3])})"
        )
    p8 += _card_run_problems(ADAPTER, adapter_code, facts.card_runs)

    # ---- P9: the merged copy answers
    merged_code, merged_code_problems = usage_code(MERGED, cfg, facts.budget, facts.local_cfg)
    p9 = list(merged_problems) + list(merged_code_problems)
    withheld_for: list[str] = []
    if not merged_built:
        p9.append(f"there is no merged copy: {COMMAND} merge")
    else:
        for label, name in (("R3", MERGED_PREDICTOR), ("R4", MERGED_FULL)):
            if facts.runs.get(label) is None:
                p9.append(f"no complete local run of {name} on {cfg.split} ({label})")
            elif label in facts.comparisons:
                difference = facts.comparisons[label]
                fewer = difference["right_reference"] - difference["right_other"]
                if fewer >= allowance:
                    withheld_for.append(
                        f"{label} has {fewer} fewer right than R1, and {allowance} or more "
                        "withholds the copy"
                    )
        p9 += withheld_for
        for label in ("merged_4bit", "merged_full"):
            p9 += _card_run_problems(label, merged_code, facts.card_runs)

    # ---- statuses, then the cards
    adapter_ok_so_far = checks[0]["passed"] and not (
        _but_the_card(adapter_p4) or adapter_p5 or adapter_p6 or p7 or p8
    )
    merged_clean = merged_built and not (
        p9 or _but_the_card(merged_p4) or merged_p5 or merged_p6
    )
    if not merged_built:
        facts.merged_status = NOT_BUILT
    elif withheld_for:
        facts.merged_status = WITHHELD
    else:
        facts.merged_status = READY if (merged_clean and adapter_ok_so_far) else NOT_READY

    cards: dict[str, str | None] = {ADAPTER: None, MERGED: None}
    p10_adapter: list[str] = []
    p10_merged: list[str] = []
    text, problems = build_card(ADAPTER, cfg, facts)
    if problems:
        p10_adapter += _named(problems, "the adapter card cannot be built")
    else:
        cards[ADAPTER] = text
        p10_adapter += scan_text(text, cfg.forbidden, "the adapter card")
        p10_adapter += _written(cfg.adapter_card, cfg.adapter_dir / CARD_NAME, text, "adapter")
    if merged_built and facts.merged_status != WITHHELD:
        text, problems = build_card(MERGED, cfg, facts)
        if problems:
            p10_merged += _named(problems, "the merged card cannot be built")
        else:
            cards[MERGED] = text
            p10_merged += scan_text(text, cfg.forbidden, "the merged card")
            p10_merged += _written(cfg.merged_card, cfg.merged_dir / CARD_NAME, text, "merged")
    elif facts.merged_status == WITHHELD:
        for path in (cfg.merged_card, cfg.merged_dir / CARD_NAME):
            if path.is_file():
                p10_merged.append(f"{path.name} exists though the merged copy is withheld")

    withheld = facts.merged_status == WITHHELD
    merged_counts = merged_built and not withheld
    checks += [
        check_result(
            "P4",
            "Package contents",
            _named(adapter_p4, "adapter") + (_named(merged_p4, "merged") if merged_counts else []),
            "each package holds the files it should and no other; the adapter files are the "
            "selected checkpoint's bytes",
        ),
        check_result(
            "P5",
            "Nothing private",
            _named(adapter_p5, "adapter") + _named(merged_p5, "merged"),
            "no text file of either package matches a forbidden pattern; the sample is the "
            "demo's",
        ),
        check_result(
            "P6",
            "Tokenizer",
            _named(adapter_p6, "adapter") + _named(merged_p6, "merged"),
            "the tokenizer files are the base model's and give the pinned fingerprint",
        ),
        check_result(
            "P7",
            "Licences",
            p7,
            f"{len(cfg.licences)} entries, each confirmed by the owner within "
            f"{cfg.licence_max_age_days} days",
        ),
        check_result(
            "P8",
            "The adapter package answers",
            p8,
            _p8_detail(facts, recorded),
        ),
        check_result(
            "P9",
            "The merged copy answers",
            p9,
            _p9_detail(facts),
        ),
        check_result(
            "P10",
            "The cards",
            p10_adapter + p10_merged,
            "each card is its template filled from today's records; the copy in the package "
            "and the committed copy are the same",
        ),
    ]

    adapter_ready = all(
        check["passed"] for check in checks if check["id"] not in ("P4", "P5", "P6", "P9", "P10")
    ) and not (adapter_p4 or adapter_p5 or adapter_p6 or p10_adapter)
    adapter_status = READY if adapter_ready else NOT_READY
    if facts.merged_status == READY and (p10_merged or not adapter_ready or merged_p4):
        facts.merged_status = NOT_READY

    record = {
        "schema": SCHEMA,
        "generated_by": f"{COMMAND} cards",
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "adapter": {
            "status": adapter_status,
            "repository": cfg.adapter_repo,
            "source": {
                "adapter": source.name,
                "weights_sha256": source.sha256,
                "selected_step": source.step,
            },
            "files": listing(cfg.adapter_dir),
        },
        "merged": {
            "status": facts.merged_status,
            "repository": cfg.merged_repo,
            "withheld_because": withheld_for,
            "weights_sha256": (facts.merged_weights or {}).get("sha256"),
            "merge": (facts.merged_meta or {}).get("merge"),
            "files": _merged_listing(cfg, facts) if merged_built else [],
        },
        "split": cfg.split,
        "noise_examples": facts.noise,
        "allowance_examples": allowance,
        "recorded": {
            "what": "arm A1 on dev, in the job that scored the checkpoints",
            "examples": recorded["examples"],
            "execution_accuracy": recorded["execution_accuracy"],
        },
        "runs": {label: facts.runs.get(label) for label in RUN_NAMES},
        "comparisons": {label: facts.comparisons.get(label) for label in ("R2", "R3", "R4")},
        "card_code": {
            label: _card_code_entry(facts.card_runs.get(label)) for label in CARD_RUNS
        },
        "licences": [entry.to_dict() for entry in cfg.licences],
        "records": {
            "freeze_sha256": records.six_arm.get("freeze_sha256"),
            "prompt_id": facts.prompt.prompt_id,
            "split_manifest_hash": facts.manifest["content_hash"],
            "base_model": {"id": facts.budget.model_id, "commit": facts.budget.revision},
        },
    }
    return State(record=record, cards=cards, facts=facts)


def _but_the_card(problems: list[str]) -> list[str]:
    """Contents problems of a package other than its card not being written yet."""
    return [problem for problem in problems if problem != f"{CARD_NAME} is missing"]


def _written(committed: Path, packaged: Path, text: str, what: str) -> list[str]:
    """Whether the card on disk, in both places, is the card as built now."""
    problems = []
    for path, where in ((committed, "the committed card"), (packaged, "the card in the package")):
        if not path.is_file():
            problems.append(f"{where} ({what}) is not written: {COMMAND} cards")
        elif read_text_lf(path) != text:
            problems.append(f"{where} ({what}) is out of date: {COMMAND} cards")
    return problems


def _merged_listing(cfg: PackagingConfig, facts: Facts) -> list[dict[str, Any]]:
    """The merged folder's files; the weight files' hashes are not computed twice."""
    known = {entry["name"]: entry for entry in (facts.merged_weights or {}).get("files") or []}
    entries = []
    for path in sorted(cfg.merged_dir.iterdir()):
        if not path.is_file():
            continue
        entries.append(
            known.get(path.name)
            or {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return entries


def _card_code_entry(run: dict[str, Any] | None) -> dict[str, Any] | None:
    if run is None:
        return None
    return {
        "code_sha256": run.get("code_sha256"),
        "problems": run.get("problems") or [],
        "program_ran": run.get("program_ran"),
        "in_the_format": run.get("in_the_format"),
        "value": run.get("value"),
    }


def _p8_detail(facts: Facts, recorded: dict[str, Any]) -> str:
    r1, r2 = facts.runs.get("R1"), facts.runs.get("R2")
    if r1 is None or r2 is None:
        return ""
    return (
        f"R2 gives the same reply as R1 on all {r2['examples']:,} examples; R1 has "
        f"{r1['counts']['execution_accuracy']} right, the recorded run "
        f"{recorded['execution_accuracy']}; the adapter card's code ran to the sample's "
        "expected result"
    )


def _p9_detail(facts: Facts) -> str:
    parts = []
    for label in ("R3", "R4"):
        difference = facts.comparisons.get(label)
        if difference is not None:
            parts.append(
                f"{label} has {difference['right_other']} right against R1's "
                f"{difference['right_reference']}"
            )
    return "; ".join(parts + ["the merged card's code ran both ways to the expected result"])


# --------------------------------------------------------------------------- writing


def render_markdown(record: dict[str, Any]) -> str:
    """The record for reading. Counts, ids, and hashes only."""
    lines = [
        "# The packages",
        "",
        f"Written by `{record['generated_by']}`. Do not edit by hand: build again. This file "
        "holds counts, ids, and hashes only: no reply and no dataset text.",
        "",
        f"- **Adapter package** (`{record['adapter']['repository']}`): "
        f"**{record['adapter']['status']}**",
        f"- **Merged copy** (`{record['merged']['repository']}`): "
        f"**{record['merged']['status']}**"
        + (
            f" ({'; '.join(record['merged']['withheld_because'])})"
            if record["merged"]["withheld_because"]
            else ""
        ),
        "",
        "Nothing is uploaded by this version. Publishing is the owner's step (req_006 v6.03).",
        "",
        "## Checks",
        "",
        table(
            ["Check", "", "Result", "Detail"],
            [
                [check["id"], check["name"], "passed" if check["passed"] else "**FAILED**",
                 check["detail"]]
                for check in record["checks"]
            ],
        ),
        "",
        f"## The runs on `{record['split']}`",
        "",
        "Each is an ordinary evaluation run on this machine. None is an arm of the "
        "comparison and none has a price.",
        "",
    ]
    rows = []
    for label, what in RUN_NAMES.items():
        run = record["runs"].get(label)
        if run is None:
            rows.append([label, what, "—", "—", "—", "—", "not run"])
            continue
        counts = run["counts"]
        rows.append(
            [
                label,
                what,
                f"{counts['execution_accuracy']:,} of {run['examples']:,}",
                f"{counts['answer_accuracy']:,}",
                f"{counts['program_match']:,}",
                f"{counts['format_validity']:,}",
                f"`{run['run_id']}`",
            ]
        )
    lines += [
        table(["", "What", "Execution", "Answer", "Program", "Format", "Run"], rows,
              right=(2, 3, 4, 5)),
        "",
        f"The recorded run of arm A1 on `{record['split']}`: "
        f"{record['recorded']['execution_accuracy']:,} of {record['recorded']['examples']:,} "
        f"right on execution. Two runs closer than {record['noise_examples']} examples are "
        "not told apart.",
        "",
        "Against R1, reply by reply:",
        "",
    ]
    rows = []
    for label in ("R2", "R3", "R4"):
        difference = record["comparisons"].get(label)
        if difference is None:
            rows.append([label, "—", "—", "—", "—"])
            continue
        gap = difference["right_other"] - difference["right_reference"]
        rows.append(
            [
                label,
                f"{difference['replies_that_differ']:,}",
                f"{gap:+d}" if gap else "0",
                f"{difference['right_to_wrong']:,}",
                f"{difference['wrong_to_right']:,}",
            ]
        )
    lines += [
        table(
            ["", "Replies that differ", "Right, against R1", "Right in R1, wrong here",
             "Wrong in R1, right here"],
            rows,
            right=(1, 2, 3, 4),
        ),
        "",
        f"The merged copy is withheld when R3 or R4 has {record['allowance_examples']} or "
        "more examples fewer right than R1.",
        "",
        "## The cards' code",
        "",
        "The code each card prints, run in an empty directory on the invented sample:",
        "",
    ]
    rows = []
    for label, entry in record["card_code"].items():
        if entry is None:
            rows.append([label, "not run", "—", "—"])
            continue
        result = "; ".join(entry["problems"]) if entry["problems"] else "ran to the expected result"
        rows.append(
            [label, result, "—" if entry["value"] is None else repr(entry["value"]),
             f"`{(entry['code_sha256'] or '')[:12]}...`"]
        )
    lines += [table(["", "Result", "Executed value", "Code"], rows), ""]
    for key, title in (("adapter", "The adapter package"), ("merged", "The merged copy")):
        entries = record[key]["files"]
        lines += [f"## {title}: files", ""]
        if not entries:
            lines += ["Not built.", ""]
            continue
        lines += [
            table(
                ["File", "Bytes", "SHA-256"],
                [[f"`{e['name']}`", f"{e['bytes']:,}", f"`{e['sha256']}`"] for e in entries],
                right=(1,),
            ),
            "",
        ]
    lines += [
        "## Licences",
        "",
        table(
            ["What", "Licence", "Source", "Read on", "Confirmed by the owner"],
            [
                [
                    f"{entry['name']}: {entry['role']}",
                    f"`{entry['identifier']}`",
                    entry["source"],
                    entry["read_on"],
                    entry["confirmed_on"] or "**not yet**",
                ]
                for entry in record["licences"]
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def write_record(cfg: PackagingConfig, record: dict[str, Any]) -> list[Path]:
    """Write the record's two files. Returns their paths."""
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    write_json_lf(cfg.record_json, record)
    write_text_lf(cfg.record_markdown, render_markdown(record))
    return [cfg.record_json, cfg.record_markdown]
