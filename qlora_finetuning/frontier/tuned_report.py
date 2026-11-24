"""The tuned Nova models beside their untuned baselines, with a verdict (req_004 v4.04).

Takes the summaries and prediction lines of the tuned runs (arms B1) and of the baseline
runs (arms B0), pairs each tuned run with the baseline run of the same model and split,
and reports per pair: both scores and their difference in points, how many questions
tuning fixed and how many it broke with an exact sign test, and the cost per example and
per correct answer. Per model it gives a verdict from the verdict split only (dev):
**met** when execution accuracy rose by the configured number of points and format
validity reached the configured share, otherwise **missed**. A miss makes v4.05 eligible;
nothing here starts anything.

Counts, example ids, tokens, and dollars only: no question, no reply, no program text.

Usage::

    from qlora_finetuning.frontier.tuned_report import build_report, write_report

    report = build_report(tuned, baselines, lines, record, gate, ledger_records, cfg, pricing,
                          prompt_id=prompt.prompt_id, manifest_hash=split_hash)
    if report["passed"]:
        write_report(report, cfg.report_dir)
    report["models"][0]["verdict"]["verdict"]        # "met" or "missed"

``scripts/nova_tuned.py`` does exactly that for run ids given on the command line. This
module calls no AWS service and reads no file.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from decimal import ROUND_CEILING, Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from qlora_finetuning.evaluation.metrics import METRICS
from qlora_finetuning.frontier.config import FrontierConfig
from qlora_finetuning.frontier.export_validation import check
from qlora_finetuning.frontier.pricing import MILLION, Pricing, inference_cost
from qlora_finetuning.frontier.tuned import TUNED_SUFFIX, export_problems
from qlora_finetuning.frontier.tuning_record import COMPLETED, Attempt, TuningRecord

SCHEMA = 1
REPORT_STEM = "nova_tuned_report"
REPORT_COMMAND = "python -m scripts.nova_tuned"
EVAL_PHASE, TRAIN_PHASE = "eval", "train"
USAGE_SOURCE = "aws_api"
MET, MISSED = "met", "missed"
HUNDRED = Decimal(100)
_POINTS = Decimal("0.01")
_DOLLARS = Decimal("0.000001")


# --------------------------------------------------------------------------- the arithmetic


def sign_test(fixed: int, broken: int) -> float | None:
    """The exact two-sided sign test on the questions whose result changed.

    Under "tuning changes nothing", a changed question is as likely fixed as broken. The
    value is the chance of a split at least this uneven. ``None`` when no question changed.
    """
    changed = fixed + broken
    if changed == 0:
        return None
    tail = sum(math.comb(changed, k) for k in range(min(fixed, broken) + 1))
    value = min(Fraction(1), Fraction(2 * tail, 2**changed))
    return float(f"{float(value):.6g}")


def paired(b0_lines: Sequence[Any], b1_lines: Sequence[Any]) -> dict[str, Any]:
    """Question by question on execution accuracy. The two runs must hold the same
    examples in the same order; ``build_report`` checks that first."""
    counts = {"both_right": 0, "both_wrong": 0, "fixed": 0, "broken": 0}
    fixed_ids, broken_ids = [], []
    for before, after in zip(b0_lines, b1_lines, strict=True):
        was = bool(before.score.get("execution_correct"))
        now = bool(after.score.get("execution_correct"))
        if was and now:
            counts["both_right"] += 1
        elif not was and not now:
            counts["both_wrong"] += 1
        elif now:
            counts["fixed"] += 1
            fixed_ids.append(after.example_id)
        else:
            counts["broken"] += 1
            broken_ids.append(after.example_id)
    return {
        **counts,
        "sign_test_p_value": sign_test(counts["fixed"], counts["broken"]),
        "broke_more_than_a_tenth_of_fixed": counts["broken"] * 10 > counts["fixed"],
        "fixed_ids": fixed_ids,
        "broken_ids": broken_ids,
    }


def _percent(count: int, total: int) -> str | None:
    if not total:
        return None
    return str((Decimal(count) * HUNDRED / total).quantize(_POINTS))


def _points(after: int, before: int, total: int) -> str | None:
    if not total:
        return None
    return str((Decimal(after - before) * HUNDRED / total).quantize(_POINTS))


def _needed(share: Decimal, total: int) -> int:
    """The smallest count that is at least ``share`` percent of ``total``."""
    return int((share * total / HUNDRED).to_integral_value(rounding=ROUND_CEILING))


def verdict(
    b0_execution: int, b1_execution: int, b1_format: int, examples: int, cfg: FrontierConfig
) -> dict[str, Any]:
    """Met or missed, from counts. A rounded percentage never decides."""
    gain, share = cfg.min_execution_gain_points, cfg.min_format_validity_percent
    gain_met = bool(examples) and Decimal(b1_execution - b0_execution) * HUNDRED >= (
        gain * examples
    )
    format_met = bool(examples) and Decimal(b1_format) * HUNDRED >= share * examples
    needed_execution = b0_execution + _needed(gain, examples)
    needed_format = _needed(share, examples)
    reasons = []
    if not examples:
        reasons.append("the run has no examples")
    else:
        if not gain_met:
            reasons.append(
                f"execution accuracy rose by {_points(b1_execution, b0_execution, examples)} "
                f"points; {gain.normalize():f} are needed ({needed_execution:,} of "
                f"{examples:,} correct, {b1_execution:,} were)"
            )
        if not format_met:
            reasons.append(
                f"format validity is {_percent(b1_format, examples)}%; {share.normalize():f}% "
                f"is needed ({needed_format:,} of {examples:,} well-formed, {b1_format:,} were)"
            )
    met = gain_met and format_met
    return {
        "verdict": MET if met else MISSED,
        "execution_gain_points": _points(b1_execution, b0_execution, examples),
        "execution_gain_met": gain_met,
        "needed_execution_count": needed_execution,
        "format_validity_percent": _percent(b1_format, examples),
        "format_validity_met": format_met,
        "needed_format_count": needed_format,
        "reasons": reasons,
        "corrective_iteration_eligible": not met,
    }


# --------------------------------------------------------------------------- reading a run


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _predictor(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    return summary.get("predictor") or {}


def model_of(summary: Mapping[str, Any]) -> str:
    """The configured model a run belongs to: ``nova-lite`` for both its B0 and its B1."""
    predictor = _predictor(summary)
    details = predictor.get("details") or {}
    if details.get("model"):
        return str(details["model"])
    return str(predictor.get("name") or "").removesuffix(TUNED_SUFFIX)


def _arm_figures(
    summary: Mapping[str, Any], price_model_id: str | None, pricing: Pricing
) -> dict[str, Any]:
    totals, usage = summary["all"], summary["usage"]
    examples = totals["examples"]
    correct = totals["execution_accuracy"]["count"]
    input_tokens, output_tokens = usage["input_tokens"], usage["output_tokens"]
    cost = per_example = per_correct = None
    if input_tokens is not None and output_tokens is not None and price_model_id in pricing.models:
        dollars = inference_cost(price_model_id, input_tokens, output_tokens, pricing)
        cost = str(dollars.quantize(_DOLLARS))
        if examples:
            per_example = str((dollars / examples).quantize(_DOLLARS))
        if correct:
            per_correct = str((dollars / correct).quantize(_DOLLARS))
    return {
        "arm": summary["arm"],
        "run_id": summary["run_id"],
        "git_commit": summary["git_commit"],
        "model_id": _predictor(summary).get("model_id"),
        "priced_as": price_model_id,
        "examples": examples,
        **{name: totals[name]["count"] for name in METRICS},
        "stop_reasons": summary["stop_reasons"],
        "error_codes": totals["error_codes"],
        "deviations": totals["deviations"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
        "cost_per_example_usd": per_example,
        "cost_per_correct_answer_usd": per_correct,
        "latency_seconds": summary["latency_seconds"],
    }


def _ids(lines: Sequence[Any]) -> list[str]:
    return [line.example_id for line in lines]


def training_cost(
    attempt: Attempt, ledger_records: Sequence[Any], pricing: Pricing
) -> dict[str, Any] | None:
    """What the model's fine-tuning job cost: the billed amount if the owner entered it,
    otherwise the ledger's estimated tokens at list price. Labelled which."""
    records = [
        r for r in ledger_records if r.phase == TRAIN_PHASE and r.external_id == attempt.job_arn
    ]
    for record in records:
        if record.billed_amount_usd is not None:
            amount = Decimal(str(record.billed_amount_usd)).quantize(_POINTS)
            return {"usd": str(amount), "basis": "billed"}
    for record in records:
        if record.training_tokens and record.model_id in pricing.models:
            price = pricing.model(record.model_id).training_per_million
            dollars = (record.training_tokens * price / MILLION).quantize(_POINTS)
            return {
                "usd": str(dollars),
                "basis": "estimate",
                "training_tokens": record.training_tokens,
            }
    return None


# --------------------------------------------------------------------------- the checks


def _pairs(
    tuned: Sequence[Mapping[str, Any]], baselines: Sequence[Mapping[str, Any]]
) -> tuple[list[tuple[Mapping[str, Any], Mapping[str, Any]]], list[str]]:
    """Each tuned run with the one baseline run of its model and split, and what is wrong."""
    pairs, problems, used = [], [], set()
    seen: set[tuple[str, str]] = set()
    for run in tuned:
        key = (model_of(run), run["split"])
        if key in seen:
            problems.append(f"more than one tuned run for {key[0]} on {key[1]}")
            continue
        seen.add(key)
        matches = [b for b in baselines if (model_of(b), b["split"]) == key]
        if len(matches) != 1:
            problems.append(
                f"tuned run {run['run_id']} ({key[0]} on {key[1]}) has {len(matches)} "
                "baseline runs, expected 1"
            )
            continue
        used.add(matches[0]["run_id"])
        pairs.append((matches[0], run))
    problems += [
        f"baseline run {b['run_id']} ({model_of(b)} on {b['split']}) pairs with no tuned run"
        for b in baselines
        if b["run_id"] not in used
    ]
    return pairs, problems


def _pair_problems(
    b0: Mapping[str, Any],
    b1: Mapping[str, Any],
    lines: Mapping[str, Sequence[Any]],
    cfg: FrontierConfig,
    prompt_id: str,
    manifest_hash: str,
) -> list[str]:
    name, split = model_of(b1), b1["split"]
    what = f"{name} on {split}"
    model = cfg.models.get(name)
    if model is None:
        return [f"tuned run {b1['run_id']} is of {name!r}, which is not a configured model"]
    problems = []
    if split not in cfg.tuned_splits:
        problems.append(f"{what}: {split} is not a split of tuned_evaluation.splits")
    if b0["arm"] != model.untuned_arm:
        problems.append(f"{what}: baseline run is arm {b0['arm']}, expected {model.untuned_arm}")
    if b1["arm"] != model.tuned_arm:
        problems.append(f"{what}: tuned run is arm {b1['arm']}, expected {model.tuned_arm}")
    if _predictor(b0).get("model_id") != model.model_id:
        problems.append(f"{what}: the baseline run did not ask {model.model_id}")
    for key, label, current in (
        ("prompt_id", "prompt wording", prompt_id),
        ("manifest_hash", "split manifest", manifest_hash),
    ):
        if not (b0[key] == b1[key] == current):
            problems.append(f"{what}: the two runs and today's {label} are not the same")
    if b0["max_new_tokens"] != b1["max_new_tokens"]:
        problems.append(f"{what}: the reply limits differ")
    if _plain(_predictor(b0).get("settings")) != _plain(_predictor(b1).get("settings")):
        problems.append(f"{what}: the two runs were made under different settings")
    if _ids(lines[b0["run_id"]]) != _ids(lines[b1["run_id"]]):
        problems.append(f"{what}: the two runs do not hold the same examples in the same order")
    return problems


def _checks(
    tuned: Sequence[Mapping[str, Any]],
    baselines: Sequence[Mapping[str, Any]],
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    pairing_problems: list[str],
    lines: Mapping[str, Sequence[Any]],
    record: TuningRecord,
    gate: Mapping[str, Any] | None,
    ledger_records: Sequence[Any],
    cfg: FrontierConfig,
    prompt_id: str,
    manifest_hash: str,
) -> list[dict[str, Any]]:
    every = [*tuned, *baselines]
    complete = [f"run {s['run_id']} is not a result" for s in every if not s["result"]]

    paired_problems = list(pairing_problems)
    for b0, b1 in pairs:
        paired_problems += _pair_problems(b0, b1, lines, cfg, prompt_id, manifest_hash)

    right_model, right_files = [], []
    for name in dict.fromkeys(model_of(run) for run in tuned):
        attempt = record.latest(name)
        if attempt is None:
            right_model.append(f"the tuning record has no attempt for {name}")
            continue
        if attempt.job_status != COMPLETED:
            right_model.append(f"the latest job of {name} is {attempt.job_status}, not {COMPLETED}")
        for run in tuned:
            if model_of(run) == name and _predictor(run).get("model_id") != (
                attempt.deployment_arn
            ):
                right_model.append(
                    f"run {run['run_id']} did not ask the deployment of {name}'s latest "
                    f"attempt ({attempt.attempt})"
                )
        right_files += export_problems(attempt, gate, prompt_id, manifest_hash)
    right_files = list(dict.fromkeys(right_files))

    usage = []
    for summary in every:
        run_id = summary["run_id"]
        model = cfg.models.get(model_of(summary))
        if summary["usage"]["predictions_without_token_counts"] or (
            summary["usage"]["input_tokens"] is None
        ):
            usage.append(f"run {run_id} has predictions without token counts")
        records = [r for r in ledger_records if r.run_id == run_id and r.phase == EVAL_PHASE]
        if not records:
            usage.append(f"run {run_id} has no ledger record")
        elif any(r.source != USAGE_SOURCE or r.input_tokens is None for r in records):
            usage.append(f"run {run_id} has a ledger record that is not {USAGE_SOURCE} tokens")
        elif model is not None and any(r.model_id != model.model_id for r in records):
            usage.append(f"run {run_id} is not recorded under {model.model_id}")

    scored = list(dict.fromkeys(model_of(run) for run in tuned))
    with_verdict = {model_of(b1) for _, b1 in pairs if b1["split"] == cfg.verdict_split}
    present = [
        f"{name} has no tuned and baseline pair on {cfg.verdict_split}"
        for name in scored
        if name not in with_verdict
    ]
    if not scored:
        present.append("no tuned run was given")

    return [
        check("V1", "Complete", complete, f"all {len(every)} runs are complete results"),
        check(
            "V2",
            "Paired",
            paired_problems,
            f"{len(pairs)} tuned runs, each with its baseline: same prompt, split, settings, "
            "reply limit, and examples in the same order",
        ),
        check(
            "V3",
            "The right model",
            right_model,
            "every tuned run asked the deployment of its model's latest, completed attempt",
        ),
        check(
            "V4",
            "The right files",
            right_files,
            f"tuned on export {(gate or {}).get('export_id')}, made for today's prompt and split",
        ),
        check(
            "V5",
            "Usage adds up",
            usage,
            "every run has Bedrock's token counts, recorded under the base model",
        ),
        check(
            "V6",
            f"{cfg.verdict_split} present",
            present,
            f"a {cfg.verdict_split} pair for: {', '.join(scored)}",
        ),
    ]


# --------------------------------------------------------------------------- the report


def _pair_entry(
    b0: Mapping[str, Any],
    b1: Mapping[str, Any],
    lines: Mapping[str, Sequence[Any]],
    cfg: FrontierConfig,
    pricing: Pricing,
) -> dict[str, Any]:
    base_id = cfg.model(model_of(b1)).model_id
    baseline = _arm_figures(b0, base_id, pricing)
    tuned = _arm_figures(b1, base_id, pricing)
    examples = tuned["examples"]
    b1_lines = lines[b1["run_id"]]
    return {
        "split": b1["split"],
        "examples": examples,
        "baseline": baseline,
        "tuned": tuned,
        "difference_points": {
            name: _points(tuned[name], baseline[name], examples) for name in METRICS
        },
        "paired": paired(lines[b0["run_id"]], b1_lines),
        "tuned_format_failure_ids": [
            line.example_id for line in b1_lines if not line.score.get("format_valid")
        ],
    }


def _model_entry(
    name: str,
    entries: Sequence[dict[str, Any]],
    record: TuningRecord,
    ledger_records: Sequence[Any],
    cfg: FrontierConfig,
    pricing: Pricing,
) -> dict[str, Any]:
    model, attempt = cfg.model(name), record.latest(name)
    deciding = next(entry for entry in entries if entry["split"] == cfg.verdict_split)
    return {
        "model": name,
        "untuned_arm": model.untuned_arm,
        "tuned_arm": model.tuned_arm,
        "verdict": verdict(
            deciding["baseline"]["execution_accuracy"],
            deciding["tuned"]["execution_accuracy"],
            deciding["tuned"]["format_validity"],
            deciding["examples"],
            cfg,
        ),
        "tuned_model": {
            "attempt": attempt.attempt,
            "base_model_id": attempt.base_model_id,
            "custom_model_name": attempt.custom_model_name,
            "job_arn": attempt.job_arn,
            "model_arn": attempt.model_arn,
            "deployment_arn": attempt.deployment_arn,
            "export_id": attempt.export_id,
            "hyperparameters": dict(attempt.hyperparameters),
            "training_loss": attempt.training_loss,
            "validation_loss": attempt.validation_loss,
        },
        "training_cost": training_cost(attempt, ledger_records, pricing),
        "pairs": sorted(entries, key=lambda entry: cfg.tuned_splits.index(entry["split"])),
    }


def build_report(
    tuned: Sequence[Mapping[str, Any]],
    baselines: Sequence[Mapping[str, Any]],
    lines: Mapping[str, Sequence[Any]],
    record: TuningRecord,
    gate: Mapping[str, Any] | None,
    ledger_records: Sequence[Any],
    cfg: FrontierConfig,
    pricing: Pricing,
    *,
    prompt_id: str,
    manifest_hash: str,
) -> dict[str, Any]:
    """The report as plain values: inputs, one entry per model, checks V1 to V6.

    ``tuned`` and ``baselines`` are summaries from ``summarize_run``; ``lines`` maps each
    run id to its prediction lines; ``ledger_records`` are the ledger's effective records.
    ``passed`` says the checks passed. ``models`` is filled only then. ``all_met`` says
    every scored model met its threshold; ``partial`` that a configured model has no run.
    """
    pairs, pairing_problems = _pairs(tuned, baselines)
    checks = _checks(
        tuned, baselines, pairs, pairing_problems, lines, record, gate, ledger_records, cfg,
        prompt_id, manifest_hash,
    )  # fmt: skip
    passed = all(item["passed"] for item in checks)
    models: list[dict[str, Any]] = []
    if passed:
        by_model: dict[str, list[dict[str, Any]]] = {}
        for b0, b1 in pairs:
            entry = _pair_entry(b0, b1, lines, cfg, pricing)
            by_model.setdefault(model_of(b1), []).append(entry)
        models = [
            _model_entry(name, by_model[name], record, ledger_records, cfg, pricing)
            for name in cfg.models
            if name in by_model
        ]
    missing = [name for name in cfg.models if name not in {m["model"] for m in models}]
    return {
        "schema": SCHEMA,
        "generated_by": REPORT_COMMAND,
        "inputs": {
            "prompt_id": prompt_id,
            "split_content_hash": manifest_hash,
            "export_id": (gate or {}).get("export_id"),
            "settings": _plain(cfg.settings),
            "verdict_split": cfg.verdict_split,
            "min_execution_gain_points": f"{cfg.min_execution_gain_points.normalize():f}",
            "min_format_validity_percent": f"{cfg.min_format_validity_percent.normalize():f}",
            "price_list": {
                "offer": pricing.offer,
                "region": pricing.region,
                "published": pricing.published,
                "read_on": pricing.read_on,
            },
        },
        "models": models,
        "checks": checks,
        "passed": passed,
        "partial": passed and bool(missing),
        "models_without_runs": missing if passed else [],
        "all_met": passed and all(m["verdict"]["verdict"] == MET for m in models),
    }


# --------------------------------------------------------------------------- rendering


def _share(count: int, total: int) -> str:
    percent = _percent(count, total)
    return f"{count:,}" if percent is None else f"{count:,} ({percent}%)"


def _signed(points: str | None) -> str:
    if points is None:
        return "-"
    return points if points.startswith("-") else f"+{points}"


def _number(value: Any) -> str:
    return "-" if value is None else f"{value:,}"


def _dollars(value: str | None, places: int = 4) -> str:
    return "-" if value is None else f"${Decimal(value):.{places}f}"


def _counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"`{name}` {count:,}" for name, count in counts.items()) or "none"


def _seconds(value: Any) -> str:
    return "-" if value is None else f"{value:.2f}"


def _p_value(value: float | None) -> str:
    if value is None:
        return "no question changed"
    return "< 0.000001" if value < 0.000001 else f"{value:.6f}"


def _loss(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def render_markdown(report: dict[str, Any]) -> str:
    inputs, models, checks = report["inputs"], report["models"], report["checks"]
    prices = inputs["price_list"]
    split = inputs["verdict_split"]
    passed = sum(item["passed"] for item in checks)
    lines = [
        "# Nova tuned-model report",
        "",
        f"Generated by `{REPORT_COMMAND}` from the runs of the fine-tuned hosted models and",
        "of their untuned baselines. Do not edit by hand.",
        "",
        "Counts, example ids, tokens, and dollars only: no question, no reply, no program.",
        "The runs are development evidence; the numbers of the comparison come from the test",
        "split in req_005. Cost is tokens times list price, at the base model's prices.",
        "",
        "## Verdict",
        "",
        f"From {split} only. A tuned model has met its threshold when its execution accuracy",
        f"is at least {inputs['min_execution_gain_points']} points above its own baseline's",
        f"and its format validity is at least {inputs['min_format_validity_percent']}%.",
        "",
        "| Model | Execution, untuned | Execution, tuned | Gain in points | Correct needed "
        "| Format validity, tuned | Well-formed needed | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model in models:
        pair = next(p for p in model["pairs"] if p["split"] == split)
        result, total = model["verdict"], pair["examples"]
        lines.append(
            f"| {model['model']} | {_share(pair['baseline']['execution_accuracy'], total)} "
            f"| {_share(pair['tuned']['execution_accuracy'], total)} "
            f"| {_signed(result['execution_gain_points'])} "
            f"| {result['needed_execution_count']:,} "
            f"| {_share(pair['tuned']['format_validity'], total)} "
            f"| {result['needed_format_count']:,} | **{result['verdict'].upper()}** |"
        )
    lines.append("")
    for model in models:
        for reason in model["verdict"]["reasons"]:
            lines.append(f"- {model['model']}: {reason}.")
        if model["verdict"]["corrective_iteration_eligible"]:
            lines.append(
                f"- {model['model']}: v4.05 (one corrective iteration) is eligible. Nothing "
                "starts by itself."
            )
    if report["partial"]:
        lines.append(
            f"- **Partial report**: no tuned run was given for "
            f"{', '.join(report['models_without_runs'])}. Milestone M6 needs every model."
        )
    lines += [
        "",
        "## Inputs",
        "",
        f"- Prompt id: `{inputs['prompt_id']}`",
        f"- Split content hash: `{inputs['split_content_hash']}`",
        f"- Export id of the training files: `{inputs['export_id']}`",
        f"- Settings: `{json.dumps(inputs['settings'], sort_keys=True)}`",
        f"- Prices: {prices['offer']} price list for {prices['region']}, published "
        f"{prices['published']}, read {prices['read_on']}",
        "",
        "## Checks",
        "",
        f"{passed} of {len(checks)} passed.",
        "",
        "| Id | Check | Result | Detail |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {c['id']} | {c['name']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |"
        for c in checks
    ]
    lines += [
        "",
        "## Scores",
        "",
        "Execution accuracy is the headline. The difference is tuned minus untuned, in points.",
        "",
        "| Model | Split | Arm | Examples | Format valid | Execution | Answer | Program match |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        for pair in model["pairs"]:
            total = pair["examples"]
            for side in ("baseline", "tuned"):
                arm = pair[side]
                cells = " | ".join(_share(arm[name], total) for name in METRICS)
                lines.append(
                    f"| {model['model']} | {pair['split']} | {arm['arm']} | {total:,} | {cells} |"
                )
            cells = " | ".join(_signed(pair["difference_points"][name]) for name in METRICS)
            lines.append(f"| {model['model']} | {pair['split']} | difference | | {cells} |")
    lines += [
        "",
        "## Question by question",
        "",
        "On execution accuracy. The sign test asks how likely a split this uneven between",
        "fixed and broken would be if tuning changed nothing. The example ids are in the",
        "JSON file.",
        "",
        "| Model | Split | Right in both | Wrong in both | Fixed by tuning | Broken by tuning "
        "| Sign test p | Broke more than a tenth of what it fixed |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for model in models:
        for pair in model["pairs"]:
            both = pair["paired"]
            lines.append(
                f"| {model['model']} | {pair['split']} | {both['both_right']:,} "
                f"| {both['both_wrong']:,} | {both['fixed']:,} | {both['broken']:,} "
                f"| {_p_value(both['sign_test_p_value'])} "
                f"| {'yes' if both['broke_more_than_a_tenth_of_fixed'] else 'no'} |"
            )
    lines += [
        "",
        "## Cost of asking",
        "",
        "| Model | Split | Arm | Input tokens | Output tokens | Cost | Per example "
        "| Per correct answer |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        for pair in model["pairs"]:
            for side in ("baseline", "tuned"):
                arm = pair[side]
                lines.append(
                    f"| {model['model']} | {pair['split']} | {arm['arm']} "
                    f"| {_number(arm['input_tokens'])} | {_number(arm['output_tokens'])} "
                    f"| {_dollars(arm['cost_usd'])} | {_dollars(arm['cost_per_example_usd'], 6)} "
                    f"| {_dollars(arm['cost_per_correct_answer_usd'], 6)} |"
                )
    lines += [
        "",
        "## Cost of tuning",
        "",
        "Paid once per model, so it is shown beside the cost of asking and not spread over",
        "the examples. Storage of a tuned model is billed monthly on top.",
        "",
        "| Model | Training cost | Basis |",
        "|---|---:|---|",
    ]
    for model in models:
        cost = model["training_cost"]
        basis = {
            None: "not in the ledger",
            "billed": "the AWS bill, entered by the owner",
            "estimate": "estimated tokens at list price; the bill has not been entered",
        }[cost["basis"] if cost else None]
        lines.append(
            f"| {model['model']} | {_dollars(cost['usd'], 2) if cost else '-'} | {basis} |"
        )
    lines += [
        "",
        "## Replies of the tuned models",
        "",
        "Latency is the caller's clock around each call, in seconds, waits included.",
        "",
        "| Model | Split | Stop reasons | Error codes | Format deviations | Not well-formed "
        "| Latency mean | median | 95th percentile |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for model in models:
        for pair in model["pairs"]:
            arm, latency = pair["tuned"], pair["tuned"]["latency_seconds"]
            lines.append(
                f"| {model['model']} | {pair['split']} | {_counts(arm['stop_reasons'])} "
                f"| {_counts(arm['error_codes'])} | {_counts(arm['deviations'])} "
                f"| {len(pair['tuned_format_failure_ids']):,} | {_seconds(latency['mean'])} "
                f"| {_seconds(latency['median'])} | {_seconds(latency['p95'])} |"
            )
    lines += ["", "## The tuned models", ""]
    for model in models:
        tuned = model["tuned_model"]
        lines += [
            f"### {model['model']}, attempt {tuned['attempt']}",
            "",
            f"- Tuned model: `{tuned['model_arn']}` (name `{tuned['custom_model_name']}`)",
            f"- Deployment asked: `{tuned['deployment_arn']}`",
            f"- Job: `{tuned['job_arn']}`",
            f"- Base model, and what the calls are billed as: `{tuned['base_model_id']}`",
            f"- Hyperparameters: `{json.dumps(tuned['hyperparameters'], sort_keys=True)}`",
            f"- Final training loss {_loss(tuned['training_loss'])}, validation loss "
            f"{_loss(tuned['validation_loss'])}",
            "",
        ]
    lines += ["## Runs", "", "| Model | Split | Arm | Run | Commit |", "|---|---|---|---|---|"]
    for model in models:
        for pair in model["pairs"]:
            for side in ("baseline", "tuned"):
                arm = pair[side]
                lines.append(
                    f"| {model['model']} | {pair['split']} | {arm['arm']} | `{arm['run_id']}` "
                    f"| `{str(arm['git_commit'])[:12]}` |"
                )
    return "\n".join(lines) + "\n"


def render_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def report_paths(report_dir: Path | str) -> tuple[Path, Path]:
    report_dir = Path(report_dir)
    return report_dir / f"{REPORT_STEM}.md", report_dir / f"{REPORT_STEM}.json"


def write_report(report: dict[str, Any], report_dir: Path | str) -> tuple[Path, Path]:
    """Write the Markdown and JSON files and return their paths."""
    markdown_path, json_path = report_paths(report_dir)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    json_path.write_text(render_json(report), encoding="utf-8", newline="\n")
    return markdown_path, json_path
