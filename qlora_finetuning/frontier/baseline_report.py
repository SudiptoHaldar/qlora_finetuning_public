"""The untuned Nova baselines as one committed report (req_004 v4.01).

Takes the summaries of the baseline runs (each configured model on quick-dev and on dev),
checks that they are what they claim to be, and reports per run: the scores, the tokens
Bedrock counted, how that compares with the Qwen count every estimate has used, the cost
at list price, and the latency. Counts, tokens, and dollars only: no reply and no example
text.

Usage::

    from qlora_finetuning.frontier.baseline_report import build_report, write_report

    report = build_report(summaries, ledger_records, cfg, pricing, qwen_prompt_tokens,
                          prompt_id=prompt.prompt_id, manifest_hash=split_hash)
    if report["passed"]:
        write_report(report, cfg.report_dir)

``scripts/nova_baseline.py`` does exactly that for run ids given on the command line.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from qlora_finetuning.evaluation.metrics import METRICS
from qlora_finetuning.frontier.config import FrontierConfig
from qlora_finetuning.frontier.pricing import Pricing, inference_cost
from qlora_finetuning.settings import ConfigError

SCHEMA = 1
REPORT_STEM = "nova_baseline_report"
REPORT_COMMAND = "python -m scripts.nova_baseline"
SPLITS = ("quick_dev", "dev")
PHASE = "eval"
USAGE_SOURCE = "aws_api"
_RATIO = Decimal("0.0001")
_DOLLARS = Decimal("0.000001")
_THOUSAND = Decimal(1000)


def load_qwen_prompt_tokens(token_budget_json: Path | str) -> dict[str, int]:
    """Prompt tokens per split, Qwen tokenizer, from the committed token budget report."""
    path = Path(token_budget_json)
    try:
        splits = json.loads(path.read_text(encoding="utf-8"))["splits"]
        return {name: int(block["prompt_tokens_in_one_pass"]) for name, block in splits.items()}
    except FileNotFoundError:
        raise ConfigError(
            f"token budget report not found: {path}. Run python -m scripts.finqa_tokens"
        ) from None
    except (ValueError, KeyError, TypeError, AttributeError):
        raise ConfigError(f"token budget report cannot be read: {path}") from None


def _check(check_id: str, name: str, problems: list[str], ok_detail: str) -> dict[str, Any]:
    return {
        "id": check_id,
        "name": name,
        "passed": not problems,
        "detail": "; ".join(problems) if problems else ok_detail,
    }


def _plain(value: Any) -> Any:
    """A value as JSON would hold it, so settings from a file and from config compare equal."""
    return json.loads(json.dumps(value, sort_keys=True))


def _name(summary: Mapping[str, Any]) -> str:
    return str((summary.get("predictor") or {}).get("name"))


def _run_entry(
    summary: Mapping[str, Any],
    cfg: FrontierConfig,
    pricing: Pricing,
    qwen_prompt_tokens: Mapping[str, int],
) -> dict[str, Any]:
    totals, usage = summary["all"], summary["usage"]
    predictor = summary.get("predictor") or {}
    examples = totals["examples"]
    input_tokens, output_tokens = usage["input_tokens"], usage["output_tokens"]
    qwen = qwen_prompt_tokens.get(summary["split"])
    model_id = predictor.get("model_id")

    ratio = cost = per_thousand = None
    if input_tokens is not None and qwen:
        ratio = str((Decimal(input_tokens) / Decimal(qwen)).quantize(_RATIO))
    if input_tokens is not None and output_tokens is not None and model_id in pricing.models:
        dollars = inference_cost(model_id, input_tokens, output_tokens, pricing)
        cost = str(dollars.quantize(_DOLLARS))
        if examples:
            per_thousand = str((dollars / examples * _THOUSAND).quantize(_DOLLARS))
    return {
        "model": _name(summary),
        "model_id": model_id,
        "arm": summary["arm"],
        "split": summary["split"],
        "run_id": summary["run_id"],
        "git_commit": summary["git_commit"],
        "result": bool(summary["result"]),
        "examples": examples,
        **{name: totals[name] for name in METRICS},
        "execution_accuracy_tight": summary["diagnostic"]["execution_accuracy_tight"],
        "error_codes": totals["error_codes"],
        "deviations": totals["deviations"],
        "mismatches": totals["mismatches"],
        "stop_reasons": summary["stop_reasons"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "qwen_prompt_tokens": qwen,
        "nova_to_qwen_input_ratio": ratio,
        "cost_usd": cost,
        "cost_per_1000_questions_usd": per_thousand,
        "latency_seconds": summary["latency_seconds"],
        "in_configured_set": _name(summary) in cfg.models and summary["split"] in SPLITS,
    }


def _order(entry: Mapping[str, Any], cfg: FrontierConfig) -> tuple[int, int, str]:
    names = list(cfg.models)
    split = SPLITS.index(entry["split"]) if entry["split"] in SPLITS else len(SPLITS)
    model = names.index(entry["model"]) if entry["model"] in names else len(names)
    return split, model, entry["run_id"]


def _checks(
    summaries: Sequence[Mapping[str, Any]],
    ledger_records: Sequence[Any],
    cfg: FrontierConfig,
    prompt_id: str,
    manifest_hash: str,
) -> list[dict[str, Any]]:
    # B1: exactly one result per configured model and split, on the model's untuned arm.
    results = []
    for split in SPLITS:
        for name, model in cfg.models.items():
            runs = [s for s in summaries if _name(s) == name and s["split"] == split]
            if len(runs) != 1:
                results.append(f"{name} on {split}: {len(runs)} runs, expected 1")
                continue
            run = runs[0]
            if not run["result"]:
                results.append(f"run {run['run_id']} is not a result")
            if run["arm"] != model.untuned_arm:
                results.append(
                    f"run {run['run_id']} is arm {run['arm']}, expected {model.untuned_arm}"
                )
    results += [
        f"run {s['run_id']} ({_name(s)} on {s['split']}) is not a baseline of a configured model"
        for s in summaries
        if _name(s) not in cfg.models or s["split"] not in SPLITS
    ]

    # B2: every run answered the same question under the same settings.
    question = []
    other_prompt = [s["run_id"] for s in summaries if s["prompt_id"] != prompt_id]
    other_split = [s["run_id"] for s in summaries if s["manifest_hash"] != manifest_hash]
    if other_prompt:
        question.append(f"made with another prompt wording: {', '.join(other_prompt)}")
    if other_split:
        question.append(f"made with another split manifest: {', '.join(other_split)}")
    limits = sorted({str(s["max_new_tokens"]) for s in summaries})
    if len(limits) > 1:
        question.append(f"reply limits differ: {', '.join(limits)}")
    settings = {
        json.dumps(_plain((s.get("predictor") or {}).get("settings")), sort_keys=True)
        for s in summaries
    }
    if len(settings) > 1:
        question.append("the runs were made under different settings")

    # B3: every token count is Bedrock's own.
    usage = []
    for summary in summaries:
        run_id = summary["run_id"]
        if summary["usage"]["predictions_without_token_counts"] or (
            summary["usage"]["input_tokens"] is None
        ):
            usage.append(f"run {run_id} has predictions without token counts")
        records = [r for r in ledger_records if r.run_id == run_id and r.phase == PHASE]
        if not records:
            usage.append(f"run {run_id} has no ledger record")
        elif any(r.source != USAGE_SOURCE or r.input_tokens is None for r in records):
            usage.append(f"run {run_id} has a ledger record that is not {USAGE_SOURCE} tokens")

    # B4: each run used the configured model id and decoding values.
    configured = []
    expected_settings = _plain(cfg.settings)
    for summary in summaries:
        model = cfg.models.get(_name(summary))
        if model is None:
            continue  # reported by B1
        predictor = summary.get("predictor") or {}
        if predictor.get("model_id") != model.model_id:
            configured.append(
                f"run {summary['run_id']} used {predictor.get('model_id')}, "
                f"config says {model.model_id}"
            )
        if _plain(predictor.get("settings")) != expected_settings:
            configured.append(f"run {summary['run_id']} used settings other than the config's")

    expected = len(cfg.models) * len(SPLITS)
    return [
        _check(
            "B1",
            f"{expected} results",
            results,
            f"one result per model ({', '.join(cfg.models)}) and split ({', '.join(SPLITS)})",
        ),
        _check(
            "B2",
            "Same question",
            question,
            f"prompt {prompt_id[:12]}..., split manifest {manifest_hash[:12]}..., "
            f"reply limit {', '.join(limits) or '-'}, one set of settings",
        ),
        _check("B3", "Usage from AWS", usage, "every prediction and ledger record has "
               "Bedrock's token counts"),
        _check("B4", "As configured", configured, "model ids and decoding values are those of "
               "config/frontier.yaml"),
    ]


def build_report(
    summaries: Sequence[Mapping[str, Any]],
    ledger_records: Sequence[Any],
    cfg: FrontierConfig,
    pricing: Pricing,
    qwen_prompt_tokens: Mapping[str, int],
    *,
    prompt_id: str,
    manifest_hash: str,
) -> dict[str, Any]:
    """The report as plain values: inputs, one entry per run, checks B1 to B4, ``passed``.

    ``summaries`` come from ``summarize_run``; ``ledger_records`` are the ledger's effective
    records. Dollar amounts are text, computed with ``Decimal`` from ``pricing``.
    """
    entries = [_run_entry(summary, cfg, pricing, qwen_prompt_tokens) for summary in summaries]
    entries.sort(key=lambda entry: _order(entry, cfg))
    checks = _checks(summaries, ledger_records, cfg, prompt_id, manifest_hash)
    return {
        "schema": SCHEMA,
        "generated_by": REPORT_COMMAND,
        "inputs": {
            "prompt_id": prompt_id,
            "split_content_hash": manifest_hash,
            "settings": _plain(cfg.settings),
            "price_list": {
                "offer": pricing.offer,
                "region": pricing.region,
                "published": pricing.published,
                "read_on": pricing.read_on,
            },
        },
        "runs": entries,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }


# --------------------------------------------------------------------------- rendering


def _share(rate: Mapping[str, Any]) -> str:
    if rate["rate"] is None:
        return f"{rate['count']:,}"
    return f"{rate['count']:,} ({100 * rate['rate']:.1f}%)"


def _number(value: Any) -> str:
    return "-" if value is None else f"{value:,}"


def _dollars(value: str | None) -> str:
    return "-" if value is None else f"${Decimal(value):.4f}"


def _counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"`{name}` {count:,}" for name, count in counts.items()) or "none"


def _seconds(value: Any) -> str:
    return "-" if value is None else f"{value:.2f}"


def render_markdown(report: dict[str, Any]) -> str:
    inputs, runs, checks = report["inputs"], report["runs"], report["checks"]
    prices = inputs["price_list"]
    passed = sum(check["passed"] for check in checks)
    lines = [
        "# Nova zero-shot baseline report",
        "",
        f"Generated by `{REPORT_COMMAND}` from the baseline runs of the untuned hosted",
        "models. Do not edit by hand.",
        "",
        "Counts, tokens, and dollars only: no reply and no example text. The runs are",
        "development evidence on dev and its quick-dev subset; the numbers of the comparison",
        "come from the test split in req_005. Cost is tokens times list price.",
        "",
        "## Inputs",
        "",
        f"- Prompt id: `{inputs['prompt_id']}`",
        f"- Split content hash: `{inputs['split_content_hash']}`",
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
        "Execution accuracy is the headline. The tight rule is a diagnostic.",
        "",
        "| Arm | Split | Examples | Format valid | Execution | Answer | Program match "
        "| Execution, tight rule |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['arm']} | {r['split']} | {r['examples']:,} | {_share(r['format_validity'])} "
        f"| {_share(r['execution_accuracy'])} | {_share(r['answer_accuracy'])} "
        f"| {_share(r['program_match'])} | {_share(r['execution_accuracy_tight'])} |"
        for r in runs
    ]
    lines += [
        "",
        "## Tokens and cost",
        "",
        "Input and output tokens are Bedrock's own counts. The Qwen figure is the same",
        "prompts counted with the Qwen3-4B tokenizer (token budget report).",
        "",
        "| Arm | Split | Input tokens | Output tokens | Qwen prompt tokens | Nova / Qwen "
        "| Cost | Cost per 1,000 questions |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['arm']} | {r['split']} | {_number(r['input_tokens'])} "
        f"| {_number(r['output_tokens'])} | {_number(r['qwen_prompt_tokens'])} "
        f"| {r['nova_to_qwen_input_ratio'] or '-'} | {_dollars(r['cost_usd'])} "
        f"| {_dollars(r['cost_per_1000_questions_usd'])} |"
        for r in runs
    ]
    lines += [
        "",
        "## Replies",
        "",
        "Latency is the caller's clock around each call, in seconds, waits included.",
        "",
        "| Arm | Split | Stop reasons | Error codes | Format deviations | Wrong-result labels "
        "| Latency mean | median | 95th percentile |",
        "|---|---|---|---|---|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r['arm']} | {r['split']} | {_counts(r['stop_reasons'])} "
        f"| {_counts(r['error_codes'])} | {_counts(r['deviations'])} "
        f"| {_counts(r['mismatches'])} | {_seconds(r['latency_seconds']['mean'])} "
        f"| {_seconds(r['latency_seconds']['median'])} "
        f"| {_seconds(r['latency_seconds']['p95'])} |"
        for r in runs
    ]
    lines += ["", "## Runs", "", "| Arm | Split | Model id | Run | Commit |"]
    lines += ["|---|---|---|---|---|"]
    lines += [
        f"| {r['arm']} | {r['split']} | `{r['model_id']}` | `{r['run_id']}` "
        f"| `{str(r['git_commit'])[:12]}` |"
        for r in runs
    ]
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
