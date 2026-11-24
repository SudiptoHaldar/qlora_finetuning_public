"""What Path B will cost, and the budget gate (req_004 v4.02).

The estimate covers the whole path: what the ledger already holds for Path B at list
price, fine-tuning both models, the evaluation runs still to come, and storing the tuned
models. Training tokens are estimated, because Amazon publishes no Nova tokenizer: the Qwen
token count of the fit split times the Nova to Qwen ratio that the v4.01 baselines
measured, times the epochs.

The outcome is the committed **gate record**: the training files' hashes, every number of
the estimate, the cap, and the verdict. A fine-tuning job (v4.03) starts only with a record
that says ``passed`` for exactly the files in S3. Without a measured ratio the gate fails:
the estimate is still shown, with an assumed ratio and the words "not measured".

Usage::

    from qlora_finetuning.frontier.training_estimate import read_gate_record

    record = read_gate_record()
    record["passed"]                          # True only when every check passed
    record["files"]["training"]["sha256"]     # compare with the object in S3
    record["estimate"]["training_usd"]        # {"amazon.nova-pro-v1:0": "52.066896", ...}

Every dollar amount is text, computed with ``Decimal`` from ``config/pricing.yaml``. Nothing
here writes a dollar amount to the ledger.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from qlora_finetuning.frontier import baseline_report
from qlora_finetuning.frontier.config import MAX_EPOCHS, FrontierConfig, load_frontier_config
from qlora_finetuning.frontier.export import Export
from qlora_finetuning.frontier.export_validation import TokenVolumes, check
from qlora_finetuning.frontier.pricing import MILLION, Pricing, inference_cost
from qlora_finetuning.settings import ConfigError

SCHEMA = 1
REPORT_STEM = "nova_training_gate"
REPORT_COMMAND = "python -m scripts.nova_export"
PATH = "B"
# Evaluation runs still to make for each model once training is priced: the tuned model on
# dev (v4.04), and the untuned and the tuned model on test (v5.01).
RUNS_TO_COME = {"dev": 1, "test": 2}
_CENT = Decimal("0.01")


@dataclass(frozen=True)
class TokenRatio:
    """Nova tokens per Qwen token, and whether it was measured."""

    value: Decimal
    measured: bool
    source: str


def _plain(amount: Decimal) -> str:
    """A decimal as exact text without an exponent: ``65.08362``."""
    return format(amount.normalize(), "f")


def _ceil(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_CEILING))


# --------------------------------------------------------------------------- the ratio


def load_token_ratio(cfg: FrontierConfig, prompt_id: str, manifest_hash: str) -> TokenRatio:
    """The largest ratio measured by the v4.01 baselines, or the assumed one.

    The measured ratio counts only if the baseline report passed its own checks and was
    made from the current prompt wording and split. Otherwise the ratio of
    ``estimate.assumed_token_ratio`` is returned with ``measured=False`` and the reason.
    """
    _, path = baseline_report.report_paths(cfg.report_dir)

    def assumed(why: str) -> TokenRatio:
        return TokenRatio(cfg.assumed_token_ratio, False, f"not measured: {why}")

    if not path.is_file():
        return assumed(
            f"no {path.name}; run the v4.01 baselines, then python -m scripts.nova_baseline"
        )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        inputs, runs = report["inputs"], report["runs"]
        ratios = [Decimal(str(run["nova_to_qwen_input_ratio"])) for run in runs]
    except (ValueError, KeyError, TypeError, InvalidOperation):
        return assumed(f"{path.name} cannot be read or has a run without a token ratio")
    if report.get("passed") is not True:
        return assumed(f"{path.name} did not pass its checks")
    if inputs.get("prompt_id") != prompt_id:
        return assumed(f"{path.name} was made from another prompt wording")
    if inputs.get("split_content_hash") != manifest_hash:
        return assumed(f"{path.name} was made from another split manifest")
    if not ratios or any(not ratio.is_finite() or ratio <= 0 for ratio in ratios):
        return assumed(f"{path.name} has no usable token ratio")
    return TokenRatio(
        max(ratios), True, f"the largest of the {len(ratios)} ratios measured in {path.name}"
    )


# --------------------------------------------------------------------------- the estimate


def estimate_path_b(
    cfg: FrontierConfig,
    pricing: Pricing,
    volumes: TokenVolumes,
    ratio: TokenRatio,
    ledger_records: Sequence[Any],
    *,
    test_examples: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """The cost of Path B as plain values; every dollar amount is exact text.

    Training tokens of one epoch are the Qwen tokens of one pass over fit times the ratio.
    The evaluation still to come is priced with every reply at the reply limit, and the
    locked test split's prompts are estimated from fit's average prompt: both err high.
    ``ledger_records`` are the ledger's effective records; only Path B token counts are read.
    """
    try:
        fit, dev = volumes.splits["fit"], volumes.splits["dev"]
    except KeyError as error:
        raise ConfigError(f"the token budget report has no {error.args[0]} split") from None
    one_epoch = _ceil(fit.tokens_in_one_pass * ratio.value)
    test_prompt_tokens = Decimal(fit.prompt_tokens_in_one_pass) / fit.records * test_examples
    eval_input = _ceil(
        (
            dev.prompt_tokens_in_one_pass * RUNS_TO_COME["dev"]
            + test_prompt_tokens * RUNS_TO_COME["test"]
        )
        * ratio.value
    )
    eval_output = (
        dev.records * RUNS_TO_COME["dev"] + test_examples * RUNS_TO_COME["test"]
    ) * max_new_tokens

    training: dict[str, Decimal] = {}
    by_epochs = {epochs: Decimal(0) for epochs in range(1, MAX_EPOCHS + 1)}
    evaluation: dict[str, Decimal] = {}
    storage: dict[str, Decimal] = {}
    for model in cfg.models.values():
        price = pricing.model(model.model_id)
        for epochs in by_epochs:
            by_epochs[epochs] += one_epoch * epochs * price.training_per_million / MILLION
        training[model.model_id] = (
            one_epoch * cfg.epoch_count * price.training_per_million / MILLION
        )
        evaluation[model.model_id] = inference_cost(
            model.model_id, eval_input, eval_output, pricing
        )
        storage[model.model_id] = price.storage_per_model_month * cfg.storage_months

    spent, spent_records, unpriced = Decimal(0), 0, 0
    for record in ledger_records:
        tokens_in, tokens_out = record.input_tokens, record.output_tokens
        if record.path != PATH or (tokens_in is None and tokens_out is None):
            continue
        if record.model_id not in pricing.models:
            unpriced += 1  # for example a tuned model's deployment; v4.04 maps it to its base
            continue
        spent += inference_cost(record.model_id, tokens_in or 0, tokens_out or 0, pricing)
        spent_records += 1

    training_total = sum(training.values(), Decimal(0))
    with_margin = training_total * (1 + cfg.margin_percent / 100)
    evaluation_total = sum(evaluation.values(), Decimal(0))
    storage_total = sum(storage.values(), Decimal(0))
    total = spent + with_margin + evaluation_total + storage_total
    return {
        "epoch_count": cfg.epoch_count,
        "token_ratio": _plain(ratio.value),
        "ratio_measured": ratio.measured,
        "ratio_source": ratio.source,
        "margin_percent": _plain(cfg.margin_percent),
        "qwen_tokens_one_pass_over_fit": fit.tokens_in_one_pass,
        "training_tokens_one_epoch": one_epoch,
        "training_tokens": {model_id: one_epoch * cfg.epoch_count for model_id in training},
        "training_usd": {model_id: _plain(amount) for model_id, amount in training.items()},
        "training_total_usd": _plain(training_total),
        "training_with_margin_usd": _plain(with_margin),
        "training_total_by_epochs_usd": {
            str(epochs): _plain(amount) for epochs, amount in by_epochs.items()
        },
        "evaluation_to_come": {
            "runs_per_model": dict(RUNS_TO_COME),
            "test_examples": test_examples,
            "reply_tokens_assumed": max_new_tokens,
            "input_tokens_per_model": eval_input,
            "output_tokens_per_model": eval_output,
        },
        "evaluation_to_come_usd": {m: _plain(amount) for m, amount in evaluation.items()},
        "evaluation_to_come_total_usd": _plain(evaluation_total),
        "storage_months": cfg.storage_months,
        "storage_usd": {m: _plain(amount) for m, amount in storage.items()},
        "storage_total_usd": _plain(storage_total),
        "spent_usd": _plain(spent),
        "spent_ledger_records": spent_records,
        "unpriced_ledger_records": unpriced,
        "total_usd": _plain(total),
        "cap_usd": _plain(cfg.path_b_cap_usd),
    }


def gate_checks(estimate: Mapping[str, Any], cfg: FrontierConfig) -> list[dict[str, Any]]:
    """G1 (the ratio was measured) and G2 (the Path B total is within the cap)."""
    total, cap = Decimal(estimate["total_usd"]), cfg.path_b_cap_usd
    ratio = estimate["token_ratio"]
    return [
        check(
            "G1",
            "Ratio measured",
            [] if estimate["ratio_measured"] else [f"{estimate['ratio_source']}"],
            f"Nova counts {ratio} tokens per Qwen token: {estimate['ratio_source']}",
        ),
        check(
            "G2",
            "Within the cap",
            []
            if total <= cap
            else [f"the Path B estimate is {usd(total)}, above the cap of {usd(cap)}"],
            f"the Path B estimate is {usd(total)} of {usd(cap)}",
        ),
    ]


# --------------------------------------------------------------------------- the gate record


def report_paths(report_dir: Path | str) -> tuple[Path, Path]:
    report_dir = Path(report_dir)
    return report_dir / f"{REPORT_STEM}.md", report_dir / f"{REPORT_STEM}.json"


def read_gate_record(cfg: FrontierConfig | None = None) -> dict[str, Any]:
    """The committed gate record. Raises ``ConfigError`` if there is none or it is damaged."""
    _, path = report_paths((cfg or load_frontier_config()).report_dir)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(
            f"no gate record ({path.name}); run python -m scripts.nova_export"
        ) from None
    except ValueError:
        raise ConfigError(f"the gate record {path.name} is damaged") from None
    if not isinstance(record, dict) or not isinstance(record.get("files"), dict):
        raise ConfigError(f"the gate record {path.name} is damaged")
    return record


def staged_before(cfg: FrontierConfig, export: Export) -> dict[str, Any] | None:
    """Where an earlier run staged exactly these files, if the committed record says so."""
    try:
        record = read_gate_record(cfg)
    except ConfigError:
        return None
    same = all(
        (record["files"].get(role) or {}).get("sha256") == file.sha256
        for role, file in export.files.items()
    )
    staged = record.get("staged")
    return staged if same and isinstance(staged, dict) else None


def build_gate_record(
    export: Export,
    checks: Sequence[Mapping[str, Any]],
    estimate: Mapping[str, Any],
    cfg: FrontierConfig,
    pricing: Pricing,
    staged: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The gate record as plain values. It holds no time, no path, and no example text,
    so the same files, prices, and ledger always give the same record."""
    return {
        "schema": SCHEMA,
        "generated_by": REPORT_COMMAND,
        "passed": all(item["passed"] for item in checks),
        "export_id": export.export_id,
        "inputs": {
            "prompt_id": export.prompt_id,
            "split_content_hash": export.split_hash,
            "schema_version": export.schema_version,
            "epoch_count": cfg.epoch_count,
            "cap_usd": _plain(cfg.path_b_cap_usd),
            "price_list": {
                "offer": pricing.offer,
                "region": pricing.region,
                "published": pricing.published,
                "read_on": pricing.read_on,
            },
        },
        "files": {
            role: {
                "name": file.path.name,
                "split": file.split,
                "records": file.records,
                "bytes": file.size,
                "sha256": file.sha256,
                "repeated_lines": file.repeated_lines,
            }
            for role, file in export.files.items()
        },
        "estimate": dict(estimate),
        "checks": [dict(item) for item in checks],
        "staged": dict(staged) if staged else None,
    }


def usd(amount: Decimal | str) -> str:
    """An amount for reading: ``$1,234.57``, half a cent rounded up."""
    return f"${Decimal(amount).quantize(_CENT, rounding=ROUND_HALF_UP):,.2f}"


def render_markdown(record: dict[str, Any]) -> str:
    inputs, files, estimate = record["inputs"], record["files"], record["estimate"]
    checks, staged, prices = record["checks"], record["staged"], record["inputs"]["price_list"]
    passed = sum(item["passed"] for item in checks)
    verdict = "PASSED: a fine-tuning job may be started on these files" if record["passed"] else (
        "NOT PASSED: no fine-tuning job may be started"
    )
    lines = [
        "# Nova training gate record",
        "",
        f"Generated by `{REPORT_COMMAND}`. Do not edit by hand.",
        "",
        "Hashes, counts, tokens, and dollars only: no line of a training file. A fine-tuning",
        "job (req_004 v4.03) starts only if this record says passed and the objects in S3 have",
        "the hashes below.",
        "",
        f"**{verdict}.**",
        "",
        "## Inputs",
        "",
        f"- Export id: `{record['export_id']}`",
        f"- Prompt id: `{inputs['prompt_id']}`",
        f"- Split content hash: `{inputs['split_content_hash']}`",
        f"- Format: `{inputs['schema_version']}`",
        f"- Epochs priced: {inputs['epoch_count']}",
        f"- Cap for Path B: {usd(inputs['cap_usd'])}",
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
        "## Files",
        "",
        "Repeated lines are records that are exact copies of an earlier one. They are kept,",
        "so that both paths train on the same examples.",
        "",
        "| File | Split | Records | Bytes | Repeated lines | SHA-256 |",
        "|---|---|---:|---:|---:|---|",
    ]
    lines += [
        f"| `{f['name']}` ({role}) | {f['split']} | {f['records']:,} | {f['bytes']:,} "
        f"| {f['repeated_lines']:,} | `{f['sha256']}` |"
        for role, f in files.items()
    ]
    lines += ["", "## Staged in S3", ""]
    if staged:
        lines += ["| File | Location | Bytes |", "|---|---|---:|"]
        lines += [
            f"| {role} | `{item['s3_uri']}` | {item['bytes']:,} |" for role, item in staged.items()
        ]
    else:
        lines += ["Not staged yet: `python -m scripts.nova_export --upload` (the owner)."]
    measured = "measured" if estimate["ratio_measured"] else "NOT MEASURED, assumed"
    lines += [
        "",
        "## Estimate",
        "",
        f"- Nova tokens per Qwen token: {estimate['token_ratio']} ({measured}). "
        f"{estimate['ratio_source']}",
        f"- Qwen tokens in one pass over fit: {estimate['qwen_tokens_one_pass_over_fit']:,}; "
        f"estimated Nova tokens in one epoch: {estimate['training_tokens_one_epoch']:,}",
        f"- Evaluation still to come, per model: the tuned model on dev and two runs on test "
        f"({estimate['evaluation_to_come']['test_examples']:,} examples, estimated from fit's "
        f"average prompt), every reply priced at "
        f"{estimate['evaluation_to_come']['reply_tokens_assumed']} tokens",
        "",
        f"| Model | Training, {estimate['epoch_count']} epochs | Evaluation to come "
        f"| Storage, {estimate['storage_months']} month(s) |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| `{model_id}` | {usd(estimate['training_usd'][model_id])} "
        f"| {usd(estimate['evaluation_to_come_usd'][model_id])} "
        f"| {usd(estimate['storage_usd'][model_id])} |"
        for model_id in estimate["training_usd"]
    ]
    lines += [
        "",
        "| Part of Path B | Amount |",
        "|---|---:|",
        f"| Already in the ledger ({estimate['spent_ledger_records']} records, at list price) "
        f"| {usd(estimate['spent_usd'])} |",
        f"| Training, both models | {usd(estimate['training_total_usd'])} |",
        f"| Training with the {estimate['margin_percent']}% margin "
        f"| {usd(estimate['training_with_margin_usd'])} |",
        f"| Evaluation to come | {usd(estimate['evaluation_to_come_total_usd'])} |",
        f"| Storage | {usd(estimate['storage_total_usd'])} |",
        f"| **Total** (ledger, training with margin, evaluation, storage) "
        f"| **{usd(estimate['total_usd'])}** |",
        f"| Cap | {usd(estimate['cap_usd'])} |",
        "",
        "Training for both models, by epoch count (without the margin):",
        "",
        "| Epochs | " + " | ".join(estimate["training_total_by_epochs_usd"]) + " |",
        "|---|" + "---:|" * len(estimate["training_total_by_epochs_usd"]),
        "| Cost | "
        + " | ".join(usd(v) for v in estimate["training_total_by_epochs_usd"].values())
        + " |",
    ]
    if estimate["unpriced_ledger_records"]:
        lines += [
            "",
            f"{estimate['unpriced_ledger_records']} Path B ledger record(s) name a model that "
            "has no price and are not in the total.",
        ]
    return "\n".join(lines) + "\n"


def render_json(record: dict[str, Any]) -> str:
    return json.dumps(record, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def write_gate_record(record: dict[str, Any], report_dir: Path | str) -> tuple[Path, Path]:
    """Write the Markdown and JSON files and return their paths."""
    markdown_path, json_path = report_paths(report_dir)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(record), encoding="utf-8", newline="\n")
    json_path.write_text(render_json(record), encoding="utf-8", newline="\n")
    return markdown_path, json_path


def stale_files(record: dict[str, Any], report_dir: Path | str) -> list[str]:
    """Names of the record files that are missing or differ from a fresh rendering (as text)."""
    stale = []
    markdown_path, json_path = report_paths(report_dir)
    for path, fresh in ((markdown_path, render_markdown(record)), (json_path, render_json(record))):
        if not path.is_file() or path.read_text(encoding="utf-8") != fresh:
            stale.append(path.name)
    return stale
