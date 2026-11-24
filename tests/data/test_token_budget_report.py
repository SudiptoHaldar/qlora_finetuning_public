"""Tests for the token budget report: statistics, volumes, and checks T1 to T6."""

from __future__ import annotations

import dataclasses
import json

import pytest

from qlora_finetuning.data import (
    load_data_config,
    load_prompt,
    load_split,
    manifest_hash,
    verify_snapshot,
)
from qlora_finetuning.data.serialization import build_record, records_digest
from qlora_finetuning.data.serialization_report import read_committed_digests
from qlora_finetuning.data.token_budget import (
    TokenBudgetError,
    TokenCounts,
    count_split,
    load_token_budget_config,
    load_tokenizer,
    tokenizer_fingerprint,
)
from qlora_finetuning.data.token_budget_report import (
    build_report,
    render_json,
    render_markdown,
    report_paths,
    run_checks,
    split_stats,
    stale_files,
    token_stats,
    write_report,
)
from tests.data.synthetic import GOLDEN_ROWS, SECRET_WORDS, FakeTokenizer, fake_budget, to_examples

SUFFIX = FakeTokenizer.THINKING_OFF_SUFFIX


@pytest.fixture(scope="module")
def prompt():
    return load_prompt()


@pytest.fixture
def records(prompt):
    return [build_record(item, prompt) for item in to_examples(list(GOLDEN_ROWS.values()))]


@pytest.fixture
def budget(tmp_path):
    return fake_budget(tmp_path)


def setup(records, budget, tokenizer=None):
    """Counts, the inputs record, and the digests a serialization report would pin."""
    tokenizer = tokenizer or FakeTokenizer()
    counts = count_split(records, tokenizer, budget.thinking_off_suffix)
    inputs = {
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "prompt_id": records[0].prompt_id,
        "split_content_hash": "h" * 64,
    }
    committed = {"prompt_id": records[0].prompt_id, "fit": records_digest(records)}
    return counts, inputs, committed


def checks(records, counts, budget, inputs, committed):
    results = run_checks(
        {"fit": (records, counts)},
        budget,
        inputs["tokenizer_fingerprint"],
        inputs["prompt_id"],
        committed,
    )
    return {item["id"]: item for item in results}


# --------------------------------------------------------------------------- statistics


def test_token_stats_use_the_nearest_rank():
    assert token_stats(list(range(1, 101))) == {
        "min": 1, "p50": 50, "p90": 90, "p98": 98, "p99": 99, "max": 100,
    }  # fmt: skip
    assert token_stats([5]) == {"min": 5, "p50": 5, "p90": 5, "p98": 5, "p99": 5, "max": 5}
    assert token_stats([]) == dict.fromkeys(("min", "p50", "p90", "p98", "p99", "max"))


def test_split_statistics(records, prompt, tmp_path):
    budget = fake_budget(tmp_path, thresholds=(280, 290), longest_listed=2)
    counts, _, _ = setup(records, budget)
    stats = split_stats(records, counts, budget)
    system = len(prompt.system.split())  # 227 words
    assert stats["records"] == 4
    assert (stats["system_text_tokens"], stats["template_tokens"]) == (system, 10)
    assert stats["user_text_tokens"] == {
        "min": 35, "p50": 35, "p90": 50, "p98": 50, "p99": 50, "max": 50,
    }  # fmt: skip
    assert stats["target_tokens_with_end_token"]["max"] == 9
    assert stats["full_sequence_tokens"]["max"] == system + 68
    assert stats["full_sequences_longer_than"] == {"280": 3, "290": 1}
    assert stats["tokens_in_one_pass"] == sum(item.total for item in counts) == 4 * system + 231
    assert stats["prompt_tokens_in_one_pass"] == sum(item.prompt for item in counts)
    assert stats["system_text_tokens_in_one_pass"] == 4 * system
    assert stats["system_text_share_percent"] == round(100 * 4 * system / (4 * system + 231), 1)
    assert stats["characters_per_token"] > 1
    assert stats["longest"] == [
        {"example_id": "ACME/2015/page_10.pdf-1", "tokens": system + 68},
        {"example_id": "CRUX/2017/page_8.pdf-1", "tokens": system + 58},
    ]


# --------------------------------------------------------------------------- checks


def test_sound_counts_pass_every_check(records, budget):
    counts, inputs, committed = setup(records, budget)
    results = checks(records, counts, budget, inputs, committed)
    assert list(results) == ["T1", "T2", "T3", "T4", "T5", "T6"]
    assert all(item["passed"] for item in results.values()), results
    assert results["T1"]["detail"] == "longest full sequence is 295 of 2,048 tokens"
    assert results["T2"]["detail"] == "longest target is 9 of 192 tokens"
    assert results["T5"]["detail"] == "the template adds 10 tokens to every record"


def test_t1_names_the_records_that_do_not_fit(records, tmp_path):
    budget = fake_budget(tmp_path, max_length=280)
    counts, inputs, committed = setup(records, budget)
    result = checks(records, counts, budget, inputs, committed)["T1"]
    assert result["detail"] == (
        "3 records are longer than 280 tokens: ACME/2015/page_10.pdf-1, CRUX/2017/page_8.pdf-1, "
        "BOLT/2016/page_3.pdf-1"
    )


def test_t2_reply_longer_than_the_limit(records, tmp_path):
    budget = fake_budget(tmp_path, max_new_tokens=8)
    counts, inputs, committed = setup(records, budget)
    result = checks(records, counts, budget, inputs, committed)["T2"]
    assert result["detail"] == (
        "1 records have a target longer than 8 tokens: BOLT/2016/page_3.pdf-1"
    )


def test_t3_another_tokenizer(records, budget):
    other = FakeTokenizer({"alpha": 0, "gamma": 7})
    counts, inputs, committed = setup(records, budget, tokenizer=other)
    result = checks(records, counts, budget, inputs, committed)["T3"]
    assert not result["passed"]
    assert "not the pinned" in result["detail"] and "Qwen/Qwen3-4B" in result["detail"]


def test_t4_prompt_without_the_empty_think_block(records, tmp_path):
    budget = fake_budget(tmp_path, thinking_off_suffix="<think>\n\n</think>\n\n")
    counts, inputs, committed = setup(records, budget)
    result = checks(records, counts, budget, inputs, committed)["T4"]
    assert result["detail"].startswith("4 records do not end with the empty think block")


def test_t5_template_that_adds_different_amounts(records, budget):
    counts, inputs, committed = setup(records, budget)
    counts[2] = dataclasses.replace(counts[2], prompt=counts[2].prompt + 3)
    result = checks(records, counts, budget, inputs, committed)["T5"]
    assert result["detail"] == "the template adds different numbers of tokens: [10, 13]"


def test_t6_needs_the_serialization_report_to_pin_the_same_records(records, budget):
    counts, inputs, committed = setup(records, budget)
    missing = checks(records, counts, budget, inputs, None)["T6"]
    assert "there is no serialization report" in missing["detail"]
    assert "python -m scripts.finqa_serialize" in missing["detail"]

    other_prompt = checks(records, counts, budget, inputs, {**committed, "prompt_id": "0" * 64})
    assert "built with another prompt id" in other_prompt["T6"]["detail"]

    other_records = checks(records, counts, budget, inputs, {**committed, "fit": "0" * 64})["T6"]
    assert (
        "the fit records differ from the ones the serialization report pins"
        in (other_records["detail"])
    )


def test_checks_on_nothing_do_not_crash(budget):
    results = run_checks({}, budget, budget.fingerprint, "p" * 64, {"prompt_id": "p" * 64})
    assert all(item["passed"] for item in results)


# --------------------------------------------------------------------------- output


def report_for(records, budget):
    counts, inputs, committed = setup(records, budget)
    by_split = {
        "fit": (records, counts),
        "dev": (records[1:2], counts[1:2]),
        "quick_dev": (records[1:2], counts[1:2]),
    }
    committed["dev"] = records_digest(records[1:2])
    return build_report(by_split, budget, inputs, committed)


def test_report_names_its_inputs_and_limits(records, budget):
    report = report_for(records, budget)
    assert report["inputs"] == {
        "tokenizer": "Qwen/Qwen3-4B",
        "tokenizer_revision": budget.revision,
        "tokenizer_fingerprint": budget.fingerprint,
        "prompt_id": records[0].prompt_id,
        "split_content_hash": "h" * 64,
    }
    assert report["limits"] == {"max_length": 2048, "max_new_tokens": 192}
    assert list(report["splits"]) == ["fit", "dev", "quick_dev"]
    assert all(item["passed"] for item in report["checks"])


def test_output_is_deterministic_and_holds_no_text_and_no_prices(records, budget, tmp_path):
    first, second = report_for(records, budget), report_for(records, budget)
    assert render_json(first) == render_json(second)
    assert render_markdown(first) == render_markdown(second)
    for text in (render_json(first), render_markdown(first)):
        for word in SECRET_WORDS:
            assert word not in text.lower(), word
        assert "$" not in text and "?" not in text
        assert text.isascii()
        assert str(tmp_path) not in text


def test_markdown_sections(records, budget):
    markdown = render_markdown(report_for(records, budget))
    assert markdown.startswith("# FinQA token budget report\n")
    assert "- Longest training sequence allowed (`max_length`): 2,048 tokens" in markdown
    assert "6 of 6 passed." in markdown
    assert "| Measure | fit | dev | quick_dev |" in markdown
    assert "| Records | 4 | 1 | 1 |" in markdown
    assert "| Full sequence tokens: `max` | 295 | 278 | 278 |" in markdown
    assert "| Full sequences longer than: `2048` | 0 | 0 | 0 |" in markdown
    assert "## Longest records in fit" in markdown and "## Longest records in quick_dev" in markdown
    assert "| `ACME/2015/page_10.pdf-1` | 295 |" in markdown


def test_write_and_stale(records, budget):
    report = report_for(records, budget)
    markdown_path, json_path = report_paths(budget.report_dir)
    assert (markdown_path.name, json_path.name) == (
        "finqa_token_budget_report.md",
        "finqa_token_budget_report.json",
    )
    assert stale_files(report, budget.report_dir) == [markdown_path.name, json_path.name]
    write_report(report, budget.report_dir)
    assert stale_files(report, budget.report_dir) == []
    markdown_path.write_bytes(markdown_path.read_bytes().replace(b"\n", b"\r\n"))
    assert stale_files(report, budget.report_dir) == []
    json_path.write_text("{}\n", encoding="utf-8")
    assert stale_files(report, budget.report_dir) == [json_path.name]


# --------------------------------------------------------------------------- repository


def test_committed_report_says_everything_fits():
    cfg = load_token_budget_config()
    report = json.loads(report_paths(cfg.report_dir)[1].read_text(encoding="utf-8"))
    assert all(item["passed"] for item in report["checks"])
    assert report["inputs"]["tokenizer_revision"] == cfg.revision
    assert report["inputs"]["tokenizer_fingerprint"] == cfg.fingerprint
    assert report["inputs"]["prompt_id"] == load_prompt().prompt_id
    assert report["inputs"]["split_content_hash"] == manifest_hash()
    assert report["limits"] == {"max_length": cfg.max_length, "max_new_tokens": cfg.max_new_tokens}
    for name, records in (("fit", 4884), ("dev", 499), ("quick_dev", 200)):
        stats = report["splits"][name]
        assert stats["records"] == records
        assert stats["full_sequence_tokens"]["max"] <= cfg.max_length
        assert stats["target_tokens_with_end_token"]["max"] <= cfg.max_new_tokens
        assert stats["full_sequences_longer_than"][str(cfg.max_length)] == 0


def test_committed_report_matches_the_real_tokenizer_and_records():
    """Skips without the training stack, the cached tokenizer, or the snapshot (CI)."""
    if not all(item.ok for item in verify_snapshot(load_data_config())):
        pytest.skip("no FinQA snapshot here; fetch it with: python -m scripts.finqa_fetch")
    cfg, prompt = load_token_budget_config(), load_prompt()
    try:
        tokenizer = load_tokenizer(cfg)
    except TokenBudgetError as exc:
        pytest.skip(f"no tokenizer here: {exc}")
    by_split = {}
    for name in ("fit", "dev", "quick_dev"):
        records = [build_record(example, prompt) for example in load_split(name)]
        by_split[name] = (records, count_split(records, tokenizer, cfg.thinking_off_suffix))
    inputs = {
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "prompt_id": prompt.prompt_id,
        "split_content_hash": manifest_hash(),
    }
    report = build_report(by_split, cfg, inputs, read_committed_digests(prompt.report_dir))
    assert all(item["passed"] for item in report["checks"]), report["checks"]
    assert stale_files(report, cfg.report_dir) == []
    assert isinstance(by_split["fit"][1][0], TokenCounts)
