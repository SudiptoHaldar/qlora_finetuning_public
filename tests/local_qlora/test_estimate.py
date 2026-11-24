"""Tests for qlora_finetuning.local_qlora.estimate: hours and dollars from measured rates."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from qlora_finetuning.local_qlora.estimate import (
    OVERHEAD,
    TRAIN,
    TUNED,
    UNTUNED,
    Rates,
    estimate_job,
    measured_rates,
    rates_of,
)
from qlora_finetuning.local_qlora.job import JOBS_DIR, RESULT_NAME
from qlora_finetuning.local_qlora.sagemaker_config import load_sagemaker_config

G5 = "ml.g5.xlarge"


def train_run(**trainer):
    values = {"seconds_per_sequence": 0.9, "train_seconds": 170}
    return {"role": "train", "status": "completed", "trainer": values | trainer}


def eval_run(role, seconds, prompts, **more):
    backend = {"generate_seconds": seconds, "prompts": prompts}
    return {"role": role, "status": "completed", "backend": backend, **more}


SMOKE_RESULT = {
    "runs": [train_run(), eval_run("eval-tuned", 30, 24), eval_run("eval-untuned", 36, 24)]
}
RATES = Rates(
    seconds={
        TRAIN: Decimal("0.9"),
        TUNED: Decimal("1.25"),
        UNTUNED: Decimal("1.5"),
        OVERHEAD: Decimal(1264),
    },
    sources={TRAIN: "job-a", TUNED: "job-b", UNTUNED: "job-a", OVERHEAD: "job-a"},
)


@pytest.fixture
def cfg(tmp_path):
    return load_sagemaker_config(root=tmp_path)


# --------------------------------------------------------------------------- the config


def test_the_committed_file_has_a_score_task_and_a_margin():
    cfg = load_sagemaker_config()
    assert cfg.task("score").max_runtime_seconds == 21600
    assert (cfg.task("score").phase, cfg.task("score").arm) == ("eval", "A1")
    assert cfg.max_cost("score") == Decimal("1.408") * 6
    assert cfg.max_cost("train", 7200) == Decimal("2.816")  # a limit given for this job
    assert cfg.runtime_margin == Decimal("1.5") and cfg.minimum_runtime_seconds == 7200


# --------------------------------------------------------------------------- one job's rates


def test_a_smoke_job_measures_training_both_kinds_of_reply_and_the_overhead():
    assert rates_of(SMOKE_RESULT, 1500) == {
        TRAIN: Decimal("0.9"),
        TUNED: Decimal("1.25"),
        UNTUNED: Decimal("1.5"),
        OVERHEAD: Decimal(1264),  # 1,500 billed, less 170 of training and 66 of generating
    }


def test_a_scoring_job_pools_its_checkpoints_and_skips_what_it_reused():
    result = {
        "runs": [
            eval_run("score-untuned", 600, 499),
            eval_run("score-checkpoint", 500, 499),
            eval_run("score-checkpoint", 498, 499),
            eval_run("score-checkpoint", 9999, 499, reused=True),  # not this job's seconds
            eval_run("eval", 100, 50, adapter="r/checkpoint-3"),
            eval_run("eval", 60, 30, adapter=None),
            {"role": "score-checkpoint", "status": "failed", "backend": {"prompts": 5}},
        ]
    }
    rates = rates_of(result, None)
    assert rates[TUNED] == Decimal(1098) / Decimal(1048)
    assert rates[UNTUNED] == Decimal(660) / Decimal(529)
    assert OVERHEAD not in rates and TRAIN not in rates  # no billable seconds, no training


def test_figures_that_are_not_positive_numbers_are_not_rates():
    result = {"runs": [train_run(seconds_per_sequence=None), eval_run("eval-tuned", 0, 24)]}
    assert rates_of(result, 100) == {}  # and 100 billed is under the 170 of work: no overhead
    assert rates_of({}, 1500) == {OVERHEAD: Decimal(1500)}


# --------------------------------------------------------------------------- across jobs


def write_result(root, name, result):
    path = root / JOBS_DIR / name / RESULT_NAME
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(result), encoding="utf-8")


def job(name, pulled=True, instance_type=G5, billable=1500):
    return {
        "name": name,
        "pulled": pulled,
        "instance_type": instance_type,
        "billable_seconds": billable,
    }


def test_each_figure_comes_from_the_newest_pulled_job_that_measured_it(tmp_path):
    write_result(tmp_path, "smoke", SMOKE_RESULT)
    write_result(tmp_path, "score", {"runs": [eval_run("score-checkpoint", 48, 24)]})
    write_result(tmp_path, "unpulled", {"runs": [eval_run("score-checkpoint", 240, 24)]})
    write_result(tmp_path, "bigger", {"runs": [eval_run("score-checkpoint", 2.4, 24)]})
    record = {
        "jobs": [
            job("smoke"),
            job("lost"),  # pulled, but its result file is not here
            job("score", billable=None),
            job("unpulled", pulled=False),
            job("bigger", instance_type="ml.g5.2xlarge"),
        ]
    }
    rates = measured_rates(record, tmp_path, G5)
    assert rates.seconds == {
        TUNED: Decimal(2),
        TRAIN: Decimal("0.9"),
        UNTUNED: Decimal("1.5"),
        OVERHEAD: Decimal(1264),
    }
    assert rates.sources == {TUNED: "score", TRAIN: "smoke", UNTUNED: "smoke", OVERHEAD: "smoke"}
    assert rates.missing((TRAIN, TUNED)) == [] and rates.get(TUNED) == Decimal(2)
    assert measured_rates({"jobs": []}, tmp_path, G5).missing((TRAIN, OVERHEAD)) == [
        TRAIN,
        OVERHEAD,
    ]


# --------------------------------------------------------------------------- the estimate


def test_a_training_job(cfg):
    estimate = estimate_job("train", cfg, RATES, sequences=4884 * 3)
    assert estimate.seconds == 14451  # 14,652 x 0.9 + 1,264, rounded up
    assert estimate.cost == cfg.cost_of(14451)
    assert estimate.proposed_max_runtime == 21720  # half as much again, to the next minute
    assert "14,652 training sequences at 0.90 s" in estimate.detail
    assert "plus 0.4 hours of overhead: about 4.0 hours" in estimate.detail
    assert estimate.sources == ("job-a",)


def test_a_scoring_job_asks_the_untuned_model_and_every_checkpoint(cfg):
    estimate = estimate_job("score", cfg, RATES, examples=499, checkpoints=10)
    assert estimate.seconds == 8250  # 499 x (1.5 + 10 x 1.25) + 1,264
    assert estimate.proposed_max_runtime == 12420
    assert "from each of 10 checkpoints at 1.25 s" in estimate.detail
    assert estimate.sources == ("job-a", "job-b")


def test_an_evaluation_and_the_least_limit_proposed(cfg):
    tuned = estimate_job("evaluate", cfg, RATES, examples=499, tuned=True)
    untuned = estimate_job("evaluate", cfg, RATES, examples=499)
    assert (tuned.seconds, untuned.seconds) == (1888, 2013)
    assert tuned.proposed_max_runtime == cfg.minimum_runtime_seconds == 7200


def test_nothing_measured_is_no_estimate(cfg):
    assert estimate_job("train", cfg, Rates(), sequences=100) is None
    only_training = Rates(
        seconds={TRAIN: Decimal("0.9"), OVERHEAD: Decimal(1264)},
        sources={TRAIN: "job-a", OVERHEAD: "job-a"},
    )
    assert estimate_job("train", cfg, only_training, sequences=100) is not None
    assert estimate_job("score", cfg, only_training, examples=499, checkpoints=10) is None
