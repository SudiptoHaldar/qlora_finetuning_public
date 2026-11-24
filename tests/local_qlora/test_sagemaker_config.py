"""Tests for qlora_finetuning.local_qlora.sagemaker_config and job_record."""

from __future__ import annotations

from decimal import Decimal

import pytest
import yaml

from qlora_finetuning.local_qlora.job_record import (
    find_job,
    is_final,
    latest_job,
    read_job_record,
    upsert_job,
    write_job_record,
)
from qlora_finetuning.local_qlora.sagemaker_config import (
    REQUIRED_KEYS,
    load_aws_names,
    load_sagemaker_config,
)
from qlora_finetuning.settings import CONFIG_DIR, REPO_ROOT, ConfigError
from qlora_finetuning.tracking.config import TrackingError


def write_configs(tmp_path, change=None, change_pricing=None):
    for name, edit in (("sagemaker", change), ("pricing", change_pricing)):
        raw = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
        if edit:
            edit(raw)
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    return tmp_path


def test_the_committed_file():
    cfg = load_sagemaker_config()
    assert cfg.image.startswith("763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:")
    assert cfg.image.endswith("-sagemaker")
    assert (cfg.instance_type, cfg.instance_count, cfg.volume_gb) == ("ml.g5.xlarge", 1, 60)
    assert cfg.name_prefix == "qlora" and cfg.smoke_examples == 24
    assert cfg.path_a_cap_usd == Decimal("100")
    assert cfg.bundle_dir == REPO_ROOT / "artifacts" / "bundle"
    assert cfg.record_file == REPO_ROOT / "reports" / "local_qlora" / "sagemaker_jobs.json"
    smoke, train, evaluate = cfg.task("smoke"), cfg.task("train"), cfg.task("evaluate")
    assert (smoke.max_runtime_seconds, smoke.phase, smoke.arm) == (3600, "other", "none")
    assert (train.max_runtime_seconds, train.phase, train.arm) == (43200, "train", "A1")
    assert (evaluate.phase, evaluate.arm) == ("eval", None)  # the predictor decides the arm
    assert cfg.tags == {"project": "qlora-finetuning"}


def test_prices_are_exact_and_come_from_the_price_file():
    cfg = load_sagemaker_config()
    assert cfg.hourly() == Decimal("1.408") and cfg.hourly("ml.g6.xlarge") == Decimal("1.127")
    assert cfg.prices_published == "2026-11-14"
    assert cfg.max_cost("smoke") == Decimal("1.408")  # one hour at most
    assert cfg.max_cost("train") == Decimal("16.896")
    assert cfg.cost_of(1500) == Decimal("1500") * Decimal("1.408") / 3600
    assert cfg.cost_of(1800, "ml.g6.xlarge", count=2) == Decimal("1.127")
    with pytest.raises(ConfigError, match="no price for ml.p4d.24xlarge"):
        cfg.hourly("ml.p4d.24xlarge")
    with pytest.raises(ConfigError, match="unknown task 'deploy'"):
        cfg.task("deploy")


def test_paths_resolve_under_the_root(tmp_path):
    cfg = load_sagemaker_config(root=tmp_path)
    assert cfg.bundle_dir == tmp_path / "artifacts" / "bundle"
    assert cfg.record_file.parent == tmp_path / "reports" / "local_qlora"


def test_the_account_names():
    names = load_aws_names()
    assert (names.profile, names.region) == ("qlora", "us-east-1")
    assert names.bucket.startswith("qlora-sagemaker-")
    assert names.role_arn.endswith(":role/qlora-sagemaker-execution")
    assert names.role_trusts == "sagemaker.amazonaws.com"
    assert (names.quota_service, names.quota_usage) == ("sagemaker", "training job usage")
    assert names.settings == {
        "profile": "qlora",
        "account_id": names.account_id,
        "region": "us-east-1",
    }


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_a_missing_key_is_named(tmp_path, key):
    def drop(raw):
        *parents, last = key.split(".")
        for parent in parents:
            raw = raw[parent]
        del raw[last]

    with pytest.raises(ConfigError, match=key.replace(".", r"\.")):
        load_sagemaker_config(config_dir=write_configs(tmp_path, drop))


def set_value(path, value):
    def change(raw):
        *parents, last = path
        for parent in parents:
            raw = raw[parent]
        raw[last] = value

    return change


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("image",), "pytorch-training:latest", "full ECR image URI with a tag"),
        (("image",), "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training", "with a tag"),
        (("instance_type",), "ml.p4d.24xlarge", "has no price"),
        (("instance_count",), 0, "instance_count"),
        (("volume_gb",), 5, "volume_gb"),
        (("workspace_prefix",), "/sagemaker/workspaces", "workspace_prefix"),
        (("code_prefix",), "a/../b", "code_prefix"),
        (("name_prefix",), "QLoRA_jobs", "name_prefix"),
        (("tasks", "smoke", "max_runtime_seconds"), 0, "max_runtime_seconds"),
        (("tasks", "smoke", "phase"), "deploy", "phase"),
        (("tasks", "train", "arm"), "B1-Pro", "arm"),
        (("tasks",), {"smoke": {}}, "tasks must name exactly"),
        (("smoke", "evaluate_examples"), 0, "evaluate_examples"),
        (("budget", "path_a_cap_usd"), 0, "path_a_cap_usd"),
        (("budget", "path_a_cap_usd"), "a lot", "path_a_cap_usd"),
        (("tags",), {"project": 7}, "tags"),
    ],
)
def test_a_value_that_would_make_a_wrong_job_is_refused(tmp_path, path, value, message):
    with pytest.raises(ConfigError, match=message):
        load_sagemaker_config(config_dir=write_configs(tmp_path, set_value(path, value)))


@pytest.mark.parametrize("price", [1.408, "free", "-1", True])
def test_a_price_must_be_a_positive_amount_as_text(tmp_path, price):
    change = set_value(("sagemaker", "training", "ml.g5.xlarge"), price)
    with pytest.raises(ConfigError, match="positive amount written as text"):
        load_sagemaker_config(config_dir=write_configs(tmp_path, change_pricing=change))


# --------------------------------------------------------------------------- the job record


def test_the_job_record_round_trip(tmp_path):
    path = tmp_path / "reports" / "sagemaker_jobs.json"
    assert read_job_record(path) == {"schema": 1, "jobs": []}
    record = upsert_job(read_job_record(path), {"name": "qlora-smoke-a-a1", "status": "InProgress"})
    upsert_job(record, {"name": "qlora-train-a-a1", "status": "InProgress"})
    upsert_job(record, {"name": "qlora-smoke-a-a1", "status": "Completed"})  # replaces, in place
    write_job_record(path, record)
    read = read_job_record(path)
    assert [job["name"] for job in read["jobs"]] == ["qlora-smoke-a-a1", "qlora-train-a-a1"]
    assert find_job(read, "qlora-smoke-a-a1")["status"] == "Completed"
    assert latest_job(read)["name"] == "qlora-train-a-a1"
    assert latest_job(read, "qlora-smoke-a-a1")["status"] == "Completed"
    assert is_final(read["jobs"][0]) and not is_final(read["jobs"][1])
    assert path.read_text(encoding="utf-8").endswith("}\n")


def test_asking_for_a_job_that_is_not_there(tmp_path):
    with pytest.raises(ConfigError, match="no job has been launched yet"):
        latest_job(read_job_record(tmp_path / "none.json"))
    record = {"schema": 1, "jobs": [{"name": "qlora-smoke-a-a1"}]}
    with pytest.raises(ConfigError, match=r"no job called x .*known: qlora-smoke-a-a1"):
        latest_job(record, "x")


def test_a_damaged_record_and_a_secret_are_refused(tmp_path):
    path = tmp_path / "jobs.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        read_job_record(path)
    path.write_text('{"schema": 9, "jobs": []}', encoding="utf-8")
    with pytest.raises(ConfigError, match="is damaged"):
        read_job_record(path)
    with pytest.raises(TrackingError, match="access key"):
        write_job_record(path, {"schema": 1, "jobs": [{"name": "AKIAABCDEFGHIJKLMNOP"}]})
