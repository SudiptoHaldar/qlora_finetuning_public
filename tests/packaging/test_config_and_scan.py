"""``config/packaging.yaml`` and the scan for what must not be published (req_006 v6.02)."""

from __future__ import annotations

import hashlib
import shutil
from datetime import date

import pytest

from qlora_finetuning.packaging.adapter_package import APACHE_TEXT
from qlora_finetuning.packaging.config import load_packaging_config
from qlora_finetuning.packaging.files import combined_sha256, listing, read_text_lf
from qlora_finetuning.packaging.scan import scan_folder, scan_text
from qlora_finetuning.settings import CONFIG_DIR, ConfigError

from .helpers import confirmed

# The SHA-256 of the Apache License 2.0 text as apache.org publishes it.
APACHE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"


def changed_config(tmp_path, old: str, new: str):
    """A copy of the config folder with one piece of ``packaging.yaml`` replaced."""
    folder = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, folder)
    path = folder / "packaging.yaml"
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return folder


def test_the_committed_config_loads_and_names_two_packages(tmp_path):
    cfg = load_packaging_config(root=tmp_path)
    assert cfg.adapter_dir == tmp_path / "artifacts" / "release" / "adapter"
    assert cfg.merged_dir == tmp_path / "artifacts" / "release" / "merged"
    assert cfg.adapter_repo.endswith("-qlora") and cfg.merged_repo.endswith("-merged")
    assert cfg.adapter_repo.split("/")[0] == cfg.merged_repo.split("/")[0] == cfg.author
    assert cfg.license == "apache-2.0" and "@" not in cfg.author
    assert [entry.name for entry in cfg.licences] == ["Qwen3-4B", "FinQA", "FinTabNet"]
    assert cfg.record_json.parent == tmp_path / "reports" / "packaging"


def test_an_unconfirmed_or_old_licence_entry_stops_a_card():
    cfg = load_packaging_config()
    today = date(2026, 11, 24)
    assert len(confirmed(cfg, None).licence_problems(today)) == 3
    assert confirmed(cfg, "2026-11-24").licence_problems(today) == []
    assert confirmed(cfg, "2026-10-25").licence_problems(today) == []  # 30 days
    old = confirmed(cfg, "2026-10-24").licence_problems(today)  # 31 days
    assert len(old) == 3 and "31 days ago" in old[0] and "read it again" in old[0]


@pytest.mark.parametrize(
    ("old", "new", "said"),
    [
        ("author: sudiptoHaldar", "author: someone@example.org", "never an email"),
        ("repo_id: sudiptoHaldar/qwen3-4b-finqa-program-qlora", "repo_id: none", "account/name"),
        ("  folder: merged", "  folder: adapter", "different folders"),
        ("compute_dtype: float32", "compute_dtype: bfloat16", "compute_dtype"),
        ('read_on: "2026-11-24"', 'read_on: "yesterday"', "YYYY-MM-DD"),
        ("code_repository: https://", "code_repository: http://", "https"),
    ],
)
def test_a_bad_value_is_refused_and_named(tmp_path, old, new, said):
    with pytest.raises(ConfigError, match=said):
        load_packaging_config(config_dir=changed_config(tmp_path, old, new))


def test_the_licence_text_is_the_standard_apache_2_0():
    text = read_text_lf(APACHE_TEXT)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == APACHE_SHA256


PLANTED = {
    "an AWS account number": "role of account 123456789012 here",
    "an AWS resource name": "arn:aws:bedrock:us-east-1::custom-model/x",
    "an S3 path": "copied to s3://some-bucket/key",
    "the project's bucket": "the bucket qlora-sagemaker-0000",
    "a SageMaker job name": "job qlora-train-958bed10-a1 finished",
    "an AWS access key": "key AKIAABCDEFGHIJKLMNOP in a file",
    "a Hugging Face token": "token hf_abcdefghijklmnopqrstuvwx",
    "an email address": "write to someone@example.org",
    "a path on someone's machine": "saved under C:\\Users\\someone\\models",
}


@pytest.mark.parametrize("name", sorted(PLANTED))
def test_each_forbidden_kind_is_found(name):
    cfg = load_packaging_config()
    assert {pattern[0] for pattern in cfg.forbidden} == set(PLANTED)
    found = scan_text(PLANTED[name], cfg.forbidden, "a file")
    assert f"a file holds {name}" in found


def test_ordinary_package_text_is_clean():
    cfg = load_packaging_config()
    text = (
        "weights a8c9561b4fc6e560a53d5104c0118e5d088fcfbb579b57b8bfb24714c559902b, "
        "3957900840 bytes, 2,898 questions, run 20261120T182710Z-train-qwen3-4b-fit-4557, "
        "eprint 2505.09388, https://huggingface.co/Qwen/Qwen3-4B, ml.g5.xlarge"
    )
    assert scan_text(text, cfg.forbidden, "a file") == []
    assert scan_text(read_text_lf(APACHE_TEXT), cfg.forbidden, "LICENSE") == []


def test_a_folder_is_scanned_but_not_its_weights_or_tokenizer(tmp_path):
    cfg = load_packaging_config()
    (tmp_path / "notes.json").write_text('{"bucket": "s3://x/y"}', encoding="utf-8")
    (tmp_path / "vocab.json").write_text('{"123456789012": 5}', encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"\xff\xfe arn:aws")
    (tmp_path / "odd.dat").write_bytes(b"\xff\xfe\xfa")
    found = scan_folder(tmp_path, cfg.forbidden, skip=cfg.tokenizer_files)
    assert found == [
        "notes.json holds an S3 path",
        "odd.dat is not a text file and is not a weight file",
    ]
    assert scan_folder(tmp_path / "absent", cfg.forbidden) == []


def test_a_listing_and_its_one_hash(tmp_path):
    (tmp_path / "b.txt").write_bytes(b"two")
    (tmp_path / "a.txt").write_bytes(b"one")
    entries = listing(tmp_path)
    assert [entry["name"] for entry in entries] == ["a.txt", "b.txt"]
    assert entries[0]["bytes"] == 3 and len(entries[0]["sha256"]) == 64
    assert combined_sha256(entries) == combined_sha256(entries[::-1])
    assert combined_sha256(entries) != combined_sha256(entries[:1])
