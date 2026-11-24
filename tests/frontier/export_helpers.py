"""Shared pieces for the v4.02 tests: invented token volumes, a baseline report, a fake S3.

Everything is written under ``tmp_path``. The split is the invented one of
``tests/evaluation/conftest.py``: fit 30, dev 8, test 22.
"""

from __future__ import annotations

import io
import json
from decimal import Decimal

from botocore.exceptions import ClientError

from qlora_finetuning.data import manifest_hash
from qlora_finetuning.frontier.baseline_report import report_paths as baseline_paths
from qlora_finetuning.frontier.export_validation import SplitVolume, TokenVolumes

RATIO = Decimal("1.5")


def volumes_for(context, **changes) -> TokenVolumes:
    """Invented Qwen token volumes for the invented split, made from the current prompt."""
    values = {
        "prompt_id": context.prompt.prompt_id,
        "split_hash": manifest_hash(context.split_cfg),
        "splits": {
            "fit": SplitVolume(
                records=30,
                tokens_in_one_pass=30_000,
                prompt_tokens_in_one_pass=27_000,
                longest=1_200,
            ),
            "dev": SplitVolume(
                records=8,
                tokens_in_one_pass=8_000,
                prompt_tokens_in_one_pass=7_200,
                longest=1_100,
            ),
        },
    }
    values.update(changes)
    return TokenVolumes(**values)


def write_baseline_report(cfg, context, ratios=("1.1000", "1.2500", "1.2000", "1.1500"), **changes):
    """A baseline report as v4.01 writes it, holding only what the gate reads."""
    report = {
        "passed": True,
        "inputs": {
            "prompt_id": context.prompt.prompt_id,
            "split_content_hash": manifest_hash(context.split_cfg),
        },
        "runs": [{"nova_to_qwen_input_ratio": ratio} for ratio in ratios],
    }
    report.update(changes)
    _, path = baseline_paths(cfg.report_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def failed(checks) -> list[str]:
    return [item["id"] for item in checks if not item["passed"]]


def detail(checks, check_id: str) -> str:
    return next(item["detail"] for item in checks if item["id"] == check_id)


class FakeS3:
    """Keeps objects in memory and answers ``head_object`` and ``put_object`` as S3 does."""

    def __init__(self):
        self.objects: dict[tuple[str, str], dict] = {}
        self.calls: list[tuple[str, str]] = []
        self.corrupt_on_put = False

    def head_object(self, Bucket, Key, ChecksumMode=None):  # noqa: N803 (boto3's names)
        self.calls.append(("head_object", Key))
        assert ChecksumMode == "ENABLED"
        stored = self.objects.get((Bucket, Key))
        if stored is None:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        return {"ContentLength": len(stored["Body"]), "ChecksumSHA256": stored["ChecksumSHA256"]}

    def put_object(self, Bucket, Key, Body, ChecksumSHA256, ContentType=None):  # noqa: N803
        self.calls.append(("put_object", Key))
        body = Body[:-1] if self.corrupt_on_put else Body
        self.objects[(Bucket, Key)] = {"Body": body, "ChecksumSHA256": ChecksumSHA256}
        return {"ETag": "stub"}

    def list_objects_v2(self, Bucket, Prefix):  # noqa: N803
        self.calls.append(("list_objects_v2", Prefix))
        keys = sorted(key for bucket, key in self.objects if bucket == Bucket)
        return {"Contents": [{"Key": key} for key in keys if key.startswith(Prefix)]}

    def get_object(self, Bucket, Key):  # noqa: N803
        self.calls.append(("get_object", Key))
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)]["Body"])}

    def count(self, name: str) -> int:
        return sum(call[0] == name for call in self.calls)
