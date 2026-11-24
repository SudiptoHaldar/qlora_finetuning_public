"""Invented inputs for the cost model's tests: round prices, round quantities.

Nothing here is a real run, a real price, or a real bill. Prices are chosen so that the
expected figures can be worked out in the head:

- ``ml.test`` costs $3.60 an hour to train on ($0.001 a second) and $7.20 an hour to host.
- Nova Lite: $0.10 and $0.20 per million input and output tokens, $1.00 per million
  training tokens. Nova Pro: $1.00, $2.00, and $10.00. Storage is $2.00 a month for each.
- Each arm answered 1,000 test questions.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from qlora_finetuning.comparison.cost import CostInputs
from qlora_finetuning.comparison.cost_config import InstancePrices
from qlora_finetuning.frontier.pricing import ModelPrice, Pricing
from qlora_finetuning.tracking.ledger import LedgerRecord

LITE, PRO, QWEN = "amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0", "Qwen/Qwen3-4B"
INSTANCE = "ml.test"
DAY = "20261101T000000Z"
TEST_RUNS = {arm: f"{DAY}-eval-{arm.lower()}-test-0001" for arm in (
    "A0", "A1", "B0-Lite", "B1-Lite", "B0-Pro", "B1-Pro"
)}  # fmt: skip
TRAIN_RUN, SCORE_RUN = f"{DAY}-train-qwen-fit-0001", f"{DAY}-eval-qwen-dev-0001"
ARN = "arn:aws:bedrock:us-east-1:000000000000:model-customization-job/{}"
QUESTIONS = 1000
CORRECT = {"A0": 10, "A1": 700, "B0-Lite": 200, "B1-Lite": 650, "B0-Pro": 400, "B1-Pro": 750}
OUTPUT = {"B0-Lite": 100_000, "B1-Lite": 50_000, "B0-Pro": 100_000, "B1-Pro": 50_000}
JOB_SECONDS = {
    "job-a0": 2200,
    "job-a1": 1100,
    "job-train": 10_000,
    "job-score": 2000,
    "job-smoke": 300,
}
RUN_SECONDS = {"A0": 2000.0, "A1": 1000.0}


def pricing() -> Pricing:
    def price(*values: str) -> ModelPrice:
        return ModelPrice(*(Decimal(value) for value in values))

    return Pricing(
        offer="AmazonBedrock",
        region="us-east-1",
        published="2026-01-01",
        read_on="2026-01-02",
        models={
            LITE: price("0.10", "0.20", "1.00", "2.00"),
            PRO: price("1.00", "2.00", "10.00", "2.00"),
        },
    )  # fmt: skip


def instances(hosting: bool = True) -> InstancePrices:
    return InstancePrices(
        training={INSTANCE: Decimal("3.60")},
        training_published="2026-01-03",
        hosting={INSTANCE: Decimal("7.20")} if hosting else {},
        hosting_published="2026-01-04",
    )


def line(run_id: str, arm: str, where: str, **fields: Any) -> LedgerRecord:
    path = "none" if arm == "none" else arm[0]
    defaults = {"phase": "eval", "source": "aws_api", "path": path}
    return LedgerRecord(run_id=run_id, arm=arm, where=where, **(defaults | fields))


def job_line(name: str, arm: str, phase: str = "eval") -> LedgerRecord:
    return line(
        f"{DAY}-other-{name}",
        arm,
        "sagemaker",
        phase=phase,
        path="A",
        external_id=name,
        instance_type=INSTANCE,
        instance_count=1,
        billable_seconds=JOB_SECONDS[name],
    )


def ledger(pro_billed: bool = False) -> list[LedgerRecord]:
    records = [
        job_line("job-smoke", "none", "other"),
        job_line("job-train", "A1", "train"),
        line(TRAIN_RUN, "A1", "sagemaker", phase="train", source="computed",
             wall_clock_seconds=9000.0, training_tokens=1_000_000, model_id=QWEN),
        job_line("job-score", "A1"),
        line(SCORE_RUN, "A1", "sagemaker", source="measured", wall_clock_seconds=900.0,
             input_tokens=1, output_tokens=1, model_id=QWEN),
        line(f"{DAY}-eval-qwen-quick-dev-0001", "A1", "local", source="measured",
             wall_clock_seconds=50.0, input_tokens=5, output_tokens=5, model_id=QWEN),
        line(f"{DAY}-eval-nova-lite-dev-0001", "B0-Lite", "bedrock", model_id=LITE,
             input_tokens=100_000, output_tokens=0),
        line(f"{DAY}-train-nova-lite-0001", "B1-Lite", "bedrock", phase="train", source="bill",
             model_id=LITE, billed_amount_usd="5.00", external_id=ARN.format("lite")),
    ]  # fmt: skip
    if pro_billed:
        records.append(
            line(f"{DAY}-train-nova-pro-0001", "B1-Pro", "bedrock", phase="train", source="bill",
                 model_id=PRO, billed_amount_usd="20.00", external_id=ARN.format("pro"))
        )  # fmt: skip
    else:
        records.append(
            line(f"{DAY}-train-nova-pro-0001", "B1-Pro", "bedrock", phase="train",
                 source="computed", model_id=PRO, training_tokens=2_000_000,
                 external_id=ARN.format("pro"))
        )  # fmt: skip
    for arm, run_id in TEST_RUNS.items():
        if arm.startswith("A"):
            records.append(job_line(f"job-{arm.lower()}", arm))
            records.append(
                line(run_id, arm, "sagemaker", source="measured", model_id=QWEN,
                     wall_clock_seconds=RUN_SECONDS[arm], input_tokens=500_000,
                     output_tokens=9000, examples=QUESTIONS)
            )  # fmt: skip
        else:
            records.append(
                line(run_id, arm, "bedrock", model_id=LITE if "Lite" in arm else PRO,
                     input_tokens=1_000_000, output_tokens=OUTPUT[arm], examples=QUESTIONS)
            )  # fmt: skip
    return records


def six_arm() -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm, run_id in TEST_RUNS.items():
        qwen = arm.startswith("A")
        arms[arm] = {
            "run_id": run_id,
            "continues": [],
            "path": arm[0],
            "model_id": QWEN if qwen else (LITE if "Lite" in arm else PRO),
            "input_tokens": 500_000 if qwen else 1_000_000,
            "output_tokens": 9000 if qwen else OUTPUT[arm],
            "all": {"examples": QUESTIONS, "execution_accuracy": {"count": CORRECT[arm]}},
        }
        if qwen:
            name = f"job-{arm.lower()}"
            arms[arm]["jobs"] = [
                {"name": name, "instance_type": INSTANCE, "billable_seconds": JOB_SECONDS[name]}
            ]
    return {"schema": 1, "passed": True, "test_examples": QUESTIONS, "arms": arms}


def job_record() -> dict[str, Any]:
    inside = {
        "job-a0": [TEST_RUNS["A0"]],
        "job-a1": [TEST_RUNS["A1"]],
        "job-train": [TRAIN_RUN],
        "job-score": [SCORE_RUN],
        "job-smoke": [],
    }
    return {
        "schema": 1,
        "jobs": [
            {
                "name": name,
                "launch_run_id": f"{DAY}-other-{name}",
                "billable_seconds": JOB_SECONDS[name],
                "pulled": True,
                "runs": [{"run_id": run_id} for run_id in runs],
            }
            for name, runs in inside.items()
        ],
    }


def tuning() -> list[dict[str, Any]]:
    return [
        {
            "model": f"nova-{kind}",
            "arm": arm,
            "attempt": 1,
            "job_arn": ARN.format(kind),
            "base_model_id": model_id,
            "ended_at": "2026-11-02T03:00:00Z",
            "model_deleted": False,
        }
        for kind, arm, model_id in (("lite", "B1-Lite", LITE), ("pro", "B1-Pro", PRO))
    ]


def inputs(pro_billed: bool = False, **changes: Any) -> CostInputs:
    """Six arms, a training job, a scoring job, a smoke job, and both customization jobs."""
    pro = Decimal("20")  # billed, or 2,000,000 tokens at $10.00 a million: the same amount
    values: dict[str, Any] = {
        "ledger": ledger(pro_billed),
        "six_arm": six_arm(),
        "jobs": job_record(),
        "selection": {"training_run": TRAIN_RUN, "selected": {"eval_run_id": SCORE_RUN}},
        "tuning": tuning(),
        "pricing": pricing(),
        "instances": instances(),
        "bills": [],
        # A: test jobs 2.20 + 1.10, training 10.00, scoring 2.00, smoke 0.30
        # B: tests 0.12 + 0.11 + 1.20 + 1.10, dev 0.01, training 5.00 + 20.00
        "committed": {"A": Decimal("15.60"), "B": Decimal("2.54") + Decimal("5") + pro},
        "caps": {"A": Decimal("100"), "B": Decimal("200")},
    }
    return CostInputs(**(values | changes))


def covering_bills() -> list[dict[str, Any]]:
    """Bill amounts that cover every run and equal the model's totals.

    Every run is dated 2026-11-01 and the tuned models exist from the 2nd, so no storage
    is owed through that day.
    """
    return [
        {"service": "sagemaker", "usd": "15.60", "through": "2026-11-01"},
        {"service": "bedrock", "usd": "27.54", "through": "2026-11-01"},
    ]
