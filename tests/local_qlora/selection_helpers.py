"""A scored training run under ``tmp_path``, for the scoring and selection tests (v3.04).

``World`` trains with the stand-in engine, scores the checkpoints with stand-in models
through the real job task and the real evaluation runner, and writes the job record and the
ledger line that a pulled SageMaker job leaves. No GPU, no network, no AWS.

The invented split has 8 dev records. A stand-in model answers the first ``right`` of them
with the gold reply and the rest with a reply that is in the format and wrong.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from qlora_finetuning.data.token_budget import load_token_budget_config
from qlora_finetuning.local_qlora.adapters import adapter_sha256, resolve_adapter
from qlora_finetuning.local_qlora.config import load_local_qlora_config
from qlora_finetuning.local_qlora.job import run_task
from qlora_finetuning.local_qlora.job_record import read_job_record, write_job_record
from qlora_finetuning.local_qlora.predictor import QwenPredictor
from qlora_finetuning.local_qlora.sagemaker_config import load_sagemaker_config
from qlora_finetuning.local_qlora.selection import select_checkpoint
from qlora_finetuning.local_qlora.training import run_training
from qlora_finetuning.tracking import Ledger, LedgerRecord
from tests.evaluation.conftest import make_context

from .conftest import StubBackend
from .training_helpers import StubEngine, StubReloader, make_training_context, with_training

WRONG_REPLY = "PROGRAM: add(1, 2)\nANSWER: 3"
NO_FORMAT = "I do not know."
TRAIN_JOB, SCORE_JOB = "qlora-train-01234567-a1", "qlora-score-01234567-a1"
# The jobs of a corrective iteration (v3.05): another commit, the first job's workspace.
TRAIN_JOB_2, SCORE_JOB_2 = "qlora-train-89abcdef-a1", "qlora-score-89abcdef-a1"
REVERSED = "PROGRAM: subtract(95, 920)\nANSWER: -825"  # the gold arguments, reversed
OTHER_NUMBER = "PROGRAM: subtract(920, 96)\nANSWER: 824"
TOO_LONG = "PROGRAM: subtract(920, 95), divide(#0, 95)\nANSWER: 8.68421"
NOT_A_PROGRAM = "PROGRAM: subtract(920 95\nANSWER: 825"
INSTANCE = "ml.g5.xlarge"


class PartlyRightBackend(StubBackend):
    """Answers the first ``right`` prompts it is given with the gold reply, the rest wrongly."""

    def __init__(self, right: int, wrong: str = WRONG_REPLY, **options: Any) -> None:
        super().__init__(**options)
        self.right, self.wrong, self.asked = right, wrong, 0

    def generate(self, prompts, max_new_tokens):
        replies = []
        for reply in super().generate(prompts, max_new_tokens):
            if self.asked >= self.right:
                reply = dataclasses.replace(reply, text=self.wrong)
            self.asked += 1
            replies.append(reply)
        return replies


class World:
    """One training run, its scoring, and the records of the jobs that made them."""

    def __init__(
        self,
        root: Path,
        *,
        right: dict[int, int] | None = None,
        untuned_right: int = 0,
        where: str = "sagemaker",
        scope: str = "measured",
        smoke: bool = False,
    ) -> None:
        self.root, self.where, self.scope = root, where, scope
        # Checkpoints at steps 2, 4, and 6. With 8 dev examples, one example is noise.
        self.training = with_training(make_training_context(root), save_steps=2, noise_examples=1)
        self.first_cfg = self.training.training_cfg
        self.runner = make_context(root)
        self.sagemaker_cfg = load_sagemaker_config(root=root)
        self.right, self.untuned_right = right or {}, untuned_right
        self.wrong: dict[int, str] = {}  # per step: what a wrong reply is
        self.built: list[str | None] = []
        self.trained = run_training(
            smoke=smoke,
            where=where,
            scope=scope,
            context=self.training,
            engine=StubEngine(),
            reloader=StubReloader(),
        )
        self.run = self.trained["run_id"]
        self.first_run = self.run
        self._own_weights()

    def _own_weights(self) -> None:
        """Give the run's checkpoints weights no other run has.

        The stand-in engine writes the same bytes at a step in every run. Real runs
        never share weights, and an adapter is known by the hash of its weights.
        """
        for step in self.steps:
            weights = self.checkpoint(step) / "adapter_model.safetensors"
            weights.write_bytes(f"{self.run} weights after step {step}".encode())

    def evaluate_checkpoint(self, step: int, split: str = "dev") -> dict[str, Any]:
        """One checkpoint on one split, as ``scripts.evaluate`` scores it."""
        return run_task(
            "evaluate",
            predictor="qwen3-4b-tuned",
            adapter=f"{self.run}/checkpoint-{step}",
            split=split,
            where=self.where,
            scope=self.scope,
            job_name="local-evaluate",
            root=self.root,
            check_pins=False,
            runner_context=self.runner,
            factory=self.factory,
        )

    # ------------------------------------------------------------------ the pieces

    @property
    def runs_dir(self) -> Path:
        return self.runner.tracking_cfg.runs_dir

    @property
    def steps(self) -> list[int]:
        folder = self.runs_dir / self.run / "checkpoints"
        return sorted(int(path.name.split("-")[1]) for path in folder.iterdir())

    def checkpoint(self, step: int) -> Path:
        return self.runs_dir / self.run / "checkpoints" / f"checkpoint-{step}"

    def factory(self, name: str, adapter: str | None, context: Any) -> QwenPredictor:
        """Stand-in models that describe themselves exactly as the real ones do."""
        self.built.append(adapter)
        cfg, budget = load_local_qlora_config(), load_token_budget_config()
        if adapter is None:
            settings = cfg.settings(budget.revision, None)
            backend = PartlyRightBackend(self.untuned_right, NO_FORMAT, settings=settings)
            return QwenPredictor(backend, cfg.predictor_name, settings)
        ref = resolve_adapter(adapter, self.training.training_cfg.adapters_dir, self.runs_dir)
        settings = cfg.settings(budget.revision, adapter_sha256(ref.path))
        backend = PartlyRightBackend(
            self.right.get(ref.checkpoint, 0),
            self.wrong.get(ref.checkpoint, WRONG_REPLY),
            settings=settings,
        )
        details = {"training_run": ref.run_id, "checkpoint": ref.checkpoint, "smoke": False}
        return QwenPredictor(backend, cfg.predictor_name + "-tuned", settings, details=details)

    def score(self, **options: Any) -> dict[str, Any]:
        """The score task, as the job runs it."""
        settings: dict[str, Any] = {
            "run": self.run,
            "split": "dev",
            "where": self.where,
            "scope": self.scope,
            "job_name": SCORE_JOB,
            "root": self.root,
            "check_pins": False,
            "runner_context": self.runner,
            "factory": self.factory,
        }
        settings.update(options)
        return run_task("score", **settings)

    def train_again(self, **changes: Any) -> str:
        """A second training run under changed settings, as a corrective iteration makes.

        The changed settings become the world's current config; ``first_cfg`` keeps the
        first run's.
        """
        self.training = with_training(self.training, **changes)
        self.trained = run_training(
            smoke=False,
            where=self.where,
            scope=self.scope,
            context=self.training,
            engine=StubEngine(),
            reloader=StubReloader(),
        )
        self.run = self.trained["run_id"]
        self._own_weights()
        return self.run

    def record_second(self, result: dict[str, Any], status: str = "Completed") -> None:
        """The job record and the ledger after the iteration's two jobs are pulled."""
        record = read_job_record(self.sagemaker_cfg.record_file)
        train = self._job(TRAIN_JOB_2, "train", [{"run_id": self.run}], 9_500, True)
        train.update(iteration=True, status=status)
        jobs = [train, self._job(SCORE_JOB_2, "score", result["runs"], 11_000, True)]
        record["jobs"] += jobs
        write_job_record(self.sagemaker_cfg.record_file, record)
        ledger = Ledger(self.runner.tracking_cfg)
        for job in jobs:
            ledger.append(
                LedgerRecord(
                    run_id=self.run,
                    phase="train" if job["task"] == "train" else "eval",
                    path="A",
                    arm="A1",
                    where="sagemaker",
                    source="aws_api",
                    external_id=job["name"],
                    instance_type=INSTANCE,
                    instance_count=1,
                    billable_seconds=job["billable_seconds"],
                )
            )

    def iterate(self, right: dict[int, int], **changes: Any) -> str:
        """Train again under ``changes``, score the new checkpoints, and record the jobs."""
        self.train_again(**changes)
        self.right = right
        self.record_second(self.score(job_name=SCORE_JOB_2))
        return self.run

    def record_jobs(self, result: dict[str, Any], pulled: bool = True, billed: bool = True) -> None:
        """What ``launch`` and ``pull`` leave for the training job and the scoring job."""
        jobs = [
            self._job(TRAIN_JOB, "train", [{"run_id": self.run}], 9_000, pulled),
            self._job(SCORE_JOB, "score", result["runs"], 12_000, pulled),
        ]
        write_job_record(self.sagemaker_cfg.record_file, {"schema": 1, "jobs": jobs})
        if billed:
            ledger = Ledger(self.runner.tracking_cfg)
            for job in jobs:
                ledger.append(
                    LedgerRecord(
                        run_id=job["runs"][0]["run_id"],
                        phase="train" if job["task"] == "train" else "eval",
                        path="A",
                        arm="A1",
                        where="sagemaker",
                        source="aws_api",
                        external_id=job["name"],
                        instance_type=INSTANCE,
                        instance_count=1,
                        billable_seconds=job["billable_seconds"],
                    )
                )

    @staticmethod
    def _job(name: str, task: str, runs: list, seconds: int, pulled: bool) -> dict[str, Any]:
        return {
            "name": name,
            "task": task,
            "workspace": TRAIN_JOB,
            "instance_type": INSTANCE,
            "instance_count": 1,
            "status": "Completed",
            "billable_seconds": seconds,
            "pulled": pulled,
            "runs": runs,
        }

    def stand_ins(self) -> dict[str, Any]:
        """What ``select_checkpoint`` reads, all of it under the temporary folder."""
        return {
            "context": self.runner,
            "training_cfg": self.training.training_cfg,
            "sagemaker_cfg": self.sagemaker_cfg,
            "budget": self.training.budget,
        }

    def select(self, **options: Any) -> dict[str, Any]:
        return select_checkpoint(self.run, **self.stand_ins(), **options)


def scored_world(root: Path, **options: Any) -> World:
    """A world whose checkpoints are scored and whose jobs are pulled: a selection can be made."""
    wrong = options.pop("wrong", {})
    world = World(root, **options)
    world.wrong = wrong
    world.record_jobs(world.score())
    return world
