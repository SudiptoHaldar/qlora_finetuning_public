"""Tests for qlora_finetuning.frontier.predictor: a stub provider, and the real runner."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from qlora_finetuning.evaluation.predictor import (
    PredictionRequest,
    Predictor,
    PredictorError,
    check_predictions,
    describe,
)
from qlora_finetuning.frontier import predictor as predictor_module
from qlora_finetuning.frontier.predictor import FrontierPredictor, build_predictor
from qlora_finetuning.frontier.provider import ProviderReply
from qlora_finetuning.settings import ConfigError
from tests.evaluation.conftest import (
    ids_of,
    ledger_records,
    read_lines,
    records_for,
    results_lines,
    run_folders,
)
from tests.frontier.conftest import (
    INPUT_TOKENS,
    OUTPUT_TOKENS,
    StubProvider,
    baseline_run,
    make_predictor,
)

PACKAGE = Path(predictor_module.__file__).resolve().parent
AWS = {"profile": "qlora", "account_id": "123456789012", "region": "us-east-1"}
TRAINING_STACK = {
    "torch", "transformers", "datasets", "peft", "trl", "accelerate", "bitsandbytes",
    "tokenizers", "safetensors", "huggingface_hub",
}  # fmt: skip


def requests_for(texts):
    return [
        PredictionRequest(example_id=f"ex-{i}", system=system, user=user, max_new_tokens=192)
        for i, (system, user) in enumerate(texts)
    ]


# --------------------------------------------------------------------------- the predictor


def test_it_describes_itself_as_the_runner_needs(cfg):
    predictor = make_predictor(cfg, "nova-pro")
    assert isinstance(predictor, Predictor)
    assert describe(predictor) == {
        "name": "nova-pro",
        "model_id": "amazon.nova-pro-v1:0",
        "settings": cfg.settings,
        "batch_size": 1,
        "usage_source": "aws_api",
    }


def test_settings_are_a_copy_so_one_predictor_cannot_change_another(cfg):
    settings = cfg.settings
    predictor = make_predictor(cfg, "nova-lite", settings=settings)
    settings["temperature"] = 0.7
    assert predictor.settings["temperature"] == 0


def test_one_prediction_per_request_in_order_with_the_texts_unchanged(cfg):
    provider = StubProvider()
    predictor = make_predictor(cfg, "nova-lite", provider=provider)
    texts = [("system one \n", "user one£ "), ("system two", "user\ttwo")]
    requests = requests_for(texts)
    predictions = check_predictions(requests, predictor.predict(requests))
    assert provider.calls == [
        ("amazon.nova-lite-v1:0", "system one \n", "user one£ ", 192),
        ("amazon.nova-lite-v1:0", "system two", "user\ttwo", 192),
    ]
    first = predictions[0]
    assert (first.input_tokens, first.output_tokens) == (INPUT_TOKENS, OUTPUT_TOKENS)
    assert (first.latency_seconds, first.stop_reason) == (0.25, "end")


def test_the_example_id_never_reaches_the_provider(cfg):
    provider = StubProvider()
    make_predictor(cfg, "nova-lite", provider=provider).predict(requests_for([("s", "u")]))
    assert "ex-0" not in repr(provider.calls)


def test_the_stop_reason_and_an_empty_reply_pass_through(cfg):
    class Cut:
        def converse(self, model_id, system, user, max_tokens):
            return ProviderReply("", 9, 192, "length", 3.0)

    predictor = make_predictor(cfg, "nova-pro", provider=Cut())
    (prediction,) = predictor.predict(requests_for([("s", "u")]))
    assert (prediction.text, prediction.stop_reason) == ("", "length")
    assert prediction.output_tokens == 192


def test_a_provider_failure_is_raised_not_turned_into_a_reply(cfg):
    predictor = make_predictor(cfg, "nova-pro", provider=StubProvider(fail_after=0))
    with pytest.raises(PredictorError, match="ThrottlingException"):
        predictor.predict(requests_for([("s", "u")]))


# --------------------------------------------------------------------------- through the runner


def test_a_run_records_bedrocks_token_counts_in_the_ledger(context, cfg):
    provider = StubProvider()
    summary = baseline_run(
        context, cfg, "nova-lite", "dev", predictor=make_predictor(cfg, "nova-lite", provider)
    )
    assert summary["result"] and summary["all"]["execution_accuracy"]["count"] == 8
    assert (summary["path"], summary["arm"], summary["where"]) == ("B", "B0-Lite", "bedrock")
    assert summary["usage"] == {
        "input_tokens": 8 * INPUT_TOKENS,
        "output_tokens": 8 * OUTPUT_TOKENS,
        "predictions_without_token_counts": 0,
    }
    assert summary["predictor"]["settings"] == cfg.settings

    # Every prompt sent is the canonical record's two texts, in dataset order.
    records = records_for("dev", context)
    assert [(call[1], call[2]) for call in provider.calls] == [
        (record.system, record.user) for record in records
    ]
    assert {call[3] for call in provider.calls} == {192}

    (record,) = ledger_records(context)
    assert (record.source, record.model_id) == ("aws_api", "amazon.nova-lite-v1:0")
    assert (record.input_tokens, record.output_tokens) == (8 * INPUT_TOKENS, 8 * OUTPUT_TOKENS)
    assert (record.arm, record.where, record.examples) == ("B0-Lite", "bedrock", 8)
    assert record.billed_amount_usd is None  # quantities only
    (line,) = results_lines(context)
    assert (line["arm"], line["model_id"]) == ("B0-Lite", "amazon.nova-lite-v1:0")


def test_a_failed_run_is_resumed_and_no_example_is_asked_twice(context, cfg):
    provider = StubProvider(fail_after=5)
    predictor = make_predictor(cfg, "nova-pro", provider=provider)
    with pytest.raises(PredictorError) as caught:
        baseline_run(context, cfg, "nova-pro", "dev", predictor=predictor)
    failed_id = caught.value.run_id
    assert run_folders(context) == [failed_id] and len(read_lines(failed_id, context)) == 5

    provider.fail_after = None
    summary = baseline_run(context, cfg, "nova-pro", "dev", predictor=predictor, resume=failed_id)
    assert summary["result"] and summary["continues"] == failed_id
    asked = [call[2] for call in provider.calls]
    assert asked == [record.user for record in records_for("dev", context)]  # each once
    lines = read_lines(summary["run_id"], context)
    assert [line["example_id"] for line in lines] == ids_of("dev", context)

    records = ledger_records(context)
    assert [(r.examples, r.source) for r in records] == [(5, "aws_api"), (3, "aws_api")]
    assert sum(r.input_tokens for r in records) == 8 * INPUT_TOKENS


def test_a_run_under_other_settings_cannot_continue_a_failed_one(context, cfg):
    from qlora_finetuning.evaluation.runner import ResumeError

    failing = make_predictor(cfg, "nova-pro", provider=StubProvider(fail_after=2))
    with pytest.raises(PredictorError) as caught:
        baseline_run(context, cfg, "nova-pro", "dev", predictor=failing)
    warmer = make_predictor(cfg, "nova-pro", settings=dict(cfg.settings, temperature=0.7))
    with pytest.raises(ResumeError, match="another predictor"):
        baseline_run(
            context, cfg, "nova-pro", "dev", predictor=warmer, resume=caught.value.run_id
        )


# --------------------------------------------------------------------------- the factory


class FakeSession:
    def __init__(self, account="123456789012"):
        self.sts = MagicMock()
        self.sts.get_caller_identity.return_value = {"Account": account}
        self.runtime = MagicMock()
        self.asked: list[str] = []

    def client(self, service, config=None):
        self.asked.append(service)
        return {"sts": self.sts, "bedrock-runtime": self.runtime}[service]


def test_the_factory_checks_the_sign_in_and_makes_no_billed_call(cfg):
    session = FakeSession()
    predictor = build_predictor("nova-lite", cfg=cfg, aws=AWS, session=session)
    assert session.asked == ["sts", "bedrock-runtime"]
    session.sts.get_caller_identity.assert_called_once_with()
    assert session.runtime.mock_calls == []  # building a predictor costs nothing
    assert describe(predictor)["model_id"] == "amazon.nova-lite-v1:0"
    assert predictor.provider.client is session.runtime and predictor.provider.profile == "qlora"


def test_the_factory_refuses_another_account_before_any_client_exists(cfg):
    session = FakeSession(account="111122223333")
    with pytest.raises(ConfigError, match="expected 123456789012"):
        build_predictor("nova-pro", cfg=cfg, aws=AWS, session=session)
    assert session.asked == ["sts"]


def test_the_factory_refuses_an_unknown_model_without_asking_aws(cfg):
    session = FakeSession()
    with pytest.raises(ConfigError, match="no model named 'nova-max'"):
        build_predictor("nova-max", cfg=cfg, aws=AWS, session=session)
    assert session.asked == []


@pytest.mark.parametrize(
    ("factory", "name"),
    [(predictor_module.nova_pro, "nova-pro"), (predictor_module.nova_lite, "nova-lite")],
)
def test_the_script_factories_name_their_model(monkeypatch, factory, name):
    seen = []
    monkeypatch.setattr(predictor_module, "build_predictor", lambda model: seen.append(model))
    factory("quick_dev", None)
    assert seen == [name]


# --------------------------------------------------------------------------- guards


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


# The modules that talk to AWS: Bedrock inference (v4.01), S3 staging (v4.02), and the
# fine-tuning jobs (v4.03).
AWS_MODULES = {"bedrock.py", "staging.py", "tuning.py"}


def test_only_the_aws_modules_import_boto3_and_nothing_imports_path_a_or_the_training_stack():
    files = sorted(PACKAGE.glob("*.py"))
    assert {path.name for path in files} >= {"predictor.py", "provider.py", "export.py"}
    assert {path.name for path in files} >= AWS_MODULES
    for path in files:
        modules = imported_modules(path)
        roots = {name.split(".")[0] for name in modules}
        if path.name not in AWS_MODULES:
            assert not roots & {"boto3", "botocore"}, path.name
            assert not any(
                name
                in (
                    "qlora_finetuning.frontier.bedrock",
                    "qlora_finetuning.frontier.staging",
                    "qlora_finetuning.frontier.tuning",
                )
                for name in module_level_imports(path)
            ), f"{path.name} would load boto3 when imported"
        assert not roots & TRAINING_STACK, path.name
        assert not any(name.startswith("qlora_finetuning.local_qlora") for name in modules)
        assert not any(name.startswith("scripts") for name in modules), path.name


def module_level_imports(path: Path) -> set[str]:
    """Modules imported at the top of a file, not inside a function."""
    names = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_loading_the_predictor_module_does_not_load_boto3():
    import subprocess
    import sys

    code = (
        "import sys, qlora_finetuning.frontier.predictor, qlora_finetuning.frontier.pricing, "
        "qlora_finetuning.frontier.baseline_report, qlora_finetuning.frontier.export, "
        "qlora_finetuning.frontier.export_validation, "
        "qlora_finetuning.frontier.training_estimate, "
        "qlora_finetuning.frontier.tuning_record, qlora_finetuning.frontier.tuned, "
        "qlora_finetuning.frontier.tuned_report; "
        "print(sorted(m for m in ('boto3', 'botocore', 'torch') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PACKAGE.parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]"


class FakeTime:
    """A clock that only moves when told to, or when slept on."""

    def __init__(self):
        self.now, self.sleeps = 100.0, []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 6))
        self.now += seconds


class SlowProvider(StubProvider):
    """Each call takes a second of the fake clock."""

    def __init__(self, time):
        super().__init__()
        self.time, self.starts = time, []

    def converse(self, *args, **kwargs):
        self.starts.append(self.time.now)
        self.time.now += 1.0
        return super().converse(*args, **kwargs)


def paced(cfg, time, rate):
    provider = SlowProvider(time)
    model = cfg.model("nova-pro")
    predictor = FrontierPredictor(
        provider,
        model.model_id,
        model.name,
        cfg.settings,
        requests_per_minute=rate,
        clock=time.clock,
        sleep=time.sleep,
    )
    return predictor, provider


def test_a_paced_predictor_spaces_the_starts_of_its_calls(cfg):
    time = FakeTime()
    predictor, provider = paced(cfg, time, 20)  # one call every 3 seconds
    request = PredictionRequest(example_id="a", system="s", user="u", max_new_tokens=8)
    predictor.predict([request, request])
    predictor.predict([request])  # the pace holds across batches
    assert provider.starts == [100.0, 103.0, 106.0]
    assert time.sleeps == [2.0, 2.0]  # the call's own second counts toward the gap
    time.now += 10  # a long pause: the next call does not wait
    predictor.predict([request])
    assert time.sleeps == [2.0, 2.0]


def test_an_unpaced_predictor_never_waits_and_pacing_is_not_described(cfg):
    time = FakeTime()
    predictor, provider = paced(cfg, time, None)
    request = PredictionRequest(example_id="a", system="s", user="u", max_new_tokens=8)
    predictor.predict([request, request])
    assert time.sleeps == [] and provider.starts == [100.0, 101.0]
    paced_one, _ = paced(cfg, time, 18)
    assert describe(paced_one) == describe(predictor)  # so a run resumes across a pace change


def test_the_built_predictors_take_the_pace_of_their_model(cfg):
    session = FakeSession()
    assert build_predictor("nova-pro", cfg=cfg, aws=AWS, session=session).requests_per_minute == (
        cfg.model("nova-pro").requests_per_minute
    )
    lite = build_predictor("nova-lite", cfg=cfg, aws=AWS, session=session)
    assert lite.requests_per_minute is None
