"""Unit tests for scripts.env_check.

Every check gets stand-ins, so nothing here needs torch, a GPU, or the network: this file runs
in CI, where the training stack is not installed. The one real run is marked ``gpu``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import env_check
from scripts.env_check import FAIL, PASS, SKIP, Environment

REPO_ROOT = Path(__file__).resolve().parents[2]
GIB = 2**30


@pytest.fixture
def cfg() -> dict:
    return {
        "version": 1,
        "python_min": "3.11",
        "pins_file": "requirements-train.txt",
        "torch": {"cuda_build": "cu126", "install_index": "https://example.invalid/whl/cu126"},
        "gpu": {"min_vram_gib": 15, "bf16_min_compute_capability": 8.0},
        "smoke": {
            "model_id": "Qwen/Qwen3-0.6B",
            "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "max_seq_length": 256,
            "max_new_tokens": 16,
            "learning_rate": 2.0e-4,
            "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"},
            "quantization": {"type": "nf4", "double_quant": True, "compute_dtype": "bfloat16"},
        },
    }


def fake_torch(
    version="2.14.1+cu126",
    available=True,
    vram_gib=15.99,
    capability=(8, 9),
    bf16=True,
    name="NVIDIA GeForce RTX 4090 Laptop GPU",
):
    """A stand-in with just the attributes the checks read."""
    return SimpleNamespace(
        __version__=version,
        version=SimpleNamespace(cuda="12.6"),
        cuda=SimpleNamespace(
            is_available=lambda: available,
            get_device_name=lambda index: name,
            get_device_properties=lambda index: SimpleNamespace(total_memory=int(vram_gib * GIB)),
            get_device_capability=lambda index: capability,
            is_bf16_supported=lambda: bf16,
        ),
    )


GOOD_PROBE = {"version": "bitsandbytes 0.50.2", "quantized": True, "finite": True}
GOOD_SMOKE = {
    "loss": 2.345,
    "trainable_parameters": 10_092_544,
    "lora_tensors_with_grad": 140,
    "base_tensors_with_grad": 0,
    "weights_changed": True,
    "thinking_in_output": False,
    "peak_vram_gib": 1.4,
    "seconds": 21.0,
}
PINS = {"torch": "2.14.1", "peft": "0.21.3"}


def good_env(**overrides) -> Environment:
    values = {
        "version_info": (3, 13, 7),
        "pins": dict(PINS),
        "installed": {"torch": "2.14.1+cu126", "peft": "0.21.3"},
        "torch": fake_torch(),
        "driver": "596.08",
        "bnb_probe": lambda: dict(GOOD_PROBE),
        "smoke_run": lambda cfg: dict(GOOD_SMOKE),
    }
    values.update(overrides)
    return Environment(**values)


def statuses(results) -> dict[str, str]:
    return {result.check_id: result.status for result in results}


# --------------------------------------------------------------------------- config


def test_committed_config_loads():
    cfg = env_check.load_env_config()
    assert cfg["python_min"] == "3.11"
    assert isinstance(cfg["python_min"], str)
    assert cfg["torch"]["cuda_build"] == "cu126"
    assert len(cfg["smoke"]["revision"]) == 40
    assert cfg["smoke"]["lora"] == {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": "all-linear",
    }
    assert isinstance(cfg["smoke"]["learning_rate"], float)


def test_config_missing_key(tmp_path):
    (tmp_path / "local_env.yaml").write_text("version: 1\npython_min: '3.11'\n", encoding="utf-8")
    with pytest.raises(env_check.ConfigError, match="smoke.model_id"):
        env_check.load_env_config(tmp_path / "local_env.yaml")


# --------------------------------------------------------------------------- E1


def test_python_pass(cfg):
    assert env_check.check_python((3, 13, 7), cfg).status == PASS


def test_python_exactly_the_minimum(cfg):
    assert env_check.check_python((3, 11, 0), cfg).status == PASS


def test_python_too_old(cfg):
    result = env_check.check_python((3, 10, 14), cfg)
    assert result.status == FAIL
    assert "3.10.14" in result.detail


def test_python_min_must_be_a_version(cfg):
    cfg["python_min"] = "three"
    with pytest.raises(env_check.ConfigError, match="python_min"):
        env_check.check_python((3, 13, 7), cfg)


# --------------------------------------------------------------------------- E2


def test_packages_pass_with_local_build_suffix():
    result = env_check.check_packages(PINS, {"torch": "2.14.1+cu126", "peft": "0.21.3"})
    assert result.status == PASS


def test_packages_missing():
    result = env_check.check_packages(PINS, {"torch": "2.14.1+cu126", "peft": None})
    assert result.status == FAIL
    assert "not installed: peft" in result.detail
    assert "runbook" in result.detail


def test_packages_wrong_version():
    result = env_check.check_packages(PINS, {"torch": "2.13.0+cu126", "peft": "0.21.3"})
    assert result.status == FAIL
    assert "torch 2.13.0+cu126 (pinned 2.14.1)" in result.detail


def test_packages_empty_pins_file():
    assert env_check.check_packages({}, {}).status == FAIL


# --------------------------------------------------------------------------- E3


def test_cuda_build_pass(cfg):
    result = env_check.check_cuda_build(fake_torch(), cfg)
    assert result.status == PASS
    assert "2.14.1+cu126" in result.detail


def test_cuda_build_cpu_wheel(cfg):
    result = env_check.check_cuda_build(fake_torch(version="2.14.1+cpu"), cfg)
    assert result.status == FAIL
    assert "cu126" in result.detail


def test_cuda_build_other_cuda(cfg):
    assert env_check.check_cuda_build(fake_torch(version="2.14.1+cu130"), cfg).status == FAIL


def test_cuda_not_available(cfg):
    result = env_check.check_cuda_build(fake_torch(available=False), cfg)
    assert result.status == FAIL
    assert "cannot reach" in result.detail


# --------------------------------------------------------------------------- E4


def test_gpu_pass_reports_name_vram_and_driver(cfg):
    result = env_check.check_gpu(fake_torch(), cfg, "596.08")
    assert result.status == PASS
    assert "RTX 4090" in result.detail
    assert "16.0 GiB" in result.detail
    assert "596.08" in result.detail


def test_gpu_too_little_vram(cfg):
    result = env_check.check_gpu(fake_torch(vram_gib=7.9), cfg, "596.08")
    assert result.status == FAIL
    assert "need at least 15 GiB" in result.detail


def test_gpu_unknown_driver_is_not_a_failure(cfg):
    result = env_check.check_gpu(fake_torch(), cfg, None)
    assert result.status == PASS
    assert "driver unknown" in result.detail


# --------------------------------------------------------------------------- E5


def test_bf16_pass(cfg):
    assert env_check.check_bf16(fake_torch(), cfg).status == PASS


def test_bf16_old_card(cfg):
    result = env_check.check_bf16(fake_torch(capability=(7, 5)), cfg)
    assert result.status == FAIL
    assert "FP16" in result.detail


def test_bf16_torch_disagrees(cfg):
    assert env_check.check_bf16(fake_torch(bf16=False), cfg).status == FAIL


# --------------------------------------------------------------------------- E6


def test_bitsandbytes_pass():
    result = env_check.check_bitsandbytes(lambda: dict(GOOD_PROBE))
    assert result.status == PASS
    assert "0.50.2" in result.detail


def test_bitsandbytes_not_quantized():
    assert env_check.check_bitsandbytes(lambda: {**GOOD_PROBE, "quantized": False}).status == FAIL


def test_bitsandbytes_non_finite_output():
    assert env_check.check_bitsandbytes(lambda: {**GOOD_PROBE, "finite": False}).status == FAIL


def test_bitsandbytes_probe_raises():
    def probe():
        raise RuntimeError("CUDA Setup failed despite GPU being available.\nlong details")

    result = env_check.check_bitsandbytes(probe)
    assert result.status == FAIL
    assert result.detail == "RuntimeError: CUDA Setup failed despite GPU being available."


# --------------------------------------------------------------------------- E7


def test_smoke_pass(cfg):
    result = env_check.check_smoke(lambda cfg: dict(GOOD_SMOKE), cfg)
    assert result.status == PASS
    assert "Qwen/Qwen3-0.6B" in result.detail
    assert "loss 2.345" in result.detail
    assert "10,092,544" in result.detail
    assert "1.4 GiB" in result.detail


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"loss": float("nan")}, "loss is nan"),
        ({"loss": float("inf")}, "loss is inf"),
        ({"lora_tensors_with_grad": 0}, "no gradient reached the LoRA weights"),
        ({"base_tensors_with_grad": 3}, "3 base tensors received gradients"),
        ({"thinking_in_output": True}, "thinking block"),
        ({"weights_changed": False}, "did not change"),
    ],
)
def test_smoke_failures(cfg, change, expected):
    result = env_check.check_smoke(lambda cfg: {**GOOD_SMOKE, **change}, cfg)
    assert result.status == FAIL
    assert expected in result.detail


def test_smoke_run_raises(cfg):
    def run(cfg):
        raise OSError("We couldn't connect to 'https://huggingface.co'")

    result = env_check.check_smoke(run, cfg)
    assert result.status == FAIL
    assert "OSError" in result.detail


# --------------------------------------------------------------------------- runner


def test_all_pass_without_smoke(cfg):
    assert statuses(env_check.run_checks(cfg, good_env())) == {
        "E1": PASS, "E2": PASS, "E3": PASS, "E4": PASS, "E5": PASS, "E6": PASS, "E7": SKIP,
    }  # fmt: skip


def test_all_pass_with_smoke(cfg):
    results = env_check.run_checks(cfg, good_env(), smoke=True)
    assert all(result.status == PASS for result in results)
    assert len(results) == 7


def test_smoke_is_not_run_unless_requested(cfg):
    def run(cfg):
        raise AssertionError("the smoke must not run without --smoke")

    results = env_check.run_checks(cfg, good_env(smoke_run=run))
    assert results[-1].status == SKIP
    assert "--smoke" in results[-1].detail


def test_missing_torch_fails_e3_and_skips_what_depends_on_it(cfg):
    env = good_env(torch=None, torch_error="torch is not installed (No module named 'torch')")
    results = env_check.run_checks(cfg, env, smoke=True)
    assert statuses(results) == {
        "E1": PASS, "E2": PASS, "E3": FAIL, "E4": SKIP, "E5": SKIP, "E6": SKIP, "E7": SKIP,
    }  # fmt: skip
    assert "runbook" in results[2].detail
    assert "needs E3" in results[3].detail


def test_small_gpu_skips_bf16_bitsandbytes_and_smoke(cfg):
    results = env_check.run_checks(cfg, good_env(torch=fake_torch(vram_gib=8)), smoke=True)
    assert statuses(results) == {
        "E1": PASS, "E2": PASS, "E3": PASS, "E4": FAIL, "E5": SKIP, "E6": SKIP, "E7": SKIP,
    }  # fmt: skip


def test_missing_bitsandbytes_is_a_failure_with_the_install_hint(cfg):
    env = good_env(bnb_probe=None, bnb_error="bitsandbytes is not installed")
    results = env_check.run_checks(cfg, env, smoke=True)
    assert statuses(results)["E6"] == FAIL
    assert "runbook" in results[5].detail
    assert statuses(results)["E7"] == SKIP


def test_wrong_package_version_does_not_block_other_checks(cfg):
    env = good_env(installed={"torch": "2.14.1+cu126", "peft": "0.20.0"})
    assert statuses(env_check.run_checks(cfg, env)) == {
        "E1": PASS, "E2": FAIL, "E3": PASS, "E4": PASS, "E5": PASS, "E6": PASS, "E7": SKIP,
    }  # fmt: skip


# --------------------------------------------------------------------------- main


def test_main_all_pass(capsys):
    assert env_check.main([], env=good_env()) == 0
    out = capsys.readouterr().out
    assert out.count("PASS") == 6
    assert "SKIP  E7  Smoke" in out
    assert "6 passed, 0 failed, 1 skipped" in out
    assert out.isascii()


def test_main_with_smoke(capsys):
    assert env_check.main(["--smoke"], env=good_env()) == 0
    assert "7 passed, 0 failed, 0 skipped" in capsys.readouterr().out


def test_main_failure_exit_code(capsys):
    assert env_check.main([], env=good_env(torch=fake_torch(bf16=False))) == 1
    assert "FAIL  E5  BF16" in capsys.readouterr().out


def test_main_bad_config_path(tmp_path, capsys):
    assert env_check.main(["--config", str(tmp_path / "absent.yaml")], env=good_env()) == 2
    assert "CONFIG ERROR" in capsys.readouterr().out


def test_main_writes_report(tmp_path, capsys):
    report_path = tmp_path / "out" / "report.json"
    assert env_check.main(["--report", str(report_path)], env=good_env()) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert [check["check_id"] for check in report["checks"]] == [f"E{n}" for n in range(1, 8)]
    assert report["checks"][0]["status"] == PASS
    assert "python" in report
    assert "packages" in report


# --------------------------------------------------------------------------- how it is run


def test_runs_as_a_module(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "scripts.env_check", "--config", str(tmp_path / "absent.yaml")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "CONFIG ERROR" in result.stdout


def test_running_by_file_path_gives_a_hint_not_a_traceback(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "env_check.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "python -m scripts.env_check" in result.stderr
    assert "Traceback" not in result.stderr


# --------------------------------------------------------------------------- this machine


@pytest.mark.gpu
def test_this_machine_passes_the_offline_checks(capsys):
    """Needs the training stack and a CUDA GPU. E7 is skipped: the smoke is run by the script."""
    assert env_check.main([]) == 0
    assert "6 passed, 0 failed, 1 skipped" in capsys.readouterr().out


def test_smoke_example_is_built_by_the_serializer():
    """The smoke step trains on the canonical layout, with the configured system text."""
    from qlora_finetuning.data import load_prompt

    system, user, target = env_check.smoke_texts()
    assert system == load_prompt().system
    assert user.startswith("Context:\n- Net revenue increased")
    assert "\n\nTable:\n| Metric ($ in millions) | 2024 | 2023 |\n|---|---|---|\n" in user
    assert user.endswith("Question: " + env_check.SMOKE_QUESTION)
    assert target == "PROGRAM: subtract(4850.0, 4474.2), divide(#0, 4474.2)\nANSWER: 0.08402"
