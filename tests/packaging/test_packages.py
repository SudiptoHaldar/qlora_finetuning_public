"""The adapter package, the merged package, and the three predictors (req_006 v6.02)."""

from __future__ import annotations

import json

import pytest

from qlora_finetuning.demo.inputs import sample_path
from qlora_finetuning.local_qlora.adapters import (
    ADAPTER_CONFIG,
    ADAPTER_WEIGHTS,
    METADATA_NAME,
    adapter_sha256,
    read_json,
    write_json,
)
from qlora_finetuning.local_qlora.scoring import expected_settings
from qlora_finetuning.local_qlora.training_config import load_training_config
from qlora_finetuning.packaging.adapter_package import (
    adapter_bytes_problems,
    build_adapter_package,
    contents_problems,
    package_names,
    sample_problems,
    tokenizer_problems,
)
from qlora_finetuning.packaging.files import read_text_lf, sha256_file
from qlora_finetuning.packaging.merged_package import (
    build_merged_package,
    merged_names,
    merged_weights,
    read_merge_metadata,
)
from qlora_finetuning.packaging.predictors import (
    MERGED,
    MERGED_FULL,
    PACKAGE,
    build_merged_predictor,
    build_package_predictor,
    merged_settings,
    package_settings,
    qwen3_4b_merged,
    qwen3_4b_merged_full,
    qwen3_4b_package,
)
from qlora_finetuning.packaging.scan import scan_folder
from qlora_finetuning.packaging.sources import records_problems, selected_source
from qlora_finetuning.settings import ConfigError

from .helpers import TOKENIZER_BYTES, make_world, stub_merge


def build_adapter(world):
    inputs = world.inputs
    return build_adapter_package(
        world.cfg,
        inputs.source,
        inputs.records,
        prompt=inputs.prompt,
        budget=inputs.budget,
        local_cfg=inputs.local_cfg,
        tokenizer_paths=world.base,
        split_cfg=inputs.context.split_cfg,
    )


def build_merged(world):
    inputs = world.inputs
    return build_merged_package(
        world.cfg,
        inputs.source,
        prompt=inputs.prompt,
        budget=inputs.budget,
        local_cfg=inputs.local_cfg,
        tokenizer_paths=world.base,
        licence_path=world.base["LICENSE"],
        end_ids=[2, 1],
        pad_token_id=1,
        merge=stub_merge,
        base=world.root / "base",
    )


# --------------------------------------------------------------------------- the source


def test_the_source_is_the_selected_checkpoint_and_never_the_runs_last_adapter(tmp_path):
    world = make_world(tmp_path)
    source = world.source
    assert source.path == world.inputs.context.checkpoint and source.name.endswith(
        f"/checkpoint-{source.step}"
    )
    assert source.sha256 == adapter_sha256(source.path)
    # The run's last adapter is another adapter, in another folder.
    training_cfg = load_training_config(root=tmp_path)
    final = training_cfg.adapters_dir / source.run_id
    assert (final / METADATA_NAME).is_file() and final != source.path.parent
    (final / ADAPTER_WEIGHTS).write_bytes(b"the weights of the last step")
    again = selected_source(world.inputs.context)  # the last adapter's weights are not read
    assert again.sha256 == source.sha256 != adapter_sha256(final)
    assert "trainable_parameters" in source.extras
    folder = build_adapter(world)
    assert adapter_sha256(folder) == source.sha256


def test_weights_that_are_not_the_selected_ones_are_refused(tmp_path):
    world = make_world(tmp_path)
    (world.source.path / ADAPTER_WEIGHTS).write_bytes(b"other weights")
    with pytest.raises(ConfigError, match="not the ones the selection was made on"):
        selected_source(world.inputs.context)


def test_a_selection_for_another_prompt_or_split_is_refused(tmp_path):
    world = make_world(tmp_path)
    path = world.inputs.context.selection_record
    for field in ("prompt_id", "manifest_hash"):
        record = read_json(path, "the selection record")
        record["key"][field] = "0" * 64
        write_json(path, record)
        with pytest.raises(ConfigError, match="another experiment"):
            selected_source(world.inputs.context)
        record["key"][field] = world.source.selection["key"][field]
        write_json(path, record)
    assert selected_source(world.inputs.context).sha256 == world.source.sha256


def test_the_committed_records_may_be_quoted_and_a_provisional_one_may_not(tmp_path):
    world = make_world(tmp_path)
    records = world.inputs.records
    assert records_problems(records) == []
    records.cost["provisional"] = True
    records.statistics["passed"] = False
    problems = records_problems(records)
    assert any("cost record is provisional" in problem for problem in problems)
    assert any("R1 to R8" in problem for problem in problems)


# --------------------------------------------------------------------------- the adapter package


def test_the_adapter_package_holds_exactly_its_files(tmp_path):
    world = make_world(tmp_path)
    folder = build_adapter(world)
    names = package_names(world.cfg)
    assert contents_problems(folder, names) == ["README.md is missing"]  # the card comes later
    assert sorted(path.name for path in folder.iterdir()) == sorted(set(names) - {"README.md"})
    for name in (ADAPTER_WEIGHTS, ADAPTER_CONFIG):
        assert (folder / name).read_bytes() == (world.source.path / name).read_bytes()
    assert adapter_bytes_problems(folder, world.source) == []
    for name, content in TOKENIZER_BYTES.items():
        assert (folder / name).read_bytes() == content
    assert tokenizer_problems(world.cfg.tokenizer_files, world.base, folder) == []
    assert sample_problems(folder) == []
    assert read_text_lf(folder / "sample_input.json") == read_text_lf(sample_path())
    assert b"\r\n" not in (folder / "LICENSE").read_bytes()
    assert b"\r\n" not in (folder / "finqa_split_manifest.json").read_bytes()


def test_the_package_says_how_the_adapter_was_made_and_nothing_private(tmp_path):
    world = make_world(tmp_path)
    folder = build_adapter(world)
    metadata = json.loads((folder / "training_metadata.json").read_text(encoding="utf-8"))
    assert metadata["adapter"]["selected_step"] == world.source.step
    assert metadata["adapter"]["weights_sha256"] == world.source.sha256
    assert metadata["base_model"]["commit"] == world.inputs.budget.revision
    assert metadata["prompt_id"] == world.inputs.prompt.prompt_id
    assert metadata["hardware"]["instance_type"] == "ml.g5.xlarge"
    prompt = json.loads((folder / "prompt.json").read_text(encoding="utf-8"))
    assert prompt["system"] == world.inputs.prompt.system and prompt["thinking"] is False
    assert prompt["max_new_tokens"] == world.inputs.budget.max_new_tokens
    assert prompt["user"]["question_label"] == world.inputs.prompt.question_label
    assert scan_folder(folder, world.cfg.forbidden, skip=world.cfg.tokenizer_files) == []
    assert str(tmp_path) not in (folder / "training_metadata.json").read_text(encoding="utf-8")


def test_a_changed_or_extra_file_is_reported(tmp_path):
    world = make_world(tmp_path)
    folder = build_adapter(world)
    (folder / "optimizer.pt").write_bytes(b"state")
    (folder / ADAPTER_CONFIG).write_text("{}", encoding="utf-8")
    (folder / "vocab.json").write_bytes(b"{}")
    (folder / "sample_input.json").write_text("{}", encoding="utf-8")
    assert "optimizer.pt does not belong in the package" in contents_problems(
        folder, package_names(world.cfg)
    )
    assert adapter_bytes_problems(folder, world.source) == [
        f"{ADAPTER_CONFIG} is not the selected checkpoint's file"
    ]
    assert tokenizer_problems(world.cfg.tokenizer_files, world.base, folder) == [
        "vocab.json is not the base model's file"
    ]
    assert sample_problems(folder) == ["sample_input.json is not the demo's sample"]
    assert contents_problems(tmp_path / "absent", ["x"])[0].endswith("build the package first")


# --------------------------------------------------------------------------- the merged package


def test_the_merged_package_carries_the_licence_a_notice_and_its_merge(tmp_path):
    world = make_world(tmp_path)
    folder = build_merged(world)
    names = merged_names(world.cfg)
    assert contents_problems(folder, names) == ["README.md is missing"]
    assert {"LICENSE", "NOTICE", "merge_metadata.json", "generation_config.json"} <= set(names)
    notice = (folder / "NOTICE").read_text(encoding="utf-8")
    assert world.inputs.budget.model_id in notice and world.source.sha256 in notice
    assert "Apache License, Version 2.0" in notice and "greedy" in notice
    assert (folder / "LICENSE").read_text(encoding="utf-8").startswith("Apache License")
    metadata = read_merge_metadata(folder)
    assert metadata["adapter"]["weights_sha256"] == world.source.sha256
    assert metadata["merge"]["saved_dtype"] == "bfloat16"
    assert metadata["generation"]["do_sample"] is False
    weights = merged_weights(folder)
    assert [entry["name"] for entry in weights["files"]] == ["model.safetensors"]
    assert weights["files"][0]["sha256"] == sha256_file(folder / "model.safetensors")
    assert merged_weights(folder)["sha256"] == weights["sha256"]
    assert scan_folder(folder, world.cfg.forbidden, skip=world.cfg.tokenizer_files) == []


def test_a_merged_copy_with_a_missing_weight_file_is_named(tmp_path):
    world = make_world(tmp_path)
    folder = build_merged(world)
    (folder / "model.safetensors").unlink()
    with pytest.raises(ConfigError, match="incomplete .model.safetensors is missing"):
        merged_weights(folder)


# --------------------------------------------------------------------------- the predictors


class StubBackend:
    model_id = "stub"

    def __init__(self, settings):
        self.settings = settings


def test_settings_name_the_weights_and_the_precision(tmp_path):
    world = make_world(tmp_path)
    budget, local_cfg, sha = world.inputs.budget, world.inputs.local_cfg, world.source.sha256
    package = package_settings(local_cfg, budget, sha)
    assert package == expected_settings(local_cfg, budget.revision, sha)  # as the adapter by name
    four = merged_settings(local_cfg, budget, weights_sha256="w" * 64, adapter=sha, quantized=True)
    full = merged_settings(local_cfg, budget, weights_sha256="w" * 64, adapter=sha, quantized=False)
    assert four["adapter"] is None
    assert four["merged"] == {"weights_sha256": "w" * 64, "adapter": sha}
    assert four["quantization"] == package["quantization"]
    assert full["quantization"] is None and full["dtype"] == local_cfg.compute_dtype
    other = merged_settings(local_cfg, budget, weights_sha256="x" * 64, adapter=sha, quantized=True)
    assert len({json.dumps(s, sort_keys=True) for s in (package, four, full, other)}) == 4
    assert four["batch_size"] == full["batch_size"] == package["batch_size"]


def test_the_package_predictor_loads_the_folder_and_refuses_other_weights(tmp_path):
    world = make_world(tmp_path)
    with pytest.raises(ConfigError, match="hf_package adapter"):
        build_package_predictor(world.cfg, world.source, load=None)
    folder = build_adapter(world)
    seen = {}

    def load(local_cfg, budget, adapter_dir):
        seen["folder"] = adapter_dir
        return StubBackend(local_cfg.settings(budget.revision, adapter_sha256(adapter_dir)))

    predictor = build_package_predictor(world.cfg, world.source, load=load)
    assert predictor.name == PACKAGE and seen["folder"] == folder
    assert predictor.settings["adapter"] == world.source.sha256
    assert predictor.details["checkpoint"] == world.source.step

    (folder / ADAPTER_WEIGHTS).write_bytes(b"other weights")
    with pytest.raises(ConfigError, match="does not hold the selected checkpoint"):
        build_package_predictor(world.cfg, world.source, load=load)


@pytest.mark.parametrize(("quantized", "name"), [(True, MERGED), (False, MERGED_FULL)])
def test_the_merged_predictors_name_their_weights(tmp_path, quantized, name):
    world = make_world(tmp_path)
    with pytest.raises(ConfigError, match="hf_package merge"):
        build_merged_predictor(world.cfg, world.source, quantized=quantized, load=None)
    folder = build_merged(world)
    seen = {}

    def load(local_cfg, budget, model_dir, *, model_id, settings, quantized):
        seen.update(folder=model_dir, model_id=model_id, quantized=quantized)
        return StubBackend(settings)

    predictor = build_merged_predictor(world.cfg, world.source, quantized=quantized, load=load)
    assert predictor.name == name and seen["folder"] == folder
    assert seen["model_id"] == world.cfg.merged_repo and seen["quantized"] is quantized
    assert predictor.settings["merged"]["weights_sha256"] == merged_weights(folder)["sha256"]
    assert (predictor.settings["quantization"] is None) is (not quantized)

    metadata = read_merge_metadata(folder)
    metadata["adapter"]["weights_sha256"] = "0" * 64
    (folder / "merge_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ConfigError, match="not made from the selected checkpoint"):
        build_merged_predictor(world.cfg, world.source, quantized=quantized, load=load)


@pytest.mark.parametrize("factory", [qwen3_4b_package, qwen3_4b_merged, qwen3_4b_merged_full])
def test_no_package_predictor_answers_the_test_split(factory):
    with pytest.raises(ConfigError, match="never answers the test split"):
        factory("test", None)  # refused before the context is even looked at
