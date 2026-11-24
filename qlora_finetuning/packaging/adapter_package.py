"""The adapter package: one folder that can be uploaded as it is (req_006 v6.02).

It holds arm A1's two adapter files byte for byte, the base model's tokenizer files, the
prompt, the split manifest, how the adapter was trained, the invented sample, and a
licence. The model card, ``README.md``, is added by the card build.

Usage::

    from qlora_finetuning.packaging.adapter_package import build_adapter_package, package_names

    build_adapter_package(cfg, source, records, prompt=prompt, budget=budget,
                          local_cfg=local_cfg, split_cfg=split_cfg, tokenizer_paths=paths)
    contents_problems(cfg.adapter_dir, package_names(cfg))      # check P4

Nothing here loads a model or imports the training stack.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qlora_finetuning.data import manifest_hash
from qlora_finetuning.data.serialization import PromptDefinition
from qlora_finetuning.data.split import load_split_config
from qlora_finetuning.data.token_budget import TokenBudgetConfig
from qlora_finetuning.demo.inputs import sample_path
from qlora_finetuning.local_qlora.adapters import ADAPTER_CONFIG, ADAPTER_WEIGHTS
from qlora_finetuning.local_qlora.config import LocalQloraConfig
from qlora_finetuning.packaging.config import PackagingConfig
from qlora_finetuning.packaging.files import (
    copy_bytes,
    copy_text,
    read_text_lf,
    sha256_file,
    write_json_lf,
    write_text_lf,
)
from qlora_finetuning.packaging.sources import Records, Source, prompt_file, training_metadata
from qlora_finetuning.settings import ConfigError

PROMPT_NAME = "prompt.json"
MANIFEST_NAME = "finqa_split_manifest.json"
METADATA_NAME = "training_metadata.json"
SAMPLE_NAME = "sample_input.json"
LICENSE_NAME = "LICENSE"
CARD_NAME = "README.md"
APACHE_TEXT = Path(__file__).with_name("apache-2.0.txt")


def package_names(cfg: PackagingConfig) -> tuple[str, ...]:
    """Every file of the finished adapter package."""
    return (
        ADAPTER_WEIGHTS,
        ADAPTER_CONFIG,
        *cfg.tokenizer_files,
        PROMPT_NAME,
        MANIFEST_NAME,
        METADATA_NAME,
        SAMPLE_NAME,
        LICENSE_NAME,
        CARD_NAME,
    )


def copy_tokenizer(
    names: Sequence[str], paths: Mapping[str, Path | str], folder: Path
) -> None:
    """The base model's tokenizer files, byte for byte."""
    for name in names:
        if name not in paths:
            raise ConfigError(f"the tokenizer file {name} was not found for the package")
        copy_bytes(paths[name], folder / name)


def tokenizer_problems(
    names: Sequence[str], paths: Mapping[str, Path | str], folder: Path
) -> list[str]:
    """Tokenizer files of a package that are not the base model's bytes."""
    problems = []
    for name in names:
        target = folder / name
        if not target.is_file():
            problems.append(f"{name} is missing")
        elif name not in paths or sha256_file(target) != sha256_file(paths[name]):
            problems.append(f"{name} is not the base model's file")
    return problems


def build_adapter_package(
    cfg: PackagingConfig,
    source: Source,
    records: Records,
    *,
    prompt: PromptDefinition,
    budget: TokenBudgetConfig,
    local_cfg: LocalQloraConfig,
    tokenizer_paths: Mapping[str, Path | str],
    split_cfg: Any = None,
) -> Path:
    """Write the adapter package, all but its card. Returns the folder.

    ``source`` is the checked checkpoint (``selected_source``). Files that are already
    there are written again; a file the package does not name is left, and check P4
    reports it.
    """
    split_cfg = split_cfg or load_split_config()
    folder = cfg.adapter_dir
    folder.mkdir(parents=True, exist_ok=True)

    for name in (ADAPTER_WEIGHTS, ADAPTER_CONFIG):
        copy_bytes(source.path / name, folder / name)
    if sha256_file(folder / ADAPTER_WEIGHTS) != source.sha256:
        raise ConfigError("the adapter weights changed while they were copied; build again")
    copy_tokenizer(cfg.tokenizer_files, tokenizer_paths, folder)
    write_json_lf(folder / PROMPT_NAME, prompt_file(prompt, budget, local_cfg))
    copy_text(split_cfg.manifest_file, folder / MANIFEST_NAME)
    write_json_lf(
        folder / METADATA_NAME,
        training_metadata(
            source,
            records,
            prompt=prompt,
            budget=budget,
            local_cfg=local_cfg,
            manifest_hash=manifest_hash(split_cfg),
        ),
    )
    copy_text(sample_path(), folder / SAMPLE_NAME)
    write_text_lf(folder / LICENSE_NAME, read_text_lf(APACHE_TEXT))
    return folder


def contents_problems(folder: Path, expected: Sequence[str]) -> list[str]:
    """What a package folder holds that it should not, or lacks (check P4)."""
    if not folder.is_dir():
        return [f"the folder {folder.name}/ does not exist: build the package first"]
    present = {path.name for path in folder.iterdir()}
    problems = [f"{name} is missing" for name in expected if name not in present]
    extra = sorted(present - set(expected))
    problems += [f"{name} does not belong in the package" for name in extra]
    return problems


def adapter_bytes_problems(folder: Path, source: Source) -> list[str]:
    """Adapter files of the package that are not the selected checkpoint's bytes."""
    problems = []
    for name in (ADAPTER_WEIGHTS, ADAPTER_CONFIG):
        target = folder / name
        if target.is_file() and sha256_file(target) != sha256_file(source.path / name):
            problems.append(f"{name} is not the selected checkpoint's file")
    return problems


def sample_problems(folder: Path) -> list[str]:
    """The package's sample must be the demo's file (with Unix line ends)."""
    target = folder / SAMPLE_NAME
    if not target.is_file():
        return []
    same = read_text_lf(target) == read_text_lf(sample_path())
    return [] if same else [f"{SAMPLE_NAME} is not the demo's sample"]
