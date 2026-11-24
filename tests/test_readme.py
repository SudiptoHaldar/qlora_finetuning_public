"""The README names things that exist (req_006 v6.02, spec D33 and D34).

A README is the first file to go stale. Every command it names must be a module of the
repository, every file it links to must exist, the generated results block must still be
between its two markers, and the dollar figures written by hand must be the cost record's.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from qlora_finetuning.comparison.cost_report import usd
from qlora_finetuning.comparison.report_build import END, START
from qlora_finetuning.packaging.config import load_packaging_config

REPO_ROOT = Path(__file__).resolve().parents[1]
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
MODULES = sorted(set(re.findall(r"python -m ([A-Za-z_][A-Za-z0-9_.]*)", README)))
LINKS = sorted(set(re.findall(r"\]\((?!https?://|#)([^)#\s]+)(?:#[^)]*)?\)", README)))
FOLDERS = sorted(set(re.findall(r"^\| `([A-Za-z_/]+/)`", README, flags=re.MULTILINE)))
ANCHORS = sorted(set(re.findall(r"\]\(#([a-z0-9-]+)\)", README)))
OWN_TOOLS = ("scripts", "qlora_finetuning")
# Until the two packages have their cards (the owner confirms the licences first), these
# links point at files that the card build will write.
WRITTEN_BY_THE_CARD_BUILD = {
    "reports/packaging/adapter_card.md",
    "reports/packaging/merged_card.md",
}


def test_the_readme_names_commands_links_and_folders_at_all():
    assert len(MODULES) >= 15 and len(LINKS) >= 6 and len(FOLDERS) >= 10


@pytest.mark.parametrize("module", MODULES)
def test_every_command_is_a_module_that_exists(module):
    if module.split(".")[0] not in OWN_TOOLS:
        assert module in ("venv", "pip", "pytest", "ruff"), module
        return
    path = REPO_ROOT.joinpath(*module.split("."))
    assert path.with_suffix(".py").is_file() or (path / "__main__.py").is_file(), module


@pytest.mark.parametrize("link", LINKS)
def test_every_linked_file_exists(link):
    if link in WRITTEN_BY_THE_CARD_BUILD and not (REPO_ROOT / link).exists():
        cfg = load_packaging_config()
        assert (REPO_ROOT / link) in (cfg.adapter_card, cfg.merged_card)
        return
    assert (REPO_ROOT / link).exists(), link


@pytest.mark.parametrize("folder", FOLDERS)
def test_every_folder_of_the_layout_table_exists(folder):
    assert (REPO_ROOT / folder).is_dir(), folder


def test_the_table_of_contents_points_at_headings():
    headings = {
        re.sub(r"[^a-z0-9 -]", "", title.lower()).strip().replace(" ", "-")
        for title in re.findall(r"^#{2,3} (.+)$", README, flags=re.MULTILINE)
    }
    assert ANCHORS and set(ANCHORS) <= headings


def test_the_results_block_is_between_its_markers_and_text_surrounds_it():
    start, end = README.index(START), README.index(END)
    assert 0 < start < end and README.count(START) == README.count(END) == 1
    assert "## Results" in README[start:end]
    assert README[:start].strip() and README[end + len(END) :].strip()


def test_the_diagrams_are_coloured_by_kind():
    blocks = re.findall(r"```mermaid\n(.*?)\n```", README, flags=re.DOTALL)
    assert [block.split()[0] for block in blocks] == ["flowchart", "sequenceDiagram"]
    colours = r"classDef (\w+) fill:#[0-9A-F]{6},stroke:#[0-9A-F]{6},color:#"
    classes = set(re.findall(colours, blocks[0]))
    assert classes == {"data", "shared", "patha", "pathb", "record", "reader", "outside"}
    used = set(re.findall(r":::(\w+)", blocks[0]))
    assert used == classes  # every class is used and every used class is defined
    nodes = re.findall(r"^\s+(\w+)\[[^\n]*\](?::::(\w+))?$", blocks[0], flags=re.MULTILINE)
    assert len(nodes) >= 18 and all(kind for _, kind in nodes)  # no node without a colour
    assert blocks[1].count("box rgba(") == 4 and blocks[1].count("participant ") == 5


def test_the_dollar_figures_are_the_cost_records():
    cost = json.loads((REPO_ROOT / "reports/comparison/cost_model.json").read_text("utf-8"))
    paths, making = cost["project"]["paths"], cost["making"]
    for amount in (
        making["A1"]["train"]["usd"],
        making["A1"]["select"]["usd"],
        paths["B"]["train"],
        paths["A"]["test"],
        paths["B"]["test"],
        paths["A"]["development"],
        paths["B"]["development"],
        paths["A"]["total"],
        paths["B"]["total"],
        cost["project"]["storage_so_far"]["B1-Pro"],
    ):
        assert usd(amount) in README, usd(amount)


def test_the_readme_holds_nothing_private_and_says_who_pays():
    cfg = load_packaging_config()
    found = [name for name, pattern in cfg.forbidden if pattern.search(README)]
    assert found == []
    assert "Your own AWS account" in README and "billed" in README
    assert "Do not report the tuned model's score on the official FinQA test split" in README
