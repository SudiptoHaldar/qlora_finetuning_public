"""Tests for the accuracy-versus-cost chart (req_005 v5.03). Invented points only."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from decimal import Decimal

import pytest

from qlora_finetuning.comparison.chart import ChartPoint, render_chart
from qlora_finetuning.comparison.report_build import chart_points
from qlora_finetuning.comparison.report_record import build_statistics
from qlora_finetuning.settings import ConfigError
from tests.comparison import report_helpers as r

SVG = "{http://www.w3.org/2000/svg}"
FAMILIES = {"One": "#2a78d6", "Two": "#eb6834"}
POINTS = [
    ChartPoint("A0", Decimal("0.50"), 10.0, 8.0, 12.0, "#2a78d6", False),
    ChartPoint("A1", Decimal("0.30"), 70.0, 68.0, 72.0, "#2a78d6", True),
    ChartPoint("B1", Decimal("0.04"), 65.0, 63.0, 67.0, "#eb6834", True),
]


def draw(points=POINTS, **options):
    settings = {
        "width": 760,
        "height": 460,
        "families": FAMILIES,
        "title": "Accuracy against cost",
        "note": "invented points",
    }
    return render_chart(points, **(settings | options))


def test_the_chart_is_one_valid_svg_with_a_title_and_a_description():
    root = ET.fromstring(draw())
    assert root.tag == f"{SVG}svg" and root.attrib["role"] == "img"
    assert root.find(f"{SVG}title").text == "Accuracy against cost"
    description = root.find(f"{SVG}desc").text
    assert "A1: 70.00% at $0.300 per 1,000 questions" in description


def test_every_point_has_a_marker_a_whisker_and_its_name():
    root = ET.fromstring(draw())
    circles = root.findall(f"{SVG}circle")
    assert len(circles) == len(FAMILIES) + 2 + 2 * len(POINTS)  # legend, then ring and marker
    assert len(root.findall(f"{SVG}path")) == len(POINTS)
    texts = [element.text for element in root.findall(f"{SVG}text")]
    assert {"A0", "A1", "B1", "tuned", "untuned", "One", "Two"} <= set(texts)
    assert "$0.05" in texts and "100%" in texts and "0%" in texts


def test_a_tuned_arm_is_filled_and_an_untuned_one_is_hollow():
    svg = draw()
    assert 'fill="#2a78d6" stroke="#2a78d6"' in svg  # A1
    assert 'fill="#fcfcfb" stroke="#2a78d6"' in svg  # A0


def test_the_same_points_give_the_same_bytes_and_end_with_a_new_line():
    assert draw() == draw() and draw().endswith("</svg>\n")


def test_a_label_moves_left_when_a_neighbour_is_close_on_the_right():
    close = [
        ChartPoint("LEFT", Decimal("0.300"), 70.0, 69.0, 71.0, "#2a78d6", True),
        ChartPoint("RIGHT", Decimal("0.330"), 71.0, 70.0, 72.0, "#eb6834", True),
        ChartPoint("FAR", Decimal("0.040"), 20.0, 19.0, 21.0, "#eb6834", False),
    ]
    root = ET.fromstring(draw(close))
    anchors = {
        element.text: element.attrib.get("text-anchor")
        for element in root.findall(f"{SVG}text")
        if element.text in {"LEFT", "RIGHT", "FAR"}
    }
    assert anchors == {"LEFT": "end", "RIGHT": "start", "FAR": "start"}


def test_the_note_says_when_the_whiskers_are_hidden_by_the_markers():
    assert "(here narrower than the markers)" in draw()
    wide = [ChartPoint("A1", Decimal("0.30"), 50.0, 20.0, 80.0, "#2a78d6", True)]
    assert "narrower than the markers" not in draw(wide)


def test_names_are_escaped():
    odd = [ChartPoint("A<1>&", Decimal("0.30"), 50.0, 40.0, 60.0, "#2a78d6", True)]
    svg = draw(odd, title="a < b")
    assert "A&lt;1&gt;&amp;" in svg and "<title>a &lt; b</title>" in svg
    ET.fromstring(svg)


def test_no_point_and_a_cost_of_zero_are_refused():
    with pytest.raises(ValueError, match="no point"):
        draw([])
    with pytest.raises(ValueError, match="above zero"):
        draw([ChartPoint("A1", Decimal("0"), 50.0, 40.0, 60.0, "#2a78d6", True)])


def test_the_points_come_from_the_record_and_the_configured_colours(tmp_path):
    cfg = r.report_cfg(tmp_path)
    record = build_statistics(r.inputs(), cfg)
    points = {point.arm: point for point in chart_points(record, cfg)}
    assert list(points) == list(record["arms"])
    a1 = points["A1"]
    assert a1.percent == 70.0 and a1.tuned and a1.color == "#2a78d6"
    assert a1.cost == Decimal(r.cost_record()["test_pass"]["A1"]["per_1000_questions"])
    assert not points["B0-Pro"].tuned and points["B0-Pro"].color == "#1baf7a"

    del record["costs"]["A1"]
    with pytest.raises(ConfigError, match="arm A1 has no cost"):
        chart_points(record, cfg)
