"""The accuracy-versus-cost chart, as SVG text (req_005 v5.03).

One point per arm: cost per 1,000 questions across, on a logarithmic axis, and execution
accuracy up, with the confidence interval as a whisker. The colour says which base model,
a filled marker says tuned and a hollow one untuned, and every point carries its arm's name
so no reading depends on colour alone. The surface is fixed and light: the file is shown as
an image, where a page's dark theme cannot reach.

The SVG is written by this module, with no plotting library, so the same points give the
same bytes on every machine.

Usage::

    from qlora_finetuning.comparison.chart import ChartPoint, render_chart

    svg = render_chart(
        [ChartPoint("A1", Decimal("0.346"), 73.19, 71.60, 74.78, "#2a78d6", True)],
        width=760, height=460, families={"Qwen3-4B": "#2a78d6"},
        title="Execution accuracy against cost", note="2,898 test questions",
    )
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from xml.sax.saxutils import escape

SURFACE, INK, SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
LEFT, RIGHT, TOP, BOTTOM = 64, 40, 92, 56
RADIUS = 5.5
LABEL_GAP = 10
NEAR_X, NEAR_Y = 70, 14  # a neighbour this close to the right sends a label to the left
STEPS = (1, 2, 5)


@dataclass(frozen=True)
class ChartPoint:
    """One arm on the chart."""

    arm: str
    cost: Decimal  # US dollars per 1,000 questions; above zero
    percent: float
    low: float
    high: float
    color: str
    tuned: bool


def _n(value: float) -> str:
    return f"{value:.1f}"


def _ticks(low: float, high: float) -> list[float]:
    ticks = []
    for exponent in range(math.floor(math.log10(low)) - 1, math.ceil(math.log10(high)) + 1):
        for step in STEPS:
            value = step * 10.0**exponent
            if low <= value <= high:
                ticks.append(value)
    return ticks


def _dollars(value: float) -> str:
    return f"${value:.2f}" if value < 1 else f"${value:,.0f}"


def render_chart(
    points: Sequence[ChartPoint],
    *,
    width: int,
    height: int,
    families: Mapping[str, str],
    title: str,
    note: str,
) -> str:
    """The chart as one SVG document. ``families`` maps a base model's name to its colour."""
    if not points:
        raise ValueError("there is no point to draw")
    if any(point.cost <= 0 for point in points):
        raise ValueError("a cost must be above zero for a logarithmic axis")
    costs = [float(point.cost) for point in points]
    x_low, x_high = min(costs) / 1.6, max(costs) * 1.6
    plot_w, plot_h = width - LEFT - RIGHT, height - TOP - BOTTOM
    span = math.log10(x_high) - math.log10(x_low)

    def x_of(cost: float) -> float:
        return LEFT + (math.log10(cost) - math.log10(x_low)) / span * plot_w

    def y_of(percent: float) -> float:
        return TOP + (100.0 - percent) / 100.0 * plot_h

    # Say so when every interval is hidden by its marker: a reader would look for whiskers.
    if all((y_of(p.low) - y_of(p.high)) / 2 <= RADIUS + 2 for p in points):
        note += " (here narrower than the markers)"
    summary = "; ".join(
        f"{p.arm}: {p.percent:.2f}% at ${p.cost:.3f} per 1,000 questions" for p in points
    )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" font-family="{FONT}">',
        f"<title>{escape(title)}</title>",
        f"<desc>{escape(note + '. ' + summary)}</desc>",
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
        f'<text x="{LEFT}" y="26" font-size="15" font-weight="600" fill="{INK}">'
        f"{escape(title)}</text>",
        f'<text x="{LEFT}" y="45" font-size="12" fill="{SECONDARY}">{escape(note)}</text>',
    ]

    # The legend: one swatch per base model, then what filled and hollow mean.
    x = float(LEFT)
    for name, color in families.items():
        out.append(f'<circle cx="{_n(x + 5)}" cy="66" r="5" fill="{color}"/>')
        out.append(
            f'<text x="{_n(x + 15)}" y="70" font-size="12" fill="{SECONDARY}">{escape(name)}</text>'
        )
        x += 15 + 7.0 * len(name) + 18
    for text, filled in (("tuned", True), ("untuned", False)):
        fill = SECONDARY if filled else SURFACE
        out.append(
            f'<circle cx="{_n(x + 5)}" cy="66" r="4.5" fill="{fill}" stroke="{SECONDARY}" '
            'stroke-width="2"/>'
        )
        out.append(
            f'<text x="{_n(x + 15)}" y="70" font-size="12" fill="{SECONDARY}">{text}</text>'
        )
        x += 15 + 7.0 * len(text) + 18

    # Grid and axes: recessive, behind the marks.
    for percent in range(0, 101, 20):
        y = y_of(percent)
        out.append(
            f'<line x1="{LEFT}" y1="{_n(y)}" x2="{LEFT + plot_w}" y2="{_n(y)}" stroke="{GRID}" '
            'stroke-width="1"/>'
        )
        out.append(
            f'<text x="{LEFT - 8}" y="{_n(y + 4)}" font-size="11" text-anchor="end" '
            f'fill="{SECONDARY}">{percent}%</text>'
        )
    for tick in _ticks(x_low, x_high):
        x_tick = x_of(tick)
        out.append(
            f'<line x1="{_n(x_tick)}" y1="{TOP}" x2="{_n(x_tick)}" y2="{TOP + plot_h}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{_n(x_tick)}" y="{TOP + plot_h + 18}" font-size="11" text-anchor="middle" '
            f'fill="{SECONDARY}">{_dollars(tick)}</text>'
        )
    out.append(
        f'<text x="{_n(LEFT + plot_w / 2)}" y="{height - 12}" font-size="12" text-anchor="middle" '
        f'fill="{SECONDARY}">Cost per 1,000 questions, US dollars (logarithmic)</text>'
    )
    out.append(
        f'<text x="16" y="{_n(TOP + plot_h / 2)}" font-size="12" text-anchor="middle" '
        f'fill="{SECONDARY}" transform="rotate(-90 16 {_n(TOP + plot_h / 2)})">'
        "Execution accuracy</text>"
    )

    placed = [(x_of(float(p.cost)), y_of(p.percent)) for p in points]
    for point, (px, _) in zip(points, placed, strict=True):
        top, bottom = y_of(point.high), y_of(point.low)
        out.append(
            f'<path d="M{_n(px)} {_n(top)}V{_n(bottom)}M{_n(px - 4)} {_n(top)}H{_n(px + 4)}'
            f'M{_n(px - 4)} {_n(bottom)}H{_n(px + 4)}" stroke="{point.color}" '
            'stroke-width="2" fill="none"/>'
        )
    for point, (px, py) in zip(points, placed, strict=True):
        # A ring of the surface colour keeps overlapping marks apart.
        out.append(f'<circle cx="{_n(px)}" cy="{_n(py)}" r="{RADIUS + 2}" fill="{SURFACE}"/>')
        fill = point.color if point.tuned else SURFACE
        out.append(
            f'<circle cx="{_n(px)}" cy="{_n(py)}" r="{RADIUS}" fill="{fill}" '
            f'stroke="{point.color}" stroke-width="2"/>'
        )
        crowded = any(
            0 < ox - px <= NEAR_X and abs(oy - py) < NEAR_Y for ox, oy in placed
        )
        anchor, lx = ("end", px - LABEL_GAP) if crowded else ("start", px + LABEL_GAP)
        out.append(
            f'<text x="{_n(lx)}" y="{_n(py + 4)}" font-size="12" font-weight="600" '
            f'text-anchor="{anchor}" fill="{INK}">{escape(point.arm)}</text>'
        )
    out.append("</svg>")
    return "\n".join(out) + "\n"
