"""Regenerate an SVG version of Figure 1 from the exact band-weight record."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path


BANDS = ("w_lo", "w_mid", "w_hi")
COLORS = {"w_lo": "#1f77b4", "w_mid": "#ff7f0e", "w_hi": "#2ca02c"}
DISPLAY_ORDER = ("Communities", "Yale", "ProteinFold")
DATASET_COLORS = {
    "Communities": COLORS["w_lo"],
    "Yale": COLORS["w_mid"],
    "ProteinFold": COLORS["w_hi"],
}
DATASET_DASHES = {"Communities": "", "Yale": "6,4", "ProteinFold": "10,4,2,4"}


def _load_rows(path: Path) -> list[tuple[str, dict[str, float]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"dataset_id", "dataset", *BANDS, "source"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} does not contain the required Figure 1 fields")

    loaded: list[tuple[str, dict[str, float]]] = []
    seen: set[str] = set()
    for row in rows:
        dataset = row["dataset"]
        if dataset in seen:
            raise ValueError(f"duplicate Figure 1 dataset: {dataset}")
        seen.add(dataset)
        values: dict[str, float] = {}
        for band in BANDS:
            value = float(row[band])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid Figure 1 weight: {row}")
            values[band] = value
        loaded.append((dataset, values))

    if tuple(dataset for dataset, _ in loaded) != DISPLAY_ORDER:
        raise ValueError(f"expected datasets in display order: {DISPLAY_ORDER}")
    return loaded


def _profile(weights: dict[str, float], frequency: float) -> float:
    low = (1.0 - frequency) ** 2
    mid = 2.0 * frequency * (1.0 - frequency)
    high = frequency**2
    return weights["w_lo"] * low + weights["w_mid"] * mid + weights["w_hi"] * high


def _svg_text(x: float, y: float, value: str, *, size: int = 14, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}px" text-anchor="{anchor}">{html.escape(value)}</text>'
    )


def _axes(
    parts: list[str],
    *,
    left: float,
    right: float,
    top: float,
    bottom: float,
    y_min: float,
    y_max: float,
    y_label: str,
    x_label: str,
) -> None:
    def y_coord(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    for tick in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00):
        y = y_coord(tick)
        parts.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{right:.1f}" y2="{y:.1f}" '
            'stroke="#dddddd" stroke-width="1"/>'
        )
        parts.append(_svg_text(left - 10, y + 5, f"{tick:.2f}", size=12, anchor="end"))
    parts.extend(
        [
            f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{bottom:.1f}" '
            'stroke="#333333" stroke-width="1.2"/>',
            f'<line x1="{left:.1f}" y1="{bottom:.1f}" x2="{right:.1f}" y2="{bottom:.1f}" '
            'stroke="#333333" stroke-width="1.2"/>',
            _svg_text(18, (top + bottom) / 2, y_label, size=14, anchor="middle"),
            _svg_text((left + right) / 2, bottom + 62, x_label, size=14, anchor="middle"),
        ]
    )


def render(input_path: Path, output_path: Path) -> None:
    rows = _load_rows(input_path)
    width, height = 1200, 520
    plot_top, plot_bottom = 82, 422
    y_min, y_max = 0.70, 1.00
    left_panel = (92.0, 548.0)
    right_panel = (662.0, 1136.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 34, "Final band weights and induced spectral profiles", size=22, anchor="middle"),
    ]

    _axes(
        parts,
        left=left_panel[0],
        right=left_panel[1],
        top=plot_top,
        bottom=plot_bottom,
        y_min=y_min,
        y_max=y_max,
        y_label="weight",
        x_label="dataset",
    )
    _axes(
        parts,
        left=right_panel[0],
        right=right_panel[1],
        top=plot_top,
        bottom=plot_bottom,
        y_min=y_min,
        y_max=y_max,
        y_label="profile a*(xi)",
        x_label="normalized disagreement frequency xi",
    )

    group_width = (left_panel[1] - left_panel[0]) / len(rows)
    bar_width = min(42.0, group_width * 0.20)
    for group_index, (dataset, weights) in enumerate(rows):
        center = left_panel[0] + (group_index + 0.5) * group_width
        for band_index, band in enumerate(BANDS):
            x = center + (band_index - 1) * (bar_width + 5) - bar_width / 2
            y = plot_bottom - (weights[band] - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{plot_bottom - y:.1f}" fill="{COLORS[band]}"/>'
            )
        parts.append(_svg_text(center, plot_bottom + 28, dataset, size=14, anchor="middle"))

    for dataset, weights in rows:
        points = []
        for index in range(101):
            frequency = index / 100.0
            x = right_panel[0] + frequency * (right_panel[1] - right_panel[0])
            value = _profile(weights, frequency)
            y = plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
            points.append(f"{x:.1f},{y:.1f}")
        color = DATASET_COLORS[dataset]
        dash = DATASET_DASHES[dataset]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
            f'stroke-width="2.5"{dash_attr}/>'
        )

    for index, label in enumerate(DISPLAY_ORDER):
        color = DATASET_COLORS[label]
        dash = DATASET_DASHES[label]
        x = right_panel[0] + 18
        y = 58 + index * 22
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<line x1="{x:.1f}" y1="{y - 5:.1f}" x2="{x + 34:.1f}" y2="{y - 5:.1f}" '
            f'stroke="{color}" stroke-width="2.5"{dash_attr}/>'
        )
        parts.append(_svg_text(x + 44, y, label, size=13))

    for index, (label, band) in enumerate((("w_lo*", "w_lo"), ("w_mid*", "w_mid"), ("w_hi*", "w_hi"))):
        x = left_panel[0] + 32 + index * 92
        y = 58
        parts.append(
            f'<rect x="{x:.1f}" y="{y - 12:.1f}" width="13" height="13" fill="{COLORS[band]}"/>'
        )
        parts.append(_svg_text(x + 18, y, label, size=13))

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "figure1_band_weights.csv")
    parser.add_argument("--output", type=Path, default=root / "figure1_band_weights.svg")
    args = parser.parse_args()
    render(args.input, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
