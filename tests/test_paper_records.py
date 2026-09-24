from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper_results.csv"
METRICS = ("ACC_percent", "ARI_percent", "NMI_percent")
DATASETS = (
    "Yale",
    "BBCSport",
    "ProteinFold",
    "COIL20",
    "Communities",
    "Obesity",
    "PAMAP2",
    "EMNIST-Balanced",
    "EMNIST-Letters",
    "Walking",
)
METHODS = (
    "HGPA",
    "EAC",
    "KCC",
    "ECPCS-HC",
    "ECPCS-MC",
    "SDGCA",
    "RANGE",
    "BEGF (Ours)",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def test_final_paper_record_shape_and_validity() -> None:
    rows = _read_csv(RESULTS)
    assert len(rows) == 80
    assert {row["dataset"] for row in rows} == set(DATASETS)
    assert {row["method"] for row in rows} == set(METHODS)

    keys = [(row["dataset_id"], row["dataset"], row["method"]) for row in rows]
    assert len(keys) == len(set(keys))

    for row in rows:
        assert row["valid"] in {"YES", "NO"}
        values = [row[name].strip() for name in METRICS]
        if row["valid"] == "YES":
            assert all(values)
            assert all(math.isfinite(float(value)) for value in values)
        else:
            assert values == ["", "", ""]


def test_final_dataset_and_method_order() -> None:
    rows = _read_csv(RESULTS)
    dataset_order = []
    method_order_by_dataset: dict[str, list[str]] = {}
    for row in rows:
        dataset = row["dataset"]
        if dataset not in method_order_by_dataset:
            dataset_order.append(dataset)
            method_order_by_dataset[dataset] = []
        method_order_by_dataset[dataset].append(row["method"])

    assert tuple(dataset_order) == DATASETS
    assert all(tuple(method_order_by_dataset[dataset]) == METHODS for dataset in DATASETS)


def test_current_manuscript_anchor_cells() -> None:
    rows = _read_csv(RESULTS)
    by_key = {(row["dataset"], row["method"]): row for row in rows}

    bbc_begf = by_key[("BBCSport", "BEGF (Ours)")]
    assert (bbc_begf["ACC_percent"], bbc_begf["ARI_percent"], bbc_begf["NMI_percent"]) == (
        "95.96",
        "89.47",
        "87.75",
    )

    bbc_sdgca = by_key[("BBCSport", "SDGCA")]
    assert (bbc_sdgca["ACC_percent"], bbc_sdgca["ARI_percent"], bbc_sdgca["NMI_percent"]) == (
        "96.51",
        "90.54",
        "88.75",
    )

    emnist_mc = by_key[("EMNIST-Balanced", "ECPCS-MC")]
    assert (emnist_mc["ACC_percent"], emnist_mc["ARI_percent"], emnist_mc["NMI_percent"]) == (
        "27.62",
        "14.32",
        "37.19",
    )

    assert by_key[("COIL20", "ECPCS-MC")]["valid"] == "NO"
    assert by_key[("Communities", "KCC")]["valid"] == "NO"
    assert by_key[("PAMAP2", "SDGCA")]["valid"] == "NO"


def test_display_level_begf_ranking_counts_match_manuscript() -> None:
    rows = _read_csv(RESULTS)
    by_key = {(row["dataset"], row["method"]): row for row in rows}

    best_or_tied_best = 0
    second_best = 0
    total_cells = 0

    for dataset in DATASETS:
        for metric in METRICS:
            values = {
                method: float(by_key[(dataset, method)][metric])
                for method in METHODS
                if by_key[(dataset, method)]["valid"] == "YES"
            }
            begf = values["BEGF (Ours)"]
            distinct = sorted(set(values.values()), reverse=True)
            if begf == distinct[0]:
                best_or_tied_best += 1
            elif len(distinct) > 1 and begf == distinct[1]:
                second_best += 1
            total_cells += 1

    assert total_cells == 30
    assert best_or_tied_best == 16
    assert second_best == 12
