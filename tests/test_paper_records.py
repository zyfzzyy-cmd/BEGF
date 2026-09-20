from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper_results.csv"
PROVENANCE = ROOT / "paper_result_provenance.csv"
METRICS = ("ACC_percent", "ARI_percent", "NMI_percent")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_paper_results_are_well_formed_and_unique() -> None:
    rows = _read_csv(RESULTS)
    assert rows
    keys = [(row["dataset_id"], row["method"]) for row in rows]
    assert len(keys) == len(set(keys))

    for row in rows:
        valid = row["valid"]
        assert valid in {"YES", "NO"}
        values = [row[name].strip() for name in METRICS]
        if valid == "YES":
            assert all(values)
            assert all(math.isfinite(float(value)) for value in values)
        else:
            assert values == ["", "", ""]


def test_unavailable_results_match_provenance_exactly() -> None:
    results = _read_csv(RESULTS)
    provenance = _read_csv(PROVENANCE)

    unavailable = {
        (row["dataset_id"], row["dataset"], row["method"])
        for row in results
        if row["valid"] == "NO"
    }
    provenance_pairs = {
        (row["dataset_id"], row["dataset"], row["method"])
        for row in provenance
    }

    assert unavailable == provenance_pairs
    assert len(provenance_pairs) == len(provenance)
    assert all(row["availability"] == "NO_VALID_RESULT" for row in provenance)


def test_usps_sdgca_matches_final_icaspp_table() -> None:
    rows = _read_csv(RESULTS)
    matches = [
        row
        for row in rows
        if row["dataset_id"] == "D6" and row["dataset"] == "USPS" and row["method"] == "SDGCA"
    ]
    assert len(matches) == 1
    row = matches[0]
    assert row["valid"] == "YES"
    assert float(row["ACC_percent"]) == 63.45
    assert float(row["ARI_percent"]) == 52.51
    assert float(row["NMI_percent"]) == 59.36

    provenance = _read_csv(PROVENANCE)
    assert not any(
        row["dataset_id"] == "D6" and row["dataset"] == "USPS" and row["method"] == "SDGCA"
        for row in provenance
    )
