import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_RESULTS = ROOT / "paper_results.csv"
FULL_RESULTS = ROOT / "paper_results_full_precision.csv"
METRICS = ("ACC", "ARI", "NMI")
CONSENSUS_METHODS = (
    "CSPA",
    "HGPA",
    "MCLA",
    "EAC",
    "KCC",
    "ECPCS-HC",
    "SDGCA",
    "RANGE",
    "BEGF (Ours)",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _key(row: dict[str, str]) -> tuple[str, str]:
    return row["dataset_id"], row["method"]


def _full_value(row: dict[str, str], metric: str) -> Decimal:
    return Decimal(row[f"{metric}_full_precision"])


def test_full_precision_roundtrip_keys_flags_and_invalid_rows() -> None:
    paper = _read_csv(PAPER_RESULTS)
    full = _read_csv(FULL_RESULTS)

    assert len(full) == len(paper) == 100
    assert len({_key(row) for row in paper}) == len(paper)
    assert len({_key(row) for row in full}) == len(full)
    assert [_key(row) for row in full] == [_key(row) for row in paper]

    for paper_row, full_row in zip(paper, full):
        for field in ("dataset", "n_samples", "n_features", "n_classes", "method", "valid"):
            assert full_row[field] == paper_row[field]
        assert full_row["source"]

        for metric in METRICS:
            full_field = f"{metric}_full_precision"
            display_field = f"{metric}_display_2dp"
            paper_field = f"{metric}_percent"
            if paper_row["valid"] == "YES":
                assert full_row[full_field]
                rounded = _full_value(full_row, metric).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                if rounded == 0:
                    rounded = Decimal("0.00")
                assert f"{rounded:.2f}" == paper_row[paper_field]
                assert full_row[display_field] == paper_row[paper_field]
            else:
                assert full_row[full_field] == ""
                assert full_row[display_field] == ""
                assert paper_row[paper_field] == ""

    usps_sdgca = next(
        row for row in full if row["dataset"] == "USPS" and row["method"] == "SDGCA"
    )
    assert usps_sdgca["valid"] == "YES"
    assert usps_sdgca["ACC_full_precision"] == "63.45450634545063"
    assert usps_sdgca["ARI_full_precision"] == "52.50896143989632"
    assert usps_sdgca["NMI_full_precision"] == "59.36109526836645"
    assert usps_sdgca["ACC_display_2dp"] == "63.45"
    assert usps_sdgca["ARI_display_2dp"] == "52.51"
    assert usps_sdgca["NMI_display_2dp"] == "59.36"
    assert "usps_sdgca_matched_begf10_seed9002_r1/metrics.json" in usps_sdgca["source"]
    assert "FINAL_PAPERSET_FIXED9002_FULL_PRECISION.csv" not in usps_sdgca["source"]


def test_full_precision_ranking_counts_and_usps_ari() -> None:
    rows = _read_csv(FULL_RESULTS)
    by_key = {(row["dataset"], row["method"]): row for row in rows}
    datasets = [row["dataset"] for row in rows if row["method"] == "BEGF (Ours)"]

    total_cells = 0
    best_or_tied_best = 0
    second_best = 0

    for dataset in datasets:
        for metric in METRICS:
            values = {
                method: _full_value(by_key[(dataset, method)], metric)
                for method in CONSENSUS_METHODS
                if by_key[(dataset, method)]["valid"] == "YES"
            }
            assert "BEGF (Ours)" in values
            distinct_descending = sorted(set(values.values()), reverse=True)
            begf_value = values["BEGF (Ours)"]
            best_or_tied_best += begf_value == distinct_descending[0]
            second_best += (
                len(distinct_descending) > 1
                and begf_value == distinct_descending[1]
            )
            total_cells += 1

    assert total_cells == 30
    assert best_or_tied_best == 18
    assert second_best == 7

    assert "Base clusterings (avg.)" not in CONSENSUS_METHODS

    usps_ari = {
        method: _full_value(by_key[("USPS", method)], "ARI")
        for method in CONSENSUS_METHODS
        if by_key[("USPS", method)]["valid"] == "YES"
    }
    usps_distinct_descending = sorted(set(usps_ari.values()), reverse=True)
    assert usps_ari["BEGF (Ours)"] < usps_distinct_descending[1]
    assert usps_ari["BEGF (Ours)"] != usps_distinct_descending[1]
    assert usps_ari["BEGF (Ours)"] == Decimal("52.483490948279166")
    assert usps_ari["MCLA"] == Decimal("52.483192907998155")
