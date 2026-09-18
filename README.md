# Band-Energy Adaptive Graph Filtering for Ensemble Clustering

This repository is the public, reproducible companion for the ICASSP 2027 paper **Band-Energy Adaptive Graph Filtering for Ensemble Clustering (BEGF)**.

The released implementation follows the final paper exactly:

\[
X_0=\frac{1}{\sqrt m}[F^{(1)},\ldots,F^{(m)}],\qquad
F^{(r)}=H^{(r)}D^{(r)-1/2},
\]

\[
L=2(I-X_0X_0^\top),\qquad
LZ=2\left[Z-X_0(X_0^\top Z)\right].
\]

The three Dirichlet band operators are evaluated from successive actions of the same matrix-free operator:

\[
R_{\mathrm{lo}}=L-L^2+\tfrac14L^3,\quad
R_{\mathrm{mid}}=L^2-\tfrac12L^3,\quad
R_{\mathrm{hi}}=\tfrac14L^3.
\]

No sample-by-sample \(n\times n\) matrix is constructed by the production code.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/demo_synthetic.py
python -m pytest -q
```

The demo is deterministic and uses only synthetic base partitions. It verifies the operator contract and runs the complete BEGF MM/CG path without requiring benchmark datasets.

Benchmark datasets and precomputed partition pools are not redistributed in this release. The public tests do not regenerate the benchmark results. The production operator stores `X0` as a sparse CSR matrix, applies it through sparse-dense products, and never materializes a \(n\times n\) sample graph. The final readout performs row-wise L2 normalization followed by k-means.

## Repository map

- `src/begf/ensemble.py`: normalized ensemble representation \(X_0\).
- `src/begf/operators.py`: matrix-free \(L\), band actions, energies, and validation.
- `src/begf/solver.py`: convex pseudo-Huber objective, MM weights, and SPD CG updates.
- `scripts/demo_synthetic.py`: small end-to-end public example.
- `tests/`: operator identities, spectrum contract, and solver checks.
- `data/README.md`: benchmark input format and release boundary.
- `paper_results.csv`
- `paper_ablation.csv`: exact Table 3 ablation record.
- `figure1_band_weights.csv`: exact band-weight source for the three Figure 1 datasets.
- `scripts/plot_figure1.py`: self-contained SVG plot generator for the band-weight record.

## ICASSP 2027 paper records

The final ICASSP 2027 paper record is represented by:

- `paper_results.csv`: Table 2 result record.
- `paper_protocol.json`: frozen reporting and BEGF configuration.
- `paper_result_provenance.csv`: provenance for unavailable Table 2 results.
- `paper_ablation.csv`: Table 3 ablation record.
- `figure1_band_weights.csv`: Figure 1 mechanism values.
- `scripts/plot_figure1.py`: Figure 1 plotting script.

The public implementation reproduces the BEGF formulation and sparse operators. Exact paper Table 2 benchmarking requires frozen or precomputed ensemble inputs that are not redistributed when licensing, size, or provenance constraints prevent their release; the public package therefore does not claim one-click reproduction of Table 2.

`paper_ablation.csv` and `figure1_band_weights.csv` transcribe verified paper diagnostics; they are not independently reproduced from raw benchmark inputs bundled in this repository.

`paper_results.csv` contains the final Table 2 results for the ten datasets reported in the ICASSP 2027 paper. Missing valid outputs are preserved as empty entries and are not imputed.

The formal comparison methods in `paper_results.csv` are:

- Base clusterings (avg.)
- CSPA
- HGPA
- MCLA
- EAC
- KCC
- ECPCS-HC
- SDGCA
- RANGE
- BEGF

See `data/README.md` for the benchmark input format and release boundary.

No software license is declared yet.
