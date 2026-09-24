# Band-Energy Adaptive Graph Filtering for Ensemble Clustering

This repository is the public companion for the ICASSP 2027 paper **Band-Energy Adaptive Graph Filtering for Ensemble Clustering (BEGF)**.

The released implementation follows the BEGF formulation used in the final paper:

\[
X_0=\frac{1}{\sqrt m}[F^{(1)},\ldots,F^{(m)}],\qquad
F^{(r)}=H^{(r)}D^{(r)-1/2},
\]

\[
L=2(I-X_0X_0^\top),\qquad
LZ=2\left[Z-X_0(X_0^\top Z)\right].
\]

The three Bernstein band operators are evaluated from successive actions of the same matrix-free operator:

\[
R_{\mathrm{lo}}=L-L^2+\tfrac14L^3,\quad
R_{\mathrm{mid}}=L^2-\tfrac12L^3,\quad
R_{\mathrm{hi}}=\tfrac14L^3.
\]

No sample-by-sample \(n\times n\) graph is constructed by the production code.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/demo_synthetic.py
python -m pytest -q
```

The demo is deterministic and uses only synthetic base partitions. It verifies the operator contract and runs the complete BEGF MM/CG path without requiring benchmark datasets.

Benchmark datasets and precomputed partition pools are not redistributed in this release. The public tests do not regenerate the benchmark results. The production operator stores `X0` as a sparse CSR matrix, applies it through sparse-dense products, and never materializes an \(n\times n\) sample graph. The final readout performs row-wise L2 normalization followed by k-means.

## Frozen ICASSP 2027 configuration

The final readout uses:

```python
labels = cluster_embedding(
    result.signal,
    n_clusters=k,
    random_state=9002,
    n_init=20,
)
```

The remaining k-means settings are `max_iter=300`, `tol=1e-4`, and Lloyd updates. The frozen benchmark protocol records CG with `rtol=1e-10`, `atol=0`, and the backend-default iteration cap.

For D1--D10, the final paper uses ensemble sizes

`30 / 20 / 120 / 10 / 10 / 10 / 70 / 10 / 10 / 10`

and dataset-specific \(\mu\) values

`1 / 0.01 / 0.01 / 1 / 1 / 100 / 10 / 10 / 10 / 10`.

The dataset order is Yale, BBCSport, ProteinFold, COIL20, Communities, Obesity, PAMAP2, EMNIST-Balanced, EMNIST-Letters, and Walking. BBCSport uses two supplied views with feature dimensions 3,183 and 3,203.

## Repository map

- `src/begf/ensemble.py`: normalized ensemble representation \(X_0\).
- `src/begf/operators.py`: matrix-free \(L\), band actions, energies, and validation.
- `src/begf/solver.py`: convex pseudo-Huber objective, MM weights, and SPD CG updates.
- `src/begf/readout.py`: row normalization and k-means consensus readout.
- `scripts/demo_synthetic.py`: small end-to-end public example.
- `tests/`: operator, solver, readout, sparse-release, and final paper-record checks.
- `data/README.md`: benchmark input format and release boundary.
- `paper_results.csv`: final displayed two-decimal Table 2 record.
- `paper_protocol.json`: final frozen paper configuration.
- `paper_ablation.csv`: final Table 3 macro-average ablation record.
- `figure1_band_weights.csv`: Figure 1 band-weight source.
- `scripts/plot_figure1.py`: Figure 1 plotting script.

## Final ICASSP 2027 paper record

`paper_results.csv` mirrors the final manuscript Table 2. The formal comparison methods are:

- HGPA
- EAC
- KCC
- ECPCS-HC
- ECPCS-MC
- SDGCA
- RANGE
- BEGF

The final manuscript reports BEGF as best or tied-best in **16 of 30** dataset-metric cells and second-best in **12** additional cells. The public CSV stores the displayed two-decimal values and validity mask used by the manuscript. It does **not** infer unreleased numerical precision from rounded values.

Unavailable method-dataset pairs remain empty with `valid=NO`; no missing result is imputed. Exact paper benchmarking still requires the frozen or precomputed ensemble inputs, which are not redistributed when licensing, size, or provenance constraints prevent release.

`paper_ablation.csv` and `figure1_band_weights.csv` transcribe the verified diagnostics reported in the manuscript; the public tests validate record consistency but do not regenerate the benchmark experiments.

See `data/README.md` for the benchmark input format and release boundary.

## License

This repository is released under the MIT License. See `LICENSE` for details.
