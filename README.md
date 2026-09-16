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

The demo is deterministic and uses only synthetic base partitions. It verifies the operator contract and runs the complete BEGF MM/CG path without requiring private datasets.

## Repository map

- `src/begf/ensemble.py`: normalized ensemble representation \(X_0\).
- `src/begf/operators.py`: matrix-free \(L\), band actions, energies, and validation.
- `src/begf/solver.py`: convex pseudo-Huber objective, MM weights, and SPD CG updates.
- `scripts/demo_synthetic.py`: small end-to-end public example.
- `tests/`: operator identities, spectrum contract, and solver checks.
- `data/README.md`: data/provenance contract. No private or benchmark data are bundled.
- `paper/BEGF_ICASSP2027_Final.pdf`: final paper PDF snapshot used to transcribe the result table.
- `paper_results.csv`: Table 2 results for the ten paper datasets, in percent. Empty metric cells preserve unavailable valid outputs; they are not imputed.

The final paper PDF is included as a convenience link: [BEGF ICASSP 2027 final PDF](paper/BEGF_ICASSP2027_Final.pdf). The corresponding final `main.tex` was not identifiable in the supplied workspace, so the paper source is intentionally not rewritten or claimed as part of this release.

The formal comparison in `paper_results.csv` contains Base clusterings (avg.), CSPA, HGPA, MCLA, EAC, KCC, ECPCS-HC, GPEC, SDGCA, OMELET, RANGE, and BEGF. Values are transcribed from the final PDF Table 2; see `data/README.md` for the reproducibility boundary around the frozen benchmark inputs.

No software license is declared yet. Add the authors' chosen license before public publication.
