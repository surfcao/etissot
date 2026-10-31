# Reproduce the manuscript results

Version 0.3.0 regenerates the 16 figures, 20 tables and 19 generated blocks of the manuscript and the package-only additional results. The manuscript sources are not yet public, so the manuscript check described below (`--manuscript`) can be run only after publication.

## Clean reproduction

```sh
python -m pip install -r requirements-reproduce.txt
python -m pip install .
python -m unittest discover -s tests -v
etissot reproduce --data data --output reproduced
```

With the manuscript sources, add `--manuscript manuscript --numerical-only` to the last command.

Use a new output directory, since the workflow regenerates outputs from declared inputs rather than prefilling finished tables, summaries or figures. Standalone `analyze` steps may seed distributed reference outputs as prerequisites; these copies are listed in `.seeded_reference.json`, and `reproduce`, `check` and manuscript verification refuse a folder in which any of them has not been regenerated. `check` re-runs the numerical checks on a folder whose last full `reproduce` completed (recorded in `reproduce_report.json`, which single `analyze` runs and checks never replace). Each run records its outcome in `run_report.json`, per-step logs and `environment.json`.

Manuscript verification accepts only tables, figures and summaries written by the current reproduction, reads the contract and baselines from the data folder and refuses an output folder inside the manuscript or the package source. It checks the required generated blocks, the complete CSP table against an independently regenerated table, the hash of the hand-written analytic table, a set of summary checks for values stated in words or shown only in figures, and a register of every number in the live prose (excluding comments and generated blocks); the register is kept with the unpublished sources. Each result number in the register is recomputed from the regenerated summaries and must match at its written precision (or as a stated bound or approximation), while design settings, formulas, citations, labels and a few figure readings are pinned. An edit that adds, removes or changes a number therefore fails until the register is updated with `python scripts/update_prose_register.py --data data --manuscript manuscript` (add `--write` after checking its report). The result is saved in `manuscript_verification.json`, and a failed verification replaces any earlier report. The `--numerical-only` option checks all of the above and the presence of all figures and reports pixel differences without failing; without it, pixel identity is mandatory.

For the manuscript check, eight newer figures were rendered with Matplotlib 3.10.9, whereas the study environment uses 3.11.0 for the historical package figures. An isolated renderer avoids changing the primary environment:

```sh
python -m pip install --no-deps --target .figure-renderer -r requirements-figures.txt
export ETISSOT_FIGURE_LIBRARIES="$PWD/.figure-renderer"
etissot reproduce --data data --output reproduced --manuscript manuscript --numerical-only
```

The figure worker records its versions in `summaries/figure_environment.json` and writes only to the output directory. Matching the renderer version does not guarantee identical pixels across platforms; in the local review, three of the eight new figures matched exactly under 3.10.9 and five retained small rasterization differences, so exact pixel verification is not reported as passed.

## Inputs and regeneration scope

- Global, randomized, dense and multiradius profiles, cached Jacobians, candidate distances and saved GP parameters remain the scientific inputs, as described below.
- `data/profiles/common_radius_0p3.npz` contains 32-bearing profiles for all thirteen checkpoints at the common radius, which `checkpoint-audits` refits to recompute the common-radius column.
- `data/measurements/review/csp_precision.json` contains the CSP site-level feature-fit and Jacobian measurements, from which the precision-error quantiles and every displayed table entry are recomputed. Fresh Jacobian evaluation requires the checkpoints.
- The `asymmetry` step refits the fixed-grid profiles in `data/profiles/global.npz` and summarizes the relative first-harmonic amplitude and its share of the residual (`outputs/summaries/asymmetry.json`).
- The `reference-accuracy` step fits the input-reference profiles on the fixed grid (`data/profiles/global.npz`, `data/profiles/input_references.npz`) and compares the fitted anisotropy with the analytic sec φ, isotropic, 2:1, Equal Earth and scaled-harmonic values (`outputs/summaries/reference_accuracy.json`, main Section 4.1).
- The `grid-smoothness` step fits the fixed-grid profiles (`data/profiles/global.npz`) and computes, for each of the thirteen checkpoints, the lag-one correlation of log anisotropy between adjacent grid nodes within 60° latitude, along longitude (excluding the antimeridian pair) and along latitude. It checks that the eight checkpoints summarized on randomized grids separate from the five fixed-grid checkpoints and writes `outputs/summaries/grid_smoothness.json`.
- `data/measurements/review/radius_screening.json` contains increment curves for the thirteen checkpoints at four fixed-grid sites (32 bearings, 34 radii from 1 m to 30 km), collected with `scripts/radius_screening.py`, which verifies each checkpoint against the study manifest. Collection scripts write to `outputs/measurements/review/`; copy the results to `data/measurements/review/` to make them declared inputs. The `radius-screening` step derives the ratios quoted in the supplement relative to the smallest radius (1 m) and, to guard against rounding there, to the screened radius nearest 10 m (11.1 m); it also records the radius used for the 10 km comparison (10.70 km on this grid) and plots `radius_screening.png`. Recollecting the curves requires the checkpoints.
- `data/measurements/review/review_followups.json` is a declared record of checkpoint evaluation audits, including activation crossings and bearing-count sensitivities. These aggregate measurements are formatted into tables; cached reproduction does not reacquire the hidden activations or the original 64-bearing model evaluations.
- `neighborhood-controls` recomputes null rankings, larger-neighborhood comparisons and the query-cluster bootstrap from `data/measurements/neighbourhoods.npz` with fixed seeds.
- `manuscript-figures` regenerates the current schematic, profiles, geographic references, global and regional fields and neighborhood maps, while the historical package figures are generated by their original reporting modules.
- The full workflow replays 360 selected GP objectives and predictions and all downstream bootstrap calculations, but it does not retrain models or reacquire the original source datasets.

Fresh model-dependent review commands are retained under `scripts/review_followups.py` and `scripts/csp_precision_check.py`; they require the pinned external model files and the CSP runtime sources and were not rerun for this release. `scripts/neighbourhood_nulls.py` and `scripts/make_revision_figures.py` retain the original research entry points, whereas the reusable, workspace-based implementations are in `etissot.diagnostics.neighborhood_controls`, `etissot.reporting.checkpoint_audits` and `etissot.reporting.manuscript_figures`.

## Manuscript sources

The manuscript sources are not included in this repository or the public release archives and will be added after publication (`scripts/build_release.py --with-manuscript` includes them in a local archive). With the sources, the commands below update and build the manuscript from a reproduction; note that they edit manuscript files:

```sh
python manuscript/scripts/sync_results.py reproduced
python manuscript/scripts/build.py
```

## Historical Corrections and Editorial Notes

The dated counts and layouts below describe earlier snapshots.

- The synthetic neighbor experiment uses 20 queries in total (five latitudes by four longitudes), not 20 queries per latitude.
- Numerically indistinguishable elongations are treated as ties at a relative tolerance of `1e-10`. The fraction of the gallery with greater encoder-selected elongation is 49–61%, which replaces the round-off-sensitive 52–62%; median elongations are unchanged at the reported precision.
- The directly recomputed maximum spherical refinement error rounds to `3.45e-5` in S2, so the bound below `3.46e-5` in the main text remains valid.

These corrections do not alter the primary encoder audit, the six figures, the downstream conclusions or the kriging comparisons.

## Matched local and global geometry studies (18 September 2026)

The `geometry-cases` analysis is included in the default reproduction. It regenerates two figures, one table and the full per-query and global summary from `data/measurements/geometry_case_studies.npz`. Fresh collection is available with `python -m etissot.diagnostics.geometry_case_studies --collect --models models`, which requires the pinned model dependencies. The cache contains newly measured full-vector distances, independent directional profiles and global feature vectors. At that time, the main manuscript had 8 empirical figures across both documents, 18 tables, 19 generated blocks and 38 tracked numerical prose checks, with 51 source tests.

The `geometry-cases` step also regenerates the neighborhood comparison of the main text for all thirteen released checkpoints and two corrected SatCLIP variants from `data/measurements/neighbourhoods.npz`; fresh collection uses `python -m etissot.diagnostics.neighbourhoods --collect --data data --models models`. Raw SINR and Climplicit features and normalized GeoCLIP, SatCLIP and CSP outputs retain the audit conventions. The global concentration and hubness study remains scoped to GeoCLIP and SatCLIP ViT.

Figure 5 maps the geographic, actual embedding and tensor-predicted selections at a common fixed query. The same `geometry-cases` step generates the main map, three detailed supplementary map sheets (all 15 variants at four longitudes), the bar chart of geographic references and the three-way overlap table, none of which requires new checkpoint measurements.
