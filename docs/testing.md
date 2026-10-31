# Unit tests

From the repository root (or the extracted release directory), run the following after installing the package:

```sh
python -m pip install ".[test]"
python -m unittest discover -s tests -v
```

The suite uses the standard-library unittest runner, the package dependencies and the `test` extra (`einops`, needed by the vendored harmonic interface). It needs no pytest, model downloads, checkpoints or study NPZ files, since the tests use analytic examples, seeded synthetic data and temporary directories.

| File | Coverage |
|---|---|
| test_geometry.py | Isotropic and rotated metrics, scale invariance, between-bearing minor axes, batch fitting, fallback flags, harmonic asymmetry, refinement convergence/failure/streak reset, geodesic distances, unit-sphere chord stretch and Equal Earth derivatives |
| test_prediction.py | Closed-form Matérn covariance, positive definiteness, nested models at zero tensor exponent, finite-difference likelihood gradients, failed-optimizer start preservation and independent prediction-score calculations |
| test_statistics.py | Weighted correlation against explicitly replicated observations, masks and missing values, weighted quantile interpolation, deterministic geographic block resampling |
| test_workflows.py | Input/output separation, reference preservation, missing data, selected CLI analysis, success/failure reports, stop-on-failure and check-only dispatch |
| test_public_api.py | Rotated tensor reconstruction, path overlap rejection and output-fit precedence |

The suite has 152 test methods in 16 files, plus parameterized subcases; the table lists the core files. Coverage also includes zero-weight and invalid-weight quantiles, every registered CLI analysis, environment preflight, round-off-tolerant dense validation that still rejects substantive changes, strict versus numerical-only figure checking and mandatory verification of relocated package documents.

For full scientific reproduction with the provided study data, run separately:

```sh
etissot reproduce --data ./data --output ./outputs
```

The unit suite uses small, independent examples, whereas the full reproduction also validates saved fits, dense profiles, tables and downstream bootstrap results. Code-review, restructuring and earlier unit-test records are kept with the author's working files rather than in this repository.

Five tests ensure that clean preparation cannot prefill finished outputs, that missing or stale manuscript blocks fail while valid blocks pass, and that isotropic neighbor geometry is recovered; the clean-wheel reproduction of 0.2.0 is recorded in `docs/validation/reproduction_v0.2.0.json`.

Additional coverage includes `measure_indicatrix`, matched neighborhood tensors and inverse-stretch ellipses, SatCLIP input rescaling, rejection of nonfinite manuscript claims, invalid fit, probe and GP inputs, numerical overflow in residual diagnostics, small-block identifier collisions, scoped loader settings, checkpoint hash rejection, preservation of input measurements, rejection of optimized Python and standalone wheel verification (22 test methods plus parameterized failure cases).

Tests of the encoder extension cover coordinate conversion, output selection, explicit normalization, arbitrary feature dimensions, equivalence of batched and scalar evaluation, per-site radii, provenance, domain checks, chunk consistency and core imports without study or ML framework dependencies.

A synthetic test of the cached SatCLIP report reproduces the complete report path and catches unescaped LaTeX braces in an interpolated reference, without real model weights or study data.

The October 2026 review added `tests/test_edge_cases.py` (bearing range, extreme scales, odd-ring fallback bound, seam and pole policy, lon/lat array layout, bootstrap pool) and `tests/test_reproducibility_guards.py` (seeded reference copies, generated-block markers, computed seam prose). A second review added tests of constant-variable correlation, float32 bearings, subnormal stretches, refinement levels that cannot be fitted, scalar Equal Earth input, combined-kernel priors and refusal of modified CSP sources. Checker tests cover the prose-number register (changed, added and removed numbers, commented copies, bounds and stated approximations, and links carried over after a register update), required generated blocks, every table environment and the analytic-table hash, failed-verification reports, output freshness and refusal of an output folder inside the manuscript.
