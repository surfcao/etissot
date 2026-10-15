# Unit tests

From the repository root or extracted release directory, after installing the package:

```sh
python -m pip install ".[test]"
python -m unittest discover -s tests -v
```

The suite uses Python's standard-library unittest runner and the package's dependencies and the `test` extra (`einops`, needed by the vendored harmonic interface). No pytest installation, model downloads, checkpoints or study NPZ files are needed. Tests use analytic examples, seeded synthetic data and temporary directories.

| File | Coverage |
|---|---|
| test_geometry.py | Isotropic and rotated metrics, scale invariance, between-bearing minor axes, batch fitting, fallback flags, harmonic asymmetry, refinement convergence/failure/streak reset, geodesic distances, unit-sphere chord stretch and Equal Earth derivatives |
| test_prediction.py | Closed-form Matérn covariance, positive definiteness, nested models at zero tensor exponent, finite-difference likelihood gradients, failed-optimizer start preservation and independent prediction-score calculations |
| test_statistics.py | Weighted correlation against explicitly replicated observations, masks and missing values, weighted quantile interpolation, deterministic geographic block resampling |
| test_workflows.py | Input/output separation, reference preservation, missing data, selected CLI analysis, success/failure reports, stop-on-failure and check-only dispatch |
| test_public_api.py | Rotated tensor reconstruction, path overlap rejection and output-fit precedence |

There are 120 test methods, with additional parameterized subcases. Coverage includes zero-weight and invalid-weight quantiles, every registered CLI analysis, environment preflight, round-off-tolerant dense validation that still rejects substantive changes, strict versus numerical-only figure checking, and mandatory verification of relocated package documents. Current code-review evidence is in `docs/validation/code_review_2026-09-30.json`. Older `UNIT_TEST_RESULTS` files retain historical installed-wheel evidence and do not describe this source snapshot.

For full scientific reproduction using the provided study data, run separately:

```sh
etissot reproduce --data ./data --output ./outputs
```

The unit suite uses small, independent examples; the full reproduction also validates saved fits, dense profiles, tables and downstream bootstrap results. The 30 September code review also reruns full scientific reproduction in a clean directory, including strict manuscript artifact comparison; its report distinguishes this cached-measurement reproduction from fresh checkpoint collection.

Workspace restructuring validation is recorded separately in `docs/maintenance/restructure_2026-09-17.json`.

The five new tests ensure that clean preparation cannot prefill finished outputs, missing or stale manuscript blocks fail, valid blocks pass, and isotropic neighbour geometry is recovered. Current clean-wheel reproduction is recorded in `docs/validation/reproduction_v0.2.0.json`.

Additional coverage includes `measure_indicatrix`, matched neighborhood tensors and inverse-stretch ellipses, SatCLIP input rescaling, rejection of nonfinite manuscript claims, invalid fit/probe/GP inputs, numerical overflow in residual diagnostics, small-block identifier collisions, scoped loader settings, checkpoint hash rejection, preservation of input measurements, optimized-Python rejection, and standalone wheel verification. The review adds 22 test methods plus parameterized failure cases.

Encoder extension tests cover coordinate conversion, output selection, explicit normalization, arbitrary feature dimensions, batched/scalar equivalence, per-site radii, provenance, domain checks, chunk consistency, and core imports without study or ML framework dependencies.

The October software review adds a synthetic cached SatCLIP report test. It reproduces the complete report path and catches unescaped LaTeX braces in an interpolated reference without real model weights or study data.
