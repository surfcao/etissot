# Public functions and workflows

## Geometry

- `etissot.fit_ellipse(profile, bearings, legacy=False)`: fit one finite, nonnegative, nonzero directional profile on at least five bearings equally spaced over a complete circle; return `A_fit`, `theta_fit`, eigenvalues, residual diagnostics and `fallback`.
- `etissot.fit_many(profiles, bearings, legacy=False)`: apply the fitter to rows of a two-dimensional array; its columns must match the bearings.
- `etissot.fitted_tensor(fit)`: reconstruct the 2-by-2 tensor in the north/east bearing basis.
- `etissot.refine_radius(profile_at_radius, initial_radius, bearings, tolerance=0.005, max_halvings=16)`: retain a full refinement trail, including failed convergence. The initial radius must be positive, tolerance nonnegative, and halving budget a nonnegative integer.
- `etissot.destination_point(lat, lon, bearing, r, R=6371.0)`: spherical destination; angles in radians and distance in kilometres.
- `etissot.geometry.references.equal_earth_metric(lat, lon)`: analytic reference anisotropy and bearing; input coordinates in degrees.

Near-isotropic bearings are not meaningful. Fallback values are sampled-minimum summaries, not ordinary harmonic tensor fits. Preserve flags in downstream displays and analysis. The legacy estimator is available only for explicit historical comparison.

## A specified encoder and location

`etissot.measure_indicatrix(encoder, latitude, longitude, radius_km, n_directions=32)` composes geodesic sampling and the manuscript's harmonic fitter. `encoder(lat, lon)` returns a one-dimensional NumPy-compatible feature vector; coordinates are radians, and radius is kilometers. The adapter determines normalization, feature selection, longitude wrapping, and evaluation mode. It should return deterministic outputs of consistent dimension at the center and all probe points.

The result contains `profile`, `bearings`, `radius_km`, `feature_dimension`, `fit`, `tensor`, `unconstrained_tensor`, and `principal_stretches`. The fit dictionary contains `A_fit` (anisotropy), `theta_fit` (bearing clockwise from north, radians modulo pi), `ne` (normalized squared-profile RMS residual), `asym` (first-harmonic asymmetry), eigenvalues, and `fallback`. Principal stretches are feature change per kilometer, in descending order; the tensor is in the north/east basis. These are finite-radius estimates. They are not a dimension-reduced embedding.

On fallback, `tensor` and `principal_stretches` are `None` to prevent using a substituted minor value as an ordinary predictive tensor. `unconstrained_tensor` preserves the harmonic fit, while `fit` preserves the fallback summaries used by the existing estimator. A constant output, nonfinite output, inconsistent dimension, invalid radius, or polar center raises `ValueError`. Near-isotropic bearings have no stable interpretation. The function does not certify continuity across seams, smoothness in the probe disc, radius convergence, or task performance; those checks remain part of the measurement protocol. See `examples/measure_encoder.py` for a runnable adapter.

## Statistics and prediction

`etissot.statistics.spatial` provides `wcorr`, `wquantile` and geographic block-resampling utilities. Its default grid matches the distributed study; do not assume these grid defaults apply to arbitrary data.

`etissot.prediction.gaussian_process.GaussianProcess(x, y, a, b, e, kind)` defines the study's Matérn covariance, likelihood, multistart optimization and prediction. `x` contains projected coordinates in kilometres, `a` anisotropy, `b` bearing in radians in that projected frame, and `e` optional embedding features. Model kinds are listed in `VARIANTS`. The evaluator transforms geographic bearings to the projected frame; passing untransformed bearings is not equivalent.

## Data and output control

`Workspace(data, output)` resolves two non-overlapping locations. `prepare()` seeds reference outputs without altering data. Workspace-based diagnostic/report modules expose `run(workspace)`; boundary and dense checkpoint evaluations also take `model_dir`. Use `pipeline.run()` for automatic preparation and ordering. Individual workspace-based module calls require a prepared workspace. The standalone case-study modules instead take data and output paths. Their `collect(data, models, output=None)` functions save new measurements outside the input data directory (by default, `outputs/measurements` beside `data`), return the new file path, and reject overlapping input/output paths. Their command-line `--collect` mode analyzes the new file; the default mode analyzes distributed measurements.

Library imports perform no study-data reads, computations over saved profiles, or model downloads. Plotting libraries are loaded when their reporting modules are requested. Model weights are loaded only through the explicit optional evaluation path.

The weighted correlation function rejects invalid weights and mismatched arrays, omits nonfinite paired observations, and returns `NaN` for constant variables, whose correlation is undefined. The bootstrap rejects undefined point correlations and uses finite replicate correlations. Study checkpoint evaluations validate files against the distributed SHA-256 manifest; direct `load_all` calls may pass `checkpoint_manifest` to request the same check.

Scientific study workflows and numerical validators require Python without `-O` or `-OO`, which disable numerical assertions. They reject optimized execution rather than reporting skipped checks as passed.

## Extending the package

`EncoderAdapter(encode, name="custom", input_units="radians", coordinate_order="latlon", wrap_longitude=False, normalization="raw", feature_transform=None, domain_check=None, metadata={})` defines a batched encoder independently of the manuscript models. Its `encode` callable receives an N-by-2 NumPy coordinate array. The selected output must convert to an N-by-d real, finite feature matrix of consistent dimension. `feature_transform` receives the model's original output, so it can select dictionary fields, concatenate components, or detach and move framework tensors to NumPy before measurement. Complex-valued encoders must explicitly represent their real and imaginary components as real feature channels.

Canonical measurement coordinates are latitude/longitude in radians. The adapter converts these to the declared input units and order; optional longitude wrapping uses [-pi, pi). Normalization is applied only when explicitly requested as `unit`; it changes the measured representation. No adapter loads a checkpoint, moves a model to a device, changes its precision or switches its evaluation mode. These operations stay in the user-supplied callable. A conditioned encoder can hold time, altitude or other inputs fixed in that callable and record them in metadata; the measured domain remains two-dimensional geography.

`domain_check(latitude, longitude, radius_km)` receives canonical coordinates and must return a Boolean for the whole probe disc. Every site's check runs before encoder evaluation starts. With no callback, admissibility is unchecked. Wrapping alone does not establish continuity across a chart boundary. Metadata must be finite JSON-compatible values; use it to identify checkpoint/version, measured layer or output head, precision and conditioning inputs. Results record the declared conventions and an independent metadata copy; they do not compute or authenticate a checkpoint hash automatically.

`measure_locations(encoder, latitudes, longitudes, radius_km, n_directions=32, *, batch_size=1024)` accepts aligned one-dimensional coordinate arrays and either a positive scalar radius or one radius per site. It evaluates rings in chunks and retains at most a chunk and one ring of full feature vectors, infers embedding dimension, checks dimensions across chunks, and returns one indicatrix dictionary per site in input order. It accepts both an `EncoderAdapter` and the original scalar callable. An invalid site or output fails the call; partial results are not returned. Each result includes `result_schema_version=1` and `encoder` provenance (or `None` for a bare scalar callable). Existing tensor/fit fields retain their meaning. No study workspace or data is required.

The generic core requires NumPy. The `study`, `models` and `test` installation extras provide the respective optional dependencies. The pinned model roster, sampling grids, cached data and numerical claim contract remain the separate manuscript workflow. See `examples/custom_encoder.py` for a complete example.

## Checkpoint acquisition

`etissot.models.checkpoints` provides standard-library-only acquisition and verification of the fixed study roster. Its installed `checkpoints.json` records exact hashes, sizes, publisher sources, source revisions, feature conventions and checkpoint redistribution decisions. These functions do not deserialize weights or load encoder frameworks:

- `load_manifest(path=None)` reads and validates the installed catalog or an explicit replacement.
- `download_models(model_dir, *, manifest=None, families=None, encoders=None, replace=False, dry_run=False, timeout=60)` obtains the selected checkpoints and any declared runtime source files, verifies them before atomic installation, preserves differing existing files unless replacement is explicit, and returns a receipt. Non-dry runs also save `acquisition.json` under the model directory. Individual failures are recorded and make the receipt's status `failed`; successful files remain reusable.
- `verify_models(model_dir, *, manifest=None, families=None, encoders=None)` checks existing checkpoint and external-source bytes and reports missing/mismatched entries without writes. Runtime dependency readiness is not assessed.
- `bundle_models(model_dir, output, *, manifest=None, families=None, encoders=None, resource_root=None, replace=False)` creates a separate ZIP containing only verified eligible checkpoints, their license evidence, attribution/catalog and checksum records. Default selection lists omitted ineligible entries; explicitly selecting an ineligible checkpoint fails. An existing output requires explicit replacement.

Families and encoder ids are lists, using the identifiers shown by `etissot models list`. Public source URLs use HTTPS; manifest paths and archive member paths must stay relative to their declared roots. The same functions are exposed through `etissot models list|download|verify|bundle`. See `docs/models.md` for the permitted companion bundle and the pinned study/runtime requirements. No network access occurs during ordinary geometry measurements or cached manuscript reproduction.
