# Public functions and workflows

## Geometry

- `etissot.fit_ellipse(profile, bearings, legacy=False)`: fit one finite, nonnegative, nonzero directional profile on at least five bearings equally spaced over a complete circle, and return `A_fit`, `theta_fit`, eigenvalues, residual diagnostics and `fallback`. The fallback bound accounts for both odd and even numbers of bearings. Profiles whose squared stretches leave the floating-point range raise an error; rescale the encoder output, since anisotropy and bearing do not depend on a common factor.
- `etissot.fit_many(profiles, bearings, legacy=False)`: apply the fitter to rows of a two-dimensional array whose columns match the bearings.
- `etissot.fitted_tensor(fit)`: reconstruct the 2-by-2 tensor in the north/east bearing basis.
- `etissot.refine_radius(profile_at_radius, initial_radius, bearings, tolerance=0.005, max_halvings=16)`: retain the refinement trail, including failed convergence. If a finer level cannot be fitted (e.g., identical low-precision features at a tiny radius), refinement stops with `converged=False`, the last fitted level and a `failed_level` entry that gives its radius and error. The initial radius must be positive, the tolerance nonnegative and the halving budget a nonnegative integer.
- `etissot.destination_point(lat, lon, bearing, r, R=6371.0)`: spherical destination (radians, kilometers); output longitudes are not wrapped.
- `etissot.geometry.probing.disc_crossings(lat, lon, r)`: whether a probe disc contains a pole or crosses the antimeridian.
- `etissot.geometry.references.equal_earth_metric(lat, lon)`: analytic reference anisotropy and bearing for input coordinates in degrees.

Bearings at near-isotropic locations are not meaningful, and fallback values are sampled-minimum summaries rather than ordinary harmonic tensor fits, so preserve these flags in downstream displays and analysis. The legacy estimator is available only for explicit historical comparison.

## A specified encoder and location

`etissot.measure_indicatrix(encoder, latitude, longitude, radius_km, n_directions=32, *, seams='warn')` combines geodesic sampling with the harmonic fitter of the manuscript. `encoder(lat, lon)` takes radians and returns a one-dimensional NumPy-compatible feature vector; the radius is in kilometers. The adapter determines normalization, feature selection, longitude wrapping and evaluation mode, and should return deterministic outputs of consistent dimension at the center and all probe points.

The result contains `profile`, `bearings`, `radius_km`, `feature_dimension`, `fit`, `tensor`, `unconstrained_tensor` and `principal_stretches`. The fit dictionary contains `A_fit` (anisotropy), `theta_fit` (bearing clockwise from north, in radians within [0, pi)), `ne` (normalized squared-profile RMS residual), `asym` (first-harmonic asymmetry), eigenvalues and `fallback`. Principal stretches are feature change per kilometer in descending order, and the tensor is in the north/east basis. These are finite-radius estimates, not a dimension-reduced embedding.

On fallback, `tensor` and `principal_stretches` are `None`, so a substituted minor value cannot be used as an ordinary predictive tensor; `unconstrained_tensor` preserves the harmonic fit, while `fit` preserves the fallback summaries used by the existing estimator. A constant, nonfinite or inconsistently dimensioned output, an invalid radius or a polar center raises `ValueError`. Bearings at near-isotropic locations have no stable interpretation. Each result also reports `contains_pole` and `crosses_antimeridian` for its probe disc; such discs warn by default, raise with `seams='error'` and pass silently with `seams='ignore'` (for encoders known to be continuous there). The function does not certify smoothness in the probe disc, radius convergence or task performance, which remain part of the measurement protocol. See `examples/measure_encoder.py` for a runnable adapter.

## Statistics and prediction

`etissot.statistics.spatial` provides `wcorr`, `wquantile` and geographic block-resampling utilities. Its default grid matches the distributed study and should not be assumed for arbitrary data.

`etissot.prediction.gaussian_process.GaussianProcess(x, y, a, b, e, kind)` defines the study's Matérn covariance, likelihood, multistart optimization and prediction. `x` contains projected coordinates in kilometers, `a` the anisotropy, `b` the bearing in radians in that projected frame and `e` optional embedding features; model kinds are listed in `VARIANTS`. The evaluator transforms geographic bearings to the projected frame, so passing untransformed bearings is not equivalent.

## Data and output control

`Workspace(data, output)` resolves two non-overlapping locations, and `prepare()` seeds reference outputs without altering the data. Workspace-based diagnostic and report modules expose `run(workspace)`, and the boundary and dense checkpoint evaluations also take `model_dir`. Use `pipeline.run()` for automatic preparation and ordering, since individual calls to workspace-based modules require a prepared workspace. Standalone case-study modules instead take data and output paths. Their `collect(data, models, output=None)` functions save new measurements outside the input data directory (by default, `outputs/measurements` beside `data`), return the path of the new file and reject overlapping input and output paths. Their command-line `--collect` mode analyzes the new file, while the default mode analyzes the distributed measurements.

Library imports perform no study-data reads, computations over saved profiles or model downloads. Plotting libraries are loaded only when their reporting modules are requested, and model weights are loaded only through the explicit optional evaluation path.

The weighted correlation function rejects invalid weights and mismatched arrays, omits nonfinite paired observations and returns `NaN` for constant variables, whose correlation is undefined. The bootstrap rejects undefined point correlations and uses the finite replicate correlations. Study checkpoint evaluations validate files against the distributed SHA-256 manifest, and direct `load_all` calls may pass `checkpoint_manifest` for the same check.

Scientific study workflows and numerical validators require Python without `-O` or `-OO` (which disable numerical assertions), and they reject optimized execution rather than report skipped checks as passed.

## Extending the package

`EncoderAdapter(encode, name="custom", input_units="radians", coordinate_order="latlon", wrap_longitude=False, normalization="raw", feature_transform=None, domain_check=None, metadata={})` defines a batched encoder independently of the manuscript models. Its `encode` callable receives an N-by-2 NumPy coordinate array, and the selected output must convert to an N-by-d real, finite feature matrix of consistent dimension. `feature_transform` receives the model's original output, so it can select dictionary fields, concatenate components, or detach framework tensors and move them to NumPy before measurement. Complex-valued encoders must represent their real and imaginary components explicitly as real feature channels.

Canonical measurement coordinates are latitude and longitude in radians, which the adapter converts to the declared input units and order; optional longitude wrapping uses [-pi, pi). Normalization is applied only when explicitly requested as `unit`, since it changes the measured representation. No adapter loads a checkpoint, moves a model to a device, changes its precision or switches its evaluation mode; these operations stay in the user-supplied callable. A conditioned encoder can hold time, altitude or other inputs fixed in that callable and record them in the metadata, while the measured domain remains two-dimensional geography.

`domain_check(latitude, longitude, radius_km)` receives canonical coordinates and must return a Boolean for the whole probe disc. Every site is checked before encoder evaluation starts; without a callback, admissibility is unchecked. Wrapping alone does not establish continuity across a chart boundary. Metadata must be finite JSON-compatible values, for example the checkpoint or version, measured layer or output head, precision and conditioning inputs. Results record the declared conventions and an independent metadata copy but do not compute or authenticate a checkpoint hash.

`measure_locations(encoder, latitudes, longitudes, radius_km, n_directions=32, *, batch_size=1024, seams='warn')` accepts aligned one-dimensional coordinate arrays and either a positive scalar radius or one radius per site. It evaluates rings in chunks, retaining at most one chunk and one ring of full feature vectors, infers the embedding dimension, checks dimensions across chunks and returns one indicatrix dictionary per site in input order. It accepts both an `EncoderAdapter` and the original scalar callable. An invalid site or output fails the call without partial results. Each result includes `result_schema_version=1` and the `encoder` provenance (or `None` for a bare scalar callable), while the existing tensor and fit fields retain their meaning. No study workspace or data is required.

The generic core requires only NumPy; the `study`, `models` and `test` extras provide the optional dependencies. The pinned model roster, sampling grids, cached data and numerical claim contract belong to the separate manuscript workflow. See `examples/custom_encoder.py` for a complete example.

## Checkpoint acquisition

`etissot.models.checkpoints` acquires and verifies the fixed study roster using only the standard library. Its installed `checkpoints.json` records exact hashes, sizes, publisher sources, source revisions, feature conventions and checkpoint redistribution decisions. These functions neither deserialize weights nor load encoder frameworks:

- `load_manifest(path=None)` reads and validates the installed catalog or an explicit replacement.
- `download_models(model_dir, *, manifest=None, families=None, encoders=None, replace=False, dry_run=False, timeout=60)` obtains the selected checkpoints and any declared runtime source files, verifies them before atomic installation, preserves differing existing files unless replacement is explicitly requested and returns a receipt. Non-dry runs also save `acquisition.json` under the model directory. Individual failures are recorded and set the receipt's status to `failed`, while successful files remain reusable.
- `verify_models(model_dir, *, manifest=None, families=None, encoders=None)` checks existing checkpoint and external-source bytes and reports missing or mismatched entries without writes; runtime dependency readiness is not assessed.
- `bundle_models(model_dir, output, *, manifest=None, families=None, encoders=None, resource_root=None, replace=False)` creates a separate ZIP that contains only verified eligible checkpoints, together with their license evidence, attribution, catalog and checksum records. The default selection lists the omitted ineligible entries, while an explicit selection of an ineligible checkpoint fails. An existing output is replaced only on request.

Families and encoder ids are lists, using the identifiers shown by `etissot models list`. Public source URLs use HTTPS, and manifest paths and archive member paths must stay relative to their declared roots. The same functions are exposed through `etissot models list|download|verify|bundle`; see `docs/models.md` for the permitted companion bundle and the pinned study and runtime requirements. No network access occurs during ordinary geometry measurements or cached manuscript reproduction.
