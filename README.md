# eTissot

**Tissot geometry for geographic embeddings.** The “e” stands for embeddings.

An installable Python package for auditing local geographic distortion in learned location encoders. Version 0.2.0 reproduces the numerical analyses accompanying the manuscript. Author: Guofeng Cao, STARlab, University of Colorado Boulder.

## Repository contents

| Folder | Purpose |
|---|---|
| `src/etissot/` | The Python package |
| `tests/`, `examples/` | Unit tests and small usage examples |
| `scripts/` | Release, figure and supplementary-analysis scripts |
| `data/` | Cached measurements used to reproduce the study (see below) |
| `docs/` | API, data, model, reproducibility and validation documentation |

Model weights are not included; `etissot models download` obtains them from their publishers with SHA-256 verification. The original SINR training and evaluation data are not redistributed.

The repository contains the cached profiles, measurements, reference summaries and Jacobians needed to regenerate the main study tables and figures. The regional prediction and downstream inputs (`data/prediction`, `data/downstream`) exceed GitHub's file-size limit and are distributed with the full release archive (`etissot-0.2.0.zip`, attached to the GitHub release). The manuscript sources used by the numerical manuscript check will be added after publication.

## Package name

eTissot uses `etissot` for Python imports, distribution filenames and the command-line tool. It replaces the earlier `localdistortion` name; update existing imports and commands accordingly. The API and data formats are unchanged. This repository and its local releases provide the package; no public package-index release has been made.

## Install

For geometry measurements with your own encoder, `python -m pip install .` installs the NumPy-based core. It does not require PyTorch, study data, or any named model. Install `.[study]` for the study workflows, `.[models]` for the pinned released-model adapters, or `.[test]` for the full test suite. A custom adapter uses the dependencies of its own encoder.

For the pinned manuscript reproduction environment, from this repository root (or the extracted shareable release):

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduce.txt
python -m pip install .
etissot --help
etissot environment
```

On Windows use `.venv\Scripts\activate`. Python 3.11+ is declared; the tested environment is Python 3.13.2 with the exact versions in `requirements-reproduce.txt`. Other compatible dependency versions have not been numerically validated. Installation needs access to the dependency packages; cached analysis makes no downloads.

The lightweight wheel installs code only. Study data stay in the separate `data/` directory supplied in this full release. This keeps the library usable on your own encoder without installing hundreds of megabytes of reference data. If using the supplied wheel, install it with `python -m pip install dist/etissot-0.2.0-py3-none-any.whl`.

## Optional encoder checkpoints

The wheel and code/data sharing ZIP contain no model weights. To obtain the exact thirteen manuscript checkpoints directly from their publishers, with SHA-256 verification:

```sh
etissot models list
etissot models download --models ./models --dry-run
etissot models download --models ./models
etissot models verify --models ./models
```

These commands do not load models or require the study dependencies. Use `--family satclip` or `--encoder satclip-vit16-l40` for a subset. CSP selection also obtains four pinned runtime source files directly from its publisher. Correct existing files are reused; differing files require explicit `--replace`.

The separate `etissot-models-0.2.0.zip` companion contains eight unmodified checkpoints—GeoCLIP, six SatCLIP variants and Climplicit—with publisher license evidence, model cards, provenance and checksums. Extract it and use the extracted folder as `--models`. SINR and CSP weights are omitted because their redistribution terms were not established; `models download` obtains them directly from the publishers. See [the model guide](docs/models.md) for details, dependencies and bundle verification. Custom encoders use `EncoderAdapter` independently of this fixed study roster.

## Use the library

```python
import numpy as np
from etissot import fit_ellipse, fitted_tensor

bearings = np.linspace(0, 2*np.pi, 32, endpoint=False)
profile = np.sqrt(9*np.cos(bearings-0.4)**2 + np.sin(bearings-0.4)**2)
result = fit_ellipse(profile, bearings)
print(result["A_fit"])  # 3.0
metric = fitted_tensor(result)
```

`geometry.probing` constructs geodesic rings. `geometry.fitting` fits the harmonic tensor and returns anisotropy, bearing, residuals and fallback flags. Bearings and coordinates passed to probe functions are radians; radius is kilometres. `fit_many` accepts an N-by-K array of directional profiles. `refine_radius` takes a callable returning a profile at a given radius; its stopping rule measures successive changes, not guaranteed error against a reference. See `examples/fit_indicatrix.py` and [the API guide](docs/api.md).

### Measure your encoder at a location

`measure_indicatrix` evaluates an encoder directly on a geodesic ring and fits its finite-radius Tissot geometry. The adapter takes latitude and longitude in radians and returns the feature vector whose geometry you want to measure:

```python
from etissot import measure_indicatrix

result = measure_indicatrix(my_encoder, latitude=0.0, longitude=0.0, radius_km=0.3)
print(result["fit"]["A_fit"], result["fit"]["theta_fit"])
print(result["tensor"], result["principal_stretches"])
```

All returned feature components are used. Apply the desired normalization in `my_encoder`; the function does not alter the vector. Bearing is in radians clockwise from north, modulo pi. The tensor uses north/east distances in kilometers. It and the principal stretches are `None` on a flagged fallback; the unconstrained harmonic tensor and fit diagnostics remain available. Nearly isotropic bearings are indeterminate. Check coordinate seams and probe-disc admissibility, and use radius refinement or independent references to assess differential accuracy. See [the runnable custom-encoder example](examples/measure_encoder.py) and [API details](docs/api.md). No study data or released checkpoints are needed for a custom encoder.

### New encoders and changed checkpoints

Wrap a batched encoder with `EncoderAdapter`; the model can come from any framework. Declare its input units/order and the output representation, and record its version or checkpoint in metadata. A feature transform can select a field from a model's output and convert tensors to NumPy. For PyTorch, the wrapper should handle its device, precision, evaluation mode and conversion with `detach().cpu().numpy()`.

```python
from etissot import EncoderAdapter, measure_locations

encoder = EncoderAdapter(
    my_model_forward, name="my-encoder-v2",
    input_units="degrees", coordinate_order="lonlat",
    normalization="raw",  # choose "unit" only when that is the measured vector
    metadata={"checkpoint": "my-checkpoint-v2", "representation": "pre-head features"},
)
results = measure_locations(encoder, latitudes_radians, longitudes_radians,
                            radius_km=0.3, n_directions=32, batch_size=1024)
```

The number of embedding components is inferred from the output. Locations, radii and angular resolution are supplied by the caller; results retain their order and adapter metadata. Changing model weights, an output head or normalization defines a new measured representation, so update that metadata when comparing versions. An optional `domain_check` can reject an inadmissible probe disc before model evaluation. Longitude wrapping is explicit and does not certify seam continuity.

See [the complete runnable example](examples/custom_encoder.py). The fixed `models.loaders.load_all` roster belongs to manuscript reproduction; new encoders use this public interface without modifying that roster. The scalar `measure_indicatrix` interface remains supported.

## Reproduce the supplied study

```sh
etissot check --data ./data --output ./outputs
etissot reproduce --data ./data --output ./reproduced --manuscript ./manuscript --numerical-only
```

The `--manuscript` verification needs the manuscript sources, which are not yet public; they will be released after publication.

The full workflow regenerates the empirical tables and all 15 figures across the current manuscript and additional package results, analytic examples, historical five-seed summaries, neighbour statistics and downstream bootstrap results. It replays all 360 selected GP predictions, then checks 20 generated manuscript blocks, the complete CSP precision table and 124 numerical claims against independent baselines. See [the complete reproduction guide](docs/reproducibility.md). It does not download model files or rerun GP fitting. Data and output directories must not overlap. Input data are never overwritten. Full reproduction does not copy finished figures, tables or summaries into outputs; standalone steps may seed reference prerequisites. Results are saved under `outputs/summaries`, `outputs/tables` and `outputs/figures`; a run report and logs record success or failure.

`environment` checks Python 3.11+ and the installed versions against `requirements-reproduce.txt` in the working directory. Reproduction records `environment.json`. Manuscript verification requires identical figure pixels by default. Add `--numerical-only` to verify all numerical content and figure presence while recording pixel differences separately; this is not an exact artifact check.

Run one analysis with, for example:

```sh
etissot analyze downstream --data ./data --output ./outputs
```

Named analyses include `geometry-cases`, `satclip-correction`, `geographic`, `shape`, `downstream`, `summary`, `audit`, `maps`, `estimator`, `historical`, `neighbours`, `harmonics`, `mixtures`, `predictions`, `dense-cached`, `replay` and `supplement`. Standalone report steps use the supplied reference summaries unless you regenerate their prerequisite analyses in that output directory.

Programmatic workflow:

```python
from etissot.workspace import Workspace
from etissot.pipeline import run

study = Workspace("data", "outputs")
report = run(study)
```

## Package layout

```text
src/etissot/
    geometry/       # geodesic probes, harmonic fitting, analytic references
    statistics/     # spatial weighting and block resampling
    diagnostics/    # geographic, shape, downstream, boundary and dense analyses
    prediction/     # reusable Gaussian process and regional evaluation
    reporting/      # summary tables, audit tables and maps
    models/         # optional external-checkpoint adapters and licensed source
    validation/     # numerical and stored-result checks
    workspace.py    # explicit input/output paths
    pipeline.py     # named reproduction workflows
    cli.py          # command-line entry point

data/
    profiles/       # global, randomized, dense and multiradius profiles
    geometry/       # cached Jacobians
    prediction/     # regional inputs, tensor fields and saved GP fits
    downstream/     # cell outcomes, feature/head profiles and masks
    reference/      # baseline summaries, tables, figures and coarse coastlines

tests/              # public API and workspace tests
examples/           # small library example independent of study data
docs/               # API, data, model and provenance documentation
```

Scientific files and modules are named by function. `docs/file_migration.json` records the old-to-new filenames for traceability; it is not needed to run the package.

## Optional GP fitting

```sh
etissot fit --data ./data --output ./new-fits --threads 2
etissot analyze summary --data ./data --output ./new-fits
```

A new output directory starts fresh; existing fits in that output directory are resumed. The input reference fits are never overwritten. `--windows CONUS` and `--limit 1` allow a bounded run. Summary reporting prefers newly generated regional fit files and requires complete sets (90 fits per region); run summaries after fitting completes. A full study has 360 fits with three starts each. Selection uses training likelihood only; repeated splits are not independent trials, and different numerical platforms can converge to different optima.

## Optional checkpoint evaluation

`evaluate-models` requires external trusted checkpoints and CSP source, not included here. See `docs/models.md`. These paths are separate from the cached reproduction and were not rerun for this release. Fresh dense evaluation writes new profiles into the output directory; the cached reproduction continues to use distributed input profiles unless you assemble a new input dataset explicitly.

## Validation, attribution and scope

Run the unit tests with `python -m unittest discover -s tests -v`. They require no study data or checkpoints; see `docs/testing.md`. Validation records are kept in `docs/validation/`. Verify both shareable ZIPs with `python scripts/verify_releases.py`. These are frozen distributions; the repository is the editable source.

The primary estimand is finite-lag geometry. All failed refinement cases and negative prediction results are retained. The shape-only GP kernel has no pooled gain (relative RMSE approximately 1.009). Reproduction uses declared measurements and selected fit parameters. The package does not retrain upstream models or historical miniature models, reacquire original rasters, or rerun unrelated exploratory experiments.

eTissot is released under the MIT License (see `LICENSE`). Third-party code, model weights and upstream datasets keep their own terms; see `RIGHTS_AND_ATTRIBUTION.md`.

## Citation

If you use eTissot, please cite the accompanying manuscript: Guofeng Cao, *Tissot indicatrix for location encoders: characterizing local distortion in learned geographic embeddings* (under review). Full citation details will be added on publication.
