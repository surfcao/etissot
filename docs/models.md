# Optional released-model evaluation

Cached reproduction is self-contained, whereas fresh evaluation requires separately obtained model files, upstream source and their dependencies. All thirteen exact checkpoints and the four CSP runtime source files have been downloaded into an empty directory and checked, and a three-location forward comparison in the study runtime returned embeddings identical to those of the original local models for every checkpoint. This check does not validate a clean dependency installation or repeat the full scientific collection with fresh data.

## Obtain and verify the exact checkpoints

The versioned acquisition catalog `src/etissot/models/checkpoints.json` covers the thirteen manuscript checkpoints with their exact SHA-256 hashes, publisher URLs, pinned repository revisions (where available), feature conventions, file sizes and recorded weight licenses. The scientific checkpoint manifest under `data/reference/summaries/` remains unchanged. The acquisition commands use only the Python standard library; they do not require the study or model extras, load weights or download training data:

```sh
etissot models list
etissot models download --models ./models --dry-run
etissot models download --models ./models
etissot models verify --models ./models
```

Select a family or checkpoint with `--family satclip` or `--encoder satclip-vit16-l40` (space-separated for several). The downloader recognizes the documented historical SatCLIP locations and installed GeoCLIP weights, verifies existing files and skips matching copies, while a differing existing file is preserved unless `--replace` is supplied. Every new file is streamed to a temporary file, checked against its declared size and SHA-256 and then installed atomically. ZIP sources are fetched once per invocation, checked as whole archives and read only for the declared members. A failed file is reported in `models/acquisition.json` and gives a nonzero exit status; rerunning skips files already acquired. `--dry-run` performs no downloads or writes. The network timeout defaults to 60 seconds (`--timeout`).

Selecting CSP also obtains its four required runtime source files directly from the publisher at commit `67f34075c944ad3845499892a23c7363691cf3bf`, with separate hashes. These files are byte-identical to the inspected source and are not redistributed in eTissot. Verification checks checkpoint and source-file bytes, not whether runtime dependencies are installed or model evaluation succeeds.

The supplied GeoCLIP file under `models/encoders/geoclip/` takes precedence over installed package weights, although its architecture still requires `geoclip==1.2.1`. When reproducing the manuscript, use the same pinned precision and feature conventions, since acquiring a matching checkpoint alone does not establish equivalent measurements.

## Optional checkpoint bundle

The companion bundle `etissot-models-0.2.0.zip` is not attached to the public release and is available from the author on request. It contains eight unmodified checkpoints (GeoCLIP, all six SatCLIP variants and Climplicit), together with the publisher model cards, license notices, author and source attribution in the acquisition catalog, a list of omissions and a checksum manifest. To use it, extract the bundle, pass the extracted `etissot-models-0.2.0` directory as `--models`, and run `models download` on that directory to obtain the omitted models from their publishers.

The bundle includes GeoCLIP under its distribution's MIT license, SatCLIP under the explicit MIT declarations in all six publisher model cards and Climplicit under its publisher's CC BY 4.0 declaration, with the original checkpoint bytes unchanged. SINR's CaltechDATA record did not state an explicit weight license in the checked metadata, and no CSP checkpoint redistribution terms were established from the publisher model link. Those five checkpoints therefore remain outside the bundle, and their code or website licenses are not treated as weight licenses. Sources and decisions were checked on 30 September 2026, and the evidence is retained under `src/etissot/models/resources/`.

To build a companion ZIP from verified local files:

```sh
etissot models bundle --models ./models --output ./etissot-models-0.2.0.zip
```

By default, only catalog entries with established redistribution terms are bundled and omissions are listed; an explicit selection containing an ineligible checkpoint is rejected. Missing or changed checkpoints or license evidence also prevent bundle creation, and license evidence is checked against the catalog's recorded hashes. The completed ZIP is read back against every member checksum and every expected checkpoint hash, while the `.sha256` sidecar verifies the ZIP as a whole. No model command uploads or publicly publishes anything.

- GeoCLIP: https://github.com/VicenteVivan/geo-clip ; pinned Python package geoclip 1.2.1 with bundled location weights.
- SatCLIP: https://github.com/microsoft/satclip ; inspected harmonic implementation commit 1ead106. The included modules retain the local operative code, including load_lightweight.py. The six original checkpoints are available through the acquisition command and optional bundle.
- SINR: https://github.com/elijahcole/sinr ; three released checkpoints; models.py and its MIT license are included.
- CSP: https://github.com/gengchenmai/csp ; source is not included because the inspected checkout has no license. The acquisition command obtains SpatialRelationEncoder.py, module.py, data_utils.py and paths.py directly from the publisher under csp/main/. Do not treat this package as granting rights to that source.
- Climplicit: https://arxiv.org/abs/2504.05089 ; rshf 0.2.2 location path and the named original checkpoint.

`data/reference/summaries/checkpoint_manifest.json` records the exact expected project-relative checkpoint paths, SHA-256 checksums, dimensions, radii and precision for the 13 checkpoints. GeoCLIP resolves its installed package directory if the historical .cache path is absent, and Climplicit concatenates months 3, 6, 9 and 12. Newer checkpoints should not be substituted when attempting the pinned comparisons. Because several publisher loaders deserialize Python objects, only trusted original releases should be used.

Install the package with its models extra (`python -m pip install ".[models]"`) and the upstream dependencies, obtain and verify the documented files, and inspect src/etissot/models/loaders.py before running:

```sh
etissot evaluate-models boundaries --data ./data --output ./fresh-models --models /path/to/model-files
etissot evaluate-models dense --data ./data --output ./fresh-models --models /path/to/model-files
```

Use a working copy. The dense generator resumes existing panels; move its output cache aside to reevaluate everything. Neither command reproduces upstream training, and cross-environment bitwise identity is not promised.

The external model directory contains `encoders/` and `csp/main/` with the documented relative paths. Licensed SatCLIP and SINR source modules are installed under `etissot/models/vendor/` and need not be copied into the external model directory. All new evaluation outputs go into the requested output directory.

## This local workspace

The external model root is now `models/`, for example:

```sh
etissot evaluate-models boundaries --data ./data --output ./fresh-models --models ./models
```

Historical checkpoint-manifest paths are relative to this external model root, and GeoCLIP still resolves its installed package weights. Upstream SINR training and evaluation data were moved to `data/raw/sinr/`, which the package checkpoint loaders do not use.

## SatCLIP zonal-normalization comparison

The opt-in function `etissot.models.satclip_correction.correct_zonal_normalization` returns a deep copy of an inspected analytic-harmonic SIREN location encoder with the degree-major zonal channels (indices `l*(l+1)`) divided by pi. It does not modify the vendor code, original checkpoints, default loaders or primary results. With `preserve_outputs=False` (the default), the learned weights are held fixed, so the correction is an input intervention rather than a retrained checkpoint. With `preserve_outputs=True`, the matching first-layer weight columns are multiplied by pi, which preserves the released function to floating-point precision. The correction does not address the cancellation of high-degree expanded polynomials, and it rejects other harmonic implementations.

To collect the comparison over six checkpoints and 35 sites from the trusted original weights:

```sh
etissot evaluate-models satclip_correction --data data --output fresh-correction --models models
```

Collection writes `fresh-correction/profiles/satclip_correction.npz` and does not replace the packaged input. The reviewed measurements are retained as `data/profiles/satclip_correction.npz`, from which the fits and the supplemental table can be recomputed:

```sh
etissot analyze satclip-correction --data data --output correction-replay
```

The full `reproduce` workflow includes this cached comparison. The sites lie at seven latitudes from -75 to 75 degrees (25-degree spacing) and five longitudes from -150 to 150 degrees (75-degree spacing), with 32 bearings at 1 and 0.5 km. Outputs are unit-normalized, whereas basis profiles are not normalized. Checkpoint hashes, all profiles, control output differences and center cosines are saved. These selected-site results do not establish global improvements or predictive accuracy.
