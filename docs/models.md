# Optional released-model evaluation

Cached reproduction is self-contained. Fresh evaluations are an advanced path requiring separately obtained model files, upstream source and any transitive dependencies. All thirteen exact checkpoints have been downloaded into an empty directory and checked, together with CSP's four runtime source files. A three-location forward comparison in the existing study runtime returned embeddings identical to the original local models for every checkpoint. This check does not validate a clean dependency installation or repeat the full fresh-data scientific collection. See `docs/validation/model_acquisition_2026-09-30.json`.

## Obtain and verify the exact checkpoints

The installed package contains a versioned acquisition catalog at `src/etissot/models/checkpoints.json`. It covers the thirteen manuscript checkpoints, their exact SHA-256 hashes, publisher URLs, pinned repository revisions where available, feature conventions, file sizes and recorded weight licenses. The scientific checkpoint manifest under `data/reference/summaries/` remains unchanged. These acquisition commands use the Python standard library and do not require the study/model extras, load weights or download training data:

```sh
etissot models list
etissot models download --models ./models --dry-run
etissot models download --models ./models
etissot models verify --models ./models
```

Select a family or checkpoint with `--family satclip` or `--encoder satclip-vit16-l40`; multiple selections are space-separated. The downloader recognizes the documented historical SatCLIP locations and installed GeoCLIP weights, verifies existing files, and skips matching copies. A differing existing file is preserved unless `--replace` is supplied. Every new file is streamed to a temporary file, checked against its declared size and SHA-256, then installed atomically. ZIP sources are fetched once per invocation, checked as whole archives and read only for the declared members. A failed file is reported in `models/acquisition.json` and gives a nonzero command exit status; rerunning skips files already acquired correctly. `--dry-run` performs no downloads or writes. Network-operation timeout defaults to 60 seconds and can be changed with `--timeout`.

Selecting CSP also obtains its four required runtime source files directly from the publisher at commit `67f34075c944ad3845499892a23c7363691cf3bf`, with separate hashes. They are byte-identical to the inspected source. They are not redistributed in eTissot. Verification checks checkpoint and selected source-file bytes; it does not establish that the runtime dependencies are installed or that model evaluation succeeds.

The supplied GeoCLIP file under `models/encoders/geoclip/` takes precedence over installed package weights. Its architecture still requires `geoclip==1.2.1`. Use the same pinned precision and feature conventions when reproducing the manuscript; acquiring a matching checkpoint alone does not establish equivalent measurements.

## Optional checkpoint bundle

`releases/etissot-models-0.2.0.zip` is separate from the Python wheel and code/data sharing ZIP. It contains eight unmodified checkpoints: GeoCLIP, all six SatCLIP variants and Climplicit. It also includes publisher model cards, license notices, author/source attribution in the acquisition catalog, a list of omissions and a checksum manifest. Extract it and pass the extracted `etissot-models-0.2.0` directory as `--models`. Then use `models download` on that directory to obtain the omitted models directly from their publishers.

The bundle includes GeoCLIP under its distribution's MIT license, SatCLIP under the explicit MIT declarations in all six publisher model cards, and Climplicit under its publisher's CC BY 4.0 declaration. The original checkpoint bytes are unchanged. SINR's CaltechDATA record did not state an explicit weight license in the checked metadata, and no CSP checkpoint redistribution terms were established from the publisher model link. Those five checkpoints remain outside the bundle; their code or website licenses are not treated as weight licenses. Sources and decisions were checked on 30 September 2026. Evidence is retained under `src/etissot/models/resources/`.

To build a companion ZIP from verified local files:

```sh
etissot models bundle --models ./models --output ./etissot-models-0.2.0.zip
```

The default bundles only catalog entries with established redistribution terms and lists omissions. An explicit selection containing an ineligible checkpoint is rejected. Missing or changed checkpoints and missing or changed license evidence also prevent bundle creation. License evidence is checked against the catalog's recorded hashes. The completed ZIP is read back against every member checksum and every expected checkpoint hash. The `.sha256` sidecar verifies the ZIP as a whole. No upload or public publication is performed by any model command.

- GeoCLIP: https://github.com/VicenteVivan/geo-clip ; pinned Python package geoclip 1.2.1 with bundled location weights.
- SatCLIP: https://github.com/microsoft/satclip ; inspected harmonic implementation commit 1ead106. The included modules retain the local operative code, including load_lightweight.py. The six original checkpoints are available through the acquisition command and optional bundle.
- SINR: https://github.com/elijahcole/sinr ; three released checkpoints; models.py and its MIT license are included.
- CSP: https://github.com/gengchenmai/csp ; source is not included because the inspected checkout has no license. The acquisition command obtains SpatialRelationEncoder.py, module.py, data_utils.py and paths.py directly from the publisher under csp/main/. Do not treat this package as granting rights to that source.
- Climplicit: https://arxiv.org/abs/2504.05089 ; rshf 0.2.2 location path and the named original checkpoint.

`data/reference/summaries/checkpoint_manifest.json` records the exact expected project-relative checkpoint paths, SHA-256 checksums, dimensions, radii and precision for 13 checkpoints. GeoCLIP resolves its installed package directory if the historical .cache path is absent. Climplicit concatenates months 3, 6, 9 and 12. Do not substitute newer checkpoints when attempting the pinned comparisons. Several publisher loaders deserialize Python objects: use trusted original releases.

Install the package with its models extra (`python -m pip install ".[models]"`) plus upstream dependencies, obtain and verify the documented files, and inspect src/etissot/models/loaders.py before running:

```sh
etissot evaluate-models boundaries --data ./data --output ./fresh-models --models /path/to/model-files
etissot evaluate-models dense --data ./data --output ./fresh-models --models /path/to/model-files
```

Use a working copy. The dense generator resumes existing panels; move its output cache aside to reevaluate everything. Neither command reproduces upstream training. Cross-environment bitwise identity is not promised.

The external model directory contains `encoders/` and `csp/main/` with the documented relative paths. Licensed SatCLIP and SINR source modules are installed under `etissot/models/vendor/`; they need not be copied into the external model directory. All new evaluation outputs go into the requested output directory.

## This local workspace

The external model root is now `models/`. For example:

```sh
etissot evaluate-models boundaries --data ./data --output ./fresh-models --models ./models
```

Historical checkpoint-manifest paths are relative to this external model root. GeoCLIP still resolves its installed package weights. Upstream SINR training/evaluation data were moved to `data/raw/sinr/`; the package checkpoint loaders do not use them.

## SatCLIP zonal-normalisation comparison

The opt-in `etissot.models.satclip_correction.correct_zonal_normalization`
returns a deep copy of an inspected analytic-harmonic SIREN location encoder.
It divides the degree-major zonal channels (indices `l*(l+1)`) by pi. It does
not modify vendor code, original checkpoints, default loaders, or primary results.
With `preserve_outputs=False` (default), learned weights are held fixed; this is
an input intervention, not a retrained checkpoint. With `preserve_outputs=True`,
matching first-layer weight columns are multiplied by pi, preserving the released
function to floating-point precision. The correction does not address high-degree
expanded-polynomial cancellation and rejects other harmonic implementations.

Collect the six-checkpoint, 35-site comparison from the trusted original weights:

```sh
etissot evaluate-models satclip_correction --data data --output fresh-correction --models models
```

Collection writes `fresh-correction/profiles/satclip_correction.npz`; it does not
replace the packaged input. The reviewed measurements are retained as
`data/profiles/satclip_correction.npz`. Recompute their fits and supplemental table:

```sh
etissot analyze satclip-correction --data data --output correction-replay
```

The full `reproduce` workflow includes this cached comparison. Sites use seven
latitudes from -75 to 75 degrees at 25-degree spacing and five longitudes from
-150 to 150 degrees at 75-degree spacing, with 32 bearings at 1 and 0.5 km.
Outputs are unit-normalised; basis profiles are unnormalised. Checkpoint hashes,
all profiles, control output differences and centre cosines are saved. These
selected-site results do not establish global improvements or predictive accuracy.
