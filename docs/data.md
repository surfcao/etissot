# Data guide

These are derived research inputs and outputs, not a distribution of the original imagery, checkpoint weights, species observation records or complete source rasters. Array names and dimensions are enumerated in data/inventory.json; scripts are the authoritative definition of how keys are joined and transformed. NumPy archives load without pickle.

| File | Role |
|---|---|
| profiles/global.npz | Fixed-grid directional profiles across encoder families and input references |
| profiles/randomized_primary.npz, profiles/randomized_satclip.npz | Randomized-grid profiles used to assess grid-phase sensitivity |
| profiles/input_references.npz | Cached input/reference and mixture profiles |
| geometry/jacobians.npz | Cached Jacobian diagnostics for matched comparisons |
| profiles/geoclip_multiradius.npz | GeoCLIP radius-dependent profiles |
| profiles/dense_regional.npz | 135,165 dense regional locations; 32-bearing profiles, coordinates, fits and fallback flags |
| prediction/regions_original.npz, prediction/regions_additional.npz | Original and additional regional GP inputs, embeddings and sampled target values |
| prediction/geometry_fields.npz | Geometry fields for GP comparisons |
| downstream/cell_outcomes.npz | SINR aggregated cell errors and adjustment covariates |
| downstream/feature_head_profiles.npz | SINR feature/head directional profiles |
| downstream/land_mask.npz | Retained Boolean land-cell membership |
| downstream/historical_cell_summaries.npz | Historical four-site cell summaries, explicitly not current-estimator validation |
| reference/maps/coastlines.npz | Coarse display-only coastline cache |

## Coordinate and statistical conventions

Geographic coordinate arrays used for displays are degrees; probe functions take radians and physical radii in kilometres, with a spherical Earth radius of 6,371 km. The directional profile arrays used by the current analyses have 32 equally spaced bearings. Fitted bearings are radians, modulo pi; inspect the relevant script before interpreting a flattened grid. Anisotropy is dimensionless. Respect fallback masks and omit bearings there.

SINR cell IDs encode one-degree cells: latitude centre = cell // 360 - 90 + 0.5; longitude centre = cell % 360 - 180 + 0.5. The retained land analysis uses 13,045 cells per model. Resampling blocks are 10 by 10 degrees with 500 paired draws per row, seed 0. Each draw refits ranks and adjustment regressions.

GP targets derive from the SINR evaluation rasters: GMTED2010 elevation, GFCC tree cover and TerraClimate soil moisture at 0.09-degree resolution. Soil moisture uses log(1 + value) for prediction. The target-derived tensor is a privileged diagnostic. New train/test splits do not establish independence from encoder pretraining information.

The land-mask and coarse coastline caches derive from global-land-mask 1.0.0. They are supplied so reproduction does not download its underlying grid. Do not interpret coarse display coastlines as the analysis mask.

## Additional manuscript measurements in 0.2.0

`measurements/released_checkpoints.json` retains radius profiles and full Jacobian metrics for replay; `measurements/neighbour_candidates.npz` retains query/candidate coordinates and measured similarities; `measurements/hidden_space_seeds.npz` retains the historical per-seed fields and held-out scores. See `data/measurements/README.md` for exact roles and units. `reference/manuscript_claims.json` is the machine-readable manuscript coverage contract. Finished reference summaries/figures/tables are optional comparison artifacts and are not copied into a full reproduction run.
