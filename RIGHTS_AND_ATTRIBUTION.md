# Rights and attribution

eTissot is copyright (c) 2026 Guofeng Cao, STARlab, Department of Geography, University of Colorado Boulder, and is released under the MIT License (see LICENSE). The MIT License covers the eTissot code and the measurements and summaries derived in this project. It does not extend to third-party code, model weights or upstream datasets, which keep their own terms as described below.

Third-party code included:
- Microsoft SatCLIP: src/etissot/models/vendor/satclip/LICENSE (MIT). Source location: https://github.com/microsoft/satclip . The local operative source includes a lightweight loader and package-relative imports, with eager training imports removed; network and harmonic formulas are unchanged; do not assume it is identical to the current upstream tree.
- Elijah Cole SINR: src/etissot/models/vendor/sinr/LICENSE (MIT). Source location: https://github.com/elijahcole/sinr . Only the required model module is included.

The Python wheel and code/data sharing ZIP do not redistribute model weights. A separate optional companion bundle contains unmodified GeoCLIP weights from the MIT-licensed geoclip distribution, six SatCLIP checkpoints with explicit publisher MIT model-card declarations, and the Climplicit checkpoint under its publisher's CC BY 4.0 declaration. Preserve the supplied publisher copyright/license notices, model cards, author citations and source URLs. The complete acquisition catalog and evidence are under src/etissot/models/checkpoints.json and src/etissot/models/resources/; the bundle records exact file hashes and its omissions.

SINR and CSP weights are not republished in the companion because their checkpoint redistribution terms were not established from the checked publisher records. The acquisition command obtains them, and CSP's pinned runtime source, directly from their publishers for local evaluation; this is separate from redistribution. CSP code, original satellite imagery, original IUCN/species records and complete environmental rasters are not redistributed. Derived data retain the provenance documented in docs/data.md and docs/models.md; this file does not assert unrestricted redistribution rights over upstream datasets. global-land-mask 1.0.0 supplies the retained mask/coastline derivations.

Associated manuscript: Tissot indicatrix for location encoders: characterizing local distortion in learned geographic embeddings. Author: Guofeng Cao. The manuscript is under review; citation details will be added on publication.
