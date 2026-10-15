# Scientific inputs

`profiles/`, `geometry/`, `prediction/`, `downstream/` and `reference/` are the byte-preserved inputs supplied with the Python release. `inventory.json` documents them. Run current workflows with `--data ./data`; generated results belong in `outputs/`.

`raw/sinr/` contains the original downloaded SINR datasets and masks. `raw/sinr_compact/` retains the original compact upstream inputs. These are not needed for cached manuscript reproduction and are excluded from the shareable release. They are retained for original data-generation work.

Unique older numerical caches live in `archive/data/`. Duplicate experiment-numbered copies of current inputs were removed only after SHA-256 agreement. See `docs/maintenance/restructure_2026-09-17.json` and `docs/file_migration.json` for the mapping.
