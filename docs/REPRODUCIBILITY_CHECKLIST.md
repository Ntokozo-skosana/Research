# Reproducibility checklist

## Included and checked
- [x] Four raw HAALSI ZIP archives included under `data/raw/`.
- [x] Frozen 404-row, 60-column analytical CSV preserved.
- [x] 42-column reference CSV preserved and explicitly labelled as a reference, not raw data.
- [x] MMSE scoring audit included.
- [x] Modelling scripts, saved results and experiment manifests included.
- [x] Cleaning/provenance documentation expanded.
- [x] Cleaning notebook expanded with source inspection and audit steps.
- [x] Standalone raw-source audit script added.
- [x] Data dictionary generated for all 60 analytical columns.

## Still required before claiming full raw-to-cleaned reproduction
- [ ] Verify that the Wave 2 Survey ZIP's V5 file is the intended release; resolve earlier V4 reference.
- [ ] Confirm source variable and value-label mappings for quality, insufficiency, smoking, alcohol and marital status.
- [ ] Confirm all covariate mappings and special missing-value codes.
- [ ] Recover the exact historical 404-participant inclusion/exclusion logic (deep availability audit completed; no tested rule exactly reproduces the IDs).
- [ ] Confirm formulas/missing-value rules for means, changes, variability and shifted bedtime.
- [ ] Verify participant-level survey and dementia assessment chronology.
- [ ] Run `python scripts/00_audit_raw_sources.py` and inspect every mismatch report.
- [ ] Build and validate a deterministic reconstruction script; compare all 404 IDs and all 60 fields before replacing anything.
- [ ] Confirm source-data licence/access conditions before sharing the raw archives.

Do not mark the remaining items complete based only on the presence of the raw ZIP files.


- [x] Smoking and alcohol recoding rules compared against source fields for all 404 participants (404/404 exact matches each).
- [ ] Original 404-participant cohort inclusion/exclusion rule recovered and reproduced. Single-rule and deep missingness/availability audits do not uniquely reproduce the cohort.
