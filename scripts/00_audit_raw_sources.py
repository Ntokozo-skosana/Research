"""Audit the supplied HAALSI raw archives against the frozen 404-row analysis cohort.

This script does NOT silently recreate or overwrite the historical 404-row dataset.
It verifies source-file structure, participant-ID coverage, direct MMSE source values,
and the presence of longitudinal sleep source variables. Any transformation that
cannot be confirmed from the archived codebook/history is left for explicit review.

Run from the package root:
    python scripts/00_audit_raw_sources.py
"""
from pathlib import Path
import io, zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "audit"
OUT.mkdir(parents=True, exist_ok=True)

ARCHIVES = {
    "dementia_w1": ("HAALSI Dementia Wave 1.zip", "HAALSI_DEMENTIA_Respondent-1.dta"),
    "dementia_w2": ("HAALSI Dementia Wave 2.zip", "HAALSI_DEMENTIA_Respondent_W2.dta"),
    "survey_w2_release": ("HAALSI Wave 2 Survey.zip", "Longitudinal Public Release Dataset V5.dta"),
    "survey_w1_w3": ("HAALSI Wave 3 Survey.zip", "HAALSI W1-3 Longitudinal Dataset.dta"),
}

def read_dta_from_zip(zip_path, suffix):
    with zipfile.ZipFile(zip_path) as archive:
        matches = [n for n in archive.namelist() if n.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {suffix!r} in {zip_path.name}; found {matches}")
        with archive.open(matches[0]) as fh:
            data = fh.read()
    return pd.read_stata(io.BytesIO(data), convert_categoricals=False), matches[0]

clean_path = PROCESSED / "haalsi_404_cleaned.csv"
if not clean_path.exists():
    raise FileNotFoundError(clean_path)
cohort = pd.read_csv(clean_path)
if "prim_key" not in cohort or cohort.prim_key.isna().any():
    raise ValueError("The cleaned cohort must contain non-missing prim_key values.")
if cohort.prim_key.duplicated().any():
    raise ValueError("Duplicate prim_key values found in cleaned cohort.")
cohort_ids = set(cohort.prim_key.astype(str))

frames, inventory, coverage = {}, [], []
for key, (archive_name, member_suffix) in ARCHIVES.items():
    archive_path = RAW / archive_name
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)
    frame, member = read_dta_from_zip(archive_path, member_suffix)
    frames[key] = frame
    id_unique = bool(frame.prim_key.is_unique) if "prim_key" in frame else False
    inventory.append({"source_key": key, "archive": archive_name, "stata_file": member,
                      "rows": len(frame), "columns": len(frame.columns),
                      "has_prim_key": "prim_key" in frame.columns,
                      "prim_key_unique": id_unique})
    if "prim_key" in frame:
        ids = set(frame.prim_key.dropna().astype(str))
        coverage.append({"source_key": key, "source_rows": len(frame),
                         "source_unique_ids": frame.prim_key.nunique(),
                         "cohort_n": len(cohort_ids),
                         "cohort_ids_found": len(cohort_ids & ids),
                         "cohort_ids_missing": len(cohort_ids - ids)})

pd.DataFrame(inventory).to_csv(OUT / "raw_source_inventory.csv", index=False)
pd.DataFrame(coverage).to_csv(OUT / "cohort_id_coverage.csv", index=False)

# Direct cognitive score comparison: these source names and labels are explicit.
w1 = frames["dementia_w1"][["prim_key", "mmse_total", "mmse_status"]].copy()
w2 = frames["dementia_w2"][["prim_key", "w2_mmse_total", "w2_mmse_status"]].copy()
mmse = cohort[["prim_key", "mmse_w1", "mmse_w2"]].merge(w1, on="prim_key", how="left", validate="one_to_one")
mmse = mmse.merge(w2, on="prim_key", how="left", validate="one_to_one")
for a, b, label in [("mmse_w1", "mmse_total", "wave1"), ("mmse_w2", "w2_mmse_total", "wave2")]:
    x, y = pd.to_numeric(mmse[a], errors="coerce"), pd.to_numeric(mmse[b], errors="coerce")
    mmse[f"{label}_source_match"] = (x.eq(y) | (x.isna() & y.isna()))
mmse.to_csv(OUT / "mmse_source_comparison.csv", index=False)

# Survey field presence and labels: candidate fields are not asserted as final cleaned
# feature mappings unless the value comparison below confirms direct correspondence.
survey = frames["survey_w1_w3"]
field_rows = []
try:
    with zipfile.ZipFile(RAW / ARCHIVES["survey_w1_w3"][0]) as archive:
        member = next(n for n in archive.namelist() if n.endswith(ARCHIVES["survey_w1_w3"][1]))
        import pandas.io.stata as stata
        reader = stata.StataReader(io.BytesIO(archive.read(member)), convert_categoricals=False)
        labels = reader.variable_labels()
except Exception:
    labels = {}
for wave in ("w1", "w2", "w3"):
    for var in (f"{wave}sp001_hour", f"{wave}sp001_minute", f"{wave}sp002_hour",
                f"{wave}sp002_minute", f"{wave}sp003", f"{wave}sp004", f"{wave}sp005"):
        field_rows.append({"variable": var, "present": var in survey.columns,
                           "source_label": labels.get(var, "")})
pd.DataFrame(field_rows).to_csv(OUT / "sleep_source_variable_inventory.csv", index=False)

# Exact source comparisons for directly verified sleep fields.
# The codebook and row-level comparisons support these mappings:
# duration = sp003 restricted to 3..16 hours; quality = sp010;
# insufficiency = 6 - sp004; bedtime = hour + minute/60, with <06:00 shifted +24.
sleep_mapping = []
for wave in ("w1", "w2", "w3"):
    indexed = survey.set_index("prim_key")
    mappings = [
        (f"duration_{wave}", f"{wave}sp003", lambda x: x.where(x.between(3, 16)), "3-16 hours inclusive; outside range set missing"),
        (f"quality_{wave}", f"{wave}sp010", lambda x: x, "direct source value"),
        (f"insufficiency_{wave}", f"{wave}sp004", lambda x: 6 - x, "reverse code: 6 - source response"),
    ]
    for clean_var, raw_var, transform, rule in mappings:
        if clean_var not in cohort or raw_var not in survey:
            continue
        raw_values = pd.to_numeric(indexed[raw_var], errors="coerce")
        expected = transform(raw_values)
        joined = cohort[["prim_key", clean_var]].merge(expected.rename("expected").reset_index(),
                  on="prim_key", how="left", validate="one_to_one")
        a = pd.to_numeric(joined[clean_var], errors="coerce")
        b = pd.to_numeric(joined["expected"], errors="coerce")
        equal = (a.sub(b).abs() < 1e-8) | (a.isna() & b.isna())
        sleep_mapping.append({"wave": wave, "cleaned_column": clean_var, "raw_source": raw_var,
            "source_label": labels.get(raw_var, ""), "transformation": rule,
            "matches_including_missing": int(equal.sum()), "n": len(joined),
            "mismatches": int((~equal).sum())})

    # Bedtime: combine hour/minute and shift early-morning values to the next day.
    clean_var = f"bedtime_{wave}"
    hour_var, minute_var = f"{wave}sp001_hour", f"{wave}sp001_minute"
    if clean_var in cohort and hour_var in indexed and minute_var in indexed:
        hour = pd.to_numeric(indexed[hour_var], errors="coerce")
        minute = pd.to_numeric(indexed[minute_var], errors="coerce")
        valid = hour.between(0, 23) & minute.between(0, 59)
        expected = (hour + minute / 60).where(valid)
        expected = expected.where(expected >= 6, expected + 24)
        joined = cohort[["prim_key", clean_var]].merge(expected.rename("expected").reset_index(),
                  on="prim_key", how="left", validate="one_to_one")
        a = pd.to_numeric(joined[clean_var], errors="coerce")
        b = pd.to_numeric(joined["expected"], errors="coerce")
        equal = (a.sub(b).abs() < 1e-8) | (a.isna() & b.isna())
        sleep_mapping.append({"wave": wave, "cleaned_column": clean_var,
            "raw_source": f"{hour_var}+{minute_var}",
            "source_label": labels.get(hour_var, "") + " / " + labels.get(minute_var, ""),
            "transformation": "hour + minute/60; add 24 when time < 06:00",
            "matches_including_missing": int(equal.sum()), "n": len(joined),
            "mismatches": int((~equal).sum())})

pd.DataFrame(sleep_mapping).to_csv(OUT / "sleep_mapping_validation.csv", index=False)

# Recheck engineered sleep summaries against explicit candidate formulas.
derived_rows = []
for feature, waves in [("duration", ["w1", "w2", "w3"]),
                       ("quality", ["w1", "w2", "w3"]),
                       ("insufficiency", ["w1", "w2", "w3"]),
                       ("bedtime", ["w1", "w2", "w3"])]:
    cols = [f"{feature}_{w}" for w in waves]
    if not all(c in cohort for c in cols):
        continue
    x = cohort[cols].apply(pd.to_numeric, errors="coerce")
    candidates = [(f"{feature}_mean", x.mean(axis=1))]
    if feature == "duration":
        candidates.append(("duration_variability", x.std(axis=1, ddof=1)))
    if feature == "bedtime":
        candidates[0] = ("bedtime_mean_shifted", x.mean(axis=1))
    candidates.append((f"{feature}_change", x.iloc[:, 2] - x.iloc[:, 0]))
    for name, expected in candidates:
        if name not in cohort:
            continue
        actual = pd.to_numeric(cohort[name], errors="coerce")
        equal = (actual.sub(expected).abs() < 1e-8) | (actual.isna() & expected.isna())
        derived_rows.append({"cleaned_feature": name, "candidate_formula": "row mean" if name.endswith("_mean") else
            ("sample standard deviation across W1-W3 (ddof=1)" if "variability" in name else ("row mean of shifted W1-W3 bedtime" if name == "bedtime_mean_shifted" else "W3 minus W1")),
            "matches_including_missing": int(equal.sum()), "n": len(cohort),
            "mismatches": int((~equal).sum())})
pd.DataFrame(derived_rows).to_csv(OUT / "derived_sleep_feature_validation.csv", index=False)

# Covariate source checks. Exact matching is required before treating a field as a direct copy.
covariate_candidates = {
    "age_w2": ["w2c_rage_calc"], "sex": ["w2c_rsex"],
    "education_w1": ["w1c_bd_educ4"], "cesd_w2": ["w2c_cd_cesd"],
    "asset_index_w2": ["w2c_hh_assetindex"], "wealth_quintile_w2": ["w2c_wealthindex"],
    "bmi_w2": ["w2c_bs_bmi"], "marital_status_w2": ["w2c_bd_mar"],
    "hypertension_w2": ["w2c_cm_hypertension_srdiag", "w2c_cm_hypertension_srtrt"],
    "diabetes_w2": ["w2c_cm_diabetes_srdiag", "w2c_cm_diabetes_srtrt"],
}
cov_rows = []
for clean_var, raw_vars in covariate_candidates.items():
    if clean_var not in cohort:
        continue
    a = pd.to_numeric(cohort[clean_var], errors="coerce")
    for raw_var in raw_vars:
        if raw_var not in survey:
            continue
        b = pd.to_numeric(pd.Series(survey.set_index("prim_key").loc[cohort.prim_key, raw_var].to_numpy()), errors="coerce")
        if clean_var in {"asset_index_w2", "bmi_w2"}:
            b = b.where(b >= 0)  # negative Stata special-missing codes -> analytical missing
        equal = (a.reset_index(drop=True).sub(b).abs() < 1e-5) | (a.reset_index(drop=True).isna() & b.isna())
        cov_rows.append({"cleaned_column": clean_var, "raw_candidate": raw_var,
            "source_label": labels.get(raw_var, ""), "matches_including_missing": int(equal.sum()),
            "n": len(cohort), "mismatches": int((~equal).sum())})
pd.DataFrame(cov_rows).to_csv(OUT / "covariate_candidate_comparison.csv", index=False)

print("Audit complete. Reports saved to data/audit/.")
print(f"Frozen cleaned cohort: {len(cohort)} participants, {len(cohort.columns)} columns")
print("No cleaned dataset was overwritten.")
