"""Audit smoking/alcohol recodes and explore possible 404-person cohort rules.

Run from the package root:
    python scripts/01_audit_behaviour_and_cohort.py

This is an audit only. It does not modify the frozen cleaned dataset.
"""
from pathlib import Path
import io, zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "audit"
OUT.mkdir(parents=True, exist_ok=True)


def read_dta(archive_name, suffix):
    path = RAW / archive_name
    with zipfile.ZipFile(path) as z:
        matches = [n for n in z.namelist() if n.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {suffix} in {archive_name}; found {matches}")
        return pd.read_stata(io.BytesIO(z.read(matches[0])), convert_categoricals=False)

clean = pd.read_csv(PROCESSED / "haalsi_404_cleaned.csv")
source = read_dta("HAALSI Wave 3 Survey.zip", "HAALSI W1-3 Longitudinal Dataset.dta")
w1 = read_dta("HAALSI Dementia Wave 1.zip", "HAALSI_DEMENTIA_Respondent-1.dta")
w2 = read_dta("HAALSI Dementia Wave 2.zip", "HAALSI_DEMENTIA_Respondent_W2.dta")

# Behavioural variables: retain only the documented Yes/No values.
merged = clean[["prim_key", "current_smoker_w2", "recent_alcohol_w2"]].merge(
    source, on="prim_key", how="left", validate="one_to_one", suffixes=("", "_src")
)

# Smoking: use current-smoking answer when present. If it is skipped/missing,
# use an explicit "never smoked" response as No. Otherwise leave missing.
smoke_current = pd.to_numeric(merged["w2cm061"], errors="coerce").map({1: 1.0, 2: 0.0})
ever_smoked = pd.to_numeric(merged["w2cm059"], errors="coerce")
smoke_expected = smoke_current.mask(smoke_current.isna() & ever_smoked.eq(2), 0.0)

# Alcohol: if never consumed alcohol, the last-30-days item is skipped and
# logically maps to No. Otherwise use the last-30-days Yes/No response.
ever_alcohol = pd.to_numeric(merged["w2cm068"], errors="coerce")
recent = pd.to_numeric(merged["w2cm069"], errors="coerce").map({1: 1.0, 2: 0.0})
alcohol_expected = recent.mask(recent.isna() & ever_alcohol.eq(2), 0.0)

rows = []
for name, expected, actual, source_fields, rule in [
    ("current_smoker_w2", smoke_expected, merged["current_smoker_w2"], "w2cm061; w2cm059",
     "w2cm061: 1=Yes, 2=No; where current status is missing, w2cm059=2 (never smoked) is coded 0; otherwise missing"),
    ("recent_alcohol_w2", alcohol_expected, merged["recent_alcohol_w2"], "w2cm068; w2cm069",
     "w2cm069: 1=Yes, 2=No; where the last-30-days item is skipped, w2cm068=2 (never consumed) is coded 0; otherwise missing"),
]:
    expected = pd.to_numeric(expected, errors="coerce")
    actual = pd.to_numeric(actual, errors="coerce")
    match = expected.eq(actual) | (expected.isna() & actual.isna())
    rows.append({"cleaned_column": name, "source_fields": source_fields, "reconstructed_rule": rule,
                 "matches_including_missing": int(match.sum()), "n": len(match),
                 "mismatches": int((~match).sum()),
                 "cleaned_0": int(actual.eq(0).sum()), "cleaned_1": int(actual.eq(1).sum()),
                 "cleaned_missing": int(actual.isna().sum())})
    detail = merged[["prim_key", name]].copy()
    detail["reconstructed_value"] = expected
    detail["matches"] = match
    detail.to_csv(OUT / f"{name}_row_validation.csv", index=False)

pd.DataFrame(rows).to_csv(OUT / "behaviour_variable_validation.csv", index=False)

# Explore plausible cohort definitions using all available source participants.
# These are diagnostics, not asserted selection rules.
all_ids = pd.DataFrame({"prim_key": source["prim_key"]}).drop_duplicates()
all_ids = all_ids.merge(w1[["prim_key", "mmse_total", "mmse_status"]], on="prim_key", how="left", validate="one_to_one")
all_ids = all_ids.merge(w2[["prim_key", "w2_mmse_total", "w2_mmse_status"]], on="prim_key", how="left", validate="one_to_one")
all_ids = all_ids.merge(source[["prim_key", "w1sp003", "w2sp003", "w3sp003",
                                "w1sp010", "w2sp010", "w3sp010"]], on="prim_key", how="left", validate="one_to_one")

# Ensure numeric tests treat Stata special missing values as missing for these fields.
for col in all_ids.columns:
    if col != "prim_key":
        all_ids[col] = pd.to_numeric(all_ids[col], errors="coerce")

cohort_ids = set(clean["prim_key"].astype(str))
all_ids["in_cleaned_404"] = all_ids["prim_key"].astype(str).isin(cohort_ids)
criteria = {
    "all_source_participants": pd.Series(True, index=all_ids.index),
    "wave1_mmse_nonmissing": all_ids["mmse_total"].notna(),
    "wave2_mmse_nonmissing": all_ids["w2_mmse_total"].notna(),
    "both_mmse_nonmissing": all_ids["mmse_total"].notna() & all_ids["w2_mmse_total"].notna(),
    "wave1_mmse_status_1": all_ids["mmse_status"].eq(1),
    "wave2_mmse_status_1": all_ids["w2_mmse_status"].eq(1),
    "both_mmse_status_1": all_ids["mmse_status"].eq(1) & all_ids["w2_mmse_status"].eq(1),
    "any_sleep_duration_wave": all_ids[["w1sp003", "w2sp003", "w3sp003"]].notna().any(axis=1),
    "all_three_sleep_duration_nonmissing": all_ids[["w1sp003", "w2sp003", "w3sp003"]].notna().all(axis=1),
    "any_sleep_quality_wave": all_ids[["w1sp010", "w2sp010", "w3sp010"]].notna().any(axis=1),
    "all_three_sleep_quality_nonmissing": all_ids[["w1sp010", "w2sp010", "w3sp010"]].notna().all(axis=1),
}
cohort_rows = []
for name, mask in criteria.items():
    selected = set(all_ids.loc[mask, "prim_key"].astype(str))
    cohort_rows.append({"candidate_rule": name, "source_n_meeting_rule": int(mask.sum()),
                        "cleaned_404_ids_meeting_rule": int(len(selected & cohort_ids)),
                        "cleaned_404_ids_not_meeting_rule": int(len(cohort_ids - selected)),
                        "source_ids_meeting_rule_not_in_cleaned_404": int(len(selected - cohort_ids)),
                        "exactly_reproduces_404_ids": bool(selected == cohort_ids)})
pd.DataFrame(cohort_rows).to_csv(OUT / "cohort_selection_candidate_audit.csv", index=False)

print("Behaviour and cohort audit complete.")
print(pd.DataFrame(rows).to_string(index=False))
print("Cohort selection report: data/audit/cohort_selection_candidate_audit.csv")
print("No cleaned dataset was overwritten.")
