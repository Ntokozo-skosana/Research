"""Profile the frozen 404-person cohort against the full HAALSI source population.

Run from package root: python scripts/02_deep_cohort_profile.py
This is descriptive only; it does not infer or apply a cohort selection rule.
"""
from pathlib import Path
import io, zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "audit"
OUT.mkdir(parents=True, exist_ok=True)

def read_dta(archive, suffix):
    with zipfile.ZipFile(RAW / archive) as z:
        hits = [n for n in z.namelist() if n.endswith(suffix)]
        if len(hits) != 1:
            raise RuntimeError(f"Expected one {suffix}; found {hits}")
        return pd.read_stata(io.BytesIO(z.read(hits[0])), convert_categoricals=False)

clean = pd.read_csv(PROC / "haalsi_404_cleaned.csv")
survey = read_dta("HAALSI Wave 3 Survey.zip", "HAALSI W1-3 Longitudinal Dataset.dta")
w1 = read_dta("HAALSI Dementia Wave 1.zip", "HAALSI_DEMENTIA_Respondent-1.dta")
w2 = read_dta("HAALSI Dementia Wave 2.zip", "HAALSI_DEMENTIA_Respondent_W2.dta")

cols = ["prim_key", "w1sp003", "w2sp003", "w3sp003", "w1sp010", "w2sp010", "w3sp010",
        "w1c_bd_educ4", "w2c_rage_calc", "w2c_cd_cesd", "w2c_bs_bmi",
        "w2c_cm_hypertension_srdiag", "w2c_cm_diabetes_srdiag", "w2cm061", "w2cm069"]
base = survey[cols].drop_duplicates("prim_key").copy()
base = base.merge(w1[["prim_key", "mmse_total", "mmse_status"]], on="prim_key", how="left", validate="one_to_one")
base = base.merge(w2[["prim_key", "w2_mmse_total", "w2_mmse_status"]], on="prim_key", how="left", validate="one_to_one")
base["in_frozen_404"] = base.prim_key.astype(str).isin(set(clean.prim_key.astype(str)))

numeric = [c for c in base.columns if c not in ["prim_key", "in_frozen_404"]]
for c in numeric:
    base[c] = pd.to_numeric(base[c], errors="coerce")

# One row per source participant: membership list supports a transparent ID audit.
base[["prim_key", "in_frozen_404"]].sort_values("prim_key").to_csv(OUT / "full_source_cohort_membership.csv", index=False)

# Compare availability patterns without pretending that any pattern is the historical rule.
base["both_mmse_nonmissing"] = base.mmse_total.notna() & base.w2_mmse_total.notna()
base["all_three_duration_nonmissing"] = base[["w1sp003", "w2sp003", "w3sp003"]].notna().all(axis=1)
base["all_three_quality_nonmissing"] = base[["w1sp010", "w2sp010", "w3sp010"]].notna().all(axis=1)
base["both_mmse_status_1"] = base.mmse_status.eq(1) & base.w2_mmse_status.eq(1)
base["all_three_duration_plausible_3_16"] = base[["w1sp003", "w2sp003", "w3sp003"]].apply(lambda s: s.between(3,16)).all(axis=1)

rules = ["both_mmse_nonmissing", "all_three_duration_nonmissing", "all_three_quality_nonmissing",
         "both_mmse_status_1", "all_three_duration_plausible_3_16"]
rows=[]
for rule in rules:
    m=base[rule].fillna(False)
    rows.append({"candidate_rule":rule,"source_n":int(m.sum()),
                 "frozen_404_n_meeting":int((m & base.in_frozen_404).sum()),
                 "frozen_404_n_not_meeting":int((~m & base.in_frozen_404).sum()),
                 "noncohort_n_meeting":int((m & ~base.in_frozen_404).sum()),
                 "exactly_reproduces_frozen_404":bool(set(base.loc[m,"prim_key"].astype(str))==set(clean.prim_key.astype(str)))})
pd.DataFrame(rows).to_csv(OUT / "deep_cohort_candidate_rules.csv",index=False)

# Missingness and basic distributions by cohort membership; counts are descriptive only.
profile_vars = ["mmse_total", "w2_mmse_total", "w1sp003", "w2sp003", "w3sp003", "w1sp010", "w2sp010", "w3sp010",
                "w1c_bd_educ4", "w2c_rage_calc", "w2c_cd_cesd", "w2c_bs_bmi",
                "w2c_cm_hypertension_srdiag", "w2c_cm_diabetes_srdiag", "w2cm061", "w2cm069"]
prof=[]
for group, frame in base.groupby("in_frozen_404"):
    for var in profile_vars:
        s=frame[var]
        prof.append({"group":"frozen_404" if group else "outside_frozen_404", "source_variable":var,
                     "n_group":len(frame),"n_nonmissing":int(s.notna().sum()),"n_missing":int(s.isna().sum()),
                     "missing_percent":round(float(s.isna().mean()*100),2),
                     "mean_nonmissing":round(float(s.mean()),4) if s.notna().any() else None,
                     "median_nonmissing":round(float(s.median()),4) if s.notna().any() else None})
pd.DataFrame(prof).to_csv(OUT / "cohort_source_availability_profile.csv",index=False)

# Compact row-level feature availability signature for all source participants.
availability = base[["prim_key","in_frozen_404"] + profile_vars].copy()
for c in profile_vars:
    availability[c+"__available"] = availability[c].notna().astype(int)
availability = availability[["prim_key","in_frozen_404"] + [c+"__available" for c in profile_vars]]
availability.to_csv(OUT / "cohort_availability_signatures.csv",index=False)

print(f"Source participants: {len(base)}; frozen cohort: {int(base.in_frozen_404.sum())}")
print(pd.DataFrame(rows).to_string(index=False))
print("Wrote membership, candidate-rule, missingness-profile and availability-signature audits.")
print("No selection rule was inferred and no analytical data were changed.")
