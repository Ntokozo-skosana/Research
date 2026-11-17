"""Preflight: whitelist-only predictors, dataset integrity and same-row CV consistency."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_CLEAN_SHA = "77E2B9D4A6B72D846A9D5767FF86F54C77413536DA30BC1B8D82827F5F46C36E"
EXPECTED_ORIGINAL_SHA = "11CE93009FC757D8523C1767769871B20BDB5190DC82F56CE457B6F4914B606A"
FORBIDDEN = {"mmse_w2", "delta_mmse", "mmse_w2_status", "mmse_w1_status",
 "mmse_both_completed", "mmse_any_partial", "duration_w1_raw", "duration_w2_raw", "duration_w3_raw",
 "survey_w2_year", "survey_w3_year", "survey_w2_pre_dementia_w1_by_year",
 "survey_w2_2019_order_uncertain", "survey_w3_pre_dementia_w2_by_year", "survey_w3_2022_order_uncertain"}
FORBIDDEN |= {f"duration_w{w}_extreme_flag" for w in (1,2,3)}
FORBIDDEN |= {"duration_extreme_any", "duration_valid_wave_count"}

def check(data, feature_sets, original=None):
    data=Path(data)
    sha=hashlib.sha256(data.read_bytes()).hexdigest().upper()
    if sha != EXPECTED_CLEAN_SHA:
        raise ValueError(f"Expected cleaned dataset checksum {EXPECTED_CLEAN_SHA}; got {sha}")
    df=pd.read_csv(data)
    if len(df)!=404 or df.prim_key.isna().any() or df.prim_key.duplicated().any():
        raise ValueError("Expected 404 distinct nonmissing prim_key IDs")
    if df[['mmse_w1','mmse_w2']].isna().any().any() or not np.allclose(df.mmse_w2-df.mmse_w1,df.delta_mmse):
        raise ValueError("Missing or inconsistent MMSE target")
    if any(set(features)&FORBIDDEN for features in feature_sets.values()):
        raise ValueError("Outcome, status, raw, flag or timing audit columns used as predictors")
    required=set().union(*[set(v) for v in feature_sets.values()])
    if missing:=required-set(df.columns):raise ValueError(f"Missing predictors: {sorted(missing)}")
    for w in (1,2,3):
        col=f'duration_w{w}';raw=f'{col}_raw';flag=f'{col}_extreme_flag'
        a=df[col].to_numpy(float);r=df[raw].to_numpy(float)
        expected=(np.isfinite(r)&((r<3)|(r>16)))
        if not np.array_equal(df[flag].to_numpy().astype(bool),expected):
            raise ValueError(f"{flag} inconsistent with screening range")
        if not np.array_equal(np.isnan(a),expected):
            raise ValueError(f"{col} missingness inconsistent with screening flags")
        if not np.allclose(a[~expected],r[~expected],equal_nan=True):
            raise ValueError(f"{col} non-extreme values changed")
    if original is not None:
        original=Path(original)
        oldsha=hashlib.sha256(original.read_bytes()).hexdigest().upper()
        if oldsha!=EXPECTED_ORIGINAL_SHA:raise ValueError(f"Original checksum mismatch: {oldsha}")
        old=pd.read_csv(original)
        if not old.prim_key.astype(str).equals(df.prim_key.astype(str)):
            raise ValueError("Row order changed; 5-fold CV assignments would differ")
        for col in ('mmse_w1','mmse_w2','age_w2','sex','education_w1','cesd_w2','wealth_quintile_w2'):
            if not old[col].equals(df[col]):raise ValueError(f"Unexpected change to {col}")
    print(f"PREFLIGHT PASSED: 404 IDs, original row order verified={original is not None}, cleaned SHA256={sha}",flush=True)
    print('Missing sleep durations:',{f'duration_w{w}':int(df[f'duration_w{w}'].isna().sum()) for w in (1,2,3)},flush=True)
    return df
