# Cleaning, merging and feature-engineering record

## 1. Purpose and provenance status

This document describes the analytical dataset supplied with this project and separates **verified source operations** from **historical transformations that cannot yet be proven**. The final modelling file is `data/processed/haalsi_404_cleaned.csv` (404 rows, 60 columns). The accompanying `haalsi_original_reference.csv` contains the same 404 participant IDs and 42 core analytical columns; it is a reference copy for integrity checks, **not a raw HAALSI dataset**.

The four original source archives are now stored under `data/raw/`. The available package does not contain the original notebook/script used at the time the 404-row file was first assembled. Therefore this is a documented reconstruction and audit trail, not a claim that the historical file has been regenerated exactly from raw data.

## 2. Source files actually present

| Archive in `data/raw/` | Stata data used for audit | Notes |
|---|---|---|
| `HAALSI Dementia Wave 1.zip` | `HAALSI_DEMENTIA_Respondent-1.dta` | 635 respondent records; includes `prim_key`, `mmse_status`, `mmse_total` and MMSE item fields |
| `HAALSI Dementia Wave 2.zip` | `HAALSI_DEMENTIA_Respondent_W2.dta` | 683 respondent records; includes `prim_key`, `w2_mmse_status`, `w2_mmse_total` and MMSE item fields |
| `HAALSI Wave 2 Survey.zip` | `Longitudinal Public Release Dataset V5.dta` | The archive contains V5, not V4; 5,059 records and 4,111 variables |
| `HAALSI Wave 3 Survey.zip` | `HAALSI W1-3 Longitudinal Dataset.dta` | Combined W1–W3 longitudinal file; 5,059 records and 6,169 variables |

**Version discrepancy to resolve:** the source register previously named Wave 2 Survey V4, but the supplied ZIP contains `Longitudinal Public Release Dataset V5.dta`. The Wave 3 ZIP contains the combined W1–W3 longitudinal file. Do not describe the included Wave 2 archive as V4 until its provenance is confirmed. Record the download page/version and date in `SOURCE_DATA_REGISTER.md`.

## 3. Participant linkage and cohort definition

- `prim_key` is the participant-level linkage field present in the dementia respondent and longitudinal survey files.
- The supplied analytical file contains 404 unique participant identifiers.
- The two dementia respondent datasets provide the two MMSE observations used in the analytical table: Wave 1 `mmse_total` and Wave 2 `w2_mmse_total`.
- The 404-row participant set is treated as a **frozen historical cohort** for audit purposes. Its exact original inclusion/exclusion logic is not established by the files currently available.
- The supplied `haalsi_original_reference.csv` is retained to check the 42 core columns and participant order. It is not used as evidence of raw-source provenance.

Run `python scripts/00_audit_raw_sources.py` to generate participant-ID coverage and direct source-comparison reports without modifying the modelling dataset.

## 4. Directly verified cognitive fields

The raw dementia respondent files label `mmse_total` as “Mini Mental: Total Score” and `mmse_status` as “Mini Mental Status”. Wave 2 uses `w2_mmse_total` and `w2_mmse_status`. The audit script joins these files to the frozen 404 cohort by `prim_key` and writes `data/audit/mmse_source_comparison.csv`.

The cleaned columns `mmse_w1` and `mmse_w2` should match the corresponding stored totals. `delta_mmse` is the arithmetic difference `mmse_w2 - mmse_w1`. This is a change in recorded score, not a diagnosis of mild cognitive impairment (MCI), and a numeric score does not itself establish that the assessment was fully completed. Completion status is retained separately in the 60-column file.

A separate item-level Wave 1 scoring audit is included for sensitivity analysis. Its reconstructed scores are provisional and must not be presented as an official correction to HAALSI scoring without confirmation from the study documentation or data custodians.

## 5. Sleep source fields and transformations

The combined W1–W3 longitudinal Stata file contains the following labelled variables for each wave:

| Source field pattern | Source label / meaning | Corresponding analytical concept |
|---|---|---|
| `wNsp001_hour`, `wNsp001_minute` | Usual time participant goes to sleep | Bedtime components |
| `wNsp002_hour`, `wNsp002_minute` | Usual time participant gets up | Wake-time components |
| `wNsp003` | Hours slept each day in past four weeks | Self-reported sleep duration |
| `wNsp004` | Whether participant got enough sleep in past four weeks | Sleep sufficiency response |
| `wNsp005` | How often participant wakes during the night in past four weeks | Night-awakening frequency |

Here `N` is `1`, `2` or `3`. These source labels are important: the existing cleaned file uses names such as `quality_w1` and `insufficiency_w1`. The exact historical recoding that produced those columns is **not confirmed** by a cleaning script. In particular, do not assume `quality_w1` is a direct copy of `sp004` or `sp005`; compare it with the source data and codebook before assigning a definitive interpretation. `sleep_candidate_comparison.csv` reports direct comparisons for duration and bedtime hour only; mismatches must be investigated rather than silently overwritten.

The earlier reconstruction work recorded the following candidate rules, which should be treated as documented candidate processing rather than proven historical code:

1. Convert valid bedtime/wake time components to decimal hours as `hour + minute / 60`, requiring hour 0–23 and minute 0–59. Preserve the original hour/minute components where available.
2. Keep self-reported duration in hours only when the value is within the raw instrument's allowable 0–24 range; treat special negative/non-response codes as missing after confirming their codebook meaning.
3. Preserve raw duration values and flag the analyst-defined 3–16 hour plausibility range separately. This range is a sensitivity flag, not a clinical definition, and should not automatically delete records.
4. Retain sleep sufficiency and night-awakening responses as coded ordinal variables until response labels and direction are confirmed from the questionnaire/codebook. Do not reverse-code or rename them based only on an assumed direction.
5. Preserve missing observations. Do not perform global complete-case deletion or impute values during raw-data cleaning. Imputation belongs inside each model's training fold as implemented by the modelling scripts.

## 6. Derived longitudinal features present in the analytical file

The file contains per-wave sleep variables and derived summaries. The column names indicate the following calculations, but exact historical treatment of missing values and special codes must be verified before these are described as fully reproduced:

- `duration_mean`: mean of available wave-specific duration values.
- `duration_change`: later-wave minus earlier-wave duration (the precise wave endpoints should be confirmed against the row-level data and original code).
- `duration_variability`: standard deviation across available wave-specific duration values; confirm whether sample or population standard deviation was used.
- `quality_mean`, `quality_change`: corresponding summaries for the variable labelled `quality` in the cleaned file; substantive interpretation depends on source mapping.
- `insufficiency_mean`, `insufficiency_change`: corresponding summaries for the variable labelled `insufficiency`; confirm source coding and direction.
- `bedtime_mean_shifted`, `bedtime_change`: bedtime summaries. Bedtime is circular (near-midnight values wrap around), so the existing shifted representation requires its original wrap/shift rule to be documented before clinical interpretation.
- `delta_mmse`: `mmse_w2 - mmse_w1`.

The audit/flag columns in the 60-column file include raw duration copies, extreme-duration flags, valid-wave counts, MMSE completion flags, survey years, and timing-order flags. These are quality-control or temporal-audit variables and must not automatically be used as model predictors. The model scripts use explicit predictor sets; continue to follow those whitelists.

## 7. Demographic and health covariates

The final file contains age, sex, education, CES-D, asset/wealth, marital status, hypertension, diabetes, BMI, smoking and alcohol fields. The combined longitudinal survey archive includes source fields with labels such as `w2c_rage_calc` (calculated age), `w2c_rsex` (respondent sex combination), `w1c_bd_educ4` (Wave 1 education category), `w2c_cd_cesd` (Wave 2 CES-D), `w2c_hh_assetindex` (Wave 2 asset index), `w2c_wealthindex` (Wave 2 wealth quintiles), `w2c_bs_bmi` (Wave 2 BMI), and Wave 2 hypertension/diabetes indicators. These are candidate source mappings based on labels; verify the exact variables, value labels, special missing codes and transformations against the data dictionary before asserting a one-to-one mapping. Smoking, alcohol and marital-status source mappings require the same verification.

The file includes separate missingness indicators for several health covariates. Do not replace missing values with zero unless the source definition explicitly means absence of a condition. The missingness flags should remain distinct from the underlying health response.

## 8. Missing data, exclusions and model preparation

- Missing and special survey responses must be distinguished from substantive zero values using the source codebooks.
- Keep missing values in the analytical CSV unless a documented variable-specific rule requires otherwise.
- Do not delete participants globally because one predictor is missing.
- Do not apply imputation before cross-validation. The training scripts should fit preprocessing/imputation only on the training partition/fold.
- Duration plausibility flags and assessment completion flags are audit fields; they are not equivalent to participant exclusion criteria.
- The 404-row cohort selection rule remains a historical provenance gap. The audit reports identify which frozen-cohort IDs are found in each source file but do not invent a selection explanation.

## 9. Temporal-ordering caution

The analytical dataset includes Wave 2 and Wave 3 sleep/covariate measurements and flags for uncertain year-level ordering. The exact interview dates of each sleep survey and dementia assessment have not been reconciled at participant level in this package. Therefore, do not describe all Wave 2/Wave 3 features as confirmed pre-outcome predictors. Wave 3 sleep must not be used as a predictor of an earlier cognitive outcome unless chronology is verified.

## 10. Reproducibility boundary and next validation

The updated package makes the raw files, source labels, cohort IDs, MMSE matching, sleep source candidates, and known/candidate cleaning rules transparent. It does **not** claim exact raw-to-404 regeneration. To close that gap:

1. Resolve the Wave 2 V4/V5 release discrepancy.
2. Verify the mappings and coding of `quality`, `insufficiency`, smoking, alcohol, marital status and all covariates.
3. Recover the exact 404-cohort inclusion/exclusion logic.
4. Recover the formula and missing-value handling for every engineered feature, especially bedtime shifting and duration variability.
5. Run the raw-source audit and review all mismatches.
6. Only then create a deterministic rebuild script that outputs a candidate 404-row file and compare every ID and field against the frozen historical CSV. Preserve the historical CSV until a full comparison passes.

**Integrity principle:** do not overwrite the 404-row analytical file with a newly inferred version. Preserve the current file and document any reconstructed candidate separately until all comparisons are explained.

## 11. Additional row-level mappings now verified against the supplied raw files

The audit was extended to test candidate transformations for all 404 frozen-cohort participants. The following mappings reproduce the cleaned columns exactly (404/404 matches, including missing values):

| Cleaned columns | Raw source | Verified transformation |
|---|---|---|
| `duration_w1`, `duration_w2`, `duration_w3` | `w1sp003`, `w2sp003`, `w3sp003` | Retain values from 3 through 16 hours inclusive; set values outside that interval to missing |
| `quality_w1`, `quality_w2`, `quality_w3` | `w1sp010`, `w2sp010`, `w3sp010` | Direct source values; source label is “RATE SLEEP QUALITY” |
| `insufficiency_w1`, `insufficiency_w2`, `insufficiency_w3` | `w1sp004`, `w2sp004`, `w3sp004` | Reverse-code as `6 - source response`; source question asks whether the respondent got enough sleep |
| `bedtime_w1`, `bedtime_w2`, `bedtime_w3` | `wNsp001_hour`, `wNsp001_minute` | `hour + minute/60`; add 24 when the resulting time is earlier than 06:00 |

The supplied W1–W3 codebook lists negative special-response values for several sleep fields (for example, `-98` and `-97`), rather than treating them as ordinary answers. The cleaned sleep comparisons above are based on numeric source values and the observed exact transformations; the cleaning script preserves non-response as missing rather than assigning it a substantive ordinal value. The codebook's descriptive labels establish what the variables ask, but the Stata file did not expose separate value-label dictionaries for all these fields. Do not invent verbal labels for the individual numeric categories without the questionnaire's response options.

### Verified derived sleep features

The following derived columns also match their candidate formulas for all 404 participants:

- `duration_mean`, `quality_mean`, `insufficiency_mean`: row-wise mean of the three wave-specific cleaned values, skipping missing values.
- `duration_variability`: sample standard deviation across the three cleaned duration values (`ddof=1`).
- `duration_change`, `quality_change`, `insufficiency_change`, `bedtime_change`: Wave 3 minus Wave 1.
- `bedtime_mean_shifted`: row-wise mean of the three shifted bedtime values.

These formulas reproduce the stored columns, but the resulting `duration_change` and `bedtime_change` still describe differences between survey waves, not necessarily changes over equal time intervals. The early-morning bedtime shift is a numerical alignment convention, not a clinical interpretation.

### Additional covariates verified

The following direct source mappings reproduce the cleaned fields exactly after treating negative Stata special-missing values as missing where applicable:

| Cleaned column | Raw source | Result |
|---|---|---|
| `age_w2` | `w2c_rage_calc` | 404/404 |
| `sex` | `w2c_rsex` | 404/404 |
| `education_w1` | `w1c_bd_educ4` | 404/404 |
| `cesd_w2` | `w2c_cd_cesd` | 404/404 |
| `asset_index_w2` | `w2c_hh_assetindex` | 404/404 after negative special-missing values are set to missing |
| `wealth_quintile_w2` | `w2c_wealthindex` | 404/404 |
| `bmi_w2` | `w2c_bs_bmi` | 404/404 after negative special-missing values are set to missing and allowing for CSV rounding |
| `marital_status_w2` | `w2c_bd_mar` | 404/404 |
| `hypertension_w2` | `w2c_cm_hypertension_srdiag` | 404/404; this is the broad definition including self-report of ever diagnosed |
| `diabetes_w2` | `w2c_cm_diabetes_srdiag` | 404/404; this is the broad definition including self-report of ever diagnosed |

The narrow hypertension and diabetes definitions do not reproduce the cleaned fields and are not the mappings used for these columns. The current smoking and recent alcohol fields still require a precise reconstruction of their handling of unanswered items and cross-wave corrections; their apparent binary coding must not be mistaken for a verified raw-to-clean rule.

### New audit outputs

Running `python scripts/00_audit_raw_sources.py` now also creates:

- `data/audit/sleep_mapping_validation.csv`
- `data/audit/derived_sleep_feature_validation.csv`
- `data/audit/covariate_candidate_comparison.csv`

These reports include match counts and mismatches for the frozen 404-row cohort. The script still does not overwrite `haalsi_404_cleaned.csv`.


## Smoking and alcohol recoding audit (verified)

A separate audit (`scripts/01_audit_behaviour_and_cohort.py`) compared the cleaned binary variables against the Wave 2 questionnaire fields for all 404 participants. Both reconstructed rules reproduce the cleaned values exactly, including missingness (404/404 matches). The audit writes row-level validation files and a summary to `data/audit/`.

- `current_smoker_w2`: use `w2cm061` (currently smoke; 1=Yes, 2=No). Where this item is missing, code 0 only when `w2cm059` (ever smoked) is 2=No. Otherwise retain missing. This rule reproduces the cleaned variable exactly. The separate `w2_cm_061_combo` corrected cross-wave field does **not** reproduce the cleaned variable exactly and should not be substituted.
- `recent_alcohol_w2`: use `w2cm069` (consumed alcohol in the last 30 days; 1=Yes, 2=No). Where the item is skipped/missing, code 0 only when `w2cm068` (ever consumed alcohol) is 2=No. Otherwise retain missing. This rule reproduces the cleaned variable exactly.

These are recoding rules supported by exact row-level comparison; they should not be interpreted as measures of lifetime use or drinking quantity.

## Cohort selection: still unresolved

The 404 participant IDs are present in all four supplied source datasets, but the original inclusion/exclusion rule that selected these 404 from the larger HAALSI source population has not been recovered. The candidate audit (`data/audit/cohort_selection_candidate_audit.csv`) shows that all 404 satisfy several broad conditions, including non-missing Wave 1 and Wave 2 MMSE totals and non-missing sleep duration across all three survey waves. However, these conditions include additional source participants and therefore do not uniquely reproduce the 404 IDs. The 404 cohort must not be described as generated by any of these candidate rules. The historical cohort selection remains undocumented pending the original selection code, a dated participant ID list with provenance, or confirmation from the researcher/supervisor.

## Deeper cohort availability investigation

A further audit (`scripts/02_deep_cohort_profile.py`) compares the frozen 404 IDs with all 5,059 unique participants in the supplied combined W1–W3 survey source. It creates a participant membership list, row-level availability signatures, candidate-rule counts, and missingness/distribution summaries by cohort membership. It is descriptive and does not change the analytical CSV.

Findings from the supplied files:

- All 404 frozen-cohort participants have non-missing Wave 1 and Wave 2 numeric MMSE totals. The broad rule `both_mmse_nonmissing` selects 506 source participants, including 102 who are not in the frozen 404.
- All 404 have non-missing raw sleep duration and sleep-quality responses in all three waves. The broad three-wave completeness rule selects 3,064 source participants, including 2,660 outside the frozen cohort.
- Requiring both MMSE status fields to equal 1 does not explain the cohort: only 44 of the 404 meet this condition, and 9 additional source participants outside the 404 meet it.
- Requiring all three raw duration responses to lie between 3 and 16 hours also does not explain the cohort: 394 of the 404 meet it, while 10 do not. This is a plausibility rule for duration values, not a defensible participant-selection rule.
- The 404 are substantially more complete than the rest of the source population on several fields. For example, all 404 have non-missing age, education, CES-D and the three waves of raw sleep duration/quality. This pattern is consistent with a selected complete-data subset, but it does not identify the exact historical selection operation.

The exact 404-person inclusion/exclusion rule therefore remains **unrecovered**. None of the tested single-variable or combined completeness rules exactly reproduces the frozen participant IDs. Do not claim that the cohort was selected by any of these rules. The next evidence needed is the original selection notebook/script, a dated participant-ID list with provenance, or confirmation from the researcher/supervisor. The frozen dataset remains unchanged.

New outputs from this audit:
- `data/audit/full_source_cohort_membership.csv`
- `data/audit/deep_cohort_candidate_rules.csv`
- `data/audit/cohort_source_availability_profile.csv`
- `data/audit/cohort_availability_signatures.csv`
