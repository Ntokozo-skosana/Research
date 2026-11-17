# Data Audit

## Overview

This directory is reserved for data-quality assessment, source-data validation and preprocessing audit outputs generated during this research.

The audit process was designed to assess the consistency, completeness and reliability of the HAALSI data used to construct the analytical dataset.

## Audit Activities

The research includes several data validation activities:

- Source dataset inventory and verification.
- Participant identifier coverage across source datasets.
- Comparison of cognitive assessment scores against their original source variables.
- Validation of sleep duration, sleep quality and sleep sufficiency variables.
- Verification of derived longitudinal sleep features.
- Assessment of demographic and health-related variable mappings.
- Investigation of missing data and potential cohort-selection criteria.

## Data Availability

Participant-level audit reports are not included in this public repository because they may contain HAALSI participant identifiers or information derived from restricted participant records.

The audit scripts are provided in the `scripts/` directory so that authorised researchers can regenerate the reports using their locally available source datasets.

## Reproducing the Audits

Run the relevant audit scripts from the project root directory after placing the authorised source data in the expected locations.

Refer to the project README and `docs/REPRODUCIBILITY_CHECKLIST.md` for instructions.

## Important Notes

- Do not commit participant-level audit reports to GitHub.
- Do not publish participant identifiers or individual-level validation records.
- Only aggregate, non-identifiable results should be considered for publication, subject to applicable data-use and research ethics requirements.
- Audit outputs should be stored securely alongside the local research data.

This directory is intentionally kept free of restricted participant-level audit outputs in the public repository.