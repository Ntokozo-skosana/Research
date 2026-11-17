# Processed Data

## Overview

This directory is reserved for the cleaned and prepared HAALSI analytical datasets generated during the data preprocessing stage of this research.

The final analytical dataset used in the experiments contains 404 participants and includes demographic, health, sleep-related and cognitive assessment variables.

## Data Availability

The processed participant-level dataset is not included in this repository because it contains information derived from restricted HAALSI participant data.

The cleaned dataset must therefore remain in an authorised local or secure research environment.

## Reproducing the Dataset

Researchers with authorised access to the original HAALSI datasets can use the cleaning and preprocessing scripts provided in the `scripts/` directory to prepare the analytical data.

The cleaning methodology, source-variable mappings and known limitations are documented in:

- `docs/CLEANING_AND_MERGING_PROVENANCE.md`
- `docs/SOURCE_DATA_REGISTER.md`
- `docs/REPRODUCIBILITY_CHECKLIST.md`

## Important Notes

- Do not commit cleaned participant-level CSV files to GitHub.
- Do not upload participant-level intermediate datasets or model inputs.
- Keep locally generated datasets within an appropriately secured research environment.
- The original 404-participant cohort selection procedure has not yet been fully established and should be considered when attempting to reproduce the historical analytical dataset.

This directory is intentionally kept free of participant-level data in the public repository.