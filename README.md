# Predicting Cognitive Performance from Longitudinal Sleep Patterns Using Interpretable Machine Learning

## Overview

This project investigates the relationship between longitudinal sleep patterns and cognitive performance using data from the **Health and Aging in Africa: A Longitudinal Study of an INDEPTH Community in South Africa (HAALSI)**.

The research applies machine learning techniques to examine whether sleep-related characteristics can help predict follow-up cognitive performance, measured using the Mini-Mental State Examination (MMSE).

The project also explores model interpretability to understand how sleep-related and other participant characteristics contribute to model predictions.

The repository contains the research code, notebooks, documentation and permitted experimental outputs. Restricted HAALSI datasets and participant-level analytical data are intentionally excluded.

## Research Objectives

The main objectives are to:

- Investigate the relationship between longitudinal sleep patterns and cognitive performance.
- Prepare and validate sleep-related and demographic variables for machine learning.
- Develop and evaluate conventional machine learning and sequence-based models.
- Compare model performance using appropriate evaluation metrics.
- Apply interpretable machine learning techniques to examine model predictions.
- Investigate the contribution of sleep-related features to predictive performance.

## Repository Structure

```text
HAALSI-Honours-Research/
│
├── data/
│   ├── raw/
│   │   └── README.md
│   ├── processed/
│   │   └── README.md
│   └── audit/
│       └── README.md
│
├── docs/
│   ├── CLEANING_AND_MERGING_PROVENANCE.md
│   ├── EXPERIMENTS_AND_RERUNNING.md
│   ├── FILE_INVENTORY.txt
│   ├── REPRODUCIBILITY_CHECKLIST.md
│   └── SOURCE_DATA_REGISTER.md
│
├── notebooks/
│   └── cleaned.ipynb
│
├── scripts/
│   ├── Data auditing and validation scripts
│   ├── Dataset preparation scripts
│   ├── Model training scripts
│   └── Explainability and sensitivity analysis scripts
│
├── results/
│   └── Permitted aggregate experimental results and figures
│
├── .gitignore
├── README.md
└── requirements.txt