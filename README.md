# COVID-19 Severity Prediction

Code for the study **"A Population-Based Retrospective Machine Learning Study of COVID-19 Severity Using Integrated Clinical and Viral Genomic Data in Jiangsu Province, China"**.

## Overview

This repository provides analysis scripts for machine-learning-based COVID-19 severity prediction using integrated clinical, regional, and viral genomic features. The code includes feature filtering, LightGBM ensemble training with rotating downsampling, temporal validation, calibration evaluation, and SHAP-based model interpretation.

The repository is intended to support methodological transparency and local implementation of the analytical workflow. It is not a deployed clinical decision-support tool or real-time web API.

## Repository structure

```text
scripts/
  01_feature_filtering.py
  02_train_evaluate_lightgbm.py
  03_shap_analysis.py

example/
  demo_input.csv
  feature_list_45.csv
```

## Requirements

* Python 3.10+
* pandas
* numpy
* scikit-learn
* lightgbm
* xgboost
* shap
* scipy
* matplotlib
* optuna
* imbalanced-learn

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

The scripts can be run locally after preparing input data according to the provided feature template.

```bash
python scripts/01_feature_filtering.py
python scripts/02_train_evaluate_lightgbm.py
python scripts/03_shap_analysis.py
```

## Data availability

The original individual-level SARS-CoV-2 genomic and clinical surveillance data are not publicly shared because they were generated through routine public health surveillance and contain sensitive clinical and epidemiological information. A demo input template is provided to illustrate the expected data structure.

Researchers may apply the workflow to their own appropriately formatted datasets.

## Citation

Citation information will be updated after publication.
