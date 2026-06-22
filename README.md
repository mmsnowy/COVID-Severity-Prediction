# COVID-19 Severity Prediction

Code for the study **"A Population-Based Retrospective Machine Learning Study of COVID-19 Severity Using Integrated Clinical and Viral Genomic Data in Jiangsu Province, China"**.

## Overview

This repository provides analysis scripts for machine-learning-based COVID-19 severity prediction using integrated clinical, regional, and viral genomic features. The code includes two-stage feature filtering, LightGBM ensemble training with rotating downsampling, temporal validation, calibration evaluation, and SHAP-based model interpretation.

The repository is intended to support methodological transparency and local implementation of the analytical workflow. It is not a deployed clinical decision-support tool or real-time web API.

## Repository structure

```text
scripts/
  01_feature_filtering.py
  02_train_evaluate_lightgbm.py
  03_shap_analysis.py

example/
  feature_template.csv

config/
  best_params_LightGBM.json
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
* seaborn
* optuna
* imbalanced-learn
* joblib

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

Prepare input data according to the provided feature template. The original surveillance data are not included in this repository.

Example commands:

```bash
python scripts/01_feature_filtering.py \
  --input data/preprocessed_data.csv \
  --output-dir results

python scripts/02_train_evaluate_lightgbm.py \
  --data-2023 data/data_2023.csv \
  --data-2024 data/data_2024.csv \
  --feature-list results/final_feature_set.csv \
  --config config/best_params_LightGBM.json \
  --output-dir results

python scripts/03_shap_analysis.py \
  --data-2023 data/data_2023.csv \
  --data-2024 data/data_2024.csv \
  --feature-list results/final_feature_set.csv \
  --config config/best_params_LightGBM.json \
  --output-dir results/shap
```

## Random seeds

The training/evaluation script uses a default random seed of 9999. The SHAP analysis script uses a default random seed of 12345, matching the attribution-analysis workflow used for model interpretation. Users may modify the `--seed` argument to test robustness under alternative random splits.

## Data availability

The original individual-level SARS-CoV-2 genomic and clinical surveillance data are not publicly shared because they were generated through routine public health surveillance and contain sensitive clinical and epidemiological information. A feature template is provided to illustrate the expected data structure.

Researchers may apply the workflow to their own appropriately formatted datasets.

## Citation

Citation information will be updated after publication.
