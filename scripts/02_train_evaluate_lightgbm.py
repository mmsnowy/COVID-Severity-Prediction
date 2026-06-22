"""
02_train_evaluate_lightgbm.py

Rotating-downsampling ensemble training and temporal validation for
COVID-19 severity prediction.

Workflow:
  - load the 2023 development cohort and 2024 temporal-validation cohort
  - split 2023 into training and internal-test sets
  - apply min-max scaling fitted on the 2023 training set
  - downsample the 2023 internal test set to 1:1 and the 2024 validation set to 2:1
  - generate rotating-downsampling rounds from the imbalanced 2023 training set
  - train one LightGBM sub-model per round
  - aggregate sub-model predictions by averaging predicted probabilities
  - evaluate discrimination and calibration metrics

Note:
  Raw individual-level surveillance data are not publicly available.
  Please prepare input tables following example/feature_template.csv.
"""

import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    brier_score_loss,
)
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier


SEED = 9999
OUTCOME = "Clinical_Type"

CONTINUOUS_VARS = [
    "Age",
    "Ct_ORFlab",
    "Ct_N",
    "Age_CtN_Interaction",
    "S_Mut_Count",
    "M_Mut_Count",
    "ORF1a_Mut_Count",
    "ORF1b_Mut_Count",
    "Charge_Change_Mut_Load",
    "Size_Change_Mut_Load",
]


DEFAULT_LIGHTGBM_PARAMS = {
    "n_estimators": 52,
    "learning_rate": 0.0183,
    "num_leaves": 96,
    "colsample_bytree": 0.5107,
    "max_depth": 16,
    "verbose": -1,
}


def downsample(df: pd.DataFrame, ratio_0_to_1: int, seed: int) -> pd.DataFrame:
    """Downsample non-severe cases to a specified non-severe:severe ratio."""
    df_1 = df[df[OUTCOME] == 1]
    df_0 = df[df[OUTCOME] == 0]

    if len(df_1) == 0:
        raise ValueError("No severe cases found in the input data.")

    n_0 = int(len(df_1) * ratio_0_to_1)
    if n_0 > len(df_0):
        raise ValueError(
            f"Requested {n_0} non-severe cases but only {len(df_0)} are available."
        )

    df_0_sampled = df_0.sample(n=n_0, random_state=seed)
    sampled = pd.concat([df_1, df_0_sampled], axis=0)
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    return sampled


def load_data(data_2023_path: str, data_2024_path: str, features, seed: int):
    """Load, split, scale, and downsample evaluation cohorts."""
    df_2023 = pd.read_csv(data_2023_path)
    df_2024 = pd.read_csv(data_2024_path)

    missing_features = [f for f in features if f not in df_2023.columns or f not in df_2024.columns]
    if missing_features:
        raise ValueError(f"Features missing from input data: {missing_features}")

    if OUTCOME not in df_2023.columns or OUTCOME not in df_2024.columns:
        raise ValueError(f"Both input files must contain outcome column: {OUTCOME}")

    train_2023, test_2023 = train_test_split(
        df_2023,
        test_size=0.25,
        stratify=df_2023[OUTCOME],
        random_state=seed,
    )

    # Fit scaler on the 2023 training set only
    cont = [c for c in CONTINUOUS_VARS if c in train_2023.columns and c in features]
    scaler = MinMaxScaler()

    if cont:
        scaler.fit(train_2023[cont])

        def apply_scaling(d):
            d = d.copy()
            d[cont] = scaler.transform(d[cont])
            return d

        train_2023 = apply_scaling(train_2023)
        test_2023 = apply_scaling(test_2023)
        df_2024 = apply_scaling(df_2024)

    # Match the evaluation setup described in the manuscript
    test_2023 = downsample(test_2023, ratio_0_to_1=1, seed=seed)
    test_2024 = downsample(df_2024, ratio_0_to_1=2, seed=seed)

    return train_2023, test_2023, test_2024, scaler


def generate_rotating_rounds(train_df: pd.DataFrame, seed: int):
    """Generate 1:1 rotating-downsampling rounds from the training cohort."""
    df_1 = train_df[train_df[OUTCOME] == 1]
    df_0 = train_df[train_df[OUTCOME] == 0].sample(frac=1, random_state=seed)

    n_1 = len(df_1)
    n_rounds = len(df_0) // n_1

    if n_rounds < 1:
        raise ValueError("Not enough non-severe cases to generate rotating rounds.")

    rounds = []
    for i in range(n_rounds):
        subset_0 = df_0.iloc[i * n_1:(i + 1) * n_1]
        round_df = pd.concat([df_1, subset_0], axis=0)
        round_df = round_df.sample(frac=1, random_state=seed + i).reset_index(drop=True)
        rounds.append(round_df)

    print(f"Generated {n_rounds} rotating-downsampling rounds")
    return rounds


def load_lightgbm_params(config_path: str, seed: int):
    """Load LightGBM hyperparameters, or use defaults if the config is absent."""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            obj = json.load(f)
        params = obj.get("params", obj)
    else:
        params = DEFAULT_LIGHTGBM_PARAMS.copy()

    params.update({
        "random_state": seed,
        "seed": seed,
        "verbose": -1,
    })
    return params


def train_ensemble(rounds, features, params):
    """Train one LightGBM sub-model per rotating-downsampling round."""
    models = []
    for i, round_df in enumerate(rounds, start=1):
        model = LGBMClassifier(**params)
        model.fit(round_df[features], round_df[OUTCOME])
        models.append(model)
        print(f"Trained sub-model {i}/{len(rounds)}")
    return models


def predict_ensemble(models, X: pd.DataFrame):
    """Average predicted probabilities across sub-models."""
    probs = np.array([m.predict_proba(X)[:, 1] for m in models])
    return probs.mean(axis=0)


def calibration_metrics(y_true, y_prob, n_bins=10):
    """Calculate Brier score, expected calibration error, and calibration slope."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    brier = brier_score_loss(y_true, y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == 0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob > lo) & (y_prob <= hi)

        if mask.sum() == 0:
            continue

        mean_prob = y_prob[mask].mean()
        observed_rate = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(observed_rate - mean_prob)

    eps = 1e-9
    p = np.clip(y_prob, eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)

    try:
        lr = LogisticRegression(solver="lbfgs", max_iter=1000)
        lr.fit(logit_p, y_true)
        cal_slope = float(lr.coef_[0][0])
    except Exception:
        cal_slope = np.nan

    return brier, ece, cal_slope


def evaluate_predictions(y_true, y_prob, threshold=0.5):
    """Evaluate discrimination and calibration metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    brier, ece, cal_slope = calibration_metrics(y_true, y_prob)

    return {
        "F1": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "AUPR": average_precision_score(y_true, y_prob),
        "ACC": accuracy_score(y_true, y_pred),
        "Brier": brier,
        "ECE": ece,
        "CalSlope": cal_slope,
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate LightGBM rotating ensemble")
    parser.add_argument("--data-2023", default="../data/data_2023.csv")
    parser.add_argument("--data-2024", default="../data/data_2024.csv")
    parser.add_argument("--feature-list", default="../results/final_feature_set.csv")
    parser.add_argument("--config", default="../config/best_params_LightGBM.json")
    parser.add_argument("--output-dir", default="../results")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    features = pd.read_csv(args.feature_list)["feature"].tolist()

    train_2023, test_2023, test_2024, scaler = load_data(
        args.data_2023,
        args.data_2024,
        features,
        seed=args.seed,
    )

    params = load_lightgbm_params(args.config, seed=args.seed)

    rounds = generate_rotating_rounds(train_2023, seed=args.seed)
    models = train_ensemble(rounds, features, params)

    prob_2023 = predict_ensemble(models, test_2023[features])
    prob_2024 = predict_ensemble(models, test_2024[features])

    metrics_2023 = evaluate_predictions(test_2023[OUTCOME], prob_2023)
    metrics_2024 = evaluate_predictions(test_2024[OUTCOME], prob_2024)

    metrics_df = pd.DataFrame([
        {"Cohort": "2023_internal_test", **metrics_2023},
        {"Cohort": "2024_temporal_validation", **metrics_2024},
    ])

    metrics_df.to_csv(os.path.join(args.output_dir, "lightgbm_ensemble_metrics.csv"), index=False)

    pred_2023 = pd.DataFrame({
        "Cohort": "2023_internal_test",
        "y_true": test_2023[OUTCOME].values,
        "y_prob": prob_2023,
        "y_pred": (prob_2023 >= 0.5).astype(int),
    })

    pred_2024 = pd.DataFrame({
        "Cohort": "2024_temporal_validation",
        "y_true": test_2024[OUTCOME].values,
        "y_prob": prob_2024,
        "y_pred": (prob_2024 >= 0.5).astype(int),
    })

    pd.concat([pred_2023, pred_2024], axis=0).to_csv(
        os.path.join(args.output_dir, "lightgbm_ensemble_predictions.csv"),
        index=False,
    )

    joblib.dump(models, os.path.join(args.output_dir, "lightgbm_rotating_ensemble.joblib"))
    joblib.dump(scaler, os.path.join(args.output_dir, "minmax_scaler.joblib"))

    print(metrics_df.round(3).to_string(index=False))
    print("Training and evaluation completed.")


if __name__ == "__main__":
    main()
