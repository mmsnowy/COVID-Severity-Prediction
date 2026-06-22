""""
03_shap_analysis.py

SHAP-based interpretation and stability analysis for the rotating-
downsampling LightGBM ensemble.

Workflow:
  - load the selected feature list and input cohorts
  - retrain the LightGBM rotating-downsampling ensemble
  - compute SHAP values for each sub-model on the 2024 validation cohort
  - average SHAP values across sub-models for the main feature-importance summary
  - assess round-to-round SHAP stability using mean |SHAP|, SD, CV, and
    approximate 95% confidence intervals across rounds

Note:
  This script supports local implementation of the SHAP analysis workflow.
  Raw individual-level surveillance data are not publicly available.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier


SEED = 12345
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
    df_2023 = pd.read_csv(data_2023_path)
    df_2024 = pd.read_csv(data_2024_path)

    missing_features = [f for f in features if f not in df_2023.columns or f not in df_2024.columns]
    if missing_features:
        raise ValueError(f"Features missing from input data: {missing_features}")

    train_2023, test_2023 = train_test_split(
        df_2023,
        test_size=0.25,
        stratify=df_2023[OUTCOME],
        random_state=seed,
    )

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

    # For SHAP interpretation, the 2024 validation cohort is downsampled
    # to match the evaluation setup used in the manuscript.
    test_2024 = downsample(df_2024, ratio_0_to_1=2, seed=seed)

    return train_2023, test_2024


def generate_rotating_rounds(train_df: pd.DataFrame, seed: int):
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
    models = []
    for i, round_df in enumerate(rounds, start=1):
        model = LGBMClassifier(**params)
        model.fit(round_df[features], round_df[OUTCOME])
        models.append(model)
        print(f"Trained sub-model {i}/{len(rounds)}")
    return models


def compute_shap_values(models, X):
    """Return one SHAP matrix per sub-model."""
    shap_list = []

    for model in models:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # For binary classification, SHAP may return either a list or an array,
        # depending on the SHAP / LightGBM version.
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_list.append(shap_values)

    return shap_list


def stability_summary(shap_list, features):
    """Summarize round-to-round stability of mean |SHAP| per feature."""
    n_rounds = len(shap_list)

    per_round = np.array([
        np.mean(np.abs(shap_values), axis=0)
        for shap_values in shap_list
    ])

    mean_abs = per_round.mean(axis=0)
    sd = per_round.std(axis=0, ddof=1)
    ci95 = 1.96 * sd / np.sqrt(n_rounds)
    cv = np.divide(sd, mean_abs, out=np.zeros_like(sd), where=mean_abs > 0)

    out = pd.DataFrame({
        "Feature": features,
        "Mean_abs_SHAP": mean_abs,
        "SD": sd,
        "Approx_95CI_halfwidth": ci95,
        "CV": cv,
    })

    out = out.sort_values("Mean_abs_SHAP", ascending=False).reset_index(drop=True)
    return out


def save_shap_summary_plot(avg_shap, X, features, output_dir):
    plt.figure(figsize=(10, 8))
    shap.summary_plot(avg_shap, X, feature_names=features, show=False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "lightgbm_shap_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(output_dir, "lightgbm_shap_summary.pdf"),
        bbox_inches="tight",
    )
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="SHAP analysis for LightGBM rotating ensemble")
    parser.add_argument("--data-2023", default="../data/data_2023.csv")
    parser.add_argument("--data-2024", default="../data/data_2024.csv")
    parser.add_argument("--feature-list", default="../results/final_feature_set.csv")
    parser.add_argument("--config", default="../config/best_params_LightGBM.json")
    parser.add_argument("--output-dir", default="../results/shap")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    features = pd.read_csv(args.feature_list)["feature"].tolist()

    train_2023, test_2024 = load_data(
        args.data_2023,
        args.data_2024,
        features,
        seed=args.seed,
    )

    params = load_lightgbm_params(args.config, seed=args.seed)

    rounds = generate_rotating_rounds(train_2023, seed=args.seed)
    models = train_ensemble(rounds, features, params)

    X_test = test_2024[features]

    shap_list = compute_shap_values(models, X_test)
    avg_shap = np.mean(shap_list, axis=0)

    save_shap_summary_plot(avg_shap, X_test, features, args.output_dir)

    stability_df = stability_summary(shap_list, features)
    stability_df.to_csv(
        os.path.join(args.output_dir, "lightgbm_shap_stability.csv"),
        index=False,
    )

    print(stability_df.head(20).round(3).to_string(index=False))
    print("SHAP analysis completed.")


if __name__ == "__main__":
    main()
