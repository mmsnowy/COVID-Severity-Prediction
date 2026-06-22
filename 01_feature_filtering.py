"""
01_feature_filtering.py

Two-stage genomic feature filtering for COVID-19 severity prediction.

The final feature set is constructed from three input groups:
  (1) individual mutation-site features,
  (2) clade-level indicator features,
  (3) predefined baseline features, including host clinical variables,
      regional indicators, engineered interaction terms, per-protein mutation
      counts, and physicochemical mutation-load metrics.

The two-stage filtering procedure is applied only to groups (1) and (2):
  Stage 1: unsupervised variance filtering for temporal robustness
           (retain features with variance >= 0.01 in all temporal cohorts).
  Stage 2: supervised binary-feature association filtering using the
           phi coefficient / chi-square association test
           (retain features with P < 0.05).

Predefined baseline features are retained without filtering.

Note:
  Raw individual-level surveillance data are not publicly available.
  Please prepare an input table following example/feature_template.csv.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


VARIANCE_THRESHOLD = 0.01
P_THRESHOLD = 0.05
OUTCOME = "Clinical_Type"      # 0 = non-severe, 1 = severe
COHORT = "Year"                # e.g., 2023 / 2024


BASELINE_FEATURES = [
    "Gender",
    "Age",
    "Ct_ORFlab",
    "Ct_N",
    "Vaccinated",
    "Prior_Infection",
    "Age_CtN_Interaction",
    "Region_Other",
    "Region_SuBei",
    "Region_SuNan",
    "Region_SuZhong",
]


def is_mutation_site_feature(col: str) -> bool:
    """Identify individual mutation-site features such as S.452 or ORF1a.690."""
    return ("." in col) and any(ch.isdigit() for ch in col) and (not col.startswith("clade_"))


def identify_feature_groups(df: pd.DataFrame):
    """Identify baseline, mutation-site, and clade-level features."""
    auto_baseline_features = [
        c for c in df.columns
        if c.endswith("_Mut_Count") or c in ["Charge_Change_Mut_Load", "Size_Change_Mut_Load"]
    ]

    baseline_features = list(dict.fromkeys(BASELINE_FEATURES + auto_baseline_features))
    baseline_features = [f for f in baseline_features if f in df.columns]

    mutation_features = [
        c for c in df.columns
        if is_mutation_site_feature(c)
    ]

    clade_features = [
        c for c in df.columns
        if c.startswith("clade_")
    ]

    # Remove outcome/cohort from all feature groups if accidentally included
    mutation_features = [c for c in mutation_features if c not in [OUTCOME, COHORT]]
    clade_features = [c for c in clade_features if c not in [OUTCOME, COHORT]]
    baseline_features = [c for c in baseline_features if c not in [OUTCOME, COHORT]]

    return baseline_features, mutation_features, clade_features


def binary_association_pvalue(feature: pd.Series, outcome: pd.Series) -> float:
    """
    P value for association between two binary variables.
    The phi coefficient is equivalent to Pearson correlation for two binary variables;
    here the P value is obtained from the corresponding chi-square test.
    """
    table = pd.crosstab(feature, outcome)

    if table.shape != (2, 2):
        return 1.0

    _, p_value, _, _ = chi2_contingency(table, correction=False)
    return float(p_value)


def two_stage_filter(df: pd.DataFrame, candidate_features):
    """Apply two-stage filtering to mutation-site and clade-level features."""
    # Stage 1: unsupervised variance filtering within each temporal cohort
    stage1 = []
    for feat in candidate_features:
        if feat not in df.columns:
            continue

        variances = df.groupby(COHORT)[feat].var()

        # Retain only if the feature has sufficient variance in all temporal cohorts
        if len(variances) > 0 and (variances >= VARIANCE_THRESHOLD).all():
            stage1.append(feat)

    print(f"Stage 1 retained {len(stage1)} / {len(candidate_features)} genomic features")

    # Stage 2: supervised binary association filtering
    stage2 = []
    association_records = []

    for feat in stage1:
        p_value = binary_association_pvalue(df[feat], df[OUTCOME])
        association_records.append({
            "feature": feat,
            "p_value": p_value,
            "selected": p_value < P_THRESHOLD,
        })

        if p_value < P_THRESHOLD:
            stage2.append(feat)

    print(f"Stage 2 retained {len(stage2)} / {len(stage1)} genomic features")

    association_df = pd.DataFrame(association_records)
    return stage1, stage2, association_df


def main():
    parser = argparse.ArgumentParser(description="Two-stage genomic feature filtering")
    parser.add_argument(
        "--input",
        default="../data/preprocessed_data.csv",
        help="Input preprocessed feature table containing outcome, cohort, baseline, mutation, and clade features.",
    )
    parser.add_argument(
        "--output-dir",
        default="../results",
        help="Directory for output files.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input)

    required_cols = [OUTCOME, COHORT]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    baseline_features, mutation_features, clade_features = identify_feature_groups(df)
    candidate_features = mutation_features + clade_features

    print(f"Baseline features retained without filtering: {len(baseline_features)}")
    print(f"Initial mutation-site features: {len(mutation_features)}")
    print(f"Initial clade-level features: {len(clade_features)}")
    print(f"Initial genomic candidate features: {len(candidate_features)}")

    stage1_features, selected_genomic_features, association_df = two_stage_filter(
        df, candidate_features
    )

    final_features = baseline_features + selected_genomic_features

    pd.Series(stage1_features, name="feature").to_csv(
        os.path.join(args.output_dir, "stage1_variance_retained_features.csv"),
        index=False,
    )

    association_df.to_csv(
        os.path.join(args.output_dir, "stage2_binary_association_results.csv"),
        index=False,
    )

    pd.Series(final_features, name="feature").to_csv(
        os.path.join(args.output_dir, "final_feature_set.csv"),
        index=False,
    )

    summary = {
        "baseline_features_retained_without_filtering": len(baseline_features),
        "initial_mutation_features": len(mutation_features),
        "initial_clade_features": len(clade_features),
        "stage1_retained_genomic_features": len(stage1_features),
        "stage2_retained_genomic_features": len(selected_genomic_features),
        "final_feature_count": len(final_features),
    }

    with open(os.path.join(args.output_dir, "feature_filtering_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Feature filtering completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
