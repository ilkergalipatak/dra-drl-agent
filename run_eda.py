#!/usr/bin/env python3
"""
Run EDA on all datasets and generate a preprocessing report.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from src.data.eda_preprocessing import DatasetEDA, DataPreprocessor

# Dataset configurations
DATASETS = {
    "student_performance": {
        "file": "ai_impact_student_performance_dataset.csv",
        "target": "performance_category",
        "drop_cols": ["student_id"],
    },
    "breast_cancer_custom": {
        "file": "breast_cancer_dataset.csv",
        "target": "diagnosis",
        "drop_cols": ["id", "Unnamed: 32"],
    },
    "credit_card_fraud": {
        "file": "credit_card_fraud_10k.csv",
        "target": "is_fraud",
        "drop_cols": [],
    },
    "customer_churn": {
        "file": "ecommerce_customer_churn_dataset.csv",
        "target": "Churned",
        "drop_cols": ["Signup_Quarter"],
    },
    "healthcare": {
        "file": "healthcare_dataset.csv",
        "target": "Test Results",
        "drop_cols": [
            "Name",
            "Date of Admission",
            "Discharge Date",
            "Doctor",
            "Hospital",
        ],
    },
    "heart_failure": {
        "file": "heart_failure_clinical_records_dataset.csv",
        "target": "DEATH_EVENT",
        "drop_cols": [],
    },
    "marketing_campaign": {
        "file": "marketing_campaign.csv",
        "target": "Response",
        "drop_cols": ["ID", "Dt_Customer", "Z_CostContact", "Z_Revenue"],
        "separator": "\t",
    },
    "mobile_price": {
        "file": "mobile_price_classification.csv",
        "target": "price_range",
        "drop_cols": [],
    },
    "credit_risk": {
        "file": "synthetic_credit_risk.csv",
        "target": "target",
        "drop_cols": [],
    },
}


def main():
    datasets_dir = Path("datasets")
    eda = DatasetEDA(verbose=True)

    print("=" * 80)
    print("EXPLORATORY DATA ANALYSIS - ALL DATASETS")
    print("=" * 80)

    results = []

    for name, config in DATASETS.items():
        file_path = datasets_dir / config["file"]

        if not file_path.exists():
            print(f"\n❌ File not found: {file_path}")
            continue

        try:
            sep = config.get("separator", ",")
            df = pd.read_csv(file_path, sep=sep)

            # Drop specified columns
            for col in config.get("drop_cols", []):
                if col in df.columns:
                    df = df.drop(columns=[col])

            target = config["target"]

            if target not in df.columns:
                print(f"\n❌ Target '{target}' not found in {name}")
                continue

            profile = eda.analyze(df, target, name)

            results.append(
                {
                    "dataset": name,
                    "samples": profile.n_samples,
                    "features": profile.n_features,
                    "classes": profile.n_classes,
                    "missing_pct": profile.missing_percentage,
                    "n_outliers": len(profile.outliers),
                    "n_categorical": sum(
                        1 for t in profile.feature_types.values() if t == "categorical"
                    ),
                }
            )

        except Exception as e:
            print(f"\n❌ Error analyzing {name}: {e}")
            import traceback

            traceback.print_exc()

    # Summary Table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    if results:
        df_summary = pd.DataFrame(results)
        print("\n")
        print(df_summary.to_string(index=False))

        # Save summary
        df_summary.to_csv("datasets/eda_summary.csv", index=False)
        print("\n✅ Summary saved to datasets/eda_summary.csv")

    print("\n" + "=" * 80)
    print("EDA COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
