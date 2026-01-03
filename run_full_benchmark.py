#!/usr/bin/env python3
"""
Full Benchmark Suite for LLM-Hybrid DRA Feature Selection

Compares three approaches execution:
1. Native: Classification using ALL features (no selection)
2. Pure DRA: Standard Binary DRA optimization
3. Hybrid DRL-LLM: DRA with DRL-guided LLM operator interventions

Generates comprehensive CSV reports and visualization plots.
"""

import sys
import argparse
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

from src.hybrid_dra_fs import HybridDRAFeatureSelector, OptimizationResult
from src.data.dataset_loader import DatasetLoader
from src.evaluation.fitness_evaluator import FitnessEvaluator, CLASSIFIERS

# Configure plotting style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["font.size"] = 12


def run_native_evaluation(
    dataset_name: str, classifier_name: str, max_samples: int = 2000
) -> Dict[str, Any]:
    """Evaluate classifier performance using ALL features (Native approach)."""
    loader = DatasetLoader(max_samples=max_samples)
    data = loader.load(dataset_name)

    fitness_func = FitnessEvaluator(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_test=data["X_test"],
        y_test=data["y_test"],
        classifier=classifier_name,
        alpha=0.99,  # Not used for accuracy, only for cost calculation
        cv_folds=5,
    )

    # All features selected (mask of 1s)
    all_features = np.ones(data["n_features"], dtype=int)

    start_time = time.time()
    results = fitness_func.evaluate_final(all_features)
    duration = time.time() - start_time

    return {
        "dataset": dataset_name,
        "classifier": classifier_name,
        "method": "native",
        "n_selected": results["n_selected"],
        "n_total": results["n_total"],
        "accuracy": results["test_accuracy"],
        "fitness": results["fitness"],
        "runtime": duration,
        "reduction_ratio": 0.0,
        "interventions": 0,
    }


def plot_results(df_results: pd.DataFrame, output_dir: Path, timestamp: str):
    """Generate comparison plots."""

    # 1. Accuracy Comparison (Bar Plot)
    plt.figure(figsize=(15, 8))
    sns.barplot(
        data=df_results, x="dataset", y="accuracy", hue="method", palette="viridis"
    )
    plt.title("Test Accuracy Comparison: Native vs Pure DRA vs Hybrid")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.ylim(0.5, 1.0)  # Zoom in on upper range
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / f"accuracy_comparison_{timestamp}.png")
    plt.close()

    # 2. Key Metrics by Method (Box Plots)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Accuracy Box Plot
    sns.boxplot(data=df_results, x="method", y="accuracy", ax=axes[0], palette="Set2")
    axes[0].set_title("Test Accuracy Distribution")

    # Feature Reduction Box Plot
    sns.boxplot(
        data=df_results, x="method", y="reduction_ratio", ax=axes[1], palette="Set2"
    )
    axes[1].set_title("Feature Reduction Ratio")

    # Runtime Box Plot
    sns.boxplot(data=df_results, x="method", y="runtime", ax=axes[2], palette="Set2")
    axes[2].set_yscale("log")
    axes[2].set_title("Runtime (Log Scale)")

    plt.tight_layout()
    plt.savefig(output_dir / f"metrics_distribution_{timestamp}.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Full LLM-Hybrid DRA Benchmark Suite")
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of runs per configuration"
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Max DRA iterations"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None, help="Specific datasets to run"
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["knn", "svm", "rf", "lr"],
        help="Classifiers to use",
    )
    parser.add_argument("--output", default="results_full", help="Output directory")
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    loader = DatasetLoader()
    datasets = args.datasets or loader.AVAILABLE_DATASETS
    classifiers = args.classifiers

    # Config for Hybrid Selector
    config_path = Path("config.yaml")
    selector = HybridDRAFeatureSelector(
        config_path=str(config_path) if config_path.exists() else None
    )
    selector.config["dra"]["max_iterations"] = args.iterations

    print("=" * 80)
    print("🚀 FULL LLM-HYBRID DRA BENCHMARK SUITE")
    print("=" * 80)
    print(f"Datasets ({len(datasets)}): {', '.join(datasets)}")
    print(f"Classifiers ({len(classifiers)}): {', '.join(classifiers)}")
    print(f"Runs: {args.runs}")
    print(f"Max Iterations: {args.iterations}")
    print("=" * 80)

    # Test LLM/DRL Connection
    print("\n[System Check]")
    llm_ok, msg = selector.test_llm_connection()
    print(f"LLM Status: {'✅' if llm_ok else '❌'} {msg}")

    agent = selector._load_prophet_agent(dim=10)
    print(
        f"DRL Agent:  {'✅' if agent else '⚠️'} {'Loaded' if agent else 'Not found/Disabled'}"
    )

    all_results = []

    # Main Benchmark Loop
    for ds_name in datasets:
        print(f"\n\n{'#'*60}")
        print(f"# Processing Dataset: {ds_name}")
        print(f"{'#'*60}")

        try:
            # Load dataset check
            data = loader.load(ds_name)
            print(f"Dims: {data['n_samples']} samples, {data['n_features']} features")
        except Exception as e:
            print(f"❌ Failed to load {ds_name}: {e}")
            continue

        for clf_name in classifiers:
            print(f"\n   📊 Classifier: {clf_name.upper()}")

            # --- 1. NATIVE (All Features) ---
            print("      Step 1/3: Native Evaluation...", end="\r")
            try:
                max_samples = selector.config.get("datasets", {}).get(
                    "max_samples", 2000
                )
                native_res = run_native_evaluation(
                    ds_name, clf_name, max_samples=max_samples
                )
                all_results.append(native_res)
                print(
                    f"      ✅ Native: Acc={native_res['accuracy']:.4f} ({native_res['runtime']:.2f}s)"
                )
            except Exception as e:
                print(f"      ❌ Native Failed: {e}")

            features_cache = []  # Store native Acc for comparison

            # Repetitions for Stochastic Methods
            for run_idx in range(args.runs):
                if args.runs > 1:
                    print(f"      Run {run_idx+1}/{args.runs}")

                # --- 2. PURE DRA (Baseline) ---
                print("      Step 2/3: Pure DRA...", end="\r")
                selector.config["feature_selection"]["classifier"] = clf_name
                try:
                    res_base = selector.run_optimization(
                        dataset_name=ds_name,
                        use_llm=False,
                        use_drl=False,
                        verbose=False,
                    )

                    all_results.append(
                        {
                            "dataset": ds_name,
                            "classifier": clf_name,
                            "method": "pure_dra",
                            "n_selected": res_base.n_selected,
                            "n_total": res_base.n_total,
                            "accuracy": res_base.test_accuracy,
                            "fitness": res_base.best_fitness,
                            "runtime": res_base.run_time,
                            "reduction_ratio": res_base.reduction_ratio,
                            "interventions": 0,
                        }
                    )
                    print(
                        f"      ✅ Pure DRA: Acc={res_base.test_accuracy:.4f} Feats={res_base.n_selected}"
                    )
                except Exception as e:
                    print(f"      ❌ Pure DRA Failed: {e}")

                # --- 3. HYBRID (LLM + DRL) ---
                print("      Step 3/3: Hybrid LLM+DRL...", end="\r")
                try:
                    res_hybrid = selector.run_optimization(
                        dataset_name=ds_name,
                        use_llm=True,
                        use_drl=True,  # Use DRL guidance if available
                        verbose=False,
                    )

                    all_results.append(
                        {
                            "dataset": ds_name,
                            "classifier": clf_name,
                            "method": "hybrid",
                            "n_selected": res_hybrid.n_selected,
                            "n_total": res_hybrid.n_total,
                            "accuracy": res_hybrid.test_accuracy,
                            "fitness": res_hybrid.best_fitness,
                            "runtime": res_hybrid.run_time,
                            "reduction_ratio": res_hybrid.reduction_ratio,
                            "interventions": res_hybrid.interventions,
                        }
                    )
                    print(
                        f"      ✅ Hybrid:   Acc={res_hybrid.test_accuracy:.4f} Feats={res_hybrid.n_selected} (Intv: {res_hybrid.interventions})"
                    )

                except Exception as e:
                    print(f"      ❌ Hybrid Failed: {e}")
                    import traceback

                    traceback.print_exc()

    # Save and Plot
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = output_dir / f"benchmark_full_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n\n✅ Results saved to {csv_path}")

        try:
            plot_results(df, output_dir, timestamp)
            print(f"✅ Plots generated in {output_dir}")
        except Exception as e:
            print(f"⚠️ Plot generation failed: {e}")

        # Summary Table
        summary = df.groupby(["dataset", "method"])["accuracy"].mean().unstack()
        print("\nSUMMARY (Mean Accuracy):")
        print(summary)

    else:
        print("\n❌ No results generated.")


if __name__ == "__main__":
    main()
