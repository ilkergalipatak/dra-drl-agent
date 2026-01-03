#!/usr/bin/env python3
"""
Run Full Benchmark - Compare Hybrid DRA vs Baseline

Runs feature selection optimization on all datasets with both
hybrid (LLM-enhanced) and baseline approaches.
"""

import sys
import argparse
from pathlib import Path
import json
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from src.hybrid_dra_fs import HybridDRAFeatureSelector, OptimizationResult
from src.data.dataset_loader import DatasetLoader
from src.evaluation.fitness_evaluator import CLASSIFIERS


def run_benchmark(
    datasets: list = None,
    classifiers: list = None,
    num_runs: int = 1,
    use_llm: bool = True,
    max_iterations: int = 100,
    population_size: int = 50,
    output_dir: str = "results",
    verbose: bool = True,
):
    """
    Run comprehensive benchmark.

    Args:
        datasets: List of datasets to test
        classifiers: List of classifiers to use
        num_runs: Number of runs per combination
        use_llm: Whether to use LLM enhancement
        max_iterations: DRA iterations per run
        population_size: DRA population size
        output_dir: Directory to save results
        verbose: Print progress
    """
    start_time = time.time()

    # Get available datasets and classifiers
    loader = DatasetLoader()
    all_datasets = loader.AVAILABLE_DATASETS
    all_classifiers = list(CLASSIFIERS.keys())

    datasets = datasets or all_datasets
    classifiers = classifiers or ["knn", "rf"]  # Default to fast classifiers

    print("=" * 80)
    print("LLM-HYBRID DRA FEATURE SELECTION BENCHMARK")
    print("=" * 80)
    print(f"\nDatasets: {len(datasets)}")
    print(f"Classifiers: {classifiers}")
    print(f"Runs per combination: {num_runs}")
    print(f"LLM Enhancement: {'Enabled' if use_llm else 'Disabled'}")
    print(f"Max iterations: {max_iterations}")
    print("=" * 80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize selector with config file
    config_path = Path(__file__).parent / "config.yaml"
    selector = HybridDRAFeatureSelector(
        config_path=str(config_path) if config_path.exists() else None
    )
    selector.config["dra"]["max_iterations"] = max_iterations
    selector.config["dra"]["population_size"] = population_size

    # Test LLM connection
    if use_llm:
        print("\n🔗 Testing LLM connection...")
        success, msg = selector.test_llm_connection()
        print(f"   {msg}")
        if not success:
            print("   ⚠️ Running without LLM enhancement")
            use_llm = False

    # Results storage
    all_results = []

    for ds_name in datasets:
        print(f"\n{'#'*70}")
        print(f"# Dataset: {ds_name}")
        print(f"{'#'*70}")

        try:
            # Load dataset info
            data = loader.load(ds_name)
            print(
                f"   Samples: {data['n_samples']}, Features: {data['n_features']}, Classes: {data['n_classes']}"
            )
        except Exception as e:
            print(f"   ❌ Error loading dataset: {e}")
            continue

        for clf_name in classifiers:
            print(f"\n   📊 Classifier: {CLASSIFIERS[clf_name]['name']}")

            hybrid_results = []
            baseline_results = []

            for run in range(num_runs):
                if verbose and num_runs > 1:
                    print(f"\n      Run {run + 1}/{num_runs}")

                # Update classifier in config
                selector.config["feature_selection"]["default_classifier"] = clf_name

                # Hybrid run (with LLM)
                if use_llm:
                    try:
                        result = selector.run_optimization(
                            dataset_name=ds_name, use_llm=True, verbose=verbose
                        )
                        hybrid_results.append(result)
                    except Exception as e:
                        print(f"      ⚠️ Hybrid run failed: {e}")

                # Baseline run (without LLM)
                try:
                    result = selector.run_baseline(
                        dataset_name=ds_name, verbose=verbose
                    )
                    baseline_results.append(result)
                except Exception as e:
                    print(f"      ⚠️ Baseline run failed: {e}")

            # Aggregate results
            if baseline_results:
                baseline_stats = {
                    "method": "baseline",
                    "dataset": ds_name,
                    "classifier": clf_name,
                    "mean_fitness": np.mean([r.best_fitness for r in baseline_results]),
                    "std_fitness": np.std([r.best_fitness for r in baseline_results]),
                    "mean_accuracy": np.mean(
                        [r.test_accuracy for r in baseline_results]
                    ),
                    "std_accuracy": np.std([r.test_accuracy for r in baseline_results]),
                    "mean_features": np.mean([r.n_selected for r in baseline_results]),
                    "mean_runtime": np.mean([r.run_time for r in baseline_results]),
                    "n_runs": len(baseline_results),
                }
                all_results.append(baseline_stats)

                print(
                    f"      Baseline: Acc={baseline_stats['mean_accuracy']:.4f} ± {baseline_stats['std_accuracy']:.4f}, "
                    f"Features={baseline_stats['mean_features']:.1f}"
                )

            if hybrid_results:
                hybrid_stats = {
                    "method": "hybrid_llm",
                    "dataset": ds_name,
                    "classifier": clf_name,
                    "mean_fitness": np.mean([r.best_fitness for r in hybrid_results]),
                    "std_fitness": np.std([r.best_fitness for r in hybrid_results]),
                    "mean_accuracy": np.mean([r.test_accuracy for r in hybrid_results]),
                    "std_accuracy": np.std([r.test_accuracy for r in hybrid_results]),
                    "mean_features": np.mean([r.n_selected for r in hybrid_results]),
                    "mean_runtime": np.mean([r.run_time for r in hybrid_results]),
                    "mean_interventions": np.mean(
                        [r.interventions for r in hybrid_results]
                    ),
                    "n_runs": len(hybrid_results),
                }
                all_results.append(hybrid_stats)

                print(
                    f"      Hybrid:   Acc={hybrid_stats['mean_accuracy']:.4f} ± {hybrid_stats['std_accuracy']:.4f}, "
                    f"Features={hybrid_stats['mean_features']:.1f}, LLM={hybrid_stats['mean_interventions']:.1f}"
                )

    # Save results
    if all_results:
        df_results = pd.DataFrame(all_results)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_path / f"benchmark_results_{timestamp}.csv"
        df_results.to_csv(csv_path, index=False)

        print(f"\n\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        # Print summary table
        print("\n" + df_results.to_string(index=False))

        print(f"\n✅ Results saved to: {csv_path}")

        # Overall comparison
        baseline_acc = df_results[df_results["method"] == "baseline"][
            "mean_accuracy"
        ].mean()
        hybrid_acc = df_results[df_results["method"] == "hybrid_llm"][
            "mean_accuracy"
        ].mean()

        if not np.isnan(hybrid_acc):
            print(f"\n📊 Overall Comparison:")
            print(f"   Baseline Mean Accuracy: {baseline_acc:.4f}")
            print(f"   Hybrid Mean Accuracy:   {hybrid_acc:.4f}")
            print(f"   Improvement:            {hybrid_acc - baseline_acc:+.4f}")

    total_time = time.time() - start_time
    print(f"\n⏱️ Total benchmark time: {total_time/60:.1f} minutes")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run LLM-Hybrid DRA Benchmark")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to benchmark (default: all)",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["knn", "rf"],
        choices=["knn", "svm", "rf", "lr"],
        help="Classifiers to use",
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="Number of runs per dataset/classifier"
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Max iterations per run"
    )
    parser.add_argument("--population", type=int, default=50, help="Population size")
    parser.add_argument(
        "--no-llm", action="store_true", help="Disable LLM (baseline only)"
    )
    parser.add_argument(
        "--output", type=str, default="results", help="Output directory"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output")

    args = parser.parse_args()

    run_benchmark(
        datasets=args.datasets,
        classifiers=args.classifiers,
        num_runs=args.runs,
        use_llm=not args.no_llm,
        max_iterations=args.iterations,
        population_size=args.population,
        output_dir=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
