#!/usr/bin/env python3
"""
LLM-Hybrid DRA Feature Selection

Main entry point for running feature selection experiments.

Usage:
    python main.py --dataset breast_cancer --use-llm
    python main.py --benchmark --datasets breast_cancer sonar ionosphere
    python main.py --train-prophet --timesteps 50000
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.hybrid_dra_fs import HybridDRAFeatureSelector, run_quick_test
from src.data.dataset_loader import DatasetLoader


def main():
    parser = argparse.ArgumentParser(
        description="LLM-Hybrid DRA Feature Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on single dataset
  python main.py --dataset breast_cancer --use-llm
  
  # Run benchmark on all datasets
  python main.py --benchmark
  
  # Run benchmark on specific datasets
  python main.py --benchmark --datasets breast_cancer sonar ionosphere
  
  # Run baseline (no LLM)
  python main.py --dataset wine --baseline
  
  # Train DRL Prophet agent
  python main.py --train-prophet --timesteps 100000
  
  # Test LLM connection
  python main.py --test-llm
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--dataset", type=str, help="Single dataset to optimize")
    mode_group.add_argument(
        "--benchmark", action="store_true", help="Run benchmark on multiple datasets"
    )
    mode_group.add_argument(
        "--train-prophet", action="store_true", help="Train DRL Prophet agent"
    )
    mode_group.add_argument(
        "--test-llm", action="store_true", help="Test LLM connection"
    )
    mode_group.add_argument(
        "--list-datasets", action="store_true", help="List available datasets"
    )

    # Options
    parser.add_argument("--datasets", nargs="+", help="Datasets for benchmark")
    parser.add_argument(
        "--use-llm", action="store_true", default=True, help="Use LLM heuristics"
    )
    parser.add_argument(
        "--baseline", action="store_true", help="Run baseline without LLM"
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="Number of runs per dataset"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100000, help="DRL training timesteps"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Configuration file"
    )
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    # List datasets
    if args.list_datasets:
        print("\nAvailable datasets:")
        print("-" * 40)
        for name in DatasetLoader.list_datasets():
            info = DatasetLoader.get_dataset_info(name)
            print(
                f"  {name:15} - {info['features']} features, {info['samples']} samples"
            )
        return

    # Check config exists
    config_path = args.config if Path(args.config).exists() else None

    # Initialize selector
    selector = HybridDRAFeatureSelector(config_path=config_path)

    # Test LLM connection
    if args.test_llm:
        print("\nTesting LLM connection...")
        success, msg = selector.test_llm_connection()
        print(f"Result: {msg}")
        if success:
            print("✓ LLM connection successful!")
        else:
            print("✗ LLM connection failed!")
        return

    # Train Prophet
    if args.train_prophet:
        print("\nTraining DRL Prophet agent...")
        print(f"Timesteps: {args.timesteps}")

        from src.agents.prophet_agent import ProphetAgent
        from src.data.dataset_loader import load_dataset
        from src.evaluation.fitness_evaluator import create_fitness_function

        # Load a training dataset
        data = load_dataset("breast_cancer")
        fitness_func = create_fitness_function(
            X_train=data["X_train"], y_train=data["y_train"], classifier="knn"
        )

        # Create and train agent
        agent = ProphetAgent(
            fitness_func=fitness_func,
            dim=data["n_features"],
            llm_generator=selector.llm_generator,
        )

        stats = agent.train(total_timesteps=args.timesteps)

        # Save model
        model_path = Path("models") / "prophet_agent"
        model_path.parent.mkdir(exist_ok=True)
        agent.save(str(model_path))

        print(f"\nTraining complete!")
        print(f"Model saved to: {model_path}")
        print(f"Stats: {stats}")
        return

    # Single dataset optimization
    if args.dataset:
        print(f"\nOptimizing on dataset: {args.dataset}")

        if args.baseline:
            result = selector.run_baseline(
                dataset_name=args.dataset, verbose=not args.quiet
            )
        else:
            result = selector.run_optimization(
                dataset_name=args.dataset, use_llm=args.use_llm, verbose=not args.quiet
            )

        # Save results
        if args.output:
            output_data = {
                "dataset": args.dataset,
                "best_fitness": result.best_fitness,
                "selected_features": result.selected_features,
                "n_selected": result.n_selected,
                "test_accuracy": result.test_accuracy,
                "train_accuracy": result.train_accuracy,
                "reduction_ratio": result.reduction_ratio,
                "interventions": result.interventions,
                "runtime": result.run_time,
                "convergence_curve": result.convergence_curve,
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")

        return

    # Benchmark mode
    if args.benchmark:
        datasets = args.datasets or DatasetLoader.list_datasets()
        print(f"\nRunning benchmark on {len(datasets)} datasets")
        print(f"Datasets: {', '.join(datasets)}")
        print(f"Runs per dataset: {args.runs}")

        results = selector.run_benchmark(
            datasets=datasets,
            num_runs=args.runs,
            compare_baseline=True,
            verbose=not args.quiet,
        )

        # Print summary
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        print(
            f"\n{'Dataset':<20} {'Hybrid Acc':<12} {'Baseline Acc':<12} {'Improvement':<12}"
        )
        print("-" * 60)

        for dataset in datasets:
            hybrid = results["hybrid"].get(dataset, {})
            baseline = (
                results["baseline"].get(dataset, {}) if results["baseline"] else {}
            )

            h_acc = hybrid.get("mean_accuracy", 0)
            b_acc = baseline.get("mean_accuracy", 0)
            improvement = h_acc - b_acc

            print(
                f"{dataset:<20} {h_acc:>10.4f}  {b_acc:>10.4f}  {improvement:>+10.4f}"
            )

        print("-" * 60)
        summary = results.get("summary", {})
        print(
            f"{'OVERALL':<20} {summary.get('hybrid_mean_accuracy', 0):>10.4f}  "
            f"{summary.get('baseline_mean_accuracy', 0):>10.4f}  "
            f"{summary.get('improvement', 0):>+10.4f}"
        )
        print(f"\nTotal LLM interventions: {summary.get('total_llm_calls', 0)}")

        # Save results
        if args.output:
            # Convert results to serializable format
            save_results = {
                "timestamp": datetime.now().isoformat(),
                "datasets": datasets,
                "num_runs": args.runs,
                "summary": {
                    k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in summary.items()
                },
                "hybrid": {
                    k: {
                        kk: float(vv) if isinstance(vv, (int, float)) else None
                        for kk, vv in v.items()
                        if kk != "runs"
                    }
                    for k, v in results["hybrid"].items()
                },
                "baseline": (
                    {
                        k: {
                            kk: float(vv) if isinstance(vv, (int, float)) else None
                            for kk, vv in v.items()
                            if kk != "runs"
                        }
                        for k, v in results["baseline"].items()
                    }
                    if results["baseline"]
                    else None
                ),
            }
            with open(args.output, "w") as f:
                json.dump(save_results, f, indent=2)
            print(f"\nResults saved to: {args.output}")

        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
