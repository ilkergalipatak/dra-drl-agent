#!/usr/bin/env python3
"""
Train DRL Prophet Agent for Hyper-Heuristic Control

Trains a PPO agent to learn when and how to request LLM-generated heuristics
during optimization.
"""

import sys
import argparse
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from src.agents.prophet_agent import ProphetAgent, RuleBasedProphet
from src.agents.prophet_env import ProphetEnv
from src.data.dataset_loader import DatasetLoader
from src.evaluation.fitness_evaluator import FitnessEvaluator
from src.llm.llm_generator import create_generator_from_config
import yaml


def train_prophet(
    dataset_name: str = "heart_failure",
    total_timesteps: int = 50000,
    config_path: str = "config.yaml",
    save_path: str = "models/prophet_agent",
    use_llm: bool = True,
    n_envs: int = 4,
    verbose: int = 1,
):
    """
    Train a Prophet agent.

    Args:
        dataset_name: Dataset to train on
        total_timesteps: Total training timesteps
        config_path: Path to config file
        save_path: Path to save trained model
        use_llm: Whether to use LLM during training
        verbose: Verbosity level
    """
    print("=" * 70)
    print("TRAINING DRL PROPHET AGENT")
    print("=" * 70)

    # Load config
    config = {}
    if Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    # Load dataset
    print(f"\n📊 Loading dataset: {dataset_name}")
    max_samples = config.get("datasets", {}).get("max_samples", 2000)
    loader = DatasetLoader(max_samples=max_samples)
    data = loader.load(dataset_name)
    print(f"   Samples: {data['n_samples']}, Features: {data['n_features']}")

    # Create fitness function
    fitness_func = FitnessEvaluator(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_test=data["X_test"],
        y_test=data["y_test"],
        classifier="knn",
    )

    # Create LLM generator (optional)
    llm_generator = None
    if use_llm:
        print("\n🔗 Initializing LLM generator...")
        try:
            llm_generator = create_generator_from_config(config)
            success, msg = llm_generator.test_connection()
            if success:
                print(f"   ✅ LLM connected: {msg}")
            else:
                print(f"   ⚠️ LLM unavailable: {msg}")
                llm_generator = None
        except Exception as e:
            print(f"   ⚠️ LLM error: {e}")
            llm_generator = None

    # Create Prophet agent
    print("\n🤖 Creating Prophet agent...")
    agent = ProphetAgent(
        fitness_func=fitness_func,
        dim=data["n_features"],
        llm_generator=llm_generator,
        population_size=30,
        max_iterations=50,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_envs=n_envs,
        verbose=verbose,
    )

    # Train
    print(f"\n🏋️ Starting training ({total_timesteps} timesteps)...")
    print("-" * 50)

    try:
        stats = agent.train(total_timesteps=total_timesteps)
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user!")
        print("🛑 Stopping training and saving model...")
        stats = {}  # Placeholder

    print("\n📈 Training Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Save model
    print(f"\n💾 Saving model to: {save_path}")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(save_path)

    # Save training info
    training_info = {
        "dataset": dataset_name,
        "total_timesteps": total_timesteps,
        "statistics": stats,
        "timestamp": datetime.now().isoformat(),
    }

    with open(f"{save_path}_training_info.json", "w") as f:
        json.dump(training_info, f, indent=2, default=str)

    print("\n✅ Training complete!")

    return agent, stats


def main():
    parser = argparse.ArgumentParser(description="Train DRL Prophet Agent")
    parser.add_argument(
        "--dataset", type=str, default="heart_failure", help="Dataset to train on"
    )
    parser.add_argument(
        "--timesteps", type=int, default=50000, help="Total training timesteps"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/prophet_agent",
        help="Path to save model",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Config file path"
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="Disable LLM during training"
    )
    parser.add_argument(
        "--n-envs", type=int, default=4, help="Number of parallel environments"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output")

    args = parser.parse_args()

    train_prophet(
        dataset_name=args.dataset,
        total_timesteps=args.timesteps,
        config_path=args.config,
        save_path=args.save_path,
        use_llm=not args.no_llm,
        n_envs=args.n_envs,
        verbose=0 if args.quiet else 1,
    )


if __name__ == "__main__":
    main()
