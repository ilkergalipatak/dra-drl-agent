"""
Hybrid DRA Feature Selector

Main integration class combining:
- Binary DRA optimizer
- DRL Prophet agent
- LLM heuristic generator
- Stagnation detection

This is the complete LLM-Hybrid DRA system for feature selection.
"""

import numpy as np
import yaml
import time
from typing import Dict, Any, Optional, Callable, List, Tuple
from pathlib import Path
from dataclasses import dataclass


from .core.dra_optimizer import BinaryDRA
from .evaluation.fitness_evaluator import FitnessEvaluator, create_fitness_function
from .utils.stagnation_detector import StagnationDetector
from .llm.llm_generator import DivineSpeech, create_generator_from_config
from .llm.prompt_templates import TEMPLATES_BY_TYPE
from .data.dataset_loader import DatasetLoader, load_dataset

# Try to import ProphetAgent
try:
    from .agents.prophet_agent import ProphetAgent

    HAS_DRL = True
except ImportError:
    HAS_DRL = False


@dataclass
class OptimizationResult:
    """Result of a feature selection optimization run."""

    best_solution: np.ndarray
    best_fitness: float
    selected_features: List[int]
    n_selected: int
    n_total: int
    reduction_ratio: float
    train_accuracy: float
    test_accuracy: float
    convergence_curve: List[float]
    run_time: float
    interventions: int
    llm_generated_heuristics: int
    final_evaluation: Dict[str, Any]
    method: str = "hybrid"  # hybrid, baseline, native


class HybridDRAFeatureSelector:
    """
    LLM-Hybrid DRA Feature Selector.

    Combines the Divine Religions Algorithm with LLM-based hyper-heuristic
    adaptation for automatic feature selection optimization.
    """

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None
    ):
        """
        Initialize the hybrid feature selector.

        Args:
            config: Configuration dictionary
            config_path: Path to YAML configuration file
        """
        # Load configuration
        if config_path:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        elif config:
            self.config = config
        else:
            self.config = self._default_config()

        # Initialize components (lazy loading)
        self._llm_generator: Optional[DivineSpeech] = None
        self._dataset_loader: Optional[DatasetLoader] = None
        self._prophet_agent = None  # Prophet Agent instance

        # Statistics
        self._total_runs = 0
        self._total_interventions = 0
        self._total_llm_calls = 0

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "llm": {
                "base_url": "http://localhost:11434",
                "model": "qwen3-coder:30b",
                "api_key": "",
                "timeout": 120,
                "max_retries": 3,
            },
            "dra": {
                "population_size": 50,
                "max_iterations": 100,
                "num_groups": 5,
                "belief_profile_rate": 0.5,
                "miracle_rate": 0.5,
                "proselytism_rate": 0.9,
                "reward_penalty_rate": 0.2,
            },
            "feature_selection": {
                "alpha": 0.99,
                "classifier": "knn",
                "knn_neighbors": 5,
                "cv_folds": 5,
                "test_size": 0.3,
            },
            "stagnation": {
                "window_size": 10,
                "threshold": 0.001,
                "diversity_threshold": 0.1,
                "intervention_threshold": 5,
            },
            "drl": {"model_path": "models/prophet_agent.zip", "enabled": True},
            "experiment": {"random_seed": 42, "num_runs": 30},
        }

    @property
    def llm_generator(self) -> DivineSpeech:
        """Lazy-load LLM generator."""
        if self._llm_generator is None:
            self._llm_generator = create_generator_from_config(self.config)
        return self._llm_generator

    @property
    def dataset_loader(self) -> DatasetLoader:
        """Lazy-load dataset loader."""
        if self._dataset_loader is None:
            fs_config = self.config.get("feature_selection", {})
            ds_config = self.config.get("datasets", {})
            self._dataset_loader = DatasetLoader(
                test_size=fs_config.get("test_size", 0.3),
                normalize=True,
                random_seed=self.config.get("experiment", {}).get("random_seed", 42),
                max_samples=ds_config.get("max_samples", 2000),
            )
        return self._dataset_loader

    def _load_prophet_agent(self, dim: int):
        """Load or initialize DRL Prophet agent."""
        if not HAS_DRL:
            print("Warning: stable-baselines3 not installed. DRL disabled.")
            return None

        drl_config = self.config.get("drl", {})
        model_path = drl_config.get("model_path", "models/prophet_agent.zip")

        # Determine dummy fitness function (not used for inference)
        def dummy_fitness(x):
            return 0.0

        if self._prophet_agent is None:
            try:
                self._prophet_agent = ProphetAgent(
                    fitness_func=dummy_fitness,
                    dim=dim,
                    llm_generator=None,  # Not needed for inference prediction
                    max_iterations=self.config["dra"]["max_iterations"],
                )

                if Path(model_path).exists():
                    self._prophet_agent.load(model_path)
                    print(f"Loaded Prophet Agent from {model_path}")
                else:
                    print(
                        f"Warning: Prophet model not found at {model_path}. Using untrained agent."
                    )
            except Exception as e:
                print(f"Failed to load Prophet Agent: {e}")
                self._prophet_agent = None

        return self._prophet_agent

    def run_optimization(
        self,
        dataset_name: str,
        use_llm: bool = True,
        use_drl: bool = False,
        callback: Optional[Callable] = None,
        verbose: bool = True,
    ) -> OptimizationResult:
        """
        Run feature selection optimization on a dataset.

        Args:
            dataset_name: Name of the dataset to optimize
            use_llm: Whether to use LLM-generated heuristics
            use_drl: Whether to use DRL-guided intervention
            callback: Optional callback for progress
            verbose: Print progress information

        Returns:
            OptimizationResult with complete results
        """
        start_time = time.time()

        # Load dataset
        data = self.dataset_loader.load(dataset_name)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name}")
            print(f"Features: {data['n_features']}, Samples: {data['n_samples']}")
            print(f"Classes: {data['n_classes']}")
            print(f"{'='*60}\n")

        # Initialize DRL agent if requested
        agent = None
        if use_drl and use_llm:
            agent = self._load_prophet_agent(data["n_features"])
            if agent is None and verbose:
                print(
                    "Proceeding with Rule-Based Stagnation Detection (DRL unavailable)."
                )

        # Create fitness function
        fs_config = self.config.get("feature_selection", {})
        fitness_func = FitnessEvaluator(
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_test=data["X_test"],
            y_test=data["y_test"],
            classifier=fs_config.get("classifier", "knn"),
            alpha=fs_config.get("alpha", 0.99),
            cv_folds=fs_config.get("cv_folds", 5),
            knn_neighbors=fs_config.get("knn_neighbors", 5),
        )

        # Create DRA optimizer
        dra_config = self.config.get("dra", {})
        optimizer = BinaryDRA(
            fitness_func=fitness_func,
            dim=data["n_features"],
            population_size=dra_config.get("population_size", 50),
            max_iterations=dra_config.get("max_iterations", 100),
            num_groups=dra_config.get("num_groups", 5),
            belief_profile_rate=dra_config.get("belief_profile_rate", 0.5),
            reward_penalty_rate=dra_config.get("reward_penalty_rate", 0.2),
            random_seed=self.config.get("experiment", {}).get("random_seed", 42),
        )

        # Create stagnation detector
        stag_config = self.config.get("stagnation", {})
        stagnation_detector = StagnationDetector(
            window_size=stag_config.get("window_size", 10),
            fitness_threshold=stag_config.get("threshold", 0.001),
            diversity_threshold=stag_config.get("diversity_threshold", 0.1),
        )

        # Initialize optimizer
        optimizer._initialize_population()
        optimizer._initialize_groups()

        # Optimization loop
        interventions = 0
        llm_heuristics = 0
        convergence_curve = []
        intervention_threshold = stag_config.get("intervention_threshold", 5)
        initial_fitness = optimizer.best_solution.cost

        max_iter = dra_config.get("max_iterations", 100)

        for iteration in range(max_iter):
            optimizer.current_iteration = iteration

            # Run one DRA iteration
            best_cost = optimizer._iteration_step()
            convergence_curve.append(best_cost)

            # Check for stagnation/metrics
            population = np.array([b.position for b in optimizer.population])
            stag_state = stagnation_detector.update(
                iteration, best_cost, population, optimizer.best_solution.position
            )

            # Decision Logic: DRL or Rule-Based
            should_intervene = False
            operator_type = None

            if use_llm:
                if agent is not None:
                    # DRL Decision
                    obs = self._get_observation(
                        optimizer.get_state(),
                        stagnation_detector.get_metrics(),
                        initial_fitness,
                    )
                    action, _ = agent.predict(obs, deterministic=True)

                    # Map action to operator: 0=None, 1=Mutation, 2=Crossover, ...
                    if action != 0:
                        should_intervene = True
                        # Map int action to string operator type
                        # 1: mutation, 2: crossover, 3: local_search, 4: diversification
                        action_map = {
                            1: "mutation",
                            2: "crossover",
                            3: "local_search",
                            4: "diversification",
                        }
                        operator_type = action_map.get(action, "diversification")
                        if verbose:
                            print(
                                f"  [Iter {iteration}] Prophet Agent chose action {action} ({operator_type})"
                            )
                else:
                    # Rule-Based Decision
                    if (
                        stag_state.is_stagnating
                        and stagnation_detector.should_intervene(intervention_threshold)
                    ):
                        should_intervene = True
                        operator_type = stag_state.recommended_action
                        if verbose:
                            print(
                                f"  [Iter {iteration}] Stagnation detected! Recommending {operator_type}"
                            )

            # Apply Intervention
            if should_intervene and operator_type:
                state = optimizer.get_state()
                state.update(stagnation_detector.get_metrics())

                templates = TEMPLATES_BY_TYPE.get(operator_type, [])
                if templates:
                    import random

                    template = random.choice(templates)

                    if verbose:
                        print(f"    Requesting {operator_type} from LLM...")

                    heuristic = self.llm_generator.generate_heuristic(
                        template, state, data["n_features"]
                    )

                    if heuristic.success and heuristic.operator:
                        improvements = optimizer.apply_external_operator(
                            heuristic.operator
                        )
                        interventions += 1
                        llm_heuristics += 1
                        self._total_llm_calls += 1

                        if verbose:
                            print(
                                f"           LLM operator applied! Improvements: {improvements}"
                            )

            # Callback
            if callback:
                callback(
                    {
                        "iteration": iteration,
                        "best_cost": best_cost,
                        "is_stagnating": stag_state.is_stagnating,
                        "interventions": interventions,
                    }
                )

            # Progress output
            if verbose and (iteration + 1) % 20 == 0:
                n_selected = np.sum(optimizer.best_solution.position)
                print(
                    f"  Iteration {iteration + 1}/{max_iter}: "
                    f"Fitness={best_cost:.6f}, "
                    f"Features={n_selected}/{data['n_features']}"
                )

        # Final evaluation
        final_eval = fitness_func.evaluate_final(optimizer.best_solution.position)

        run_time = time.time() - start_time
        self._total_runs += 1
        self._total_interventions += interventions

        if verbose:
            print(f"\n{'='*60}")
            print(f"Optimization Complete!")
            print(f"Best Fitness: {optimizer.best_solution.cost:.6f}")
            print(
                f"Selected Features: {final_eval['n_selected']}/{final_eval['n_total']}"
            )
            print(f"Train Accuracy: {final_eval['train_accuracy']:.4f}")
            print(f"Test Accuracy: {final_eval['test_accuracy']:.4f}")
            print(f"LLM Interventions: {interventions}")
            print(f"Runtime: {run_time:.2f}s")
            print(f"{'='*60}\n")

        method_name = (
            "hybrid_drl"
            if (use_drl and agent)
            else ("hybrid_rule" if use_llm else "baseline")
        )

        return OptimizationResult(
            best_solution=optimizer.best_solution.position,
            best_fitness=optimizer.best_solution.cost,
            selected_features=final_eval["selected_features"],
            n_selected=final_eval["n_selected"],
            n_total=final_eval["n_total"],
            reduction_ratio=final_eval["reduction_ratio"],
            train_accuracy=final_eval["train_accuracy"],
            test_accuracy=final_eval["test_accuracy"],
            convergence_curve=convergence_curve,
            run_time=run_time,
            interventions=interventions,
            llm_generated_heuristics=llm_heuristics,
            final_evaluation=final_eval,
            method=method_name,
        )

    def _get_observation(
        self,
        state: Dict[str, Any],
        stag_metrics: Dict[str, Any],
        initial_fitness: float,
    ) -> np.ndarray:
        """Construct normalized observation vector for DRL Agent."""
        # Normalize all values to [0, 1]
        # Must match ProphetEnv observation space
        obs = np.array(
            [
                state.get("progress", 0),
                min(1.0, state.get("best_cost", 1) / (initial_fitness + 1e-10)),
                min(1.0, state.get("mean_cost", 1) / (initial_fitness + 1e-10)),
                min(1.0, state.get("std_cost", 0)),
                state.get("diversity", 0.5),
                state.get("feature_ratio", 0.5),
                min(1.0, abs(state.get("recent_improvement", 0)) * 100),
                float(state.get("is_stagnating", False)),
                min(1.0, stag_metrics.get("stagnation_count", 0) / 50),
                stag_metrics.get("current_diversity", 0.5),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, 0.0, 1.0)

    def run_baseline(
        self, dataset_name: str, verbose: bool = True
    ) -> OptimizationResult:
        """
        Run baseline DRA without LLM enhancement.

        Args:
            dataset_name: Name of dataset
            verbose: Print progress

        Returns:
            OptimizationResult
        """
        return self.run_optimization(
            dataset_name=dataset_name, use_llm=False, use_drl=False, verbose=verbose
        )

    def run_benchmark(
        self,
        datasets: Optional[List[str]] = None,
        num_runs: int = 1,
        compare_baseline: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run benchmark on multiple datasets.

        Args:
            datasets: List of dataset names (None = all)
            num_runs: Number of runs per dataset
            compare_baseline: Also run baseline for comparison
            verbose: Print progress

        Returns:
            Benchmark results dictionary
        """
        if datasets is None:
            datasets = DatasetLoader.list_datasets()

        results = {
            "hybrid": {},
            "baseline": {} if compare_baseline else None,
            "summary": {},
        }

        for dataset in datasets:
            print(f"\n{'#'*70}")
            print(f"# Processing: {dataset}")
            print(f"{'#'*70}")

            hybrid_results = []
            baseline_results = []

            for run in range(num_runs):
                if verbose:
                    print(f"\n--- Run {run + 1}/{num_runs} ---")

                # Hybrid run - Use DRL if available
                # Note: We rely on Default Config or passed config for DRL toggle.
                # Here we default to requesting DRL if it's in the intent
                hybrid = self.run_optimization(
                    dataset_name=dataset, use_llm=True, use_drl=True, verbose=verbose
                )
                hybrid_results.append(hybrid)

                # Baseline run
                if compare_baseline:
                    baseline = self.run_baseline(dataset_name=dataset, verbose=verbose)
                    baseline_results.append(baseline)

            # Aggregate results
            results["hybrid"][dataset] = {
                "mean_fitness": np.mean([r.best_fitness for r in hybrid_results]),
                "std_fitness": np.std([r.best_fitness for r in hybrid_results]),
                "mean_accuracy": np.mean([r.test_accuracy for r in hybrid_results]),
                "std_accuracy": np.std([r.test_accuracy for r in hybrid_results]),
                "mean_features": np.mean([r.n_selected for r in hybrid_results]),
                "mean_runtime": np.mean([r.run_time for r in hybrid_results]),
                "mean_interventions": np.mean(
                    [r.interventions for r in hybrid_results]
                ),
                "runs": hybrid_results,
            }

            if compare_baseline:
                results["baseline"][dataset] = {
                    "mean_fitness": np.mean([r.best_fitness for r in baseline_results]),
                    "std_fitness": np.std([r.best_fitness for r in baseline_results]),
                    "mean_accuracy": np.mean(
                        [r.test_accuracy for r in baseline_results]
                    ),
                    "std_accuracy": np.std([r.test_accuracy for r in baseline_results]),
                    "mean_features": np.mean([r.n_selected for r in baseline_results]),
                    "mean_runtime": np.mean([r.run_time for r in baseline_results]),
                    "runs": baseline_results,
                }

        # Summary
        if results["baseline"]:
            hybrid_acc = np.mean(
                [r["mean_accuracy"] for r in results["hybrid"].values()]
            )
            baseline_acc = np.mean(
                [r["mean_accuracy"] for r in results["baseline"].values()]
            )

            results["summary"] = {
                "hybrid_mean_accuracy": hybrid_acc,
                "baseline_mean_accuracy": baseline_acc,
                "improvement": hybrid_acc - baseline_acc,
                "total_interventions": self._total_interventions,
                "total_llm_calls": self._total_llm_calls,
            }

        return results

    def test_llm_connection(self) -> Tuple[bool, str]:
        """Test LLM API connection."""
        return self.llm_generator.test_connection()

    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        return {
            "total_runs": self._total_runs,
            "total_interventions": self._total_interventions,
            "total_llm_calls": self._total_llm_calls,
            "llm_stats": self.llm_generator.get_stats() if self._llm_generator else {},
        }


def run_quick_test(dataset: str = "breast_cancer", config_path: str = None):
    """Quick test function."""
    selector = HybridDRAFeatureSelector(config_path=config_path)

    print("Testing LLM connection...")
    success, msg = selector.test_llm_connection()
    print(f"LLM Connection: {msg}")

    # Test DRL loading
    print("Testing DRL Agent loading...")
    agent = selector._load_prophet_agent(dim=10)
    print(f"DRL Agent loaded: {agent is not None}")

    if not success:
        print("Warning: LLM not available, running baseline only")
        return selector.run_baseline(dataset)

    return selector.run_optimization(dataset, use_drl=True)
