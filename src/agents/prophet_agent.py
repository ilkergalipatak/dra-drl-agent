"""
Prophet Agent - DRL Agent for Hyper-Heuristic Control

The Prophet (Peygamber) uses PPO to learn when and how to request
new heuristics from the LLM during optimization.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import json

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        EvalCallback,
        CheckpointCallback,
    )
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        VecNormalize,
        SubprocVecEnv,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    PPO = None

from .prophet_env import ProphetEnv, make_prophet_env
from ..llm.llm_generator import DivineSpeech


class ProphetCallback(BaseCallback):
    """Callback for tracking Prophet training progress."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.interventions = []
        self.best_fitness_values = []

    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get("dones") is not None:
            for idx, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][idx]
                    self.episode_rewards.append(info.get("total_reward", 0))
                    self.episode_lengths.append(info.get("step", 0))
                    self.interventions.append(info.get("interventions", 0))
                    self.best_fitness_values.append(
                        info.get("best_fitness", float("inf"))
                    )
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        if not self.episode_rewards:
            return {}

        return {
            "mean_reward": np.mean(self.episode_rewards[-100:]),
            "mean_length": np.mean(self.episode_lengths[-100:]),
            "mean_interventions": np.mean(self.interventions[-100:]),
            "mean_best_fitness": np.mean(self.best_fitness_values[-100:]),
            "total_episodes": len(self.episode_rewards),
        }


class ProphetAgent:
    """
    Prophet Agent using PPO for hyper-heuristic control.

    The Prophet learns when to request LLM-generated heuristics
    and which type of heuristic to request based on optimizer state.
    """

    def __init__(
        self,
        fitness_func: Callable[[np.ndarray], float],
        dim: int,
        llm_generator: Optional[DivineSpeech] = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        population_size: int = 20,
        max_iterations: int = 50,
        n_envs: int = 1,
        device: str = "auto",
        verbose: int = 1,
    ):
        """
        Initialize Prophet Agent.

        Args:
            fitness_func: Fitness function for optimization
            dim: Problem dimension
            llm_generator: LLM heuristic generator
            learning_rate: PPO learning rate
            n_steps: Steps per update
            batch_size: Minibatch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            population_size: DRA population size
            max_iterations: Max iterations per episode
            n_envs: Number of parallel environments
            device: Device for training ('auto', 'cuda', 'cpu')
            verbose: Verbosity level
        """
        if not HAS_SB3:
            raise ImportError(
                "stable-baselines3 required. Install with: pip install stable-baselines3[extra]"
            )

        self.fitness_func = fitness_func
        self.dim = dim
        self.llm_generator = llm_generator
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.n_envs = n_envs
        self.verbose = verbose

        # Create environment
        self.env = self._create_env()

        # PPO hyperparameters
        self.ppo_params = {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "verbose": verbose,
            "device": device,
        }

        # Create PPO model
        self.model = PPO("MlpPolicy", self.env, **self.ppo_params)

        # Training callback
        self.callback = ProphetCallback(verbose=verbose)

        # Training state
        self._is_trained = False
        self._training_stats = {}

    def _create_env(self) -> Any:
        """Create vectorized environment."""

        def make_env(rank: int, seed: int = 0):
            def _init():
                env = ProphetEnv(
                    fitness_func=self.fitness_func,
                    dim=self.dim,
                    population_size=self.population_size,
                    max_iterations=self.max_iterations,
                    llm_generator=self.llm_generator,
                )
                env.reset(seed=seed + rank)
                return Monitor(env)

            return _init

        if self.n_envs > 1:
            # Use SubprocVecEnv for parallel execution
            return SubprocVecEnv([make_env(i) for i in range(self.n_envs)])
        else:
            # Use DummyVecEnv for sequential execution
            return DummyVecEnv([make_env(0)])

    def train(self, total_timesteps: int = 100000) -> Dict[str, Any]:
        """
        Train the Prophet agent.

        Args:
            total_timesteps: Total training timesteps

        Returns:
            Training statistics
        """
        # Callbacks
        callbacks = [self.callback]

        # Add checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=1024,
            save_path="./models/checkpoints/",
            name_prefix="prophet_model",
        )
        callbacks.append(checkpoint_callback)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=self.verbose > 0,
        )

        self._is_trained = True
        self._training_stats = self.callback.get_stats()

        return self._training_stats

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> Tuple[int, Optional[np.ndarray]]:
        """
        Predict action for given observation.

        Args:
            observation: Normalized optimizer state
            deterministic: Whether to use deterministic policy

        Returns:
            Tuple of (action, action_probabilities)
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action), None

    def optimize_with_guidance(
        self, callback: Optional[Callable] = None
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Run optimization with Prophet guidance.

        Args:
            callback: Optional callback for progress tracking

        Returns:
            Tuple of (best_solution, best_fitness, stats)
        """
        # Reset environment
        obs, info = self.env.reset()

        total_interventions = 0
        intervention_history = []
        fitness_history = []

        done = False
        while not done:
            # Get Prophet's decision
            action, _ = self.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step([action])
            done = terminated[0] or truncated[0]

            # Track metrics
            if action != 0:
                total_interventions += 1
            intervention_history.append(action)
            fitness_history.append(info[0].get("best_fitness", float("inf")))

            if callback:
                callback(info[0])

        # Get final results
        env_unwrapped = self.env.envs[0].unwrapped
        best_solution = env_unwrapped.optimizer.best_solution.position
        best_fitness = env_unwrapped.optimizer.best_solution.cost

        stats = {
            "total_interventions": total_interventions,
            "intervention_history": intervention_history,
            "fitness_history": fitness_history,
            "final_fitness": best_fitness,
        }

        return best_solution, best_fitness, stats

    def save(self, path: str):
        """Save trained model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

        # Save metadata
        metadata = {
            "dim": self.dim,
            "population_size": self.population_size,
            "max_iterations": self.max_iterations,
            "ppo_params": self.ppo_params,
            "training_stats": self._training_stats,
            "is_trained": self._is_trained,
        }

        metadata_path = path.parent / f"{path.stem}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def load(self, path: str):
        """Load trained model."""
        self.model = PPO.load(path, env=self.env)
        self._is_trained = True

        # Load metadata if exists
        metadata_path = Path(path).parent / f"{Path(path).stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                self._training_stats = metadata.get("training_stats", {})

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return self._training_stats


class RuleBasedProphet:
    """
    Rule-based Prophet for comparison/fallback.

    Uses simple heuristics instead of learned policy.
    Useful for baseline comparison or when DRL training is not feasible.
    """

    def __init__(
        self,
        intervention_threshold: int = 10,
        diversity_threshold: float = 0.1,
        max_interventions_per_episode: int = 20,
    ):
        """
        Initialize rule-based prophet.

        Args:
            intervention_threshold: Stagnation steps before intervention
            diversity_threshold: Diversity below which to diversify
            max_interventions_per_episode: Maximum allowed interventions
        """
        self.intervention_threshold = intervention_threshold
        self.diversity_threshold = diversity_threshold
        self.max_interventions = max_interventions_per_episode
        self._intervention_count = 0

    def reset(self):
        """Reset for new episode."""
        self._intervention_count = 0

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> Tuple[int, None]:
        """
        Predict action based on rules.

        Args:
            observation: Normalized state
            deterministic: Ignored (always deterministic)

        Returns:
            Action and None for compatibility
        """
        if self._intervention_count >= self.max_interventions:
            return 0, None  # No more interventions allowed

        # Parse observation
        progress = observation[0]
        is_stagnating = observation[7] > 0.5
        stagnation_count = observation[8] * 50  # Denormalize
        diversity = observation[9]

        # Rule-based decision
        if not is_stagnating:
            return 0, None  # No intervention needed

        if stagnation_count < self.intervention_threshold / 2:
            return 0, None  # Wait a bit more

        self._intervention_count += 1

        # Choose intervention type
        if diversity < self.diversity_threshold:
            return 4, None  # Diversification

        if progress < 0.3:
            return 1, None  # Mutation (exploration)
        elif progress > 0.7:
            return 3, None  # Local search (exploitation)
        else:
            return 2, None  # Crossover
