"""
Prophet Environment - Gymnasium Environment for DRL Agent

Creates a Gymnasium-compatible environment that wraps the DRA optimization
process, allowing a DRL agent to learn when and how to intervene.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple, Callable
from ..core.dra_optimizer import BinaryDRA
from ..utils.stagnation_detector import StagnationDetector
from ..llm.llm_generator import DivineSpeech
from ..llm.prompt_templates import TEMPLATES_BY_TYPE, ALL_TEMPLATES


class ProphetEnv(gym.Env):
    """
    Gymnasium environment for the Prophet (DRL Agent).

    The Prophet observes the optimization process and decides:
    1. Whether to intervene
    2. What type of heuristic to request from LLM

    Action space: Discrete actions for intervention type
    Observation space: Normalized optimizer state metrics
    """

    metadata = {"render_modes": ["human"]}

    # Action definitions
    ACTION_NONE = 0
    ACTION_MUTATION = 1
    ACTION_CROSSOVER = 2
    ACTION_LOCAL_SEARCH = 3
    ACTION_DIVERSIFICATION = 4

    ACTION_NAMES = {
        0: "none",
        1: "mutation",
        2: "crossover",
        3: "local_search",
        4: "diversification",
    }

    def __init__(
        self,
        fitness_func: Callable[[np.ndarray], float],
        dim: int,
        population_size: int = 50,
        max_iterations: int = 100,
        llm_generator: Optional[DivineSpeech] = None,
        intervention_cost: float = 0.01,
        improvement_reward_scale: float = 10.0,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize Prophet environment.

        Args:
            fitness_func: Fitness function for optimization
            dim: Number of features (dimensions)
            population_size: DRA population size
            max_iterations: Maximum optimization iterations
            llm_generator: LLM generator for heuristics
            intervention_cost: Penalty for each intervention
            improvement_reward_scale: Scale factor for improvement rewards
            render_mode: Rendering mode ('human' or None)
        """
        super().__init__()

        self.fitness_func = fitness_func
        self.dim = dim
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.llm_generator = llm_generator
        self.intervention_cost = intervention_cost
        self.improvement_reward_scale = improvement_reward_scale
        self.render_mode = render_mode

        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)

        # Observation space: normalized optimizer state
        # [progress, best_cost, mean_cost, std_cost, diversity,
        #  feature_ratio, recent_improvement, stagnation_severity]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )

        # Initialize components (reset in reset())
        self.optimizer: Optional[BinaryDRA] = None
        self.stagnation_detector: Optional[StagnationDetector] = None

        # Episode tracking
        self._current_step = 0
        self._total_reward = 0
        self._interventions = 0
        self._initial_fitness = float("inf")

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Create new optimizer
        self.optimizer = BinaryDRA(
            fitness_func=self.fitness_func,
            dim=self.dim,
            population_size=self.population_size,
            max_iterations=self.max_iterations,
            random_seed=seed,
        )

        # Initialize optimizer
        self.optimizer._initialize_population()
        self.optimizer._initialize_groups()

        # Create stagnation detector
        self.stagnation_detector = StagnationDetector()

        # Reset tracking
        self._current_step = 0
        self._total_reward = 0
        self._interventions = 0
        self._initial_fitness = self.optimizer.best_solution.cost

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step (one optimization iteration).

        Args:
            action: Intervention action (0-4)

        Returns:
            observation, reward, terminated, truncated, info
        """
        assert self.optimizer is not None, "Environment not reset"

        # Store pre-step fitness
        pre_fitness = self.optimizer.best_solution.cost

        # Apply action (intervention)
        reward = 0.0
        if action != self.ACTION_NONE:
            reward -= self.intervention_cost  # Penalty for intervention
            self._interventions += 1

            if self.llm_generator is not None:
                # Generate and apply LLM heuristic
                operator_type = self.ACTION_NAMES[action]
                state = self.optimizer.get_state()

                templates = TEMPLATES_BY_TYPE.get(operator_type, [])
                if templates:
                    import random

                    template = random.choice(templates)
                    heuristic = self.llm_generator.generate_heuristic(
                        template, state, self.dim
                    )

                    if heuristic.success and heuristic.operator:
                        improvements = self.optimizer.apply_external_operator(
                            heuristic.operator
                        )
                        # Bonus for successful LLM operator
                        if improvements > 0:
                            reward += 0.1 * improvements

        # Run one DRA iteration
        self.optimizer.current_iteration = self._current_step
        best_cost = self.optimizer._iteration_step()

        # Calculate improvement reward
        post_fitness = self.optimizer.best_solution.cost
        improvement = pre_fitness - post_fitness

        if improvement > 0:
            reward += self.improvement_reward_scale * improvement

        # Update stagnation detector
        population = np.array([b.position for b in self.optimizer.population])
        stagnation_state = self.stagnation_detector.update(
            self._current_step,
            post_fitness,
            population,
            self.optimizer.best_solution.position,
        )

        # Penalize stagnation (encourage prevention)
        if stagnation_state.is_stagnating:
            reward -= 0.001 * stagnation_state.stagnation_count

        self._current_step += 1
        self._total_reward += reward

        # Check termination
        terminated = self._current_step >= self.max_iterations
        truncated = False

        observation = self._get_observation()
        info = self._get_info()
        info["stagnation_state"] = stagnation_state

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get normalized observation from optimizer state."""
        if self.optimizer is None:
            return np.zeros(10, dtype=np.float32)

        state = self.optimizer.get_state()
        stag_metrics = (
            self.stagnation_detector.get_metrics() if self.stagnation_detector else {}
        )

        # Normalize all values to [0, 1]
        obs = np.array(
            [
                state.get("progress", 0),
                min(1.0, state.get("best_cost", 1) / (self._initial_fitness + 1e-10)),
                min(1.0, state.get("mean_cost", 1) / (self._initial_fitness + 1e-10)),
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

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info for debugging."""
        if self.optimizer is None:
            return {}

        return {
            "step": self._current_step,
            "best_fitness": self.optimizer.best_solution.cost,
            "interventions": self._interventions,
            "total_reward": self._total_reward,
            "optimizer_state": self.optimizer.get_state(),
        }

    def render(self):
        """Render current state."""
        if self.render_mode == "human":
            info = self._get_info()
            print(
                f"Step {info['step']}/{self.max_iterations} | "
                f"Best: {info['best_fitness']:.6f} | "
                f"Interventions: {info['interventions']} | "
                f"Reward: {info['total_reward']:.4f}"
            )

    def close(self):
        """Clean up resources."""
        self.optimizer = None
        self.stagnation_detector = None


def make_prophet_env(
    fitness_func: Callable,
    dim: int,
    llm_generator: Optional[DivineSpeech] = None,
    **kwargs,
) -> ProphetEnv:
    """Factory function to create Prophet environment."""
    return ProphetEnv(
        fitness_func=fitness_func, dim=dim, llm_generator=llm_generator, **kwargs
    )
