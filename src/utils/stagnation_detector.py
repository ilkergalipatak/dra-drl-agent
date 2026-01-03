"""
Stagnation Detector for DRA Optimization

Monitors the optimization process and detects when the algorithm
is stuck in local optima or losing diversity.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque
from dataclasses import dataclass


@dataclass
class StagnationState:
    """Current stagnation state information."""

    is_stagnating: bool
    stagnation_type: str  # 'none', 'fitness', 'diversity', 'both'
    stagnation_count: int
    severity: float  # 0-1, higher means worse
    recommended_action: str


class StagnationDetector:
    """
    Detects various types of stagnation in the optimization process.

    Monitors:
    - Fitness improvement over time
    - Population diversity (Hamming distance for binary)
    - Best solution changes
    """

    def __init__(
        self,
        window_size: int = 10,
        fitness_threshold: float = 0.001,
        diversity_threshold: float = 0.1,
        severe_stagnation_count: int = 20,
    ):
        """
        Initialize stagnation detector.

        Args:
            window_size: Number of iterations to consider for detection
            fitness_threshold: Minimum improvement to not be stagnating
            diversity_threshold: Minimum diversity to not be stagnating
            severe_stagnation_count: Number of iterations for severe stagnation
        """
        self.window_size = window_size
        self.fitness_threshold = fitness_threshold
        self.diversity_threshold = diversity_threshold
        self.severe_stagnation_count = severe_stagnation_count

        # History buffers
        self.fitness_history: deque = deque(maxlen=window_size * 2)
        self.diversity_history: deque = deque(maxlen=window_size)
        self.best_solutions: deque = deque(maxlen=window_size)

        # Stagnation tracking
        self._stagnation_counter = 0
        self._last_improvement_iteration = 0
        self._best_fitness_ever = float("inf")
        self._current_iteration = 0

    def update(
        self,
        iteration: int,
        best_fitness: float,
        population: np.ndarray,
        best_solution: np.ndarray,
    ) -> StagnationState:
        """
        Update detector with current state and check for stagnation.

        Args:
            iteration: Current iteration number
            best_fitness: Current best fitness value
            population: Binary population matrix (n_individuals x n_features)
            best_solution: Current best solution (binary array)

        Returns:
            StagnationState with detection results
        """
        self._current_iteration = iteration

        # Update fitness history
        self.fitness_history.append(best_fitness)

        # Check for improvement
        if best_fitness < self._best_fitness_ever - self.fitness_threshold:
            self._best_fitness_ever = best_fitness
            self._last_improvement_iteration = iteration
            self._stagnation_counter = 0
        else:
            self._stagnation_counter += 1

        # Calculate diversity
        diversity = self._calculate_diversity(population)
        self.diversity_history.append(diversity)

        # Store best solution
        self.best_solutions.append(best_solution.copy())

        # Detect stagnation type
        return self._detect_stagnation(diversity)

    def _calculate_diversity(self, population: np.ndarray) -> float:
        """
        Calculate population diversity using average Hamming distance.

        Args:
            population: Binary population matrix

        Returns:
            Diversity value between 0 and 1
        """
        n_individuals = len(population)
        if n_individuals < 2:
            return 0.0

        n_features = (
            population.shape[1] if len(population.shape) > 1 else len(population[0])
        )

        total_distance = 0
        count = 0

        for i in range(n_individuals):
            for j in range(i + 1, n_individuals):
                hamming = np.sum(population[i] != population[j])
                total_distance += hamming
                count += 1

        if count == 0:
            return 0.0

        # Normalize by maximum possible distance
        avg_distance = total_distance / count
        diversity = avg_distance / n_features

        return diversity

    def _detect_stagnation(self, current_diversity: float) -> StagnationState:
        """Analyze current state and detect stagnation."""

        # Check fitness stagnation
        fitness_stagnating = False
        if len(self.fitness_history) >= self.window_size:
            recent = list(self.fitness_history)[-self.window_size :]
            improvement = (recent[0] - recent[-1]) / (abs(recent[0]) + 1e-10)
            fitness_stagnating = improvement < self.fitness_threshold

        # Check diversity stagnation
        diversity_stagnating = current_diversity < self.diversity_threshold

        # Determine stagnation type
        if fitness_stagnating and diversity_stagnating:
            stagnation_type = "both"
        elif fitness_stagnating:
            stagnation_type = "fitness"
        elif diversity_stagnating:
            stagnation_type = "diversity"
        else:
            stagnation_type = "none"

        is_stagnating = stagnation_type != "none"

        # Calculate severity
        severity = min(1.0, self._stagnation_counter / self.severe_stagnation_count)

        # Recommend action
        recommended_action = self._recommend_action(
            stagnation_type, severity, current_diversity
        )

        return StagnationState(
            is_stagnating=is_stagnating,
            stagnation_type=stagnation_type,
            stagnation_count=self._stagnation_counter,
            severity=severity,
            recommended_action=recommended_action,
        )

    def _recommend_action(
        self, stagnation_type: str, severity: float, diversity: float
    ) -> str:
        """Recommend action based on stagnation analysis."""

        if stagnation_type == "none":
            return "continue"

        if stagnation_type == "diversity" or diversity < 0.05:
            # Very low diversity - need explosion
            return "diversification"

        if severity > 0.7:
            # Severe stagnation - need major change
            return "diversification"

        if severity > 0.4:
            # Moderate stagnation - try new operator
            if stagnation_type == "fitness":
                return "local_search"
            else:
                return "mutation"

        # Mild stagnation - minor adjustment
        return "crossover"

    def get_metrics(self) -> Dict[str, Any]:
        """Get current stagnation metrics."""
        diversity = self.diversity_history[-1] if self.diversity_history else 0.5

        fitness_improvement = 0.0
        if len(self.fitness_history) >= 2:
            recent = list(self.fitness_history)[-self.window_size :]
            fitness_improvement = (recent[0] - recent[-1]) / (abs(recent[0]) + 1e-10)

        return {
            "stagnation_count": self._stagnation_counter,
            "iterations_since_improvement": self._current_iteration
            - self._last_improvement_iteration,
            "current_diversity": diversity,
            "mean_diversity": (
                np.mean(list(self.diversity_history)) if self.diversity_history else 0.5
            ),
            "fitness_improvement": fitness_improvement,
            "best_fitness_ever": self._best_fitness_ever,
        }

    def reset(self):
        """Reset detector state."""
        self.fitness_history.clear()
        self.diversity_history.clear()
        self.best_solutions.clear()
        self._stagnation_counter = 0
        self._last_improvement_iteration = 0
        self._best_fitness_ever = float("inf")
        self._current_iteration = 0

    def should_intervene(self, min_iterations: int = 5) -> bool:
        """
        Check if intervention is recommended.

        Args:
            min_iterations: Minimum stagnation iterations before intervention

        Returns:
            True if intervention is recommended
        """
        return self._stagnation_counter >= min_iterations
