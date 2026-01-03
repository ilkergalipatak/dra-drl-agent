"""
Binary Divine Religions Algorithm (DRA) for Feature Selection

Based on the original DRA by Nima Khodadadi, adapted for binary optimization.
This implementation converts the continuous DRA to work with feature selection
problems using transfer functions.

Reference:
    Khodadadi, N., et al. "Divine Religions Algorithm: A Social-Inspired
    Metaheuristic for Engineering Optimization"
"""

import numpy as np
from typing import Callable, Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from .transfer_functions import TransferFunction, ensure_at_least_one_feature


@dataclass
class Belief:
    """Represents a belief (solution) in DRA terminology."""

    position: np.ndarray  # Binary feature mask
    continuous: np.ndarray  # Continuous representation for operators
    cost: float = float("inf")

    def copy(self) -> "Belief":
        return Belief(
            position=self.position.copy(),
            continuous=self.continuous.copy(),
            cost=self.cost,
        )


@dataclass
class Group:
    """Represents a religious group with missionary and followers."""

    missionary: Belief = None
    followers: List[Belief] = field(default_factory=list)

    @property
    def num_followers(self) -> int:
        return len(self.followers)

    @property
    def total_cost(self) -> float:
        costs = [f.cost for f in self.followers]
        if self.missionary:
            costs.append(self.missionary.cost)
        return sum(costs) if costs else 0


class BinaryDRA:
    """
    Binary Divine Religions Algorithm for Feature Selection.

    The algorithm organizes the population into groups (religious schools),
    each led by a missionary. Various operators (miracle, proselytism,
    reward/penalty) are used to evolve the population.
    """

    def __init__(
        self,
        fitness_func: Callable[[np.ndarray], float],
        dim: int,
        population_size: int = 50,
        max_iterations: int = 100,
        num_groups: int = 5,
        belief_profile_rate: float = 0.5,
        miracle_rate: float = 0.5,
        proselytism_rate: float = 0.9,
        reward_penalty_rate: float = 0.2,
        transfer_function: str = "s1",
        random_seed: Optional[int] = None,
    ):
        """
        Initialize Binary DRA.

        Args:
            fitness_func: Function to minimize (lower is better)
            dim: Number of features (dimensions)
            population_size: Size of the belief profile (population)
            max_iterations: Maximum number of iterations
            num_groups: Number of religious groups
            belief_profile_rate: Belief Profile Consideration Rate (BPSP)
            miracle_rate: Initial miracle rate (MP)
            proselytism_rate: Proselytism consideration rate (PP)
            reward_penalty_rate: Reward/Penalty consideration rate (RP)
            transfer_function: Transfer function name for binary conversion
            random_seed: Random seed for reproducibility
        """
        self.fitness_func = fitness_func
        self.dim = dim
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.num_groups = num_groups
        self.belief_profile_rate = belief_profile_rate
        self.initial_miracle_rate = miracle_rate
        self.proselytism_rate = proselytism_rate
        self.reward_penalty_rate = reward_penalty_rate
        self.transfer = TransferFunction(transfer_function)

        if random_seed is not None:
            np.random.seed(random_seed)

        # Population (Belief Profile)
        self.population: List[Belief] = []
        self.groups: List[Group] = []
        self.leader: Belief = None

        # History tracking
        self.convergence_curve: List[float] = []
        self.best_solution: Belief = None
        self.current_iteration: int = 0

        # For external operators (LLM-generated)
        self.external_operators: List[Callable] = []

    def _create_random_belief(self) -> Belief:
        """Create a random belief (solution)."""
        continuous = np.random.uniform(-1, 1, self.dim)
        binary = self.transfer.to_binary(continuous)
        binary = ensure_at_least_one_feature(binary)
        cost = self.fitness_func(binary)
        return Belief(position=binary, continuous=continuous, cost=cost)

    def _initialize_population(self):
        """Initialize the belief profile (population)."""
        self.population = [
            self._create_random_belief() for _ in range(self.population_size)
        ]

        # Sort by cost and find leader
        self.population.sort(key=lambda x: x.cost)
        self.leader = self.population[0].copy()
        self.best_solution = self.leader.copy()

    def _initialize_groups(self):
        """Initialize groups and assign missionaries & followers."""
        num_followers = self.population_size - self.num_groups

        # Create groups
        self.groups = [Group() for _ in range(self.num_groups)]

        # Assign missionaries (first num_groups individuals)
        for k in range(self.num_groups):
            self.groups[k].missionary = self.population[k].copy()

        # Assign followers randomly to groups
        for j in range(num_followers):
            k = np.random.randint(0, self.num_groups)
            self.groups[k].followers.append(self.population[self.num_groups + j].copy())

    def _miracle_operator(self, belief: Belief, mp: float) -> np.ndarray:
        """
        Miracle operator - Exploration phase.
        Creates variation using cosine-based transformation.
        """
        if np.random.random() <= 0.5:
            # Variant 1
            new_continuous = (
                belief.continuous
                * np.cos(np.pi / 2)
                * (np.random.random() - np.cos(np.random.random()))
            )
        else:
            # Variant 2
            new_continuous = belief.continuous + np.random.random() * (
                belief.continuous - round(1 ** np.random.random()) * belief.continuous
            )
        return new_continuous

    def _proselytism_operator(self, belief: Belief, mp: float) -> np.ndarray:
        """
        Proselytism operator - Exploitation phase.
        Guides solutions toward the leader using mean-based transformation.
        """
        if np.random.random() < (1 - mp):
            # Variant 1
            mean_val = np.mean(belief.continuous)
            new_continuous = (
                belief.continuous * (1 - mp)
                + (1 - mean_val)
                - (np.random.random() - 4 * np.sin(np.sin(3.14 * np.random.random())))
            )
        else:
            # Variant 2
            mean_val = np.mean(belief.continuous)
            new_continuous = (
                belief.continuous
                * (np.random.random() - 2 * np.random.random() - mean_val)
                * (2 * np.random.random() - (1 - mp))
            )
        return new_continuous

    def _reward_operator(self, belief: Belief) -> np.ndarray:
        """Reward operator - adds positive variation."""
        return belief.continuous + (1 - np.random.randn())

    def _penalty_operator(self, belief: Belief) -> np.ndarray:
        """Penalty operator - adds negative variation."""
        return belief.continuous - (1 + np.random.randn())

    def _leader_influence_operator(self) -> np.ndarray:
        """Leader influence using belief profile consideration."""
        return self.leader.continuous * (
            np.random.random() - np.cos(np.random.random())
        )

    def _new_follower_operator(self, belief: Belief) -> np.ndarray:
        """Generate new follower belief using sine transformation."""
        return belief.continuous * (np.random.random() - np.sin(np.random.random()))

    def _apply_bounds(self, continuous: np.ndarray) -> np.ndarray:
        """Keep continuous values within bounds."""
        return np.clip(continuous, -10, 10)

    def _evaluate_and_update(self, belief: Belief, new_continuous: np.ndarray) -> bool:
        """
        Convert continuous to binary, evaluate, and update if better.

        Returns:
            True if the belief was updated, False otherwise.
        """
        new_continuous = self._apply_bounds(new_continuous)
        new_binary = self.transfer.to_binary(new_continuous, belief.position)
        new_binary = ensure_at_least_one_feature(new_binary)
        new_cost = self.fitness_func(new_binary)

        if new_cost < belief.cost:
            belief.continuous = new_continuous
            belief.position = new_binary
            belief.cost = new_cost
            return True
        return False

    def apply_external_operator(self, operator: Callable) -> int:
        """
        Apply an external (LLM-generated) operator to the population.

        Args:
            operator: Function that takes (population, leader, iteration)
                     and returns modified continuous values for each individual

        Returns:
            Number of improvements made
        """
        improvements = 0
        try:
            for belief in self.population:
                new_continuous = operator(
                    belief.continuous.copy(),
                    self.leader.continuous.copy(),
                    self.current_iteration,
                    self.max_iterations,
                )
                if self._evaluate_and_update(belief, new_continuous):
                    improvements += 1
        except Exception as e:
            print(f"External operator failed: {e}")
        return improvements

    def _iteration_step(self) -> float:
        """
        Perform one iteration of DRA.

        Returns:
            Best cost found in this iteration.
        """
        # Calculate dynamic miracle rate
        mp = (
            np.random.random()
            * (1 - (self.current_iteration / self.max_iterations * 2))
            * np.random.random()
        )
        mp = max(0, min(1, mp))

        # Find current leader
        min_idx = np.argmin([b.cost for b in self.population])
        self.leader = self.population[min_idx].copy()

        # Generate new follower (absorption)
        new_follower = self._create_random_belief()

        # Belief Profile consideration
        if np.random.random() <= self.belief_profile_rate:
            x_continuous = self._leader_influence_operator()
        else:
            x_continuous = None

        # Main loop over population
        for i, belief in enumerate(self.population):
            # Exploration vs Exploitation
            if np.random.random() <= mp:
                # Exploration: Miracle Operator
                new_continuous = self._miracle_operator(belief, mp)
            else:
                # Exploitation: Proselytism Operator
                new_continuous = self._proselytism_operator(belief, mp)

            self._evaluate_and_update(belief, new_continuous)

        # Apply external operators if any
        for operator in self.external_operators:
            self.apply_external_operator(operator)

        # Calculate new follower
        random_idx = np.random.randint(0, self.population_size)
        new_follower_continuous = self._new_follower_operator(
            self.population[random_idx]
        )

        # Reward or Penalty operator
        random_idx = np.random.randint(0, self.population_size)
        if np.random.random() >= self.reward_penalty_rate:
            x_continuous = self._reward_operator(self.population[random_idx])
        else:
            x_continuous = self._penalty_operator(self.population[random_idx])

        # Update the continuous value but evaluate later
        if x_continuous is not None:
            self._evaluate_and_update(self.population[random_idx], x_continuous)

        # Replacement operator (swap missionary and follower in a random group)
        if self.groups:
            random_group_idx = np.random.randint(0, self.num_groups)
            group = self.groups[random_group_idx]
            if group.followers:
                # Swap missionary with a random follower
                follower_idx = np.random.randint(0, len(group.followers))
                group.missionary, group.followers[follower_idx] = (
                    group.followers[follower_idx],
                    group.missionary,
                )

        # Update last individual with new follower
        new_follower_binary = self.transfer.to_binary(new_follower_continuous)
        new_follower_binary = ensure_at_least_one_feature(new_follower_binary)
        self.population[-1].continuous = new_follower_continuous
        self.population[-1].position = new_follower_binary
        self.population[-1].cost = self.fitness_func(new_follower_binary)

        # Find best solution
        min_cost = float("inf")
        min_idx = 0
        for i, belief in enumerate(self.population):
            if belief.cost < min_cost:
                min_cost = belief.cost
                min_idx = i

        if min_cost < self.best_solution.cost:
            self.best_solution = self.population[min_idx].copy()

        return self.best_solution.cost

    def optimize(
        self, callback: Optional[Callable] = None
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run the DRA optimization.

        Args:
            callback: Optional callback function called after each iteration
                     with signature callback(iteration, best_cost, best_solution)

        Returns:
            Tuple of (best_solution, best_cost, convergence_curve)
        """
        # Initialize
        self._initialize_population()
        self._initialize_groups()

        self.convergence_curve = []

        # Main optimization loop
        for iteration in range(self.max_iterations):
            self.current_iteration = iteration
            best_cost = self._iteration_step()
            self.convergence_curve.append(best_cost)

            if callback:
                callback(iteration, best_cost, self.best_solution.position)

        return (
            self.best_solution.position,
            self.best_solution.cost,
            self.convergence_curve,
        )

    def get_state(self) -> Dict[str, Any]:
        """
        Get current state of the optimizer for DRL observation.

        Returns:
            Dictionary containing optimizer state metrics.
        """
        costs = [b.cost for b in self.population]
        positions = np.array([b.position for b in self.population])

        # Calculate diversity (average Hamming distance)
        diversity = 0
        n = len(self.population)
        if n > 1:
            for i in range(n):
                for j in range(i + 1, n):
                    diversity += np.sum(positions[i] != positions[j])
            diversity /= n * (n - 1) / 2 * self.dim

        # Stagnation detection
        if len(self.convergence_curve) >= 10:
            recent = self.convergence_curve[-10:]
            improvement = (recent[0] - recent[-1]) / (abs(recent[0]) + 1e-10)
        else:
            improvement = 1.0

        return {
            "iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "progress": self.current_iteration / self.max_iterations,
            "best_cost": (
                self.best_solution.cost if self.best_solution else float("inf")
            ),
            "mean_cost": np.mean(costs),
            "std_cost": np.std(costs),
            "min_cost": np.min(costs),
            "max_cost": np.max(costs),
            "diversity": diversity,
            "num_selected_features": (
                np.sum(self.best_solution.position) if self.best_solution else 0
            ),
            "feature_ratio": (
                np.sum(self.best_solution.position) / self.dim
                if self.best_solution
                else 1
            ),
            "recent_improvement": improvement,
            "is_stagnating": improvement < 0.001,
        }

    def inject_solution(self, binary_solution: np.ndarray, replace_worst: bool = True):
        """
        Inject a solution into the population.

        Args:
            binary_solution: Binary feature mask to inject
            replace_worst: If True, replace worst individual; else replace random
        """
        binary_solution = ensure_at_least_one_feature(binary_solution)
        cost = self.fitness_func(binary_solution)
        continuous = np.where(
            binary_solution == 1,
            np.random.uniform(0.5, 1, self.dim),
            np.random.uniform(-1, -0.5, self.dim),
        )

        new_belief = Belief(position=binary_solution, continuous=continuous, cost=cost)

        if replace_worst:
            worst_idx = np.argmax([b.cost for b in self.population])
        else:
            worst_idx = np.random.randint(0, self.population_size)

        self.population[worst_idx] = new_belief

        if cost < self.best_solution.cost:
            self.best_solution = new_belief.copy()
