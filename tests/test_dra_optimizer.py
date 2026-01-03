"""
Tests for DRA Optimizer
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.transfer_functions import (
    TransferFunction,
    s_shaped_v1,
    v_shaped_v1,
    binary_conversion_s,
    ensure_at_least_one_feature,
)
from src.core.dra_optimizer import BinaryDRA, Belief


class TestTransferFunctions:
    """Test transfer function implementations."""

    def test_sigmoid_bounds(self):
        """Test sigmoid output is in [0, 1]."""
        x = np.random.randn(100)
        result = s_shaped_v1(x)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_v_shaped_bounds(self):
        """Test V-shaped output is in [0, 1]."""
        x = np.random.randn(100)
        result = v_shaped_v1(x)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_transfer_function_class(self):
        """Test TransferFunction class."""
        tf = TransferFunction("s1")
        assert tf.is_s_shaped

        tf_v = TransferFunction("v1")
        assert not tf_v.is_s_shaped

    def test_binary_conversion(self):
        """Test binary conversion produces 0/1 values."""
        x = np.random.randn(30)
        binary = binary_conversion_s(x)
        assert np.all((binary == 0) | (binary == 1))

    def test_ensure_at_least_one_feature(self):
        """Test that at least one feature is selected."""
        zeros = np.zeros(30)
        result = ensure_at_least_one_feature(zeros)
        assert np.sum(result) >= 1


class TestBinaryDRA:
    """Test Binary DRA optimizer."""

    @pytest.fixture
    def simple_fitness(self):
        """Simple fitness function: minimize selected features."""
        return lambda x: np.sum(x) / len(x)

    def test_initialization(self, simple_fitness):
        """Test DRA initialization."""
        dra = BinaryDRA(
            fitness_func=simple_fitness, dim=10, population_size=20, max_iterations=50
        )
        dra._initialize_population()
        dra._initialize_groups()

        assert len(dra.population) == 20
        assert len(dra.groups) == 5
        assert dra.best_solution is not None

    def test_population_binary(self, simple_fitness):
        """Test that population is binary."""
        dra = BinaryDRA(fitness_func=simple_fitness, dim=10, population_size=20)
        dra._initialize_population()

        for belief in dra.population:
            assert np.all((belief.position == 0) | (belief.position == 1))

    def test_optimization_improves(self, simple_fitness):
        """Test that optimization improves fitness."""
        dra = BinaryDRA(
            fitness_func=simple_fitness,
            dim=10,
            population_size=20,
            max_iterations=30,
            random_seed=42,
        )

        solution, fitness, curve = dra.optimize()

        # Should improve or stay same
        assert curve[-1] <= curve[0] + 0.01
        assert np.all((solution == 0) | (solution == 1))

    def test_get_state(self, simple_fitness):
        """Test state extraction for DRL."""
        dra = BinaryDRA(fitness_func=simple_fitness, dim=10, population_size=20)
        dra._initialize_population()
        dra._initialize_groups()

        state = dra.get_state()

        assert "iteration" in state
        assert "diversity" in state
        assert "best_cost" in state
        assert "is_stagnating" in state

    def test_external_operator(self, simple_fitness):
        """Test applying external operator."""
        dra = BinaryDRA(fitness_func=simple_fitness, dim=10, population_size=20)
        dra._initialize_population()
        dra._initialize_groups()

        # Simple operator that moves toward zeros
        def custom_op(pos, leader, iteration, max_iter):
            return pos * 0.5

        improvements = dra.apply_external_operator(custom_op)
        assert isinstance(improvements, int)


class TestBelief:
    """Test Belief dataclass."""

    def test_belief_copy(self):
        """Test belief copying."""
        belief = Belief(
            position=np.array([1, 0, 1, 0]),
            continuous=np.array([0.5, -0.5, 0.5, -0.5]),
            cost=0.5,
        )

        copy = belief.copy()

        assert np.array_equal(copy.position, belief.position)
        assert copy.cost == belief.cost
        # Should be independent
        copy.position[0] = 0
        assert belief.position[0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
