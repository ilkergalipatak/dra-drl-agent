"""
Integration Tests for Hybrid DRA Feature Selection
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hybrid_dra_fs import HybridDRAFeatureSelector, OptimizationResult
from src.core.dra_optimizer import BinaryDRA
from src.evaluation.fitness_evaluator import FitnessEvaluator
from src.data.dataset_loader import load_dataset
from src.utils.stagnation_detector import StagnationDetector
from src.llm.code_sandbox import CodeSandbox, create_fallback_operator


class TestIntegration:
    """Integration tests for the hybrid system."""

    def test_full_optimization_baseline(self):
        """Test full optimization without LLM."""
        selector = HybridDRAFeatureSelector()

        # Override config for faster test
        selector.config["dra"]["max_iterations"] = 20
        selector.config["dra"]["population_size"] = 20

        result = selector.run_baseline("iris", verbose=False)

        assert isinstance(result, OptimizationResult)
        assert result.n_selected > 0
        assert result.n_selected <= result.n_total
        assert 0 <= result.test_accuracy <= 1

    def test_stagnation_detection_integration(self):
        """Test stagnation detection with optimizer."""
        data = load_dataset("iris")

        fitness_func = FitnessEvaluator(
            X_train=data["X_train"], y_train=data["y_train"]
        )

        optimizer = BinaryDRA(
            fitness_func=fitness_func,
            dim=data["n_features"],
            population_size=20,
            max_iterations=30,
        )

        detector = StagnationDetector()
        optimizer._initialize_population()
        optimizer._initialize_groups()

        stagnation_detected = False
        for i in range(30):
            optimizer.current_iteration = i
            optimizer._iteration_step()

            population = np.array([b.position for b in optimizer.population])
            state = detector.update(
                i,
                optimizer.best_solution.cost,
                population,
                optimizer.best_solution.position,
            )

            if state.is_stagnating:
                stagnation_detected = True

        # Stagnation should be detected in some iterations
        metrics = detector.get_metrics()
        assert "current_diversity" in metrics


class TestCodeSandbox:
    """Test code sandbox for LLM-generated code."""

    def test_valid_operator(self):
        """Test compiling valid operator code."""
        sandbox = CodeSandbox()

        code = """
def custom_operator(position, leader, iteration, max_iteration):
    progress = iteration / max_iteration
    return position * (1 - progress) + leader * progress
"""

        operator, error = sandbox.compile_operator(code)
        assert operator is not None
        assert error == ""

    def test_invalid_syntax(self):
        """Test handling invalid syntax."""
        sandbox = CodeSandbox()

        code = "def custom_operator(x: this is invalid"

        operator, error = sandbox.compile_operator(code)
        assert operator is None
        assert "Syntax error" in error

    def test_forbidden_code(self):
        """Test blocking forbidden patterns."""
        sandbox = CodeSandbox()

        code = """
def custom_operator(position, leader, iteration, max_iteration):
    import os
    os.system("rm -rf /")
    return position
"""

        operator, error = sandbox.compile_operator(code)
        assert operator is None
        assert "Forbidden" in error

    def test_operator_execution(self):
        """Test executing compiled operator."""
        sandbox = CodeSandbox()

        code = """
def custom_operator(position, leader, iteration, max_iteration):
    return position * 0.9 + leader * 0.1
"""

        operator, success, error = sandbox.compile_and_test(code, dim=10)
        assert success
        assert operator is not None

        # Test execution
        pos = np.random.randn(10)
        leader = np.random.randn(10)
        result = operator(pos, leader, 50, 100)

        assert result.shape == pos.shape

    def test_fallback_operators(self):
        """Test fallback operator creation."""
        for op_type in ["mutation", "crossover", "local_search", "diversification"]:
            operator = create_fallback_operator(op_type)

            pos = np.random.randn(10)
            leader = np.random.randn(10)
            result = operator(pos, leader, 50, 100)

            assert result.shape == pos.shape


class TestStagnationDetector:
    """Test stagnation detector."""

    def test_diversity_calculation(self):
        """Test diversity calculation."""
        detector = StagnationDetector()

        # Create diverse population
        population = np.random.randint(0, 2, (10, 20))

        # Simulate updates
        for i in range(5):
            state = detector.update(
                i, 1.0 - i * 0.1, population, population[0]  # Improving
            )

        metrics = detector.get_metrics()
        assert 0 <= metrics["current_diversity"] <= 1

    def test_stagnation_after_no_improvement(self):
        """Test stagnation is detected when no improvement."""
        detector = StagnationDetector(window_size=5, fitness_threshold=0.001)

        population = np.ones((10, 20))  # No diversity

        for i in range(20):
            state = detector.update(i, 0.5, population, population[0])  # Same fitness

        assert state.is_stagnating
        assert state.stagnation_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
