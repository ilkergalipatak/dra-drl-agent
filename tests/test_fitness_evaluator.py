"""
Tests for Fitness Evaluator
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.fitness_evaluator import FitnessEvaluator, create_fitness_function
from src.data.dataset_loader import DatasetLoader, load_dataset


class TestFitnessEvaluator:
    """Test fitness evaluator."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    def test_initialization(self, sample_data):
        """Test evaluator initialization."""
        X, y = sample_data
        evaluator = FitnessEvaluator(X, y)

        assert evaluator.n_features == 20
        assert evaluator.alpha == 0.99

    def test_evaluate_all_features(self, sample_data):
        """Test evaluation with all features."""
        X, y = sample_data
        evaluator = FitnessEvaluator(X, y)

        mask = np.ones(20, dtype=int)
        fitness = evaluator.evaluate(mask)

        assert 0 <= fitness <= 1

    def test_evaluate_single_feature(self, sample_data):
        """Test evaluation with single feature."""
        X, y = sample_data
        evaluator = FitnessEvaluator(X, y)

        mask = np.zeros(20, dtype=int)
        mask[0] = 1
        fitness = evaluator.evaluate(mask)

        assert 0 <= fitness <= 1

    def test_evaluate_no_features(self, sample_data):
        """Test evaluation with no features."""
        X, y = sample_data
        evaluator = FitnessEvaluator(X, y)

        mask = np.zeros(20, dtype=int)
        fitness = evaluator.evaluate(mask)

        assert fitness == 1.0  # Worst possible

    def test_caching(self, sample_data):
        """Test that caching works."""
        X, y = sample_data
        evaluator = FitnessEvaluator(X, y, use_cache=True)

        mask = np.ones(20, dtype=int)

        # First evaluation
        evaluator.evaluate(mask)
        stats1 = evaluator.get_stats()

        # Second evaluation (should hit cache)
        evaluator.evaluate(mask)
        stats2 = evaluator.get_stats()

        assert stats2["cache_hits"] > stats1["cache_hits"]

    def test_final_evaluation(self, sample_data):
        """Test final evaluation with detailed results."""
        X, y = sample_data
        X_train, X_test = X[:70], X[70:]
        y_train, y_test = y[:70], y[70:]

        evaluator = FitnessEvaluator(X_train, y_train, X_test, y_test)

        mask = np.ones(20, dtype=int)
        result = evaluator.evaluate_final(mask)

        assert "train_accuracy" in result
        assert "test_accuracy" in result
        assert "n_selected" in result
        assert result["n_selected"] == 20

    def test_different_classifiers(self, sample_data):
        """Test with different classifiers."""
        X, y = sample_data

        for clf in ["knn", "svm", "rf"]:
            evaluator = FitnessEvaluator(X, y, classifier=clf)
            fitness = evaluator.evaluate(np.ones(20, dtype=int))
            assert 0 <= fitness <= 1


class TestDatasetLoader:
    """Test dataset loader."""

    def test_list_datasets(self):
        """Test listing available datasets."""
        datasets = DatasetLoader.list_datasets()
        assert len(datasets) == 10
        assert "breast_cancer" in datasets

    def test_load_breast_cancer(self):
        """Test loading breast cancer dataset."""
        data = load_dataset("breast_cancer")

        assert "X_train" in data
        assert "y_train" in data
        assert data["n_features"] == 30
        assert data["n_classes"] == 2

    def test_load_all_datasets(self):
        """Test loading all datasets."""
        loader = DatasetLoader()

        for name in loader.AVAILABLE_DATASETS:
            data = loader.load(name)
            assert data["n_features"] > 0
            assert data["n_samples"] > 0

    def test_normalization(self):
        """Test that data is normalized."""
        data = load_dataset("breast_cancer", normalize=True)

        # Check roughly normalized (mean ~ 0, std ~ 1)
        mean = np.mean(data["X_train"])
        std = np.std(data["X_train"])

        assert abs(mean) < 1
        assert 0.5 < std < 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
