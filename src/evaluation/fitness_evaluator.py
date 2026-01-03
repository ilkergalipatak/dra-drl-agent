"""
Fitness Evaluator for Feature Selection

Wrapper-based fitness evaluation using classification accuracy.
Supports multiple classifiers: KNN, SVM, Random Forest, Logistic Regression.
"""

import numpy as np
from typing import Optional, Dict, Any, Literal, List
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import hashlib
import warnings

warnings.filterwarnings("ignore")


# Available classifiers
CLASSIFIERS = {
    "knn": {
        "name": "K-Nearest Neighbors",
        "class": KNeighborsClassifier,
        "default_params": {"n_neighbors": 5},
    },
    "svm": {
        "name": "Support Vector Machine",
        "class": SVC,
        "default_params": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
    },
    "rf": {
        "name": "Random Forest",
        "class": RandomForestClassifier,
        "default_params": {"n_estimators": 100, "max_depth": 10},
    },
    "lr": {
        "name": "Logistic Regression",
        "class": LogisticRegression,
        "default_params": {"max_iter": 1000, "solver": "lbfgs"},
    },
}


class FitnessEvaluator:
    """
    Wrapper-based fitness evaluator for feature selection.

    Evaluates feature subsets using classification accuracy.
    Lower fitness values are better (for minimization).

    Fitness = 1 - (alpha * accuracy + (1-alpha) * feature_reduction)
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        classifier: Literal["knn", "svm", "rf", "lr"] = "knn",
        alpha: float = 0.99,
        cv_folds: int = 5,
        knn_neighbors: int = 5,
        use_cache: bool = True,
        random_seed: int = 42,
    ):
        """
        Initialize fitness evaluator.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional, for final evaluation)
            y_test: Test labels (optional)
            classifier: Classifier type ('knn', 'svm', 'rf', 'lr')
            alpha: Trade-off between accuracy and feature reduction
                   Higher alpha prioritizes accuracy
            cv_folds: Number of cross-validation folds
            knn_neighbors: Number of neighbors for KNN
            use_cache: Whether to cache fitness evaluations
            random_seed: Random seed for reproducibility
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_features = X_train.shape[1]
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.use_cache = use_cache

        # Initialize classifier
        self.classifier_type = classifier
        self.knn_neighbors = knn_neighbors

        # Validate classifier type
        if classifier not in CLASSIFIERS:
            raise ValueError(
                f"Unknown classifier: {classifier}. "
                f"Available: {list(CLASSIFIERS.keys())}"
            )

        # Cache for evaluated solutions
        self._cache: Dict[str, float] = {}
        self._evaluation_count = 0
        self._cache_hits = 0

    def _create_classifier(self):
        """Create classifier instance based on type."""
        config = CLASSIFIERS[self.classifier_type]
        params = config["default_params"].copy()

        # Add random state if applicable
        if "random_state" in config["class"]().get_params():
            params["random_state"] = self.random_seed

        # Special handling for KNN
        if self.classifier_type == "knn":
            params["n_neighbors"] = min(self.knn_neighbors, len(self.y_train) // 2)

        return config["class"](**params)

    def _solution_hash(self, feature_mask: np.ndarray) -> str:
        """Generate hash for a feature mask."""
        return hashlib.md5(feature_mask.astype(np.int8).tobytes()).hexdigest()

    def evaluate(self, feature_mask: np.ndarray) -> float:
        """
        Evaluate a feature subset.

        Args:
            feature_mask: Binary array where 1 = feature selected

        Returns:
            Fitness value (lower is better)
        """
        # Ensure binary mask
        feature_mask = np.asarray(feature_mask).astype(int)

        # Check cache
        if self.use_cache:
            cache_key = self._solution_hash(feature_mask)
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key]

        self._evaluation_count += 1

        # Get selected feature indices
        selected_indices = np.where(feature_mask == 1)[0]

        # Edge case: no features selected
        if len(selected_indices) == 0:
            fitness = 1.0  # Worst possible fitness
            if self.use_cache:
                self._cache[cache_key] = fitness
            return fitness

        # Select features
        X_selected = self.X_train[:, selected_indices]

        # Evaluate using cross-validation
        try:
            clf = self._create_classifier()

            # Adjust cv_folds if needed
            n_samples = len(self.y_train)
            min_class_count = min(np.bincount(self.y_train))
            cv = min(self.cv_folds, min_class_count, n_samples // 2)
            cv = max(2, cv)  # At least 2 folds

            cv_scores = cross_val_score(
                clf, X_selected, self.y_train, cv=cv, scoring="accuracy", n_jobs=-1
            )
            accuracy = np.mean(cv_scores)
        except Exception as e:
            # If evaluation fails, return worst fitness
            accuracy = 0.0

        # Calculate feature reduction ratio
        feature_reduction = 1 - (len(selected_indices) / self.n_features)

        # Calculate fitness (minimize this)
        fitness = 1 - (self.alpha * accuracy + (1 - self.alpha) * feature_reduction)

        # Cache result
        if self.use_cache:
            self._cache[cache_key] = fitness

        return fitness

    def evaluate_final(self, feature_mask: np.ndarray) -> Dict[str, Any]:
        """
        Perform final evaluation on test set.

        Args:
            feature_mask: Binary feature mask

        Returns:
            Dictionary with detailed evaluation metrics
        """
        feature_mask = np.asarray(feature_mask).astype(int)
        selected_indices = np.where(feature_mask == 1)[0]

        if len(selected_indices) == 0:
            return {
                "train_accuracy": 0.0,
                "test_accuracy": 0.0,
                "cv_accuracy": 0.0,
                "n_selected": 0,
                "n_total": self.n_features,
                "reduction_ratio": 1.0,
                "fitness": 1.0,
            }

        X_train_selected = self.X_train[:, selected_indices]

        # Cross-validation accuracy
        clf = self._create_classifier()
        min_class_count = min(np.bincount(self.y_train))
        cv = min(self.cv_folds, min_class_count, len(self.y_train) // 2)
        cv = max(2, cv)

        cv_scores = cross_val_score(
            clf, X_train_selected, self.y_train, cv=cv, scoring="accuracy", n_jobs=-1
        )
        cv_accuracy = np.mean(cv_scores)

        # Train on full training set
        clf = self._create_classifier()
        clf.fit(X_train_selected, self.y_train)
        train_accuracy = clf.score(X_train_selected, self.y_train)

        # Test accuracy (if test set available)
        test_accuracy = 0.0
        if self.X_test is not None and self.y_test is not None:
            X_test_selected = self.X_test[:, selected_indices]
            test_accuracy = clf.score(X_test_selected, self.y_test)

        reduction_ratio = 1 - (len(selected_indices) / self.n_features)
        fitness = self.evaluate(feature_mask)

        return {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "cv_accuracy": cv_accuracy,
            "cv_std": np.std(cv_scores),
            "n_selected": len(selected_indices),
            "n_total": self.n_features,
            "reduction_ratio": reduction_ratio,
            "selected_features": selected_indices.tolist(),
            "fitness": fitness,
            "classifier": self.classifier_type,
            "classifier_name": CLASSIFIERS[self.classifier_type]["name"],
            "alpha": self.alpha,
        }

    def __call__(self, feature_mask: np.ndarray) -> float:
        """Make evaluator callable."""
        return self.evaluate(feature_mask)

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        return {
            "total_evaluations": self._evaluation_count,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "cache_hit_ratio": self._cache_hits
            / max(1, self._evaluation_count + self._cache_hits),
        }

    def clear_cache(self):
        """Clear the evaluation cache."""
        self._cache.clear()
        self._cache_hits = 0


class MultiClassifierEvaluator:
    """
    Evaluates feature subsets using multiple classifiers.

    Useful for comprehensive benchmarking.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        classifiers: List[str] = None,
        alpha: float = 0.99,
        cv_folds: int = 5,
        random_seed: int = 42,
    ):
        """
        Initialize multi-classifier evaluator.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            classifiers: List of classifier names (default: all)
            alpha: Trade-off parameter
            cv_folds: Number of CV folds
            random_seed: Random seed
        """
        if classifiers is None:
            classifiers = list(CLASSIFIERS.keys())

        self.classifiers = classifiers
        self.evaluators = {
            clf: FitnessEvaluator(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                classifier=clf,
                alpha=alpha,
                cv_folds=cv_folds,
                random_seed=random_seed,
            )
            for clf in classifiers
        }

    def evaluate_all(self, feature_mask: np.ndarray) -> Dict[str, float]:
        """Evaluate using all classifiers."""
        return {
            clf: evaluator.evaluate(feature_mask)
            for clf, evaluator in self.evaluators.items()
        }

    def evaluate_final_all(self, feature_mask: np.ndarray) -> Dict[str, Dict]:
        """Final evaluation with all classifiers."""
        return {
            clf: evaluator.evaluate_final(feature_mask)
            for clf, evaluator in self.evaluators.items()
        }

    def get_best_classifier(self, feature_mask: np.ndarray) -> str:
        """Get classifier with best performance for given features."""
        results = self.evaluate_all(feature_mask)
        return min(results, key=results.get)


def create_fitness_function(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    classifier: str = "knn",
    alpha: float = 0.99,
    **kwargs,
) -> FitnessEvaluator:
    """
    Create a fitness function for feature selection.

    Convenience function matching the interface expected by DRA optimizer.
    """
    return FitnessEvaluator(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        classifier=classifier,
        alpha=alpha,
        **kwargs,
    )


def get_available_classifiers() -> Dict[str, str]:
    """Get list of available classifiers with descriptions."""
    return {k: v["name"] for k, v in CLASSIFIERS.items()}
