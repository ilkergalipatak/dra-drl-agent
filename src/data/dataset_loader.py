"""
Dataset Loader for Feature Selection Benchmarks

Loads and preprocesses custom CSV datasets and UCI benchmark datasets.
Supports both local CSV files and sklearn built-in datasets.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List, Literal
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits


# Dataset configuration: name -> (filename, target_column, drop_columns)
CUSTOM_DATASETS = {
    "student_performance": {
        "file": "ai_impact_student_performance_dataset.csv",
        "target": "performance_category",
        "drop_cols": ["student_id"],
        "description": "AI Impact on Student Performance",
    },
    "breast_cancer_custom": {
        "file": "breast_cancer_dataset.csv",
        "target": "diagnosis",
        "drop_cols": ["id", "Unnamed: 32"],
        "description": "Wisconsin Breast Cancer (Custom)",
    },
    "credit_card_fraud": {
        "file": "credit_card_fraud_10k.csv",
        "target": "is_fraud",
        "drop_cols": [],
        "description": "Credit Card Fraud Detection",
    },
    "customer_churn": {
        "file": "ecommerce_customer_churn_dataset.csv",
        "target": "Churned",
        "drop_cols": ["Signup_Quarter"],
        "description": "E-commerce Customer Churn",
    },
    "healthcare": {
        "file": "healthcare_dataset.csv",
        "target": "Test Results",
        "drop_cols": [
            "Name",
            "Date of Admission",
            "Discharge Date",
            "Doctor",
            "Hospital",
        ],
        "description": "Healthcare Test Results",
    },
    "heart_failure": {
        "file": "heart_failure_clinical_records_dataset.csv",
        "target": "DEATH_EVENT",
        "drop_cols": [],
        "description": "Heart Failure Clinical Records",
    },
    "marketing_campaign": {
        "file": "marketing_campaign.csv",
        "target": "Response",
        "drop_cols": ["ID", "Dt_Customer", "Z_CostContact", "Z_Revenue"],
        "separator": "\t",
        "description": "Marketing Campaign Response",
    },
    "mobile_price": {
        "file": "mobile_price_classification.csv",
        "target": "price_range",
        "drop_cols": [],
        "description": "Mobile Price Classification",
    },
    "credit_risk": {
        "file": "synthetic_credit_risk.csv",
        "target": "target",
        "drop_cols": [],
        "description": "Synthetic Credit Risk",
    },
}

# Sklearn built-in datasets
SKLEARN_DATASETS = {
    "breast_cancer": {
        "loader": load_breast_cancer,
        "description": "Wisconsin Breast Cancer (sklearn)",
    },
    "wine": {
        "loader": load_wine,
        "description": "Wine Recognition",
    },
    "iris": {
        "loader": load_iris,
        "description": "Iris Species",
    },
    "digits": {
        "loader": load_digits,
        "description": "Handwritten Digits",
    },
}


class DatasetLoader:
    """
    Unified dataset loader for feature selection benchmarks.

    Supports:
    - Custom CSV datasets from the datasets folder
    - Sklearn built-in datasets
    """

    def __init__(
        self,
        datasets_dir: Optional[str] = None,
        test_size: float = 0.3,
        normalize: bool = True,
        random_seed: int = 42,
        max_samples: int = 10000,  # Limit large datasets
    ):
        """
        Initialize dataset loader.

        Args:
            datasets_dir: Directory containing CSV datasets
            test_size: Proportion of data for testing
            normalize: Whether to standardize features
            random_seed: Random seed for reproducibility
            max_samples: Maximum samples to use (for large datasets)
        """
        if datasets_dir is None:
            # Default to datasets folder in project root
            datasets_dir = Path(__file__).parent.parent.parent / "datasets"
        self.datasets_dir = Path(datasets_dir)

        self.test_size = test_size
        self.normalize = normalize
        self.random_seed = random_seed
        self.max_samples = max_samples
        self.scaler = StandardScaler() if normalize else None
        self.label_encoder = LabelEncoder()

    @property
    def AVAILABLE_DATASETS(self) -> List[str]:
        """List all available datasets."""
        return list(CUSTOM_DATASETS.keys()) + list(SKLEARN_DATASETS.keys())

    def _load_csv_dataset(self, name: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load a custom CSV dataset."""
        config = CUSTOM_DATASETS[name]
        file_path = self.datasets_dir / config["file"]

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Read CSV with appropriate separator
        separator = config.get("separator", ",")
        df = pd.read_csv(file_path, sep=separator)

        # Drop specified columns
        drop_cols = config.get("drop_cols", [])
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Get target column
        target_col = config["target"]
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in {name}")

        y = df[target_col]
        X = df.drop(columns=[target_col])

        # Handle categorical features
        categorical_cols = X.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        # Handle missing values
        X = X.fillna(X.median())

        # Convert to numpy
        X = X.values.astype(np.float64)

        # Encode target
        y = self.label_encoder.fit_transform(y.astype(str))

        info = {
            "name": name,
            "description": config.get("description", name),
            "source": "custom_csv",
        }

        return X, y, info

    def _load_sklearn_dataset(self, name: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load a sklearn built-in dataset."""
        config = SKLEARN_DATASETS[name]
        data = config["loader"]()

        X = data.data
        y = data.target

        info = {
            "name": name,
            "description": config.get("description", name),
            "source": "sklearn",
        }

        return X, y, info

    def load(self, dataset_name: str) -> Dict:
        """
        Load and preprocess a dataset.

        Args:
            dataset_name: Name of the dataset to load

        Returns:
            Dictionary with X_train, X_test, y_train, y_test, and metadata
        """
        # Determine source and load
        if dataset_name in CUSTOM_DATASETS:
            X, y, info = self._load_csv_dataset(dataset_name)
        elif dataset_name in SKLEARN_DATASETS:
            X, y, info = self._load_sklearn_dataset(dataset_name)
        else:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {self.AVAILABLE_DATASETS}"
            )

        # Subsample if too large
        if len(X) > self.max_samples:
            indices = np.random.RandomState(self.random_seed).choice(
                len(X), self.max_samples, replace=False
            )
            X, y = X[indices], y[indices]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_seed, stratify=y
        )

        # Normalize
        if self.normalize:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "n_classes": len(np.unique(y)),
            **info,
        }

    def load_all(self, include_sklearn: bool = True) -> Dict[str, Dict]:
        """
        Load all available datasets.

        Args:
            include_sklearn: Whether to include sklearn datasets

        Returns:
            Dictionary mapping dataset names to their data
        """
        datasets = list(CUSTOM_DATASETS.keys())
        if include_sklearn:
            datasets += list(SKLEARN_DATASETS.keys())

        results = {}
        for name in datasets:
            try:
                results[name] = self.load(name)
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")

        return results

    @staticmethod
    def list_datasets() -> List[str]:
        """List all available datasets."""
        return list(CUSTOM_DATASETS.keys()) + list(SKLEARN_DATASETS.keys())

    @staticmethod
    def get_dataset_info(dataset_name: str = None) -> Dict:
        """Get information about dataset(s)."""
        if dataset_name:
            if dataset_name in CUSTOM_DATASETS:
                return CUSTOM_DATASETS[dataset_name]
            elif dataset_name in SKLEARN_DATASETS:
                return SKLEARN_DATASETS[dataset_name]
            return {}

        return {**CUSTOM_DATASETS, **SKLEARN_DATASETS}


def load_dataset(
    name: str,
    datasets_dir: Optional[str] = None,
    test_size: float = 0.3,
    normalize: bool = True,
    random_seed: int = 42,
) -> Dict:
    """
    Convenience function to load a single dataset.

    Args:
        name: Dataset name
        datasets_dir: Directory containing CSV datasets
        test_size: Test set proportion
        normalize: Whether to normalize features
        random_seed: Random seed

    Returns:
        Dataset dictionary
    """
    loader = DatasetLoader(
        datasets_dir=datasets_dir,
        test_size=test_size,
        normalize=normalize,
        random_seed=random_seed,
    )
    return loader.load(name)
