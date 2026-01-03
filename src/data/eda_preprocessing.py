"""
Exploratory Data Analysis (EDA) and Preprocessing Module

Provides comprehensive EDA and preprocessing for feature selection datasets:
- Dataset statistics and profiling
- Missing value analysis and handling
- Categorical encoding
- Outlier detection
- Correlation analysis
- Feature scaling
- Report generation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


@dataclass
class DatasetProfile:
    """Profile information about a dataset."""

    name: str
    n_samples: int
    n_features: int
    n_classes: int
    class_distribution: Dict[str, int]
    feature_types: Dict[str, str]  # feature_name -> type (numeric, categorical)
    missing_values: Dict[str, int]
    missing_percentage: float
    numeric_stats: Dict[str, Dict[str, float]]  # feature -> {mean, std, min, max, ...}
    categorical_stats: Dict[str, Dict[str, int]]  # feature -> {value: count}
    correlations: Optional[np.ndarray] = None
    outliers: Dict[str, int] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class DatasetEDA:
    """
    Exploratory Data Analysis for feature selection datasets.

    Provides comprehensive analysis and preprocessing recommendations.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize EDA analyzer.

        Args:
            verbose: Print analysis results
        """
        self.verbose = verbose

    def analyze(
        self, df: pd.DataFrame, target_col: str, dataset_name: str = "dataset"
    ) -> DatasetProfile:
        """
        Perform comprehensive EDA on a dataset.

        Args:
            df: Pandas DataFrame
            target_col: Name of target column
            dataset_name: Name for identification

        Returns:
            DatasetProfile with all analysis results
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"EDA: {dataset_name}")
            print(f"{'='*60}")

        # Basic info
        n_samples, n_cols = df.shape
        n_features = n_cols - 1  # Exclude target

        # Separate features and target
        X = df.drop(columns=[target_col]) if target_col in df.columns else df
        y = df[target_col] if target_col in df.columns else None

        # Class distribution
        class_dist = {}
        n_classes = 0
        if y is not None:
            class_dist = y.value_counts().to_dict()
            n_classes = len(class_dist)

        # Feature types
        feature_types = {}
        for col in X.columns:
            if X[col].dtype in ["object", "category", "bool"]:
                feature_types[col] = "categorical"
            else:
                feature_types[col] = "numeric"

        # Missing values
        missing = X.isnull().sum().to_dict()
        missing_pct = (X.isnull().sum().sum() / (n_samples * n_features)) * 100

        # Numeric statistics
        numeric_cols = [c for c, t in feature_types.items() if t == "numeric"]
        numeric_stats = {}
        for col in numeric_cols:
            col_data = X[col].dropna()
            if len(col_data) > 0:
                numeric_stats[col] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "median": float(col_data.median()),
                    "skewness": float(col_data.skew()) if len(col_data) > 2 else 0,
                    "kurtosis": float(col_data.kurtosis()) if len(col_data) > 3 else 0,
                }

        # Categorical statistics
        cat_cols = [c for c, t in feature_types.items() if t == "categorical"]
        categorical_stats = {}
        for col in cat_cols:
            categorical_stats[col] = X[col].value_counts().to_dict()

        # Outlier detection (IQR method)
        outliers = {}
        for col in numeric_cols:
            col_data = X[col].dropna()
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                n_outliers = ((col_data < lower) | (col_data > upper)).sum()
                if n_outliers > 0:
                    outliers[col] = int(n_outliers)

        # Correlations (for numeric features only)
        correlations = None
        if len(numeric_cols) > 1:
            try:
                correlations = X[numeric_cols].corr().values
            except:
                pass

        # Generate recommendations
        recommendations = self._generate_recommendations(
            missing,
            missing_pct,
            outliers,
            feature_types,
            numeric_stats,
            class_dist,
            n_samples,
        )

        profile = DatasetProfile(
            name=dataset_name,
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            class_distribution=class_dist,
            feature_types=feature_types,
            missing_values=missing,
            missing_percentage=missing_pct,
            numeric_stats=numeric_stats,
            categorical_stats=categorical_stats,
            correlations=correlations,
            outliers=outliers,
            recommendations=recommendations,
        )

        if self.verbose:
            self._print_profile(profile)

        return profile

    def _generate_recommendations(
        self,
        missing: Dict,
        missing_pct: float,
        outliers: Dict,
        feature_types: Dict,
        numeric_stats: Dict,
        class_dist: Dict,
        n_samples: int,
    ) -> List[str]:
        """Generate preprocessing recommendations."""
        recommendations = []

        # Missing values
        if missing_pct > 0:
            if missing_pct > 20:
                recommendations.append(
                    f"⚠️ High missing rate ({missing_pct:.1f}%). Consider imputation or dropping features."
                )
            else:
                recommendations.append(
                    f"ℹ️ {missing_pct:.1f}% missing values. Will use median/mode imputation."
                )

        # Outliers
        total_outliers = sum(outliers.values())
        if total_outliers > 0:
            pct_outliers = (total_outliers / (n_samples * len(feature_types))) * 100
            if pct_outliers > 5:
                recommendations.append(
                    f"⚠️ {len(outliers)} features have outliers ({pct_outliers:.1f}% total). Consider clipping or robust scaling."
                )

        # Class imbalance
        if class_dist:
            max_class = max(class_dist.values())
            min_class = min(class_dist.values())
            imbalance_ratio = max_class / min_class if min_class > 0 else float("inf")
            if imbalance_ratio > 3:
                recommendations.append(
                    f"⚠️ Class imbalance detected (ratio {imbalance_ratio:.1f}:1). Consider stratified sampling."
                )

        # High cardinality categoricals
        cat_cols = [c for c, t in feature_types.items() if t == "categorical"]
        if cat_cols:
            recommendations.append(
                f"ℹ️ {len(cat_cols)} categorical features will be label encoded."
            )

        # Skewed distributions
        skewed = []
        for col, stats in numeric_stats.items():
            if abs(stats.get("skewness", 0)) > 2:
                skewed.append(col)
        if skewed:
            recommendations.append(
                f"ℹ️ {len(skewed)} features are highly skewed. Consider log transformation."
            )

        if not recommendations:
            recommendations.append("✅ Dataset looks clean and ready for processing.")

        return recommendations

    def _print_profile(self, profile: DatasetProfile):
        """Print profile summary."""
        print(f"\n📊 Basic Statistics:")
        print(f"   Samples: {profile.n_samples}")
        print(f"   Features: {profile.n_features}")
        print(f"   Classes: {profile.n_classes}")

        print(f"\n📈 Class Distribution:")
        for cls, count in profile.class_distribution.items():
            pct = count / profile.n_samples * 100
            print(f"   {cls}: {count} ({pct:.1f}%)")

        print(f"\n🔧 Feature Types:")
        numeric = sum(1 for t in profile.feature_types.values() if t == "numeric")
        categorical = len(profile.feature_types) - numeric
        print(f"   Numeric: {numeric}")
        print(f"   Categorical: {categorical}")

        if profile.missing_percentage > 0:
            print(f"\n❓ Missing Values: {profile.missing_percentage:.2f}%")

        if profile.outliers:
            print(f"\n🔍 Outliers detected in {len(profile.outliers)} features")

        print(f"\n💡 Recommendations:")
        for rec in profile.recommendations:
            print(f"   {rec}")


class DataPreprocessor:
    """
    Preprocessing pipeline for feature selection datasets.

    Handles:
    - Missing value imputation
    - Categorical encoding
    - Outlier handling
    - Feature scaling
    """

    def __init__(
        self,
        handle_missing: str = "median",  # median, mean, mode, drop
        handle_outliers: str = "clip",  # clip, remove, none
        scaling: str = "standard",  # standard, minmax, none
        encode_categorical: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize preprocessor.

        Args:
            handle_missing: Strategy for missing values
            handle_outliers: Strategy for outliers
            scaling: Scaling method
            encode_categorical: Whether to encode categorical features
            verbose: Print processing info
        """
        self.handle_missing = handle_missing
        self.handle_outliers = handle_outliers
        self.scaling = scaling
        self.encode_categorical = encode_categorical
        self.verbose = verbose

        # Fitted transformers
        self._imputers: Dict[str, SimpleImputer] = {}
        self._encoders: Dict[str, LabelEncoder] = {}
        self._scaler: Optional[StandardScaler] = None
        self._outlier_bounds: Dict[str, Tuple[float, float]] = {}
        self._feature_names: List[str] = []

    def fit_transform(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Fit preprocessor and transform data.

        Args:
            df: Input DataFrame
            target_col: Target column name

        Returns:
            Tuple of (X, y, feature_names)
        """
        df = df.copy()

        # Separate features and target
        y_raw = df[target_col] if target_col in df.columns else None
        X_df = df.drop(columns=[target_col]) if target_col in df.columns else df

        # Store original feature names
        self._feature_names = list(X_df.columns)

        if self.verbose:
            print(f"\n🔄 Preprocessing ({len(self._feature_names)} features)...")

        # 1. Handle missing values
        X_df = self._handle_missing_values(X_df, fit=True)

        # 2. Encode categorical features
        X_df = self._encode_categoricals(X_df, fit=True)

        # 3. Handle outliers
        X_df = self._handle_outliers_df(X_df, fit=True)

        # Convert to numpy
        X = X_df.values.astype(np.float64)

        # 4. Scale features
        X = self._scale_features(X, fit=True)

        # Encode target
        y = None
        if y_raw is not None:
            self._target_encoder = LabelEncoder()
            y = self._target_encoder.fit_transform(y_raw.astype(str))

        if self.verbose:
            print(f"   ✅ Preprocessing complete. Shape: {X.shape}")

        return X, y, self._feature_names

    def transform(
        self, df: pd.DataFrame, target_col: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform new data using fitted preprocessor.

        Args:
            df: Input DataFrame
            target_col: Target column name (optional)

        Returns:
            Tuple of (X, y) or just X if no target
        """
        df = df.copy()

        y_raw = None
        if target_col and target_col in df.columns:
            y_raw = df[target_col]
            df = df.drop(columns=[target_col])

        # Apply same transformations
        df = self._handle_missing_values(df, fit=False)
        df = self._encode_categoricals(df, fit=False)
        df = self._handle_outliers_df(df, fit=False)

        X = df.values.astype(np.float64)
        X = self._scale_features(X, fit=False)

        y = None
        if y_raw is not None:
            y = self._target_encoder.transform(y_raw.astype(str))

        return X, y

    def _handle_missing_values(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Handle missing values."""
        if self.handle_missing == "drop":
            return df.dropna()

        for col in df.columns:
            if df[col].isnull().any():
                if fit:
                    if df[col].dtype in ["object", "category"]:
                        strategy = "most_frequent"
                    elif self.handle_missing == "median":
                        strategy = "median"
                    else:
                        strategy = "mean"

                    imputer = SimpleImputer(strategy=strategy)
                    df[col] = imputer.fit_transform(df[[col]]).ravel()
                    self._imputers[col] = imputer
                else:
                    if col in self._imputers:
                        df[col] = self._imputers[col].transform(df[[col]]).ravel()

        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Encode categorical features."""
        if not self.encode_categorical:
            return df

        for col in df.columns:
            if df[col].dtype in ["object", "category", "bool"]:
                if fit:
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col].astype(str))
                    self._encoders[col] = encoder
                else:
                    if col in self._encoders:
                        # Handle unseen categories
                        df[col] = df[col].astype(str)
                        known = set(self._encoders[col].classes_)
                        df[col] = df[col].apply(
                            lambda x: (
                                x if x in known else self._encoders[col].classes_[0]
                            )
                        )
                        df[col] = self._encoders[col].transform(df[col])

        return df

    def _handle_outliers_df(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Handle outliers in DataFrame."""
        if self.handle_outliers == "none":
            return df

        for col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                if fit:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    self._outlier_bounds[col] = (lower, upper)

                if col in self._outlier_bounds:
                    lower, upper = self._outlier_bounds[col]
                    if self.handle_outliers == "clip":
                        df[col] = df[col].clip(lower, upper)
                    elif self.handle_outliers == "remove":
                        df = df[(df[col] >= lower) & (df[col] <= upper)]

        return df

    def _scale_features(self, X: np.ndarray, fit: bool) -> np.ndarray:
        """Scale features."""
        if self.scaling == "none":
            return X

        if fit:
            if self.scaling == "standard":
                self._scaler = StandardScaler()
            elif self.scaling == "minmax":
                self._scaler = MinMaxScaler()
            X = self._scaler.fit_transform(X)
        else:
            if self._scaler is not None:
                X = self._scaler.transform(X)

        return X

    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing."""
        return self._feature_names


def run_eda_on_all_datasets(datasets_dir: str, output_dir: Optional[str] = None):
    """
    Run EDA on all datasets in a directory.

    Args:
        datasets_dir: Directory containing CSV datasets
        output_dir: Directory to save reports (optional)
    """
    from .dataset_loader import CUSTOM_DATASETS

    datasets_path = Path(datasets_dir)
    eda = DatasetEDA(verbose=True)

    profiles = {}

    print("=" * 70)
    print("EXPLORATORY DATA ANALYSIS - ALL DATASETS")
    print("=" * 70)

    for name, config in CUSTOM_DATASETS.items():
        file_path = datasets_path / config["file"]
        if file_path.exists():
            try:
                sep = config.get("separator", ",")
                df = pd.read_csv(file_path, sep=sep)

                # Drop specified columns
                for col in config.get("drop_cols", []):
                    if col in df.columns:
                        df = df.drop(columns=[col])

                target = config["target"]
                profiles[name] = eda.analyze(df, target, name)

            except Exception as e:
                print(f"\n❌ Error analyzing {name}: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"\n{'Dataset':<25} {'Samples':<10} {'Features':<10} {'Classes':<10} {'Missing%':<10}"
    )
    print("-" * 70)

    for name, profile in profiles.items():
        print(
            f"{name:<25} {profile.n_samples:<10} {profile.n_features:<10} "
            f"{profile.n_classes:<10} {profile.missing_percentage:<10.2f}"
        )

    return profiles
