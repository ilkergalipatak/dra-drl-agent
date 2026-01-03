#!/usr/bin/env python3
"""Quick test script for datasets and classifiers."""

from src.data.dataset_loader import DatasetLoader
from src.evaluation.fitness_evaluator import FitnessEvaluator, get_available_classifiers
import numpy as np

print("=" * 60)
print("TESTING DATASETS AND CLASSIFIERS")
print("=" * 60)

print("\nAvailable classifiers:", get_available_classifiers())

# Load datasets
loader = DatasetLoader()
print("\nAvailable datasets:", len(loader.AVAILABLE_DATASETS))
for ds in loader.AVAILABLE_DATASETS:
    print(f"  - {ds}")

print("\n" + "=" * 60)
print("Testing classifiers on heart_failure dataset")
print("=" * 60)

data = loader.load("heart_failure")
print(f"\nDataset: heart_failure")
print(f"  Samples: {data['n_samples']}")
print(f"  Features: {data['n_features']}")
print(f"  Classes: {data['n_classes']}")

# Test each classifier with all features
mask = np.ones(data["n_features"], dtype=int)

print("\nClassifier Performance (all features):")
print("-" * 50)

for clf_name in ["knn", "svm", "rf", "lr"]:
    evaluator = FitnessEvaluator(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_test=data["X_test"],
        y_test=data["y_test"],
        classifier=clf_name,
    )
    result = evaluator.evaluate_final(mask)
    print(
        f"  {clf_name.upper():3s}: Test Acc = {result['test_accuracy']:.4f}, "
        f"CV Acc = {result['cv_accuracy']:.4f}"
    )

print("\n" + "=" * 60)
print("Testing dataset loading for all custom datasets")
print("=" * 60 + "\n")

for ds_name in loader.AVAILABLE_DATASETS[:10]:  # First 10
    try:
        data = loader.load(ds_name)
        print(
            f"OK  {ds_name}: {data['n_samples']} samples, "
            f"{data['n_features']} features, {data['n_classes']} classes"
        )
    except Exception as e:
        print(f"ERR {ds_name}: {e}")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED!")
print("=" * 60)
