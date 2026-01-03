"""
Transfer Functions for Binary Optimization

These functions convert continuous values to binary (0/1) for feature selection.
Two main families: S-shaped (sigmoid-based) and V-shaped (tanh-based).
"""

import numpy as np
from typing import Callable


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Standard sigmoid function with overflow protection."""
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def s_shaped_v1(x: np.ndarray) -> np.ndarray:
    """S-shaped transfer function V1 (standard sigmoid)."""
    return sigmoid(x)


def s_shaped_v2(x: np.ndarray) -> np.ndarray:
    """S-shaped transfer function V2."""
    return sigmoid(x / 2)


def s_shaped_v3(x: np.ndarray) -> np.ndarray:
    """S-shaped transfer function V3."""
    return sigmoid(x / 3)


def s_shaped_v4(x: np.ndarray) -> np.ndarray:
    """S-shaped transfer function V4."""
    return sigmoid(x / 4)


def v_shaped_v1(x: np.ndarray) -> np.ndarray:
    """V-shaped transfer function V1 (absolute tanh)."""
    return np.abs(np.tanh(x))


def v_shaped_v2(x: np.ndarray) -> np.ndarray:
    """V-shaped transfer function V2."""
    return np.abs(x / np.sqrt(1 + x**2))


def v_shaped_v3(x: np.ndarray) -> np.ndarray:
    """V-shaped transfer function V3."""
    return np.abs(np.tanh(x / 2))


def v_shaped_v4(x: np.ndarray) -> np.ndarray:
    """V-shaped transfer function V4."""
    return np.abs(x / np.sqrt(3 + x**2))


def binary_conversion_s(
    x: np.ndarray, transfer_func: Callable = s_shaped_v1
) -> np.ndarray:
    """
    Convert continuous values to binary using S-shaped transfer.

    For S-shaped functions, a random threshold determines conversion:
    - If transfer(x) > rand, then 1
    - Else 0
    """
    probabilities = transfer_func(x)
    random_values = np.random.random(x.shape)
    return (probabilities > random_values).astype(int)


def binary_conversion_v(
    x: np.ndarray, x_prev: np.ndarray, transfer_func: Callable = v_shaped_v1
) -> np.ndarray:
    """
    Convert continuous values to binary using V-shaped transfer.

    For V-shaped functions, the conversion depends on previous position:
    - If rand < transfer(x), flip the bit from previous
    - Else keep the same
    """
    probabilities = transfer_func(x)
    random_values = np.random.random(x.shape)
    flip_mask = random_values < probabilities
    result = np.where(flip_mask, 1 - x_prev, x_prev)
    return result.astype(int)


class TransferFunction:
    """Wrapper class for transfer functions with utility methods."""

    FUNCTIONS = {
        "s1": s_shaped_v1,
        "s2": s_shaped_v2,
        "s3": s_shaped_v3,
        "s4": s_shaped_v4,
        "v1": v_shaped_v1,
        "v2": v_shaped_v2,
        "v3": v_shaped_v3,
        "v4": v_shaped_v4,
    }

    def __init__(self, name: str = "s1"):
        """
        Initialize transfer function.

        Args:
            name: Transfer function name ('s1', 's2', 's3', 's4', 'v1', 'v2', 'v3', 'v4')
        """
        if name not in self.FUNCTIONS:
            raise ValueError(
                f"Unknown transfer function: {name}. "
                f"Available: {list(self.FUNCTIONS.keys())}"
            )

        self.name = name
        self.func = self.FUNCTIONS[name]
        self.is_s_shaped = name.startswith("s")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply transfer function."""
        return self.func(x)

    def to_binary(self, x: np.ndarray, x_prev: np.ndarray = None) -> np.ndarray:
        """
        Convert continuous values to binary.

        Args:
            x: Continuous values
            x_prev: Previous binary solution (required for V-shaped)
        """
        if self.is_s_shaped:
            return binary_conversion_s(x, self.func)
        else:
            if x_prev is None:
                x_prev = np.zeros_like(x)
            return binary_conversion_v(x, x_prev, self.func)


def ensure_at_least_one_feature(binary_solution: np.ndarray) -> np.ndarray:
    """
    Ensure at least one feature is selected.

    If all features are 0, randomly select one feature.
    """
    result = binary_solution.copy()
    if np.sum(result) == 0:
        random_idx = np.random.randint(0, len(result))
        result[random_idx] = 1
    return result
