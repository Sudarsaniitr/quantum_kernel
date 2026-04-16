"""
experiments/toy_problem.py
===========================
Toy problem from Eq.(11) of the paper.

Training states:
  |x1⟩ = (i/√2)|0⟩ + (1/√2)|1⟩   with label y1=0
  |x2⟩ = (i/√2)|0⟩ - (1/√2)|1⟩   with label y2=1

Test state:
  |x̃(θ)⟩ = cos(θ/2)|0⟩ + i sin(θ/2)|1⟩

Hadamard baseline fails because Re⟨x̃|x_m⟩ = 0 for all θ here.
Swap-test succeeds by using |⟨x̃|x_m⟩|².
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Training data definition (paper Eq. 11)
# ---------------------------------------------------------------------------


def get_training_data() -> Tuple[List[np.ndarray], List[int]]:
    """Return the two training states and labels from Eq.(11)."""
    x1 = np.array([1j / np.sqrt(2), 1.0 / np.sqrt(2)], dtype=complex)
    x2 = np.array([1j / np.sqrt(2), -1.0 / np.sqrt(2)], dtype=complex)
    return [x1, x2], [0, 1]


def get_test_state(theta: float) -> np.ndarray:
    """Return |x̃(θ)⟩ = cos(θ/2)|0⟩ + i sin(θ/2)|1⟩."""
    return np.array([np.cos(theta / 2), 1j * np.sin(theta / 2)], dtype=complex)


def get_theta_range(
    n_points: int = 63,
    start: float = 0.0,
    end: float = 2 * np.pi,
) -> np.ndarray:
    """Return theta values over [start, end] (paper-like sweep uses ~63 points)."""
    return np.linspace(start, end, n_points)


def get_test_states(thetas: np.ndarray) -> List[np.ndarray]:
    """Return a list of test states for each θ value."""
    return [get_test_state(theta) for theta in thetas]


# ---------------------------------------------------------------------------
# Analytical / ground truth
# ---------------------------------------------------------------------------


def analytical_swap_kernel(
    theta: float,
    n_copies: int = 1,
    w1: float = 0.5,
    w2: float = 0.5,
) -> float:
    """
    Analytical swap-test expectation (Eq. 12):

      ⟨σz(a)σz(l)⟩ = w1|⟨x̃|x1⟩|^(2n) - w2|⟨x̃|x2⟩|^(2n)
    """
    x_test = get_test_state(theta)
    x_train, _ = get_training_data()

    f1 = abs(np.conj(x_test) @ x_train[0]) ** (2 * n_copies)
    f2 = abs(np.conj(x_test) @ x_train[1]) ** (2 * n_copies)

    return float(w1 * f1 - w2 * f2)


def analytical_hadamard_kernel(
    theta: float,
    w1: float = 0.5,
    w2: float = 0.5,
) -> float:
    """
    Analytical Hadamard classifier kernel (Eq. 13):

      ⟨σz(a)σz(l)⟩_H = w1 Re⟨x̃|x1⟩ - w2 Re⟨x̃|x2⟩

    For this toy problem it is identically zero.
    """
    x_test = get_test_state(theta)
    x_train, _ = get_training_data()

    overlap1 = np.conj(x_test) @ x_train[0]
    overlap2 = np.conj(x_test) @ x_train[1]

    return float(w1 * np.real(overlap1) - w2 * np.real(overlap2))


def true_classification(theta: float) -> int:
    """
    Ground truth oracle label from Eq.(11):

      class 0 if |⟨x̃|x1⟩|² ≥ |⟨x̃|x2⟩|², else class 1.

    Boundaries occur at θ = kπ (k integer), where fidelities are equal.
    Returns -1 on exact boundary.
    """
    k = analytical_swap_kernel(theta, n_copies=1)
    if abs(k) < 1e-12:
        return -1
    return 0 if k > 0 else 1


def compute_classification_accuracy(thetas: np.ndarray, predicted: List[int]) -> float:
    """Compute fraction of correctly classified test points (boundary counts as correct)."""
    correct = sum(
        1
        for theta, pred in zip(thetas, predicted)
        if pred == true_classification(theta) or pred == -1
    )
    return float(correct / len(thetas))


# ---------------------------------------------------------------------------
# Inner product analysis (why Hadamard fails)
# ---------------------------------------------------------------------------


def analyse_inner_products(theta: float) -> dict:
    """Return inner product diagnostics for a given θ."""
    x_test = get_test_state(theta)
    x_train, _ = get_training_data()

    ip1 = np.conj(x_test) @ x_train[0]
    ip2 = np.conj(x_test) @ x_train[1]

    return {
        "theta": theta,
        "inner_prod_1": ip1,
        "inner_prod_2": ip2,
        "real_part_1": np.real(ip1),
        "real_part_2": np.real(ip2),
        "fidelity_1": abs(ip1) ** 2,
        "fidelity_2": abs(ip2) ** 2,
        "hadamard_kernel": np.real(ip1) / 2 - np.real(ip2) / 2,
        "swap_kernel": abs(ip1) ** 2 / 2 - abs(ip2) ** 2 / 2,
        "true_label": true_classification(theta),
    }
