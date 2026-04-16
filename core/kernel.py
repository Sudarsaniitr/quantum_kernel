"""
core/kernel.py
==============
Classical computation of the quantum kernels described in the paper.

Key equations:
  - Eq.(6):  Hadamard kernel  = Σ_m (-1)^y_m w_m Re⟨x̃|x_m⟩
  - Eq.(9):  Swap-test kernel = Σ_m (-1)^y_m w_m |⟨x̃|x_m⟩|^(2n)
  - Eq.(14): Limit n→∞ gives a highly localized kernel response.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Kernel functions (analytical / classical computation)
# ---------------------------------------------------------------------------


def hadamard_kernel(
    x_test: np.ndarray,
    x_train: List[np.ndarray],
    labels: List[int],
    weights: Optional[List[float]] = None,
) -> float:
    """
    Compute the Hadamard classifier kernel value (Eq. 6).

    K_H(x̃) = Σ_m (-1)^y_m · w_m · Re⟨x̃|x_m⟩
    """
    M = len(x_train)
    if M == 0:
        raise ValueError("x_train must contain at least one state.")
    if len(labels) != M:
        raise ValueError("labels and x_train must have equal length.")

    if weights is None:
        weights = [1.0 / M] * M

    result = 0.0
    for m in range(M):
        overlap = np.conj(x_test) @ x_train[m]
        result += ((-1) ** labels[m]) * weights[m] * float(np.real(overlap))
    return float(result)


def swap_test_kernel(
    x_test: np.ndarray,
    x_train: List[np.ndarray],
    labels: List[int],
    weights: Optional[List[float]] = None,
    n_copies: int = 1,
) -> float:
    """
    Compute the swap-test classifier kernel value (Eq. 9).

    K_S(x̃) = Σ_m (-1)^y_m · w_m · |⟨x̃|x_m⟩|^(2n)
    """
    if n_copies < 1:
        raise ValueError("n_copies must be >= 1.")

    M = len(x_train)
    if M == 0:
        raise ValueError("x_train must contain at least one state.")
    if len(labels) != M:
        raise ValueError("labels and x_train must have equal length.")

    if weights is None:
        weights = [1.0 / M] * M

    result = 0.0
    for m in range(M):
        overlap = np.conj(x_test) @ x_train[m]
        fidelity_n = abs(overlap) ** (2 * n_copies)
        result += ((-1) ** labels[m]) * weights[m] * float(fidelity_n)
    return float(result)


def classify(kernel_value: float) -> int:
    """
    Apply Eq.(7)-style sign decision rule.

    Returns
    -------
    0 if kernel_value > 0
    1 if kernel_value < 0
    raises ValueError if kernel_value == 0 (exact boundary)
    """
    if kernel_value > 0:
        return 0
    if kernel_value < 0:
        return 1
    raise ValueError("Kernel value is exactly zero — ambiguous classification.")


def true_label(
    x_test: np.ndarray,
    x_train: List[np.ndarray],
    labels: List[int],
    power: int = 2,
) -> int:
    """
    Ground-truth oracle for binary toy-style setup:
      class 0 if Σ_{y=0}|⟨x|x_m⟩|^power >= Σ_{y=1}|⟨x|x_m⟩|^power else class 1.
    """
    if power <= 0:
        raise ValueError("power must be positive.")

    class0 = [x_train[m] for m in range(len(x_train)) if labels[m] == 0]
    class1 = [x_train[m] for m in range(len(x_train)) if labels[m] == 1]

    f0 = sum(abs(np.conj(x_test) @ xm) ** power for xm in class0)
    f1 = sum(abs(np.conj(x_test) @ xm) ** power for xm in class1)

    return 0 if f0 >= f1 else 1


# ---------------------------------------------------------------------------
# Kernel matrix construction
# ---------------------------------------------------------------------------


def kernel_matrix(states: List[np.ndarray], n_copies: int = 1) -> np.ndarray:
    """
    Compute N×N Gram matrix K[i,j] = |⟨x_i|x_j⟩|^(2n).
    """
    if n_copies < 1:
        raise ValueError("n_copies must be >= 1.")

    N = len(states)
    K = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            overlap = np.conj(states[i]) @ states[j]
            K[i, j] = abs(overlap) ** (2 * n_copies)
    return K


def helstrom_operator(
    x_train: List[np.ndarray],
    labels: List[int],
    weights: Optional[List[float]] = None,
    n_copies: int = 1,
) -> np.ndarray:
    """
    Construct Helstrom operator (Eq. 16):

      A = Σ_{y_m=0} w_m |x_m⟩⟨x_m|^⊗n - Σ_{y_m=1} w_m |x_m⟩⟨x_m|^⊗n
    """
    if n_copies < 1:
        raise ValueError("n_copies must be >= 1.")

    M = len(x_train)
    if M == 0:
        raise ValueError("x_train must contain at least one state.")
    if len(labels) != M:
        raise ValueError("labels and x_train must have equal length.")

    if weights is None:
        weights = [1.0 / M] * M

    d = len(x_train[0])
    d_n = d**n_copies

    A = np.zeros((d_n, d_n), dtype=complex)
    for m in range(M):
        xm_n = x_train[m].copy()
        for _ in range(n_copies - 1):
            xm_n = np.kron(xm_n, x_train[m])

        rho_n = np.outer(xm_n, np.conj(xm_n))
        A += ((-1) ** labels[m]) * weights[m] * rho_n
    return A


def helstrom_expectation(
    x_test: np.ndarray,
    helstrom_op: np.ndarray,
    n_copies: int = 1,
) -> float:
    """
    Compute ⟨x̃^⊗n | A | x̃^⊗n⟩ (Eq. 17).
    """
    if n_copies < 1:
        raise ValueError("n_copies must be >= 1.")

    x_test_n = x_test.copy()
    for _ in range(n_copies - 1):
        x_test_n = np.kron(x_test_n, x_test)

    return float(np.real(np.conj(x_test_n) @ helstrom_op @ x_test_n))
