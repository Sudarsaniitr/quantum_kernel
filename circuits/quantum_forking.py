"""
circuits/quantum_forking.py
===========================
Product-state (quantum-forking style) helper utilities.

This module provides a minimal analytical equivalent for the product-state
variant discussed in the paper: n-copy fidelity sharpening without explicitly
constructing all entangled pre-processing circuits.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from core.kernel import swap_test_kernel
from core.quantum_gates import normalize


def product_state_expectation(
    x_test: np.ndarray,
    x_train: List[np.ndarray],
    labels: List[int],
    n_copies: int,
    weights: Optional[List[float]] = None,
) -> float:
    """
    Return expected ⟨σz(a)σz(l)⟩ for the product-state n-copy setup.

    In the ideal noiseless setting, this equals the swap-test kernel with n copies.
    """
    return swap_test_kernel(
        normalize(x_test),
        [normalize(x) for x in x_train],
        labels,
        weights=weights,
        n_copies=n_copies,
    )
