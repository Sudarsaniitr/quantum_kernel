"""
circuits/hadamard_classifier.py
================================
Hadamard classifier (prior work baseline).

This classifier uses only the real part of inner products:

  ⟨σz(a)σz(l)⟩ = Σ_m (-1)^y_m w_m Re⟨x̃|x_m⟩

For the paper's complex toy states, this baseline collapses to ~0 everywhere.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from core.kernel import classify, hadamard_kernel
from core.quantum_gates import H, apply, embed_single, expectation_ZZ, ket, normalize, tensor


class HadamardClassifier:
    """
    Distance-based quantum classifier using Hadamard interference.

    Qubit layout for M=2, d=2:
      qubit 0: ancilla
      qubit 1: data
      qubit 2: label
      qubit 3: index
    """

    def __init__(self, n_data_qubits: int = 1, n_index_qubits: int = 1):
        self.n_data = n_data_qubits
        self.n_index = n_index_qubits
        self.n_qubits = 1 + n_data_qubits + 1 + n_index_qubits
        self.dim = 2**self.n_qubits

        self.q_ancilla = 0
        self.q_data = list(range(1, 1 + n_data_qubits))
        self.q_label = 1 + n_data_qubits
        self.q_index = list(range(2 + n_data_qubits, 2 + n_data_qubits + n_index_qubits))

    def prepare_input_state(
        self,
        x_train: List[np.ndarray],
        labels: List[int],
        x_test: np.ndarray,
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Prepare |Ψ_h⟩ for M=2:
          |Ψ_h⟩ = Σ_m sqrt(w_m)/sqrt(2) (|0,x_m,y_m,m⟩ + |1,x̃,y_m,m⟩)
        """
        M = len(x_train)
        if M == 0:
            raise ValueError("x_train cannot be empty.")
        if len(labels) != M:
            raise ValueError("labels and x_train must have equal length.")

        if weights is None:
            weights = [1.0 / M] * M

        state = np.zeros(self.dim, dtype=complex)

        for m, (xm, ym, wm) in enumerate(zip(x_train, labels, weights)):
            ancilla_0 = ket(0, 1)
            data_xm = normalize(xm)
            label_ym = ket(ym, 1)
            index_m = ket(m, self.n_index)
            term0 = tensor(ancilla_0, data_xm, label_ym, index_m)

            ancilla_1 = ket(1, 1)
            data_xt = normalize(x_test)
            term1 = tensor(ancilla_1, data_xt, label_ym, index_m)

            state += np.sqrt(wm) / np.sqrt(2) * (term0 + term1)

        return normalize(state)

    def run(
        self,
        x_train: List[np.ndarray],
        labels: List[int],
        x_test: np.ndarray,
        weights: Optional[List[float]] = None,
    ) -> dict:
        """Run Hadamard classifier and return circuit + analytical outputs."""
        psi_h = self.prepare_input_state(x_train, labels, x_test, weights)

        H_full = embed_single(H, self.q_ancilla, self.n_qubits)
        psi_after_H = apply(H_full, psi_h)

        exp_zz = expectation_ZZ(psi_after_H, self.q_ancilla, self.q_label, self.n_qubits)

        kernel_classical = hadamard_kernel(
            normalize(x_test),
            [normalize(x) for x in x_train],
            labels,
            weights,
        )

        predicted = classify(exp_zz) if abs(exp_zz) > 1e-12 else -1

        return {
            "state_before_H": psi_h,
            "state_after_H": psi_after_H,
            "expectation_ZZ": exp_zz,
            "kernel_classical": kernel_classical,
            "predicted_label": predicted,
            "classical_match": abs(exp_zz - kernel_classical) < 1e-10,
        }

    def sweep(
        self,
        x_train: List[np.ndarray],
        labels: List[int],
        test_states: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> List[dict]:
        """Run classifier on a list of test states."""
        return [self.run(x_train, labels, xt, weights) for xt in test_states]
