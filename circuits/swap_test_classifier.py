"""
circuits/swap_test_classifier.py
=================================
Swap-test classifier (paper's main contribution).

Circuit summary for n=1, M=2, d=2:
  qubit 0: ancilla (a)
  qubit 1: test data x̃
  qubit 2: training data x_m
  qubit 3: label (l)
  qubit 4: index (m)

Decision rule:
  class 0 if ⟨σz(a)σz(l)⟩ > 0
  class 1 if ⟨σz(a)σz(l)⟩ < 0
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from core.kernel import classify, swap_test_kernel
from core.quantum_gates import (
    H,
    apply,
    controlled_swap,
    embed_single,
    expectation_ZZ,
    ket,
    normalize,
    tensor,
)


class SwapTestClassifier:
    """
    Quantum classifier based on swap-test with fidelity kernel (Eq. 9).
    """

    def __init__(
        self,
        n_copies: int = 1,
        n_data_qubits: int = 1,
        n_index_qubits: int = 1,
    ):
        if n_copies < 1:
            raise ValueError("n_copies must be >= 1")
        self.n = n_copies
        self.n_data = n_data_qubits
        self.n_index = n_index_qubits

        # Qubit layout: ancilla | n×test | n×train | label | index
        self.n_qubits = 1 + 2 * n_copies * n_data_qubits + 1 + n_index_qubits
        self.dim = 2**self.n_qubits

        self.q_ancilla = 0
        self.q_test = [1 + i * n_data_qubits for i in range(n_copies)]

        offset = 1 + n_copies * n_data_qubits
        self.q_train = [offset + i * n_data_qubits for i in range(n_copies)]
        self.q_label = offset + n_copies * n_data_qubits
        self.q_index = [self.q_label + 1 + i for i in range(n_index_qubits)]

    def prepare_input_state(
        self,
        x_train: List[np.ndarray],
        labels: List[int],
        x_test: np.ndarray,
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Prepare Eq.(8)-style input superposition:
          |Ψ⟩ = Σ_m sqrt(w_m) |0⟩_a |x̃^⊗n⟩ |x_m^⊗n⟩ |y_m⟩ |m⟩
        """
        M = len(x_train)
        if M == 0:
            raise ValueError("x_train cannot be empty.")
        if len(labels) != M:
            raise ValueError("labels and x_train must have equal length.")

        if weights is None:
            weights = [1.0 / M] * M

        state = np.zeros(self.dim, dtype=complex)
        ancilla = ket(0, 1)

        xt_norm = normalize(x_test)
        xt_n = xt_norm.copy()
        for _ in range(self.n - 1):
            xt_n = np.kron(xt_n, xt_norm)

        for m, (xm, ym, wm) in enumerate(zip(x_train, labels, weights)):
            xm_norm = normalize(xm)
            xm_n = xm_norm.copy()
            for _ in range(self.n - 1):
                xm_n = np.kron(xm_n, xm_norm)

            label_ym = ket(ym, 1)
            index_m = ket(m, self.n_index)
            term = tensor(ancilla, xt_n, xm_n, label_ym, index_m)
            state += np.sqrt(wm) * term

        return normalize(state)

    def run(
        self,
        x_train: List[np.ndarray],
        labels: List[int],
        x_test: np.ndarray,
        weights: Optional[List[float]] = None,
        add_noise: bool = False,
        noise_params: Optional[dict] = None,
    ) -> dict:
        """Execute swap-test classifier circuit and return diagnostics."""
        psi_0 = self.prepare_input_state(x_train, labels, x_test, weights)

        H_gate = embed_single(H, self.q_ancilla, self.n_qubits)
        psi_1 = apply(H_gate, psi_0)

        if add_noise and noise_params:
            psi_1 = self._apply_depolarising(psi_1, self.q_ancilla, noise_params.get("p1", 0.001))

        psi_2 = psi_1.copy()
        for copy_i in range(self.n):
            for qubit_j in range(self.n_data):
                cswap = controlled_swap(
                    self.q_ancilla,
                    self.q_test[copy_i] + qubit_j,
                    self.q_train[copy_i] + qubit_j,
                    self.n_qubits,
                )
                psi_2 = apply(cswap, psi_2)

                if add_noise and noise_params:
                    psi_2 = self._apply_depolarising(
                        psi_2, self.q_ancilla, noise_params.get("p2", 0.01)
                    )

        psi_3 = apply(H_gate, psi_2)

        if add_noise and noise_params:
            psi_3 = self._apply_depolarising(psi_3, self.q_ancilla, noise_params.get("p1", 0.001))

        exp_zz = expectation_ZZ(psi_3, self.q_ancilla, self.q_label, self.n_qubits)

        kernel_classical = swap_test_kernel(
            normalize(x_test),
            [normalize(x) for x in x_train],
            labels,
            weights,
            self.n,
        )

        predicted = classify(exp_zz) if abs(exp_zz) > 1e-12 else -1

        return {
            "psi_initial": psi_0,
            "psi_after_H1": psi_1,
            "psi_after_cswap": psi_2,
            "psi_final": psi_3,
            "expectation_ZZ": exp_zz,
            "kernel_classical": kernel_classical,
            "predicted_label": predicted,
            "classical_match": abs(exp_zz - kernel_classical) < 1e-8,
            "n_copies": self.n,
            "n_qubits": self.n_qubits,
        }

    def _apply_depolarising(self, state: np.ndarray, qubit: int, p: float) -> np.ndarray:
        """
        First-order statevector approximation of depolarising noise.

        This is intentionally lightweight; exact noise is modeled in
        `experiments/noise_simulation.py` via density matrices.
        """
        from core.quantum_gates import X

        if p <= 0:
            return state
        p = min(max(p, 0.0), 1.0)

        noise_factor = np.sqrt(1 - p)
        x_state = apply(embed_single(X, qubit, self.n_qubits), state)
        noisy = noise_factor * state + np.sqrt(p / 3) * x_state

        # re-normalize to keep this statevector approximation stable
        norm = np.linalg.norm(noisy)
        return noisy / norm if norm > 1e-15 else noisy

    def sweep(
        self,
        x_train: List[np.ndarray],
        labels: List[int],
        test_states: List[np.ndarray],
        weights: Optional[List[float]] = None,
        add_noise: bool = False,
        noise_params: Optional[dict] = None,
    ) -> List[dict]:
        """Run classifier over multiple test states."""
        return [
            self.run(x_train, labels, xt, weights, add_noise, noise_params)
            for xt in test_states
        ]
