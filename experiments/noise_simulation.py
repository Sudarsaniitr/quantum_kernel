"""
experiments/noise_simulation.py
================================
Noise simulation utilities for the swap-test classifier.

Provides:
- Lightweight amplitude-scaled expectation model (fast sweeps)
- Shot-based multinomial measurement simulation (paper-style 8192 shots)
- Optional density-matrix helper class for channel-level experimentation
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# IBM Q 5 Ourense parameters (from paper supplementary tables)
# ---------------------------------------------------------------------------

IBMQ_OURENSE_PARAMS = {
    "T1": [58.3, 49.0, 76.9, 60.7, 46.4],
    "T2": [90.2, 71.7, 109.0, 37.9, 86.5],
    "single_qubit_error": 4.5e-4,
    "cx_error": {
        (0, 1): 0.0118,
        (1, 0): 0.0118,
        (1, 2): 0.0253,
        (2, 1): 0.0253,
        (1, 3): 0.0147,
        (3, 1): 0.0147,
        (3, 4): 0.0162,
        (4, 3): 0.0162,
    },
    "readout_error": [
        (0.0278, 0.0444),
        (0.0328, 0.0550),
        (0.0222, 0.0550),
        (0.0678, 0.0678),
        (0.0678, 0.0778),
    ],
    "single_qubit_gate_time": 50,
    "cx_gate_time": 300,
}


class DensityMatrixSimulator:
    """Small density-matrix helper for channel-level experimentation."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits

    def statevector_to_dm(self, psi: np.ndarray) -> np.ndarray:
        return np.outer(psi, np.conj(psi))

    def apply_gate_dm(self, rho: np.ndarray, U: np.ndarray) -> np.ndarray:
        return U @ rho @ np.conj(U.T)

    def _embed_kraus(self, K2: np.ndarray, qubit: int) -> np.ndarray:
        from core.quantum_gates import I

        ops = [I] * self.n_qubits
        ops[qubit] = K2
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    def apply_single_depolarising(self, rho: np.ndarray, qubit: int, p: float) -> np.ndarray:
        from core.quantum_gates import X, Y, Z, embed_single

        p = min(max(p, 0.0), 1.0)
        paulis = [X, Y, Z]
        new_rho = (1 - 4 * p / 3) * rho
        for P in paulis:
            P_full = embed_single(P, qubit, self.n_qubits)
            new_rho += (p / 3) * (P_full @ rho @ np.conj(P_full.T))
        return new_rho

    def apply_two_qubit_depolarising(self, rho: np.ndarray, q0: int, q1: int, p: float) -> np.ndarray:
        p_single = 1 - np.sqrt(max(0.0, 1 - p))
        rho = self.apply_single_depolarising(rho, q0, p_single)
        rho = self.apply_single_depolarising(rho, q1, p_single)
        return rho

    def apply_thermal_relaxation(
        self,
        rho: np.ndarray,
        qubit: int,
        T1: float,
        T2: float,
        gate_time: float,
    ) -> np.ndarray:
        t = gate_time * 1e-3  # ns -> us

        gamma1 = 1 - np.exp(-t / T1) if T1 > 0 else 1.0
        gamma2 = 1 - np.exp(-t / T2) if T2 > 0 else 1.0
        gamma_phi = max(0.0, gamma2 - gamma1 / 2)

        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma1)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(gamma1)], [0, 0]], dtype=complex)

        K0_full = self._embed_kraus(K0, qubit)
        K1_full = self._embed_kraus(K1, qubit)

        rho_ad = K0_full @ rho @ np.conj(K0_full.T) + K1_full @ rho @ np.conj(K1_full.T)

        Kd0 = np.sqrt(1 - gamma_phi / 2) * np.eye(2, dtype=complex)
        Kd1 = np.sqrt(gamma_phi / 2) * np.array([[1, 0], [0, -1]], dtype=complex)
        Kd0_full = self._embed_kraus(Kd0, qubit)
        Kd1_full = self._embed_kraus(Kd1, qubit)

        return Kd0_full @ rho_ad @ np.conj(Kd0_full.T) + Kd1_full @ rho_ad @ np.conj(Kd1_full.T)

    def expectation_ZZ_dm(self, rho: np.ndarray, qubit_a: int, qubit_l: int) -> float:
        from core.quantum_gates import Z, embed_single

        Za = embed_single(Z, qubit_a, self.n_qubits)
        Zl = embed_single(Z, qubit_l, self.n_qubits)
        ZZ = Za @ Zl
        return float(np.real(np.trace(ZZ @ rho)))


# ---------------------------------------------------------------------------
# Fast noise sweep models
# ---------------------------------------------------------------------------


def simulate_with_noise(
    thetas: np.ndarray,
    device_params: Optional[dict] = None,
    amplitude_reduction: float = 0.82,
) -> np.ndarray:
    """
    Fast analytical approximation:
      noisy(θ) ≈ amplitude_reduction × theory(θ)

    Paper-reported scales:
      simulation ~0.82, experiment ~0.65.
    """
    from experiments.toy_problem import analytical_swap_kernel

    _ = device_params  # reserved for future use
    theory = np.array([analytical_swap_kernel(theta) for theta in thetas])
    return amplitude_reduction * theory


def compute_noise_statistics(
    thetas: np.ndarray,
    n_shots: int = 8192,
    noise_level: float = 0.01,
    amplitude_reduction: float = 0.82,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Simulate shot-based measurements of ⟨σz(a)σz(l)⟩.

    Uses multinomial sampling over outcomes (00,01,10,11).
    """
    from experiments.toy_problem import analytical_swap_kernel

    rng = np.random.default_rng(seed)

    results = {
        "thetas": thetas,
        "expectation_noisy": [],
        "expectation_theoretical": [],
        "counts": [],
        "std_errors": [],
        "noise_level": noise_level,
    }

    for theta in thetas:
        theory = analytical_swap_kernel(theta)
        noisy = amplitude_reduction * theory

        # Optional extra shrink from configurable synthetic noise_level
        noisy *= (1 - min(max(noise_level, 0.0), 0.49))

        p_pp = (1 + noisy) / 4
        p_pm = (1 - noisy) / 4
        p_mp = (1 - noisy) / 4
        p_mm = (1 + noisy) / 4

        probs = np.array([p_pp, p_pm, p_mp, p_mm], dtype=float)
        probs = np.clip(probs, 0.0, 1.0)
        probs /= probs.sum()

        counts = rng.multinomial(n_shots, probs)
        c00, c01, c10, c11 = counts

        measured_exp = (c00 - c01 - c10 + c11) / n_shots
        std_error = np.sqrt(max(0.0, (1 - measured_exp**2) / n_shots))

        results["expectation_noisy"].append(measured_exp)
        results["expectation_theoretical"].append(theory)
        results["counts"].append({"c00": c00, "c01": c01, "c10": c10, "c11": c11})
        results["std_errors"].append(std_error)

    results["expectation_noisy"] = np.array(results["expectation_noisy"])
    results["expectation_theoretical"] = np.array(results["expectation_theoretical"])
    results["std_errors"] = np.array(results["std_errors"])

    return results
