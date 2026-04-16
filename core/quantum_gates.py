"""
core/quantum_gates.py
=====================
Pure-numpy implementation of quantum gates and statevector helpers.

Gate convention: qubits are indexed 0 = most-significant (leftmost in ket notation).
The full Hilbert space of n qubits has dimension 2^n.
A statevector |ψ⟩ is a complex numpy array of shape (2^n,).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Single-qubit gate matrices
# ---------------------------------------------------------------------------

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli-X
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)  # Pauli-Y
Z = np.array([[1, 0], [0, -1]], dtype=complex)  # Pauli-Z
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)  # Hadamard
S = np.array([[1, 0], [0, 1j]], dtype=complex)  # Phase gate


def Rx(theta: float) -> np.ndarray:
    """Rotation about x-axis: Rx(θ) = exp(-i θ/2 X)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def Ry(theta: float) -> np.ndarray:
    """Rotation about y-axis: Ry(θ) = exp(-i θ/2 Y)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def Rz(theta: float) -> np.ndarray:
    """Rotation about z-axis: Rz(θ) = exp(-i θ/2 Z)."""
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=complex,
    )


# ---------------------------------------------------------------------------
# Multi-qubit gate construction helpers
# ---------------------------------------------------------------------------


def _validate_qubit_index(index: int, n_qubits: int, name: str) -> None:
    if not (0 <= index < n_qubits):
        raise ValueError(f"{name}={index} out of range for n_qubits={n_qubits}.")


def embed_single(gate: np.ndarray, target: int, n_qubits: int) -> np.ndarray:
    """
    Embed a single-qubit gate into an n-qubit space.

    Returns a 2^n × 2^n unitary matrix.
    """
    _validate_qubit_index(target, n_qubits, "target")
    if gate.shape != (2, 2):
        raise ValueError("single-qubit gate must have shape (2, 2).")

    ops = [I] * n_qubits
    ops[target] = gate
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def controlled_gate(gate: np.ndarray, control: int, target: int, n_qubits: int) -> np.ndarray:
    """
    Build a controlled-U gate (CU) for an n-qubit system.

    CU = |0⟩⟨0|_c ⊗ I_t + |1⟩⟨1|_c ⊗ U_t, embedded in the full space.
    """
    _validate_qubit_index(control, n_qubits, "control")
    _validate_qubit_index(target, n_qubits, "target")
    if control == target:
        raise ValueError("control and target must be different qubits.")
    if gate.shape != (2, 2):
        raise ValueError("controlled gate target operation must have shape (2, 2).")

    proj0 = np.array([[1, 0], [0, 0]], dtype=complex)
    proj1 = np.array([[0, 0], [0, 1]], dtype=complex)

    ops0 = [I] * n_qubits
    ops0[control] = proj0
    t0 = ops0[0]
    for op in ops0[1:]:
        t0 = np.kron(t0, op)

    ops1 = [I] * n_qubits
    ops1[control] = proj1
    ops1[target] = gate
    t1 = ops1[0]
    for op in ops1[1:]:
        t1 = np.kron(t1, op)

    return t0 + t1


def swap_gate(q0: int, q1: int, n_qubits: int) -> np.ndarray:
    """
    Build a SWAP gate between qubits q0 and q1 in an n-qubit space.

    SWAP|...b_q0...b_q1...⟩ = |...b_q1...b_q0...⟩.
    """
    _validate_qubit_index(q0, n_qubits, "q0")
    _validate_qubit_index(q1, n_qubits, "q1")

    dim = 2**n_qubits
    mat = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bits = list(format(i, f"0{n_qubits}b"))
        bits[q0], bits[q1] = bits[q1], bits[q0]
        j = int("".join(bits), 2)
        mat[j, i] = 1.0
    return mat


def controlled_swap(control: int, q0: int, q1: int, n_qubits: int) -> np.ndarray:
    """
    Fredkin (controlled-SWAP) gate.

    Swaps q0 and q1 only when control qubit = |1⟩.
    """
    _validate_qubit_index(control, n_qubits, "control")
    _validate_qubit_index(q0, n_qubits, "q0")
    _validate_qubit_index(q1, n_qubits, "q1")
    if len({control, q0, q1}) < 3:
        raise ValueError("control, q0, and q1 must be distinct qubits.")

    dim = 2**n_qubits
    mat = np.eye(dim, dtype=complex)
    for i in range(dim):
        bits = list(format(i, f"0{n_qubits}b"))
        if bits[control] == "1":
            bits[q0], bits[q1] = bits[q1], bits[q0]
            j = int("".join(bits), 2)
            mat[j, i] = 1.0
            mat[i, i] = 0.0
    return mat


# ---------------------------------------------------------------------------
# Statevector helpers
# ---------------------------------------------------------------------------


def ket(index: int, n_qubits: int) -> np.ndarray:
    """Return computational basis state |index⟩ for n_qubits."""
    dim = 2**n_qubits
    if not (0 <= index < dim):
        raise ValueError(f"index={index} out of range for n_qubits={n_qubits}.")
    v = np.zeros(dim, dtype=complex)
    v[index] = 1.0
    return v


def tensor(*vecs: np.ndarray) -> np.ndarray:
    """Tensor product of multiple vectors: |a⟩ ⊗ |b⟩ ⊗ ..."""
    if len(vecs) == 0:
        raise ValueError("tensor requires at least one vector.")
    result = vecs[0]
    for v in vecs[1:]:
        result = np.kron(result, v)
    return result


def apply(gate: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Apply a gate matrix to a statevector."""
    return gate @ state


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a statevector."""
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        raise ValueError("Cannot normalize a zero vector.")
    return v / norm


def inner_product(bra: np.ndarray, ket_: np.ndarray) -> complex:
    """⟨bra|ket⟩ = bra†·ket."""
    return np.conj(bra) @ ket_


def fidelity(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """Quantum state fidelity: F(a,b) = |⟨a|b⟩|²."""
    return float(abs(inner_product(state_a, state_b)) ** 2)


def expectation_ZZ(state: np.ndarray, qubit_a: int, qubit_l: int, n_qubits: int) -> float:
    """
    Compute ⟨σz(a) ⊗ σz(l)⟩ for a pure statevector.

    σz eigenvalue is +1 for |0⟩ and -1 for |1⟩.
    Eigenvalue product for basis state |...a...l...⟩ is (-1)^(bit_a + bit_l).
    """
    _validate_qubit_index(qubit_a, n_qubits, "qubit_a")
    _validate_qubit_index(qubit_l, n_qubits, "qubit_l")

    expected_len = 2**n_qubits
    if len(state) != expected_len:
        raise ValueError(
            f"state has length {len(state)} but expected {expected_len} for n_qubits={n_qubits}."
        )

    expectation = 0.0
    for idx in range(expected_len):
        bits = format(idx, f"0{n_qubits}b")
        bit_a = int(bits[qubit_a])
        bit_l = int(bits[qubit_l])
        eigenvalue = (-1) ** (bit_a + bit_l)
        expectation += eigenvalue * abs(state[idx]) ** 2
    return float(expectation)
