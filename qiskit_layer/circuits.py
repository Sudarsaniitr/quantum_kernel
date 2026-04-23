"""
qiskit_layer/circuits.py
========================
Qiskit circuit builders for both paper-faithful classifier circuits.

The two circuits correspond to the two main evaluations in the paper:

  build_swap_test_toy_circuit(theta)
      Implements the 5-qubit swap-test classifier (paper Eq. 9, n=1).
      Qubit layout: index(m) | label(l) | training-data(d) | test-data(in) | ancilla(a)
      Measurement: ancilla → c[0],  label → c[1]
      ⟨σz^(a) σz^(l)⟩ from counts gives the kernel value.

  build_product_state_n_copies_circuit(theta, copies)
      Implements the n-copy product-state circuit (paper Fig. 3, Eq. 12 for n>1).
      Register layout: a(1) | m(1) | d(copies) | l(1) | in(copies)
      Running copies=3 physically requires 3× as many data qubits but gives a
      sharper kernel K_3 = Σ w_m |⟨x̃|x_m⟩|^6 that separates classes better.
      This is the circuit that VCE (mitigation.py) avoids running at large n.

Training state preparation details
-----------------------------------
Training states from Eq.(11):
  |x1⟩ = (i/√2)|0⟩ + (1/√2)|1⟩   class 0
  |x2⟩ = (i/√2)|0⟩ − (1/√2)|1⟩   class 1

The index qubit is prepared in sqrt(w1)|0⟩ + sqrt(w2)|1⟩ via RY(2·arcsin(√w2)).
Training data qubit is prepared as H → Rz(-π) → S, then conditionally phase-
flipped by CZ with the index qubit to encode the ±1/√2 sign difference.
Test state |x̃(θ)⟩ = cos(θ/2)|0⟩ + i·sin(θ/2)|1⟩ is prepared by RX(-θ).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _require_qiskit() -> None:
    try:
        import qiskit  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Qiskit is required for paper-mode execution. "
            "Install with: pip install -r requirements-qiskit.txt"
        ) from exc


def normalize_weights(weights: Sequence[float]) -> tuple[float, float]:
    """Return validated, normalized (w1, w2) for 2-class toy problem."""
    if len(weights) != 2:
        raise ValueError("weights must have length 2 for the toy problem.")
    w1, w2 = float(weights[0]), float(weights[1])
    if w1 < 0 or w2 < 0:
        raise ValueError("weights must be non-negative.")
    s = w1 + w2
    if s <= 0:
        raise ValueError("sum(weights) must be > 0.")
    return w1 / s, w2 / s


def index_superposition_angle(weights: Sequence[float]) -> float:
    """
    RY angle that prepares sqrt(w1)|0> + sqrt(w2)|1> from |0>.

    For RY(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>, we need:
      cos(theta/2)=sqrt(w1), sin(theta/2)=sqrt(w2)
      => theta = 2*arcsin(sqrt(w2)).
    """
    _, w2 = normalize_weights(weights)
    return float(2 * np.arcsin(np.sqrt(w2)))


def build_swap_test_toy_circuit(
    theta: float,
    weights: Sequence[float] = (0.5, 0.5),
    use_barriers: bool = False,
    measure: bool = True,
):
    """
    Build the 5-qubit swap-test toy circuit.

    Qubit map:
      q0 = index (m)
      q1 = label (l)
      q2 = training data (d)
      q3 = test data (in)
      q4 = ancilla (a)

    Classical bits:
      c0 <- ancilla, c1 <- label
    """
    _require_qiskit()
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    q = QuantumRegister(5, "q")
    c = ClassicalRegister(2, "c")
    qc = QuantumCircuit(q, c, name="swap_test_toy")

    qb_m, qb_l, qb_d, qb_in, qb_a = q[0], q[1], q[2], q[3], q[4]

    # Index superposition with desired class weights
    angle = index_superposition_angle(weights)
    qc.ry(angle, qb_m)
    if use_barriers:
        qc.barrier()

    # Prepare training states conditioned by index
    # This sequence mirrors supplemental logic for x1/x2 toy states
    qc.h(qb_d)
    qc.rz(-np.pi, qb_d)
    qc.s(qb_d)
    qc.cz(qb_m, qb_d)
    if use_barriers:
        qc.barrier()

    # Label register: y_m = m
    qc.cx(qb_m, qb_l)
    if use_barriers:
        qc.barrier()

    # Test state: |x~(theta)> = cos(theta/2)|0> + i sin(theta/2)|1>
    qc.rx(-theta, qb_in)
    if use_barriers:
        qc.barrier()

    # Swap-test
    qc.h(qb_a)
    qc.cswap(qb_a, qb_d, qb_in)
    qc.h(qb_a)

    if measure:
        if use_barriers:
            qc.barrier()
        qc.measure(qb_a, c[0])
        qc.measure(qb_l, c[1])

    return qc


def build_product_state_n_copies_circuit(
    theta: float,
    copies: int = 1,
    weights: Sequence[float] = (0.5, 0.5),
    use_barriers: bool = False,
    measure: bool = True,
):
    """
    Build product-state n-copy swap-test circuit (quantum forking style).

    Register layout:
      a(1), m(1), d(copies), l(1), in(copies), c(2)
    """
    if copies < 1:
        raise ValueError("copies must be >= 1")

    _require_qiskit()
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    a = QuantumRegister(1, "a")
    m = QuantumRegister(1, "m")
    d = QuantumRegister(copies, "d")
    l = QuantumRegister(1, "l")
    inp = QuantumRegister(copies, "in")
    c = ClassicalRegister(2, "c")

    qc = QuantumCircuit(a, m, d, l, inp, c, name=f"swap_test_product_{copies}")

    angle = index_superposition_angle(weights)
    qc.ry(angle, m[0])
    if use_barriers:
        qc.barrier()

    for i in range(copies):
        qc.rx(-theta, inp[i])

    if use_barriers:
        qc.barrier()

    for i in range(copies):
        qc.h(d[i])
        qc.rz(-np.pi, d[i])
        qc.s(d[i])
        qc.cz(m[0], d[i])

    qc.cx(m[0], l[0])

    if use_barriers:
        qc.barrier()

    qc.h(a[0])
    for i in range(copies):
        qc.cswap(a[0], d[i], inp[i])
    qc.h(a[0])

    if measure:
        if use_barriers:
            qc.barrier()
        qc.measure(a[0], c[0])
        qc.measure(l[0], c[1])

    return qc
