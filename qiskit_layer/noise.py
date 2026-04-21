"""
qiskit_layer/noise.py
=====================
Noise model builders for Qiskit paper-mode runs.
"""

from __future__ import annotations


def _require_aer_noise() -> None:
    try:
        import qiskit_aer.noise  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "qiskit-aer is required for noise simulation. "
            "Install with: pip install -r requirements-qiskit.txt"
        ) from exc


def build_noise_model_from_backend(backend):
    """Build an Aer noise model from an IBM backend calibration snapshot."""
    _require_aer_noise()
    from qiskit_aer.noise import NoiseModel

    return NoiseModel.from_backend(backend)


def build_simple_depolarizing_noise_model(
    p1: float = 4.5e-4,
    p2: float = 0.012,
):
    """
    Build a lightweight depolarizing noise model.

    Defaults are close to the order of magnitude discussed for single-/two-qubit
    errors in the paper's hardware context.
    """
    _require_aer_noise()
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    p1 = min(max(float(p1), 0.0), 1.0)
    p2 = min(max(float(p2), 0.0), 1.0)

    noise_model = NoiseModel()
    err1 = depolarizing_error(p1, 1)
    err2 = depolarizing_error(p2, 2)

    single_qubit_gates = ["x", "sx", "rz", "h", "rx", "ry", "u", "u1", "u2", "u3"]
    two_qubit_gates = ["cx", "cz", "swap"]

    for gate in single_qubit_gates:
        try:
            noise_model.add_all_qubit_quantum_error(err1, [gate])
        except Exception:
            pass

    for gate in two_qubit_gates:
        try:
            noise_model.add_all_qubit_quantum_error(err2, [gate])
        except Exception:
            pass

    return noise_model
