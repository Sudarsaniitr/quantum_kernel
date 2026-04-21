"""
qiskit_layer/runner.py
======================
Execution runners for paper-mode Qiskit experiments.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Sequence

import numpy as np

from .backends import backend_name, get_aer_simulator, get_ibm_backend, get_ibm_runtime_config
from .circuits import build_product_state_n_copies_circuit, build_swap_test_toy_circuit
from .noise import build_noise_model_from_backend, build_simple_depolarizing_noise_model


def _require_qiskit_transpile() -> None:
    try:
        import qiskit  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Qiskit is required for paper-mode execution. "
            "Install with: pip install -r requirements-qiskit.txt"
        ) from exc


def expectation_from_counts(counts: dict[str, int]) -> float:
    """
    Compute <Z_a Z_l> from 2-bit measurement counts.

    Matches the supplemental code formula:
      (c00 - c01 - c10 + c11) / shots
    """
    shots = sum(counts.values())
    if shots <= 0:
        return 0.0
    return float(
        (counts.get("00", 0) - counts.get("01", 0) - counts.get("10", 0) + counts.get("11", 0))
        / float(shots)
    )


def _serialize_counts(counts_list: Sequence[dict[str, int]]) -> list[dict[str, int]]:
    return [{str(k): int(v) for k, v in c.items()} for c in counts_list]


def run_swaptest_theta_sweep_qiskit(
    thetas: np.ndarray,
    weights: Sequence[float] = (0.5, 0.5),
    shots: int = 8192,
    mode: str = "simulator",
    circuit_family: str = "swap_test",
    copies: int = 1,
    backend_name_value: Optional[str] = None,
    use_noise: bool = True,
    token: Optional[str] = None,
    instance: Optional[str] = None,
    env_file: Optional[str] = None,
    optimization_level: int = 1,
    seed_simulator: int = 1234,
    wait_for_result: bool = True,
) -> dict:
    """
    Run swap-test toy classifier sweep with Qiskit.

    Parameters
    ----------
    mode :
      - "simulator": AerSimulator (+ optional noise)
      - "hardware": IBM backend execution
    wait_for_result :
      For hardware mode, if False only submits and returns job metadata.
    """
    _require_qiskit_transpile()
    from qiskit import transpile

    mode = mode.lower().strip()
    if mode not in {"simulator", "hardware"}:
        raise ValueError("mode must be one of {'simulator', 'hardware'}")

    circuit_family = circuit_family.lower().strip()
    if circuit_family not in {"swap_test", "product_state"}:
        raise ValueError("circuit_family must be one of {'swap_test', 'product_state'}")

    if copies < 1:
        raise ValueError("copies must be >= 1")

    if circuit_family == "swap_test":
        circuits = [build_swap_test_toy_circuit(theta=float(theta), weights=weights) for theta in thetas]
    else:
        circuits = [
            build_product_state_n_copies_circuit(
                theta=float(theta),
                copies=int(copies),
                weights=weights,
            )
            for theta in thetas
        ]

    meta = {
        "mode": mode,
        "circuit_family": circuit_family,
        "copies": int(copies),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "shots": int(shots),
        "optimization_level": int(optimization_level),
        "n_circuits": len(circuits),
        "weights": [float(weights[0]), float(weights[1])],
        "thetas": [float(t) for t in thetas],
    }

    if mode == "simulator":
        noise_model = None
        noise_source = "none"

        if use_noise:
            # Try backend-calibrated noise if backend name is provided and IBM access works.
            if backend_name_value:
                try:
                    cfg = get_ibm_runtime_config(token=token, instance=instance, env_file=env_file)
                    ibm_backend = get_ibm_backend(backend_name_value, cfg)
                    noise_model = build_noise_model_from_backend(ibm_backend)
                    noise_source = f"from_backend:{backend_name(ibm_backend)}"
                except Exception:
                    noise_model = build_simple_depolarizing_noise_model()
                    noise_source = "fallback:simple_depolarizing"
            else:
                noise_model = build_simple_depolarizing_noise_model()
                noise_source = "simple_depolarizing"

        sim = get_aer_simulator(noise_model=noise_model, seed_simulator=seed_simulator)
        tqc = transpile(circuits, backend=sim, optimization_level=optimization_level)
        job = sim.run(tqc, shots=shots)
        result = job.result()

        counts_list = [result.get_counts(i) for i in range(len(circuits))]
        expectations = [expectation_from_counts(c) for c in counts_list]

        meta.update(
            {
                "backend": backend_name(sim),
                "noise_enabled": bool(use_noise),
                "noise_source": noise_source,
                "job_id": str(job.job_id()),
            }
        )

        return {
            "metadata": meta,
            "counts": _serialize_counts(counts_list),
            "expectation": [float(x) for x in expectations],
        }

    # hardware mode
    cfg = get_ibm_runtime_config(token=token, instance=instance, env_file=env_file)
    if not backend_name_value:
        backend_name_value = "ibmq_qasm_simulator"

    backend = get_ibm_backend(backend_name_value, cfg)
    tqc = transpile(circuits, backend=backend, optimization_level=optimization_level)
    job = backend.run(tqc, shots=shots)

    meta.update(
        {
            "backend": backend_name(backend),
            "noise_enabled": False,
            "noise_source": "hardware",
            "job_id": str(job.job_id()),
        }
    )

    if not wait_for_result:
        return {
            "metadata": meta,
            "counts": [],
            "expectation": [],
            "note": "Hardware job submitted. Re-run with wait_for_result=True to block for results.",
        }

    result = job.result()
    counts_list = [result.get_counts(i) for i in range(len(circuits))]
    expectations = [expectation_from_counts(c) for c in counts_list]

    return {
        "metadata": meta,
        "counts": _serialize_counts(counts_list),
        "expectation": [float(x) for x in expectations],
    }


def summarize_sign_accuracy(expectation: Sequence[float], truth: Sequence[float]) -> float:
    """Return sign-based agreement ratio between measured and reference curves."""
    if len(expectation) != len(truth):
        raise ValueError("expectation and truth must have equal length")
    if not expectation:
        return 0.0

    correct = 0
    for e, t in zip(expectation, truth):
        se = np.sign(e)
        st = np.sign(t)
        if se == 0 or st == 0 or se == st:
            correct += 1
    return float(correct / len(expectation))
