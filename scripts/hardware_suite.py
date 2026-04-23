"""
scripts/hardware_suite.py
=========================
Consolidated hardware runner for the full comparison suite.

Minimises IBM global-catalog calls by building ONE QiskitRuntimeService and
ONE backend reference, then reusing them for every theta sweep. Each sweep
is saved to results/ as soon as it completes so transient network errors
cannot wipe out the earlier successful runs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Sequence

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv  # noqa: E402
from qiskit import transpile  # noqa: E402
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler  # noqa: E402

from qiskit_layer.circuits import (  # noqa: E402
    build_product_state_n_copies_circuit,
    build_swap_test_toy_circuit,
)
from qiskit_layer.runner import (  # noqa: E402
    _extract_counts_from_sampler_pub,
    _serialize_counts,
    expectation_from_counts,
)
from experiments.toy_problem import analytical_swap_kernel, get_theta_range  # noqa: E402
from qiskit_layer.mitigation import build_vce_curves, curve_error_metrics  # noqa: E402


def _results_dir() -> str:
    p = os.path.join(ROOT, "results")
    os.makedirs(p, exist_ok=True)
    return p


def _save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _make_service(max_attempts: int = 5) -> QiskitRuntimeService:
    load_dotenv(os.path.join(ROOT, ".env"))
    token = os.getenv("QISKIT_IBM_TOKEN")
    instance = os.getenv("QISKIT_IBM_INSTANCE") or None
    channel = os.getenv("QISKIT_IBM_CHANNEL", "ibm_quantum_platform")

    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[svc] attempt {attempt}/{max_attempts} ...", flush=True)
            svc = QiskitRuntimeService(channel=channel, token=token, instance=instance)
            print("[svc] service ready", flush=True)
            return svc
        except Exception as exc:
            last_err = exc
            msg = str(exc)
            print(f"[svc] failed: {type(exc).__name__}: {msg[:220]}", flush=True)
            if attempt < max_attempts:
                delay = min(30, 2 * attempt)
                print(f"[svc] sleeping {delay}s before retry", flush=True)
                time.sleep(delay)
    assert last_err is not None
    raise last_err


def _run_sweep(
    sampler: Sampler,
    backend,
    circuit_family: str,
    copies: int,
    shots: int,
    thetas: np.ndarray,
    optimization_level: int = 1,
) -> dict:
    if circuit_family == "swap_test":
        circuits = [build_swap_test_toy_circuit(theta=float(t)) for t in thetas]
    else:
        circuits = [
            build_product_state_n_copies_circuit(theta=float(t), copies=int(copies))
            for t in thetas
        ]

    tqc = transpile(circuits, backend=backend, optimization_level=optimization_level)
    print(
        f"[job] submit family={circuit_family} copies={copies} shots={shots} circuits={len(circuits)}",
        flush=True,
    )
    job = sampler.run(tqc, shots=shots)
    job_id = str(job.job_id())
    print(f"[job] id={job_id} waiting...", flush=True)

    result = job.result()
    try:
        pubs = list(result)
    except Exception:
        pubs = [result[i] for i in range(len(circuits))]

    counts_list = []
    for i in range(len(circuits)):
        if i < len(pubs):
            counts_list.append(_extract_counts_from_sampler_pub(pubs[i]))
        else:
            counts_list.append({})

    if all(len(c) == 0 for c in counts_list):
        raise RuntimeError("Hardware result returned no extractable counts.")

    expectations = [expectation_from_counts(c) for c in counts_list]

    meta = {
        "mode": "hardware",
        "circuit_family": circuit_family,
        "copies": int(copies),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "shots": int(shots),
        "optimization_level": int(optimization_level),
        "n_circuits": int(len(circuits)),
        "weights": [0.5, 0.5],
        "thetas": [float(t) for t in thetas],
        "backend": getattr(backend, "name", None) if isinstance(getattr(backend, "name", None), str)
                    else backend.name(),
        "noise_enabled": False,
        "noise_source": "hardware",
        "job_id": job_id,
    }

    return {
        "metadata": meta,
        "counts": _serialize_counts(counts_list),
        "expectation": [float(x) for x in expectations],
    }


def main(backend_name_value: str = "ibm_kingston", quick: bool = True) -> int:
    thetas = get_theta_range(30 if quick else 63)
    results_dir = _results_dir()

    svc = _make_service()
    print(f"[svc] resolving backend {backend_name_value}...", flush=True)
    backend = svc.backend(backend_name_value)
    print(f"[svc] backend={backend.name} resolved", flush=True)
    sampler = Sampler(mode=backend)

    # ---------------- shots comparison (swap-test n=1) -----------------
    shots_runs: dict[int, dict] = {}
    for shots in (256, 1024):
        run = _run_sweep(sampler, backend, "swap_test", 1, shots, thetas)
        path = f"10_qiskit_swap_test_hardware_shots_{shots}_results.json"
        _save_json(os.path.join(results_dir, path), run)
        print(f"[save] results/{path}", flush=True)
        shots_runs[shots] = run

    # shots-comparison summary + plot
    theory_n1 = np.array([analytical_swap_kernel(t, n_copies=1) for t in thetas], dtype=float)
    shot_curves: dict[int, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    for shots, run in shots_runs.items():
        measured = np.array(run["expectation"], dtype=float)
        shot_curves[int(shots)] = measured
        metrics[str(shots)] = curve_error_metrics(measured, theory_n1)

    summary_shots = {
        "metadata": {
            "mode": "hardware",
            "comparison": "old_architecture_shots",
            "shot_values": sorted(shots_runs.keys()),
            "thetas": [float(t) for t in thetas],
            "n_theta": int(len(thetas)),
            "backend": backend_name_value,
        },
        "theory_n1": [float(x) for x in theory_n1],
        "runs": {str(k): v for k, v in shots_runs.items()},
        "metrics_vs_theory_n1": metrics,
        "curves_by_shot": {str(k): [float(v) for v in vals] for k, vals in shot_curves.items()},
    }
    _save_json(
        os.path.join(results_dir, "11_qiskit_swap_test_hardware_shots_comparison.json"),
        summary_shots,
    )
    print("[save] results/11_qiskit_swap_test_hardware_shots_comparison.json", flush=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from visualization.plots import plot_qiskit_shots_comparison

        fig = plot_qiskit_shots_comparison(
            thetas,
            theory_n1,
            measured_by_shot=shot_curves,
            title="Old architecture (swap-test n=1) shots comparison [hardware]",
            save_path=os.path.join(results_dir, "12_qiskit_swap_test_hardware_shots_comparison.png"),
        )
        plt.close(fig)
        print("[save] results/12_qiskit_swap_test_hardware_shots_comparison.png", flush=True)
    except Exception as e:
        print(f"[warn] shots comparison plot failed: {e}", flush=True)

    # ------------- VCE novelty sweeps (product_state n=1,2,3) -------------
    physical_curves: dict[int, np.ndarray] = {}
    raw_runs_nov: dict[str, dict] = {}
    for copies in (1, 2, 3):
        run = _run_sweep(sampler, backend, "product_state", copies, 1024, thetas)
        path = f"13_qiskit_product_state_hardware_copies_{copies}_shots_1024_results.json"
        _save_json(os.path.join(results_dir, path), run)
        print(f"[save] results/{path}", flush=True)
        raw_runs_nov[str(copies)] = run
        physical_curves[copies] = np.array(run["expectation"], dtype=float)

    # VCE summary
    summary_vce: dict = {
        "metadata": {
            "mode": "hardware",
            "shots": 1024,
            "physical_copies": [1, 2, 3],
            "target_copies": 5,
            "thetas": [float(t) for t in thetas],
            "n_theta": int(len(thetas)),
            "backend": backend_name_value,
        },
        "raw_runs": raw_runs_nov,
        "physical_curves": {str(k): [float(x) for x in v] for k, v in physical_curves.items()},
        "theory_curves": {},
        "virtual_curves": {},
        "metrics": {},
    }

    for copies in sorted(physical_curves.keys()):
        th = np.array([analytical_swap_kernel(t, n_copies=copies) for t in thetas], dtype=float)
        summary_vce["theory_curves"][str(copies)] = [float(x) for x in th]
        summary_vce["metrics"][f"physical_n{copies}_vs_theory_n{copies}"] = curve_error_metrics(
            physical_curves[copies], th
        )

    theory_target = np.array([analytical_swap_kernel(t, n_copies=5) for t in thetas], dtype=float)
    summary_vce["theory_curves"]["5"] = [float(x) for x in theory_target]

    vce_curves = build_vce_curves(physical_curves, target_copies=5)
    summary_vce["virtual_curves"] = vce_curves

    virtual_n3 = np.array(vce_curves["virtual_n3_from_12"], dtype=float)
    theory_n3 = np.array([analytical_swap_kernel(t, n_copies=3) for t in thetas], dtype=float)
    summary_vce["theory_curves"]["3"] = [float(x) for x in theory_n3]
    summary_vce["metrics"]["virtual_n3_from_12_vs_theory_n3"] = curve_error_metrics(
        virtual_n3, theory_n3
    )
    virtual_target = np.array(vce_curves["virtual_target_from_123"], dtype=float)
    summary_vce["metrics"]["virtual_n5_from_123_vs_theory_n5"] = curve_error_metrics(
        virtual_target, theory_target
    )
    summary_vce["metrics"]["pre_novelty_physical_n3_vs_theory_n3"] = curve_error_metrics(
        physical_curves[3], theory_n3
    )
    summary_vce["metrics"]["post_novelty_virtual_n3_vs_theory_n3"] = curve_error_metrics(
        virtual_n3, theory_n3
    )

    _save_json(
        os.path.join(results_dir, "14_qiskit_vce_hardware_shots_1024_summary.json"),
        summary_vce,
    )
    print("[save] results/14_qiskit_vce_hardware_shots_1024_summary.json", flush=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from visualization.plots import plot_vce_target_comparison

        fig = plot_vce_target_comparison(
            thetas,
            theory_target=theory_n3,
            physical_target=physical_curves[3],
            virtual_target=virtual_n3,
            baseline_n1=shot_curves.get(1024),
            title="Pre/Post novelty comparison at 1024 shots [hardware] (target n=3)",
            save_path=os.path.join(
                results_dir, "15_qiskit_vce_hardware_shots_1024_pre_post.png"
            ),
        )
        plt.close(fig)
        print("[save] results/15_qiskit_vce_hardware_shots_1024_pre_post.png", flush=True)
    except Exception as e:
        print(f"[warn] vce plot failed: {e}", flush=True)

    # combined suite summary
    suite = {
        "metadata": {
            "mode": "hardware",
            "high_shots": 1024,
            "backend": backend_name_value,
            "requested": {
                "shots_comparison": [256, 1024],
                "vce_physical_copies": [1, 2, 3],
                "vce_target_copies": 5,
            },
        },
        "shots_comparison_summary": summary_shots,
        "vce_summary": summary_vce,
    }
    _save_json(
        os.path.join(results_dir, "16_qiskit_hardware_comparison_suite_summary.json"),
        suite,
    )
    print("[save] results/16_qiskit_hardware_comparison_suite_summary.json", flush=True)
    print("[done] hardware suite complete", flush=True)
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_kingston")
    ap.add_argument("--quick", action="store_true", default=True)
    args = ap.parse_args()
    raise SystemExit(main(args.backend, args.quick))
