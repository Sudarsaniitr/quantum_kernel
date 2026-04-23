"""
main.py
=======
Main entry point for the quantum classifier implementation.

Usage:
  python main.py              # run verification + full experiment suite + summary
  python main.py --quick      # fewer theta points, same math
  python main.py --verify     # only run verification checks
  python main.py --report     # only generate text summary report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence, Tuple

import numpy as np

# make sure local packages resolve
sys.path.insert(0, os.path.dirname(__file__))


def section(title: str):
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _results_dir() -> str:
    p = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(p, exist_ok=True)
    return p


def _save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_verification() -> bool:
    """Run mathematical and implementation consistency checks."""
    section("VERIFICATION: Mathematical Properties")

    from circuits.hadamard_classifier import HadamardClassifier
    from circuits.swap_test_classifier import SwapTestClassifier
    from core.kernel import (
        helstrom_expectation,
        helstrom_operator,
        kernel_matrix,
        swap_test_kernel,
    )
    from experiments.toy_problem import (
        analytical_hadamard_kernel,
        analytical_swap_kernel,
        get_test_state,
        get_theta_range,
        get_training_data,
        true_classification,
    )

    x_train, labels = get_training_data()
    test_thetas = get_theta_range(20)

    pass_count = 0
    fail_count = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal pass_count, fail_count
        if condition:
            pass_count += 1
            print(f"  ✅ {name}")
        else:
            fail_count += 1
            print(f"  ❌ {name}")
            if detail:
                print(f"      {detail}")

    print("\n[1] Training state normalization")
    for m, xm in enumerate(x_train):
        check(
            f"‖x_{m+1}‖ = 1",
            abs(np.linalg.norm(xm) - 1.0) < 1e-14,
            f"norm={np.linalg.norm(xm):.16f}",
        )

    print("\n[2] Hadamard kernel = 0 for all θ (Eq. 13 behavior)")
    had_vals = [analytical_hadamard_kernel(t) for t in test_thetas]
    check(
        "Hadamard kernel ≡ 0",
        all(abs(v) < 1e-14 for v in had_vals),
        f"max |value| = {max(abs(v) for v in had_vals):.2e}",
    )

    print("\n[3] Swap kernel range and periodicity")
    swap_vals = [analytical_swap_kernel(t) for t in test_thetas]
    check(
        "Swap kernel in [-0.5, 0.5]",
        all(-0.5 - 1e-12 <= v <= 0.5 + 1e-12 for v in swap_vals),
        f"range=[{min(swap_vals):.4f}, {max(swap_vals):.4f}]",
    )
    v1 = analytical_swap_kernel(0.5)
    v2 = analytical_swap_kernel(0.5 + 2 * np.pi)
    check("2π periodicity", abs(v1 - v2) < 1e-12, f"Δ={abs(v1-v2):.3e}")

    print("\n[4] Helstrom equivalence (Eq. 16-17)")
    for n in [1, 2, 3]:
        A = helstrom_operator(x_train, labels, n_copies=n)
        diffs = []
        for theta in test_thetas:
            xt = get_test_state(theta)
            lhs = swap_test_kernel(xt, x_train, labels, n_copies=n)
            rhs = helstrom_expectation(xt, A, n_copies=n)
            diffs.append(abs(lhs - rhs))
        check(f"n={n} swap-test == Helstrom", max(diffs) < 1e-12, f"max diff={max(diffs):.2e}")

    print("\n[5] Boundary checks")
    k0 = analytical_swap_kernel(0.0)
    kpi = analytical_swap_kernel(np.pi)
    check("k(0) ≈ 0 (boundary)", abs(k0) < 1e-12, f"k(0)={k0:.3e}")
    check("k(π) ≈ 0 (boundary)", abs(kpi) < 1e-12, f"k(π)={kpi:.3e}")

    print("\n[6] Kernel matrix validity")
    states = x_train + [get_test_state(t) for t in get_theta_range(12)]
    K = kernel_matrix(states, n_copies=1)
    eigvals = np.linalg.eigvalsh(K)
    check("Kernel matrix PSD", np.all(eigvals >= -1e-10), f"min eig={eigvals.min():.2e}")
    check("Kernel diagonal = 1", np.allclose(np.diag(K), 1.0), "diagonal deviates from 1")

    print("\n[7] Circuit-vs-analytical agreement")
    clf_swap = SwapTestClassifier(n_copies=1)
    clf_had = HadamardClassifier()
    swap_errors = []
    had_errors = []

    for theta in test_thetas:
        xt = get_test_state(theta)
        r_swap = clf_swap.run(x_train, labels, xt)
        r_had = clf_had.run(x_train, labels, xt)
        swap_errors.append(abs(r_swap["expectation_ZZ"] - analytical_swap_kernel(theta)))
        had_errors.append(abs(r_had["expectation_ZZ"] - analytical_hadamard_kernel(theta)))

    check("Swap circuit matches analytical", max(swap_errors) < 1e-8, f"max err={max(swap_errors):.2e}")
    check("Hadamard circuit matches analytical", max(had_errors) < 1e-8, f"max err={max(had_errors):.2e}")

    print("\n[8] Noiseless classification accuracy")
    total = len(test_thetas)
    correct = 0
    for theta in test_thetas:
        xt = get_test_state(theta)
        pred = clf_swap.run(x_train, labels, xt)["predicted_label"]
        truth = true_classification(theta)
        if pred == truth or pred == -1 or truth == -1:
            correct += 1
    acc = correct / total
    check("Swap-test accuracy = 100% (allowing boundary)", correct == total, f"accuracy={acc*100:.1f}%")

    print(f"\n  Results: {pass_count} passed, {fail_count} failed")
    if fail_count == 0:
        print("  ✅ Verification complete: all checks passed.")
    else:
        print("  ⚠️  Verification finished with failures. See logs above.")

    return fail_count == 0


def run_all_experiments(quick: bool = False):
    """Run full experiment suite and generate figures."""
    section("RUNNING FULL EXPERIMENT SUITE")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.noise_simulation import compute_noise_statistics
    from experiments.toy_problem import get_theta_range
    from visualization.plots import (
        plot_bloch_sphere,
        plot_circuit_verification,
        plot_hadamard_vs_swaptest,
        plot_helstrom_equivalence,
        plot_kernel_matrix,
        plot_n_copies_effect,
        plot_theory_vs_noisy,
    )

    n_points = 30 if quick else 63
    thetas = get_theta_range(n_points)
    results_dir = _results_dir()

    section("EXPERIMENT 1: Effect of n Copies (Fig. 3)")
    fig = plot_n_copies_effect(thetas, save_path=os.path.join(results_dir, "01_n_copies_effect.png"))
    plt.close(fig)
    print("  → Saved: results/01_n_copies_effect.png")

    section("EXPERIMENT 2: Theory vs Noise (Fig. 5)")
    noise_results = compute_noise_statistics(thetas, n_shots=8192, amplitude_reduction=0.82)
    fig = plot_theory_vs_noisy(
        thetas,
        noise_results,
        save_path=os.path.join(results_dir, "02_theory_vs_noisy.png"),
    )
    plt.close(fig)
    print("  → Saved: results/02_theory_vs_noisy.png")

    section("EXPERIMENT 3: Hadamard vs Swap-Test")
    fig = plot_hadamard_vs_swaptest(
        thetas,
        save_path=os.path.join(results_dir, "03_hadamard_vs_swaptest.png"),
    )
    plt.close(fig)
    print("  → Saved: results/03_hadamard_vs_swaptest.png")

    section("EXPERIMENT 4: Bloch Sphere")
    fig = plot_bloch_sphere(thetas, save_path=os.path.join(results_dir, "04_bloch_sphere.png"))
    plt.close(fig)
    print("  → Saved: results/04_bloch_sphere.png")

    section("EXPERIMENT 5: Kernel Matrix")
    fig = plot_kernel_matrix(save_path=os.path.join(results_dir, "05_kernel_matrix.png"))
    plt.close(fig)
    print("  → Saved: results/05_kernel_matrix.png")

    section("EXPERIMENT 6: Helstrom Equivalence")
    fig = plot_helstrom_equivalence(
        thetas,
        save_path=os.path.join(results_dir, "06_helstrom_equivalence.png"),
    )
    plt.close(fig)
    print("  → Saved: results/06_helstrom_equivalence.png")

    section("EXPERIMENT 7: Circuit Verification")
    fig = plot_circuit_verification(
        thetas,
        save_path=os.path.join(results_dir, "07_circuit_verification.png"),
    )
    plt.close(fig)
    print("  → Saved: results/07_circuit_verification.png")


def run_qiskit_paper_mode(
    quick: bool = False,
    backend_mode: str = "simulator",
    circuit_family: str = "swap_test",
    copies: int = 1,
    backend_name_value: str | None = None,
    shots: int = 8192,
    use_noise: bool = True,
    wait_for_result: bool = True,
    env_file: str | None = ".env",
) -> dict:
    """Run paper-faithful Qiskit sweep (simulator or IBM hardware backend)."""
    section("PAPER MODE: Qiskit Execution")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.toy_problem import analytical_swap_kernel, get_theta_range
    from qiskit_layer.runner import run_swaptest_theta_sweep_qiskit, summarize_sign_accuracy
    from visualization.plots import plot_qiskit_vs_theory

    thetas = get_theta_range(30 if quick else 63)
    result = run_swaptest_theta_sweep_qiskit(
        thetas=thetas,
        shots=shots,
        mode=backend_mode,
        circuit_family=circuit_family,
        copies=copies,
        backend_name_value=backend_name_value,
        use_noise=use_noise,
        wait_for_result=wait_for_result,
        env_file=env_file,
    )

    results_dir = _results_dir()
    json_name = f"08_qiskit_{circuit_family}_{backend_mode}_results.json"
    json_path = os.path.join(results_dir, json_name)
    _save_json(json_path, result)
    print(f"  → Saved: results/{json_name}")

    if result.get("expectation"):
        measured = np.array(result["expectation"], dtype=float)
        theory = np.array([analytical_swap_kernel(theta, n_copies=max(1, copies)) for theta in thetas], dtype=float)
        sign_acc = summarize_sign_accuracy(measured.tolist(), theory.tolist())
        max_abs_diff = float(np.max(np.abs(measured - theory)))

        fig_name = f"09_qiskit_{circuit_family}_{backend_mode}_vs_theory.png"
        fig = plot_qiskit_vs_theory(
            thetas,
            measured,
            theory,
            title=f"Qiskit {circuit_family} ({backend_mode}) vs analytical theory",
            save_path=os.path.join(results_dir, fig_name),
        )
        plt.close(fig)
        print(f"  → Saved: results/{fig_name}")
        print(f"  Sign agreement with theory: {sign_acc*100:.2f}%")
        print(f"  Max |Qiskit - theory|: {max_abs_diff:.4e}")
    else:
        print("  Hardware job submitted without waiting for result.")
        print(f"  Job ID: {result['metadata'].get('job_id', 'unknown')}")

    return result


def _parse_copy_list(copies_text: str) -> list[int]:
    vals = []
    for token in (copies_text or "").split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(int(token))

    if not vals:
        raise ValueError("At least one copy value must be provided.")
    if any(v < 1 for v in vals):
        raise ValueError("All copy values must be >= 1.")
    return sorted(set(vals))


def run_qiskit_shots_comparison(
    quick: bool,
    backend_mode: str,
    shots_values: Sequence[int],
    backend_name_value: str | None = None,
    use_noise: bool = True,
    wait_for_result: bool = True,
    env_file: str | None = ".env",
) -> dict:
    """Run old-architecture paper model at multiple shot counts and compare."""
    section("COMPARISON: Old architecture shots sweep")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.toy_problem import analytical_swap_kernel, get_theta_range
    from qiskit_layer.mitigation import curve_error_metrics
    from qiskit_layer.runner import run_swaptest_theta_sweep_qiskit
    from visualization.plots import plot_qiskit_shots_comparison

    thetas = get_theta_range(30 if quick else 63)
    theory_n1 = np.array([analytical_swap_kernel(theta, n_copies=1) for theta in thetas], dtype=float)
    results_dir = _results_dir()

    shot_curves: dict[int, np.ndarray] = {}
    run_records: dict[str, dict] = {}
    metrics: dict[str, dict] = {}

    for shot in sorted(set(int(s) for s in shots_values)):
        run = run_swaptest_theta_sweep_qiskit(
            thetas=thetas,
            shots=int(shot),
            mode=backend_mode,
            circuit_family="swap_test",
            copies=1,
            backend_name_value=backend_name_value,
            use_noise=use_noise,
            wait_for_result=wait_for_result,
            env_file=env_file,
        )

        run_file = f"10_qiskit_swap_test_{backend_mode}_shots_{shot}_results.json"
        _save_json(os.path.join(results_dir, run_file), run)
        print(f"  → Saved: results/{run_file}")

        run_records[str(shot)] = run

        if run.get("expectation"):
            measured = np.array(run["expectation"], dtype=float)
            shot_curves[int(shot)] = measured
            metrics[str(shot)] = curve_error_metrics(measured, theory_n1)

    summary = {
        "metadata": {
            "mode": backend_mode,
            "comparison": "old_architecture_shots",
            "shot_values": [int(s) for s in sorted(set(int(x) for x in shots_values))],
            "thetas": [float(t) for t in thetas],
            "n_theta": int(len(thetas)),
        },
        "theory_n1": [float(x) for x in theory_n1],
        "runs": run_records,
        "metrics_vs_theory_n1": metrics,
        "curves_by_shot": {str(k): [float(v) for v in vals] for k, vals in shot_curves.items()},
    }

    summary_file = f"11_qiskit_swap_test_{backend_mode}_shots_comparison.json"
    _save_json(os.path.join(results_dir, summary_file), summary)
    print(f"  → Saved: results/{summary_file}")

    if shot_curves:
        fig_file = f"12_qiskit_swap_test_{backend_mode}_shots_comparison.png"
        fig = plot_qiskit_shots_comparison(
            thetas,
            theory_n1,
            measured_by_shot=shot_curves,
            title=f"Old architecture (swap-test n=1) shots comparison [{backend_mode}]",
            save_path=os.path.join(results_dir, fig_file),
        )
        plt.close(fig)
        print(f"  → Saved: results/{fig_file}")

    return summary


def run_vce_novelty_comparison(
    quick: bool,
    backend_mode: str,
    shots: int,
    physical_copies: Sequence[int],
    target_copies: int,
    backend_name_value: str | None = None,
    use_noise: bool = True,
    wait_for_result: bool = True,
    env_file: str | None = ".env",
    baseline_old_arch_curve: Sequence[float] | None = None,
) -> dict:
    """Run novelty VCE pipeline and compare pre-/post-novelty at fixed shots."""
    section("NOVELTY: Virtual Copy Extrapolation (VCE)")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.toy_problem import analytical_swap_kernel, get_theta_range
    from qiskit_layer.mitigation import build_vce_curves, curve_error_metrics
    from qiskit_layer.runner import run_swaptest_theta_sweep_qiskit
    from visualization.plots import plot_vce_target_comparison

    if target_copies < 1:
        raise ValueError("target_copies must be >= 1")

    copy_list = sorted(set(int(c) for c in physical_copies))
    if any(c < 1 for c in copy_list):
        raise ValueError("physical_copies must contain integers >= 1")

    thetas = get_theta_range(30 if quick else 63)
    results_dir = _results_dir()

    raw_runs: dict[str, dict] = {}
    physical_curves: dict[int, np.ndarray] = {}

    for copy in copy_list:
        run = run_swaptest_theta_sweep_qiskit(
            thetas=thetas,
            shots=int(shots),
            mode=backend_mode,
            circuit_family="product_state",
            copies=int(copy),
            backend_name_value=backend_name_value,
            use_noise=use_noise,
            wait_for_result=wait_for_result,
            env_file=env_file,
        )

        run_file = f"13_qiskit_product_state_{backend_mode}_copies_{copy}_shots_{int(shots)}_results.json"
        _save_json(os.path.join(results_dir, run_file), run)
        print(f"  → Saved: results/{run_file}")

        raw_runs[str(copy)] = run
        if run.get("expectation"):
            physical_curves[int(copy)] = np.array(run["expectation"], dtype=float)

    summary = {
        "metadata": {
            "mode": backend_mode,
            "shots": int(shots),
            "physical_copies": [int(c) for c in copy_list],
            "target_copies": int(target_copies),
            "thetas": [float(t) for t in thetas],
            "n_theta": int(len(thetas)),
        },
        "raw_runs": raw_runs,
        "physical_curves": {str(k): [float(x) for x in v] for k, v in physical_curves.items()},
        "theory_curves": {},
        "virtual_curves": {},
        "metrics": {},
    }

    if 1 not in physical_curves or 2 not in physical_curves:
        summary_file = f"14_qiskit_vce_{backend_mode}_shots_{int(shots)}_summary.json"
        _save_json(os.path.join(results_dir, summary_file), summary)
        print("  VCE metrics skipped: missing physical n=1 and/or n=2 expectations.")
        print(f"  → Saved: results/{summary_file}")
        return summary

    for copy in sorted(physical_curves.keys()):
        th = np.array([analytical_swap_kernel(theta, n_copies=copy) for theta in thetas], dtype=float)
        summary["theory_curves"][str(copy)] = [float(x) for x in th]
        summary["metrics"][f"physical_n{copy}_vs_theory_n{copy}"] = curve_error_metrics(physical_curves[copy], th)

    theory_target = np.array([analytical_swap_kernel(theta, n_copies=target_copies) for theta in thetas], dtype=float)
    summary["theory_curves"][str(target_copies)] = [float(x) for x in theory_target]

    vce_curves = build_vce_curves(physical_curves, target_copies=target_copies)
    summary["virtual_curves"] = vce_curves

    virtual_n3 = np.array(vce_curves["virtual_n3_from_12"], dtype=float)
    theory_n3 = np.array([analytical_swap_kernel(theta, n_copies=3) for theta in thetas], dtype=float)
    summary["theory_curves"]["3"] = [float(x) for x in theory_n3]
    summary["metrics"]["virtual_n3_from_12_vs_theory_n3"] = curve_error_metrics(virtual_n3, theory_n3)

    virtual_target = np.array(vce_curves["virtual_target_from_123"], dtype=float)
    summary["metrics"][f"virtual_n{target_copies}_from_123_vs_theory_n{target_copies}"] = curve_error_metrics(
        virtual_target,
        theory_target,
    )

    if 3 in physical_curves:
        summary["metrics"]["pre_novelty_physical_n3_vs_theory_n3"] = curve_error_metrics(
            physical_curves[3],
            theory_n3,
        )
        summary["metrics"]["post_novelty_virtual_n3_vs_theory_n3"] = curve_error_metrics(
            virtual_n3,
            theory_n3,
        )

    baseline_curve = None
    if baseline_old_arch_curve is not None:
        baseline_curve = np.array(baseline_old_arch_curve, dtype=float)

    if 3 in physical_curves:
        physical_target_curve = physical_curves[3]
        virtual_target_curve = virtual_n3
        theory_for_plot = theory_n3
        plot_title = f"Pre/Post novelty comparison at {shots} shots [{backend_mode}] (target n=3)"
    else:
        physical_target_curve = physical_curves[max(physical_curves.keys())]
        virtual_target_curve = virtual_target
        theory_for_plot = theory_target
        plot_title = (
            f"Pre/Post novelty comparison at {shots} shots [{backend_mode}] "
            f"(target n={target_copies})"
        )

    fig_file = f"15_qiskit_vce_{backend_mode}_shots_{int(shots)}_pre_post.png"
    fig = plot_vce_target_comparison(
        thetas,
        theory_target=theory_for_plot,
        physical_target=physical_target_curve,
        virtual_target=virtual_target_curve,
        baseline_n1=baseline_curve,
        title=plot_title,
        save_path=os.path.join(results_dir, fig_file),
    )
    plt.close(fig)
    print(f"  → Saved: results/{fig_file}")

    summary_file = f"14_qiskit_vce_{backend_mode}_shots_{int(shots)}_summary.json"
    _save_json(os.path.join(results_dir, summary_file), summary)
    print(f"  → Saved: results/{summary_file}")

    return summary


def _summary_stats(thetas_full: np.ndarray) -> Tuple[float, float, float, float]:
    from circuits.swap_test_classifier import SwapTestClassifier
    from experiments.toy_problem import get_test_state, get_training_data, true_classification

    x_train, labels = get_training_data()
    clf = SwapTestClassifier(n_copies=1)

    correct = 0
    boundary = 0
    for theta in thetas_full:
        pred = clf.run(x_train, labels, get_test_state(theta))["predicted_label"]
        truth = true_classification(theta)
        if truth == -1:
            boundary += 1
        if pred == truth or pred == -1 or truth == -1:
            correct += 1

    acc = correct / len(thetas_full)
    return acc, float(correct), float(len(thetas_full)), float(boundary)


def print_summary_report(thetas_full: np.ndarray):
    """Create and save text summary report."""
    from core.kernel import helstrom_expectation, helstrom_operator
    from experiments.toy_problem import (
        analytical_hadamard_kernel,
        analytical_swap_kernel,
        get_test_state,
        get_training_data,
    )

    x_train, labels = get_training_data()

    had_vals = np.array([analytical_hadamard_kernel(t) for t in thetas_full])
    swap_vals = np.array([analytical_swap_kernel(t) for t in thetas_full])

    A1 = helstrom_operator(x_train, labels, n_copies=1)
    hel_vals = np.array([helstrom_expectation(get_test_state(t), A1, n_copies=1) for t in thetas_full])
    hel_diff_max = float(np.max(np.abs(swap_vals - hel_vals)))

    acc, correct, total, boundary = _summary_stats(thetas_full)

    lines = []
    lines.append("=" * 74)
    lines.append("Quantum Classifier Summary Report")
    lines.append("=" * 74)
    lines.append("")
    lines.append("1) Hadamard baseline")
    lines.append(f"   max |Hadamard kernel| over sweep: {np.max(np.abs(had_vals)):.3e}")
    lines.append("   interpretation: approximately zero everywhere for the toy problem.")
    lines.append("")
    lines.append("2) Swap-test classifier")
    lines.append(f"   min / max kernel value: {np.min(swap_vals):.6f} / {np.max(swap_vals):.6f}")
    lines.append("   interpretation: sign changes cleanly across boundaries, enabling classification.")
    lines.append("")
    lines.append("3) Helstrom equivalence")
    lines.append(f"   max |swap - helstrom|: {hel_diff_max:.3e}")
    lines.append("   interpretation: numerical agreement supports Eq.(16-17) equivalence.")
    lines.append("")
    lines.append("4) Classification performance (noiseless)")
    lines.append(f"   correct: {int(correct)}/{int(total)}")
    lines.append(f"   boundary points: {int(boundary)}")
    lines.append(f"   effective accuracy: {acc*100:.2f}%")
    lines.append("")
    lines.append("5) Output artifacts")
    lines.append("   - results/01_n_copies_effect.png")
    lines.append("   - results/02_theory_vs_noisy.png")
    lines.append("   - results/03_hadamard_vs_swaptest.png")
    lines.append("   - results/04_bloch_sphere.png")
    lines.append("   - results/05_kernel_matrix.png")
    lines.append("   - results/06_helstrom_equivalence.png")
    lines.append("   - results/07_circuit_verification.png")
    lines.append("")
    lines.append("=" * 74)

    text = "\n".join(lines)
    out = os.path.join(_results_dir(), "summary_report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)

    section("SUMMARY REPORT")
    print(text)
    print(f"\n  → Saved: {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantum classifier experiment runner")
    parser.add_argument("--quick", action="store_true", help="Use fewer theta points")
    parser.add_argument("--verify", action="store_true", help="Run verification checks only")
    parser.add_argument("--report", action="store_true", help="Generate summary report only")
    parser.add_argument("--paper-mode", action="store_true", help="Run Qiskit paper-faithful execution mode")
    parser.add_argument("--paper-only", action="store_true", help="Skip NumPy experiment plots and run only paper-mode")
    parser.add_argument("--paper-backend", choices=["simulator", "hardware"], default="simulator",
                        help="Paper-mode backend type")
    parser.add_argument("--paper-circuit", choices=["swap_test", "product_state"], default="swap_test",
                        help="Paper-mode circuit family")
    parser.add_argument("--paper-copies", type=int, default=1,
                        help="Number of copies for product_state circuit family")
    parser.add_argument("--ibm-backend", default=None,
                        help="IBM backend name for hardware runs or backend-calibrated simulator noise")
    parser.add_argument("--shots", type=int, default=8192, help="Number of shots for Qiskit runs")
    parser.add_argument("--no-paper-noise", action="store_true",
                        help="Disable noise model in paper-mode simulator runs")
    parser.add_argument("--hardware-wait", action="store_true",
                        help="In hardware mode, wait for result instead of submit-only")
    parser.add_argument("--env-file", default=".env",
                        help="Path to env file with QISKIT_IBM_* settings")
    parser.add_argument("--compare-suite", action="store_true",
                        help="Run full comparison suite: old-architecture shots + novelty VCE")
    parser.add_argument("--vce-physical-copies", default="1,2,3",
                        help="Comma-separated physical copies for novelty run, e.g. 1,2,3")
    parser.add_argument("--vce-target-copies", type=int, default=5,
                        help="Virtual target copies for extrapolation summaries")
    args = parser.parse_args()

    if args.paper_only and not args.paper_mode:
        parser.error("--paper-only requires --paper-mode")
    if args.paper_circuit == "swap_test" and args.paper_copies != 1:
        print("Note: --paper-copies is ignored for swap_test circuit family.")

    run_only_report = args.report and not args.verify

    if args.verify:
        ok = run_verification()
        if not args.report and not args.paper_mode:
            return 0 if ok else 1

    if not run_only_report and not args.verify and not args.paper_only and not args.compare_suite:
        ok = run_verification()
        if not ok:
            print("\nStopping because verification failed.")
            return 1
        run_all_experiments(quick=args.quick)

    if args.paper_mode:
        run_qiskit_paper_mode(
            quick=args.quick,
            backend_mode=args.paper_backend,
            circuit_family=args.paper_circuit,
            copies=args.paper_copies,
            backend_name_value=args.ibm_backend,
            shots=args.shots,
            use_noise=not args.no_paper_noise,
            wait_for_result=args.hardware_wait,
            env_file=args.env_file,
        )

    if args.compare_suite:
        high_shots = int(args.shots)
        if high_shots <= 256:
            high_shots = 1024

        shots_cmp = run_qiskit_shots_comparison(
            quick=args.quick,
            backend_mode=args.paper_backend,
            shots_values=(256, high_shots),
            backend_name_value=args.ibm_backend,
            use_noise=not args.no_paper_noise,
            wait_for_result=args.hardware_wait,
            env_file=args.env_file,
        )

        baseline_curve = None
        baseline_key = str(high_shots)
        if baseline_key in shots_cmp.get("curves_by_shot", {}):
            baseline_curve = shots_cmp["curves_by_shot"][baseline_key]

        vce_cmp = run_vce_novelty_comparison(
            quick=args.quick,
            backend_mode=args.paper_backend,
            shots=high_shots,
            physical_copies=_parse_copy_list(args.vce_physical_copies),
            target_copies=int(args.vce_target_copies),
            backend_name_value=args.ibm_backend,
            use_noise=not args.no_paper_noise,
            wait_for_result=args.hardware_wait,
            env_file=args.env_file,
            baseline_old_arch_curve=baseline_curve,
        )

        suite_summary = {
            "metadata": {
                "mode": args.paper_backend,
                "high_shots": int(high_shots),
                "requested": {
                    "shots_comparison": [256, int(high_shots)],
                    "vce_physical_copies": _parse_copy_list(args.vce_physical_copies),
                    "vce_target_copies": int(args.vce_target_copies),
                },
            },
            "shots_comparison_summary": shots_cmp,
            "vce_summary": vce_cmp,
        }

        suite_file = f"16_qiskit_{args.paper_backend}_comparison_suite_summary.json"
        _save_json(os.path.join(_results_dir(), suite_file), suite_summary)
        print(f"  → Saved: results/{suite_file}")

    from experiments.toy_problem import get_theta_range

    if (not args.paper_only) or args.report:
        n_points = 30 if args.quick else 63
        print_summary_report(get_theta_range(n_points))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
