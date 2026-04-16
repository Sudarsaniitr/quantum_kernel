"""
visualization/plots.py
=======================
Reproduces key figures from the paper and adds diagnostics.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "lines.linewidth": 2,
    }
)


def _save_if_requested(fig, save_path: Optional[str]) -> None:
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Fig 3: Effect of n copies
# ---------------------------------------------------------------------------


def plot_n_copies_effect(thetas: np.ndarray, save_path: str | None = None):
    from experiments.toy_problem import analytical_swap_kernel

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    n_vals = [1, 10, 100]

    ax = axes[0]
    base = np.array([analytical_swap_kernel(t) for t in thetas])

    for n, color in zip(n_vals, colors):
        vals = np.array([analytical_swap_kernel(t, n_copies=n) for t in thetas])
        ax.plot(thetas, vals, color=color, label=f"n={n}")

    ax.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.fill_between(thetas, 0, np.where(base > 0, 0.6, 0), alpha=0.07, color="blue")
    ax.fill_between(thetas, np.where(base < 0, -0.6, 0), 0, alpha=0.07, color="red")

    ax.set_xlabel("θ (radians)")
    ax.set_ylabel(r"$\langle \sigma_z^{(a)} \sigma_z^{(l)} \rangle$")
    ax.set_title("Effect of Data Copies n (Fig. 3 behavior)")
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax.set_ylim(-0.65, 0.65)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    ax2 = axes[1]
    for n, color in zip(n_vals, colors):
        vals = np.array([analytical_swap_kernel(t, n_copies=n) for t in thetas])
        preds = np.array([1 if v > 1e-10 else -1 if v < -1e-10 else 0 for v in vals])
        ax2.step(thetas, preds, where="mid", color=color, label=f"n={n}")

    ax2.set_xlabel("θ (radians)")
    ax2.set_ylabel("Predicted class (sign)")
    ax2.set_title("Classification decision vs θ")
    ax2.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax2.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(["Class 1", "Boundary", "Class 0"])
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Fig 5: Theory vs noisy simulation
# ---------------------------------------------------------------------------


def _to_pred(v: float) -> int:
    if v > 1e-10:
        return 0
    if v < -1e-10:
        return 1
    return -1


def _acc(preds: list[int], truth: list[int]) -> float:
    c = 0
    for p, t in zip(preds, truth):
        if p == t or p == -1 or t == -1:
            c += 1
    return c / len(truth) if truth else 0.0


def plot_theory_vs_noisy(
    thetas: np.ndarray,
    noise_results: dict,
    save_path: str | None = None,
):
    from experiments.toy_problem import analytical_swap_kernel, true_classification

    theory = np.array([analytical_swap_kernel(theta) for theta in thetas])
    sim = noise_results["expectation_noisy"]
    exp = 0.65 * theory + np.random.default_rng(99).normal(0, 0.01, len(thetas))

    fig, axes = plt.subplots(2, 1, figsize=(10, 9))

    ax = axes[0]
    ax.plot(thetas, theory, "k-", linewidth=2.5, label="Theory (noiseless)", zorder=5)
    ax.plot(
        thetas,
        sim,
        "bs-",
        markersize=4,
        linewidth=1.5,
        markevery=max(1, len(thetas) // 15),
        label="Simulation (noise model)",
    )
    ax.plot(
        thetas,
        exp,
        "r^",
        markersize=4,
        linewidth=1.5,
        linestyle="--",
        markevery=max(1, len(thetas) // 15),
        label="Experiment-like trace",
    )

    std = noise_results["std_errors"]
    ax.fill_between(thetas, sim - std, sim + std, alpha=0.15, color="blue", label="Simulation ±1σ")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("θ (radians)")
    ax.set_ylabel(r"$\langle \sigma_z^{(a)} \sigma_z^{(l)} \rangle$")
    ax.set_title("Theory vs noisy simulation vs experiment-style trace (Fig. 5 behavior)")
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax.set_ylim(-0.65, 0.65)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)

    ax2 = axes[1]
    true_labels = [true_classification(theta) for theta in thetas]
    pred_sim = [_to_pred(v) for v in sim]
    pred_exp = [_to_pred(v) for v in exp]

    for i, theta in enumerate(thetas):
        sim_ok = pred_sim[i] == true_labels[i] or pred_sim[i] == -1 or true_labels[i] == -1
        exp_ok = pred_exp[i] == true_labels[i] or pred_exp[i] == -1 or true_labels[i] == -1

        ax2.scatter(theta, -0.5, c="green" if sim_ok else "red", s=20)
        ax2.scatter(theta, 0.5, c="green" if exp_ok else "red", s=20)

    sim_acc = _acc(pred_sim, true_labels)
    exp_acc = _acc(pred_exp, true_labels)

    ax2.set_yticks([-0.5, 0.5])
    ax2.set_yticklabels([f"Simulation ({sim_acc*100:.1f}% acc.)", f"Experiment-like ({exp_acc*100:.1f}% acc.)"])
    ax2.set_xlabel("θ (radians)")
    ax2.set_title("Classification correctness (green=correct, red=incorrect)")
    ax2.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax2.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Hadamard vs swap-test
# ---------------------------------------------------------------------------


def plot_hadamard_vs_swaptest(thetas: np.ndarray, save_path: str | None = None):
    from experiments.toy_problem import analytical_hadamard_kernel, analytical_swap_kernel, get_test_state, get_training_data

    swap = np.array([analytical_swap_kernel(theta) for theta in thetas])
    had = np.array([analytical_hadamard_kernel(theta) for theta in thetas])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(thetas, swap, "b-", linewidth=2.5, label="Swap-test kernel (Eq. 9)")
    ax.plot(thetas, had, "r--", linewidth=2.5, label="Hadamard kernel (Eq. 6)")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("θ (radians)")
    ax.set_ylabel(r"$\langle \sigma_z^{(a)} \sigma_z^{(l)} \rangle$")
    ax.set_title("Hadamard vs swap-test")
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.annotate(
        "Hadamard ~ 0 everywhere",
        xy=(np.pi, 0),
        xytext=(np.pi, 0.25),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red",
        ha="center",
        fontsize=10,
    )

    x_train, _ = get_training_data()
    ip_real = np.array([np.real(np.conj(get_test_state(t)) @ x_train[0]) for t in thetas])
    ip_imag = np.array([np.imag(np.conj(get_test_state(t)) @ x_train[0]) for t in thetas])
    ip_abs = np.array([abs(np.conj(get_test_state(t)) @ x_train[0]) for t in thetas])

    ax2 = axes[1]
    ax2.plot(thetas, ip_real, "r-", label=r"Re$\langle\tilde{x}|x_1\rangle$")
    ax2.plot(thetas, ip_imag, "b--", label=r"Im$\langle\tilde{x}|x_1\rangle$")
    ax2.plot(thetas, ip_abs, "g-", label=r"$|\langle\tilde{x}|x_1\rangle|$")
    ax2.axhline(0, color="k", linewidth=0.8)
    ax2.set_xlabel("θ (radians)")
    ax2.set_ylabel("Inner product component")
    ax2.set_title("Inner-product decomposition")
    ax2.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax2.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Bloch sphere
# ---------------------------------------------------------------------------


def state_to_bloch(psi: np.ndarray):
    """Convert a single-qubit state to Bloch coordinates (x, y, z)."""
    psi = psi / np.linalg.norm(psi)
    alpha, beta = psi[0], psi[1]
    x = 2 * np.real(np.conj(alpha) * beta)
    y = 2 * np.imag(np.conj(alpha) * beta)
    z = abs(alpha) ** 2 - abs(beta) ** 2
    return float(x), float(y), float(z)


def plot_bloch_sphere(thetas: np.ndarray, save_path: str | None = None):
    from experiments.toy_problem import get_test_state, get_training_data

    fig = plt.figure(figsize=(12, 5))

    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    x_train, _ = get_training_data()
    bx1 = state_to_bloch(x_train[0])
    bx2 = state_to_bloch(x_train[1])
    btest = np.array([state_to_bloch(get_test_state(t)) for t in thetas])

    views = [
        (131, 30, 45, "Bloch sphere (3D)"),
        (132, 90, 0, "Top view (XY)"),
        (133, 0, 0, "Front view (XZ)"),
    ]

    for panel, elev, azim, title in views:
        ax = fig.add_subplot(panel, projection="3d")
        ax.plot_surface(xs, ys, zs, alpha=0.08, linewidth=0)
        ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.12, linewidth=0.3)

        ax.plot(btest[:, 0], btest[:, 1], btest[:, 2], color="black", linewidth=1.5, alpha=0.8)
        ax.scatter(*bx1, color="blue", s=80, marker="*", label="|x1⟩")
        ax.scatter(*bx2, color="red", s=80, marker="*", label="|x2⟩")

        # class-colored test points
        class0 = btest[::2]
        class1 = btest[1::2]
        ax.scatter(class0[:, 0], class0[:, 1], class0[:, 2], color="dodgerblue", s=10, alpha=0.8)
        ax.scatter(class1[:, 0], class1[:, 1], class1[:, 2], color="tomato", s=10, alpha=0.8)

        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim([-1.05, 1.05])
        ax.set_ylim([-1.05, 1.05])
        ax.set_zlim([-1.05, 1.05])
        ax.view_init(elev=elev, azim=azim)

    legend_handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="blue", markersize=10, label="Train |x1⟩"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red", markersize=10, label="Train |x2⟩"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="dodgerblue", markersize=6, label="Test points"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _save_if_requested(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Kernel matrix
# ---------------------------------------------------------------------------


def plot_kernel_matrix(save_path: str | None = None):
    from core.kernel import kernel_matrix
    from experiments.toy_problem import get_test_states, get_theta_range, get_training_data

    x_train, _ = get_training_data()
    test_states = get_test_states(get_theta_range(18))
    states = x_train + test_states

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, n in zip(axes, [1, 2, 5]):
        K = kernel_matrix(states, n_copies=n)
        im = ax.imshow(K, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"Kernel matrix, n={n}")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Helstrom equivalence
# ---------------------------------------------------------------------------


def plot_helstrom_equivalence(thetas: np.ndarray, save_path: str | None = None):
    from core.kernel import helstrom_expectation, helstrom_operator
    from experiments.toy_problem import analytical_swap_kernel, get_test_state, get_training_data

    x_train, labels = get_training_data()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    ax = axes[0]
    ax_diff = axes[1]

    for n, color in zip([1, 2, 3], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        A = helstrom_operator(x_train, labels, n_copies=n)
        swap_vals = np.array([analytical_swap_kernel(t, n_copies=n) for t in thetas])
        hel_vals = np.array([helstrom_expectation(get_test_state(t), A, n_copies=n) for t in thetas])
        diff = swap_vals - hel_vals

        ax.plot(thetas, swap_vals, color=color, linewidth=2, label=f"Swap kernel, n={n}")
        ax.plot(thetas, hel_vals, color=color, linestyle="--", linewidth=1.2, alpha=0.8, label=f"Helstrom, n={n}")
        ax_diff.plot(thetas, diff, color=color, linewidth=1.4, label=f"Δ, n={n}")

    ax.set_title("Swap-test kernel vs Helstrom expectation")
    ax.set_xlabel("θ (radians)")
    ax.set_ylabel("Expectation")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    ax_diff.axhline(0, color="k", linewidth=0.8)
    ax_diff.set_title("Numerical difference (should be ~0)")
    ax_diff.set_xlabel("θ (radians)")
    ax_diff.set_ylabel("Swap - Helstrom")
    ax_diff.grid(True, alpha=0.3)
    ax_diff.legend()

    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Circuit verification
# ---------------------------------------------------------------------------


def plot_circuit_verification(thetas: np.ndarray, save_path: str | None = None):
    from circuits.hadamard_classifier import HadamardClassifier
    from circuits.swap_test_classifier import SwapTestClassifier
    from experiments.toy_problem import (
        analytical_hadamard_kernel,
        analytical_swap_kernel,
        get_test_state,
        get_training_data,
    )

    x_train, labels = get_training_data()
    clf_swap = SwapTestClassifier(n_copies=1)
    clf_had = HadamardClassifier()

    swap_circ = []
    swap_theory = []
    had_circ = []
    had_theory = []

    for theta in thetas:
        xt = get_test_state(theta)
        r_swap = clf_swap.run(x_train, labels, xt)
        r_had = clf_had.run(x_train, labels, xt)

        swap_circ.append(r_swap["expectation_ZZ"])
        had_circ.append(r_had["expectation_ZZ"])
        swap_theory.append(analytical_swap_kernel(theta))
        had_theory.append(analytical_hadamard_kernel(theta))

    swap_circ = np.array(swap_circ)
    swap_theory = np.array(swap_theory)
    had_circ = np.array(had_circ)
    had_theory = np.array(had_theory)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(thetas, swap_theory, "k-", linewidth=2.5, label="Swap theory")
    ax.plot(thetas, swap_circ, "bo", markersize=3, alpha=0.8, label="Swap circuit")
    ax.plot(thetas, had_theory, "r--", linewidth=2.0, label="Hadamard theory")
    ax.plot(thetas, had_circ, "rs", markersize=2.7, alpha=0.8, label="Hadamard circuit")
    ax.set_title("Circuit outputs vs analytical formulas")
    ax.set_xlabel("θ (radians)")
    ax.set_ylabel(r"$\langle \sigma_z^{(a)} \sigma_z^{(l)} \rangle$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax2 = axes[1]
    ax2.plot(thetas, np.abs(swap_circ - swap_theory), "b-", label="|Swap error|")
    ax2.plot(thetas, np.abs(had_circ - had_theory), "r-", label="|Hadamard error|")
    ax2.set_yscale("log")
    ax2.set_title("Absolute numerical error")
    ax2.set_xlabel("θ (radians)")
    ax2.set_ylabel("Absolute error")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig
