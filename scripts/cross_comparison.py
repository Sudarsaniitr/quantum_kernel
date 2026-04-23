"""
scripts/cross_comparison.py
===========================
Build simulator-vs-hardware cross comparison artifacts.

Inputs (must already exist):
  results/14_qiskit_vce_simulator_shots_1024_summary.json
  results/14_qiskit_vce_hardware_shots_1024_summary.json

Outputs:
  results/17_qiskit_sim_vs_hw_novelty_comparison.json
  results/18_qiskit_sim_vs_hw_pre_post_comparison.png
"""

from __future__ import annotations

import json
import os
import sys
from typing import Sequence

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _metric_delta(pre: dict, post: dict) -> dict:
    out = {}
    for k in ("sign_agreement", "max_abs_diff", "mean_abs_diff", "rmse"):
        if k not in pre or k not in post:
            continue
        delta_abs = float(post[k] - pre[k])
        delta_rel = float((post[k] - pre[k]) / pre[k]) if pre[k] != 0 else 0.0
        out[k] = {
            "pre": float(pre[k]),
            "post": float(post[k]),
            "delta_abs": delta_abs,
            "delta_rel": delta_rel,
        }
    return out


def main() -> int:
    res_dir = os.path.join(ROOT, "results")
    sim_path = os.path.join(res_dir, "14_qiskit_vce_simulator_shots_1024_summary.json")
    hw_path = os.path.join(res_dir, "14_qiskit_vce_hardware_shots_1024_summary.json")
    for p in (sim_path, hw_path):
        if not os.path.exists(p):
            print(f"[err] missing input: {p}")
            return 1

    sim = _load(sim_path)
    hw = _load(hw_path)

    thetas = np.array(sim["metadata"]["thetas"], dtype=float)
    theory_n3 = np.array(sim["theory_curves"]["3"], dtype=float)

    sim_phys_n3 = np.array(sim["physical_curves"]["3"], dtype=float)
    sim_virtual_n3 = np.array(sim["virtual_curves"]["virtual_n3_from_12"], dtype=float)
    hw_phys_n3 = np.array(hw["physical_curves"]["3"], dtype=float)
    hw_virtual_n3 = np.array(hw["virtual_curves"]["virtual_n3_from_12"], dtype=float)

    # ----- aggregated cross-comparison summary -----
    cross = {
        "metadata": {
            "thetas": sim["metadata"]["thetas"],
            "shots": sim["metadata"]["shots"],
            "simulator_backend": sim.get("metadata", {}).get("backend", "aer"),
            "hardware_backend": hw["metadata"].get("backend"),
            "target_copies": 3,
        },
        "simulator": {
            "pre_novelty_physical_n3": sim["metrics"]["pre_novelty_physical_n3_vs_theory_n3"],
            "post_novelty_virtual_n3": sim["metrics"]["post_novelty_virtual_n3_vs_theory_n3"],
            "improvement_vs_theory_n3": _metric_delta(
                sim["metrics"]["pre_novelty_physical_n3_vs_theory_n3"],
                sim["metrics"]["post_novelty_virtual_n3_vs_theory_n3"],
            ),
        },
        "hardware": {
            "pre_novelty_physical_n3": hw["metrics"]["pre_novelty_physical_n3_vs_theory_n3"],
            "post_novelty_virtual_n3": hw["metrics"]["post_novelty_virtual_n3_vs_theory_n3"],
            "improvement_vs_theory_n3": _metric_delta(
                hw["metrics"]["pre_novelty_physical_n3_vs_theory_n3"],
                hw["metrics"]["post_novelty_virtual_n3_vs_theory_n3"],
            ),
        },
        "cross_simulator_vs_hardware": {
            "physical_n3": {
                "sim_vs_theory": sim["metrics"]["physical_n3_vs_theory_n3"],
                "hw_vs_theory": hw["metrics"]["physical_n3_vs_theory_n3"],
            },
            "virtual_n3": {
                "sim_vs_theory": sim["metrics"]["virtual_n3_from_12_vs_theory_n3"],
                "hw_vs_theory": hw["metrics"]["virtual_n3_from_12_vs_theory_n3"],
            },
        },
        "curves": {
            "theory_n3": [float(x) for x in theory_n3],
            "sim_physical_n3": [float(x) for x in sim_phys_n3],
            "sim_virtual_n3": [float(x) for x in sim_virtual_n3],
            "hw_physical_n3": [float(x) for x in hw_phys_n3],
            "hw_virtual_n3": [float(x) for x in hw_virtual_n3],
        },
    }

    out_json = os.path.join(res_dir, "17_qiskit_sim_vs_hw_novelty_comparison.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(cross, f, indent=2)
    print(f"[save] results/17_qiskit_sim_vs_hw_novelty_comparison.json")

    # ----- side-by-side plot -----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    def _panel(ax, measured_phys, measured_virt, title):
        ax.plot(thetas, theory_n3, "k-", linewidth=2.2, label="Theory n=3")
        ax.plot(thetas, measured_phys, "o-", color="#1f77b4", markersize=3, linewidth=1.4,
                label="Before novelty (physical n=3)")
        ax.plot(thetas, measured_virt, "o-", color="#d62728", markersize=3, linewidth=1.4,
                label="After novelty (virtual n=3 from n=1,2)")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(title)
        ax.set_xlabel("θ (radians)")
        ax.set_ylabel(r"$\langle \sigma_z^{(a)} \sigma_z^{(l)} \rangle$")
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    def _err_panel(ax, measured_phys, measured_virt, title):
        ax.plot(thetas, np.abs(measured_phys - theory_n3), color="#1f77b4", linewidth=1.8,
                label="|physical - theory|")
        ax.plot(thetas, np.abs(measured_virt - theory_n3), color="#d62728", linewidth=1.8,
                label="|virtual - theory|")
        ax.set_title(title)
        ax.set_xlabel("θ (radians)")
        ax.set_ylabel("Absolute error")
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    _panel(axes[0, 0], sim_phys_n3, sim_virtual_n3, "Simulator: pre vs post novelty")
    _panel(axes[0, 1], hw_phys_n3, hw_virtual_n3,
           f"Hardware [{hw['metadata'].get('backend','')}]: pre vs post novelty")
    _err_panel(axes[1, 0], sim_phys_n3, sim_virtual_n3, "Simulator: error magnitude")
    _err_panel(axes[1, 1], hw_phys_n3, hw_virtual_n3, "Hardware: error magnitude")

    plt.tight_layout()
    out_png = os.path.join(res_dir, "18_qiskit_sim_vs_hw_pre_post_comparison.png")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] results/18_qiskit_sim_vs_hw_pre_post_comparison.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
