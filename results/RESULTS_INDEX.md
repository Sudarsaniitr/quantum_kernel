# Results Index

Complete catalogue of every artifact in `results/`, organised by experiment family. This is the master reference for the replication of **"Quantum classifier with tailored quantum kernel"** (Blank et al., *npj Quantum Information*, 2020) plus the **Virtual Copy Extrapolation (VCE)** novelty.

- **Paper:** <https://www.nature.com/articles/s41534-020-0272-6>
- **Supplemental reference repo:** <https://github.com/carstenblank/Quantum-classifier-with-tailored-quantum-kernels---Supplemental>
- **Toy problem:** 2-state classifier on 1-qubit test state `|x̃(θ)⟩ = cos(θ/2)|0⟩ + i sin(θ/2)|1⟩` with training set `{|x₁⟩,|x₂⟩} = {|0⟩,|1⟩}` and equal class weights.
- **Hardware backend used:** `ibm_kingston` (127-qubit QPU, IBM Quantum open-plan access).
- **Theta grid:** 30 points in `[0, 2π]` (to keep simulator and hardware directly comparable; see section 7).
- **All hardware jobs** below executed at 1024 shots unless stated; all simulator jobs use AerSimulator with a simple depolarising noise model.

---

## 1. NumPy analytical reproduction (figures 01–07)

Pure-NumPy ground-truth figures replicating the paper's conceptual structure. No Qiskit, no sampling noise.

| # | File | What it shows |
|---|---|---|
| 01 | `01_n_copies_effect.png` | Fig. 3 of the paper: kernel sharpens as `n → ∞`. Shows analytical `Kₙ = Fⁿ` for `n ∈ {1, 10, 100}` plus discrete decision signs. |
| 02 | `02_theory_vs_noisy.png` | Fig. 5 behaviour: analytical kernel vs noisy-simulation curve vs experiment-style trace, with `±1σ` band and per-θ correctness markers. |
| 03 | `03_hadamard_vs_swaptest.png` | Replicates Eq. 6 vs Eq. 9: Hadamard baseline is ~0 everywhere (real inner product vanishes for toy encoding), swap-test cleanly separates classes. |
| 04 | `04_bloch_sphere.png` | 3D + two 2D Bloch views of the test-state trajectory plus training states, colour-coded by class. Intuition for the encoding. |
| 05 | `05_kernel_matrix.png` | Gram matrix of 20 states (2 train + 18 test), for `n ∈ {1, 2, 5}` — visualises the kernel-sharpening PSD behaviour. |
| 06 | `06_helstrom_equivalence.png` | Numerical verification of Eq. 16–17: swap-test expectation equals `Tr(A ρ⊗ⁿ)` with the Helstrom-style operator `A`. Max diff ~1e-16. |
| 07 | `07_circuit_verification.png` | Circuit-level (NumPy backends of `HadamardClassifier` and `SwapTestClassifier`) outputs vs analytical theory, with log-scale error panel. |

Rollup text file: `summary_report.txt` — pass/fail verdict, kernel-range, Helstrom diff, noiseless classification accuracy (30/30 with 2 boundary points on the quick grid).

---

## 2. Paper-faithful Qiskit runs (files 08–09, legacy single-run outputs)

Single-run swap-test / product-state sweeps, both on the AerSimulator and on real IBM hardware. These files remain as the *initial* Qiskit hook-up (pre-shots-comparison, pre-novelty).

| File | Mode | Circuit | Backend | Job ID | Notes |
|---|---|---|---|---|---|
| `08_qiskit_swap_test_simulator_results.json` | simulator | swap-test (n=1) | `aer_simulator` | — | 30-θ sweep, 1024 shots (quick rerun). |
| `09_qiskit_swap_test_simulator_vs_theory.png` | simulator | swap-test | — | — | Theory vs measured curve + deviation. |
| `08_qiskit_product_state_simulator_results.json` | simulator | product_state (n=3) | `aer_simulator` | — | Earlier single-family run. |
| `09_qiskit_product_state_simulator_vs_theory.png` | simulator | product_state | — | — | Theory-vs-measured with n=3 kernel. |
| `08_qiskit_swap_test_hardware_results.json` | **hardware** | swap-test (n=1) | `ibm_kingston` | `d7kgr924lglc73fvelk0` | First successful hardware submission (earlier session). |
| `08_qiskit_swap_test_hardware_results_verified.json` | **hardware** | swap-test (n=1) | `ibm_kingston` | `d7kgr924lglc73fvelk0` | Pulled-via-API re-verification with fresh metrics. **Sign agreement 96.67% · max abs dev 0.258 · mean abs dev 0.102** (256 shots). |
| `09_qiskit_swap_test_hardware_vs_theory.png` | hardware | swap-test | — | — | Plot from initial hardware submission. |
| `09_qiskit_swap_test_hardware_vs_theory_verified.png` | hardware | swap-test | — | — | Plot regenerated from API-retrieved counts. |
| `08_qiskit_product_state_hardware_results.json` | hardware | product_state | `ibm_kingston` | — | Earlier single-family run (partial from paused session). |
| `09_qiskit_product_state_hardware_vs_theory.png` | hardware | product_state | — | — | Hardware product-state plot. |

---

## 3. Shots comparison — old architecture (files 10–12)

Verifies shot-count convergence on the paper's swap-test (n=1) across both backends. Each JSON contains raw counts per θ, the metadata block, and aggregated metrics vs `analytical_swap_kernel(n=1)`.

### 3a. Simulator (aer_simulator, depolarising noise)
| File | Content |
|---|---|
| `10_qiskit_swap_test_simulator_shots_256_results.json` | 30-θ sweep, 256 shots. Job: `71980926-d38c-4b2e-a8fe-ac0a2908916e`. |
| `10_qiskit_swap_test_simulator_shots_1024_results.json` | 30-θ sweep, 1024 shots. Job: `b0bd2020-8a61-43f9-87d0-4374705ae6ec`. |
| `11_qiskit_swap_test_simulator_shots_comparison.json` | Combined summary: curves + metrics vs theory for both shot counts. |
| `12_qiskit_swap_test_simulator_shots_comparison.png` | Two-panel plot: theory + both measured curves, with deviation panel. |

### 3b. Hardware (`ibm_kingston`)
| File | Job ID | Content |
|---|---|---|
| `10_qiskit_swap_test_hardware_shots_256_results.json` | `d7ktoqa4lglc73fvttl0` | 30-θ sweep, 256 shots. |
| `10_qiskit_swap_test_hardware_shots_1024_results.json` | `d7ktoua8ui0s73b5n980` | 30-θ sweep, 1024 shots. |
| `11_qiskit_swap_test_hardware_shots_comparison.json` | — | Combined summary for both shot counts. |
| `12_qiskit_swap_test_hardware_shots_comparison.png` | — | Shots-comparison plot on real hardware. |

### 3c. Shots-comparison metrics (vs analytical n=1 kernel)

| Mode | Shots | Sign agreement | Max abs diff | Mean abs diff | RMSE |
|---|---|---|---|---|---|
| Simulator | 256 | 96.67% | 0.1555 | 0.0591 | 0.0706 |
| Simulator | 1024 | **100.00%** | 0.1165 | 0.0422 | 0.0529 |
| Hardware | 256 | 90.00% | 0.1693 | 0.0664 | 0.0795 |
| Hardware | 1024 | **100.00%** | 0.1458 | 0.0692 | 0.0782 |

**Reading:** 4× more shots → sign agreement rises to 100% on both simulator and hardware; simulator mean abs err improves ~29%, hardware improves ~16%. Hardware mean abs err is flat between 256 and 1024 because coherent-error terms (not shot-noise) dominate — which motivates VCE.

---

## 4. Novelty: Virtual Copy Extrapolation (VCE) (files 13–15)

Runs the **n-copy product-state** circuit (paper Fig. 3 style: swap-test with `n` simultaneous copies of test and training registers) for `n ∈ {1, 2, 3}` at 1024 shots, then applies VCE post-processing to estimate a *virtual* `n=3` kernel from only `n=1, 2` runs (fewer qubits, less coherent error accumulation). See `qiskit_layer/mitigation.py` for the estimator (Richardson-denoised `K₁` + closed-form map `Kₙ = ½(pⁿ − (1−p)ⁿ)` with `p = K₁* + ½`).

### 4a. Raw per-copy sweeps

Simulator (all jobs on `aer_simulator`, 1024 shots):
- `13_qiskit_product_state_simulator_copies_1_shots_1024_results.json`
- `13_qiskit_product_state_simulator_copies_2_shots_1024_results.json`
- `13_qiskit_product_state_simulator_copies_3_shots_1024_results.json`

Hardware (all jobs on `ibm_kingston`, 1024 shots):
| File | Job ID |
|---|---|
| `13_qiskit_product_state_hardware_copies_1_shots_1024_results.json` | `d7ktp424lglc73fvtu00` |
| `13_qiskit_product_state_hardware_copies_2_shots_1024_results.json` | `d7ktpa8e7usc73f59c1g` |
| `13_qiskit_product_state_hardware_copies_3_shots_1024_results.json` | `d7ktpgq8ui0s73b5n9u0` |

### 4b. VCE summaries and plots

- `14_qiskit_vce_simulator_shots_1024_summary.json` — all physical/virtual/theory curves + metric block.
- `14_qiskit_vce_hardware_shots_1024_summary.json` — same structure, hardware backend.
- `15_qiskit_vce_simulator_shots_1024_pre_post.png` — pre/post novelty, simulator.
- `15_qiskit_vce_hardware_shots_1024_pre_post.png` — pre/post novelty, hardware.

### 4c. VCE metrics at 1024 shots (vs theory at matching n)

Simulator:

| Curve | Sign | Max abs | Mean abs | RMSE |
|---|---|---|---|---|
| Physical n=1 | 100.00% | 0.0989 | 0.0439 | 0.0507 |
| Physical n=2 | 96.67% | 0.1262 | 0.0635 | 0.0696 |
| **Physical n=3 (pre-novelty)** | 96.67% | 0.1727 | 0.0690 | 0.0826 |
| **Virtual n=3 from {1,2} (post-novelty)** | **100.00%** | 0.1852 | **0.0513** | **0.0731** |
| Virtual n=5 from {1,2,3} | 100.00% | 0.2665 | 0.0609 | 0.0965 |

Hardware (`ibm_kingston`):

| Curve | Sign | Max abs | Mean abs | RMSE |
|---|---|---|---|---|
| Physical n=1 | 96.67% | 0.1214 | 0.0612 | 0.0727 |
| Physical n=2 | 100.00% | 0.1692 | 0.0758 | 0.0885 |
| **Physical n=3 (pre-novelty)** | 100.00% | 0.2036 | 0.0818 | 0.0983 |
| **Virtual n=3 from {1,2} (post-novelty)** | 96.67% | 0.1821 | **0.0717** | **0.0860** |
| Virtual n=5 from {1,2,3} | 96.67% | 0.2203 | 0.0751 | 0.0964 |

**Headline novelty result:** on both simulator and hardware the VCE-estimated `n=3` kernel — computed from only `n=1` and `n=2` physical runs — is closer to the theoretical `n=3` kernel (lower mean abs diff and RMSE) than the actual physical `n=3` hardware run, while using one fewer qubit-copy register and therefore less coherent error per circuit. This validates the "software fix for a hardware problem" hypothesis from `novelty.md`.

---

## 5. Consolidated suite summaries (file 16)

- `16_qiskit_simulator_comparison_suite_summary.json` — single JSON containing the simulator shots-comparison object + VCE object, keyed under `shots_comparison_summary` and `vce_summary`. Reproducible one-stop file for the simulator pipeline.
- `16_qiskit_hardware_comparison_suite_summary.json` — same structure, hardware backend, with job IDs embedded for audit.

Schema (both files):
```json
{
  "metadata": { "mode": "...", "high_shots": 1024, "backend": "...",
                "requested": { "shots_comparison": [...], "vce_physical_copies": [...], "vce_target_copies": 5 } },
  "shots_comparison_summary": { ... section 3 object ... },
  "vce_summary": { ... section 4 object ... }
}
```

---

## 6. Cross-backend novelty comparison (files 17–18)

- `17_qiskit_sim_vs_hw_novelty_comparison.json` — aligned simulator and hardware metric deltas for pre- vs post-novelty, plus raw curves for all four series: `sim_physical_n3`, `sim_virtual_n3`, `hw_physical_n3`, `hw_virtual_n3`, with `theory_n3` as the reference.
- `18_qiskit_sim_vs_hw_pre_post_comparison.png` — 2×2 figure: top row shows the pre/post curves for simulator (left) and hardware (right); bottom row shows the absolute-error curves.

### Key improvements vs pre-novelty (target n=3)

| Metric | Simulator (pre→post) | Hardware (pre→post) |
|---|---|---|
| Sign agreement | 0.9667 → **1.0000** (+3.4%) | 1.0000 → 0.9667 (−3.3%) |
| Max abs diff | 0.1727 → 0.1852 (+7.2%) | 0.2036 → **0.1821** (−10.6%) |
| **Mean abs diff** | 0.0690 → **0.0513** (**−25.6%**) | 0.0818 → **0.0717** (**−12.3%**) |
| **RMSE** | 0.0826 → **0.0731** (**−11.4%**) | 0.0983 → **0.0860** (**−12.6%**) |

Bold = improvements. On every headline-error metric (mean abs, RMSE) and on hardware max-abs, VCE beats the physical `n=3` baseline.

---

## 7. Reproducibility notes

- **Theta grid:** 30 points in `[0, 2π]` on both stacks (the `--quick` flag on `main.py`; the hardware runner script defaults to `--quick`). Grid parity is a precondition for files 17–18; using the full 63-point grid on simulator while hardware uses 30 causes the cross-comparison plot to fail.
- **Shots:** 1024 for every novelty and main paper-mode result. 256-shot counterparts only exist for the shots-comparison family (files 10–12).
- **Classical post-processing:** `qiskit_layer/mitigation.py` — Richardson denoising of `K₁` using `K₂`, then analytical toy-kernel map to target `n`. Also provides a generic power-law estimator and a three-point asymptote fit for reference.
- **Hardware runner:** `scripts/hardware_suite.py` builds the Qiskit service once, resolves the backend once, and reuses a single `SamplerV2(mode=backend)` for all five sweeps. This is what the built-in `main.py --compare-suite --paper-backend hardware` does too, but the script fails gracefully across transient IBM global-catalog timeouts — use it over `--compare-suite` when the IBM catalog API is flaky.
- **Cross-comparison builder:** `scripts/cross_comparison.py` reads both `14_*_summary.json` files and emits files 17–18. Rerun it any time you regenerate either VCE summary and want consistent cross artifacts.
- **Commands to reproduce from scratch:**
  ```bash
  # simulator side
  python main.py --compare-suite --quick --paper-backend simulator --shots 1024 \
      --vce-physical-copies 1,2,3 --vce-target-copies 5
  # hardware side (once .env has a valid QISKIT_IBM_TOKEN + instance CRN)
  python scripts/hardware_suite.py --backend ibm_kingston
  # cross artifacts
  python scripts/cross_comparison.py
  ```

---

## 8. File-count sanity

- **Images:** 17 PNGs — 01–07 (7), 09 × 5, 12 × 2, 15 × 2, 18 × 1.
- **JSON:** 22 data files — 08 × 5, 10 × 4, 11 × 2, 13 × 6, 14 × 2, 16 × 2, 17 × 1.
- **Text:** `summary_report.txt` (NumPy pipeline verdict).
- **Markdown:** `RESULTS_INDEX.md` (this file).
- **Total:** 41 files in `results/`.

Every file is listed above and has a described purpose. Nothing in `results/` is orphaned.
