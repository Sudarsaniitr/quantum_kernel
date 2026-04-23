# Quantum Classifier with Tailored Quantum Kernel

**Course project implementing and extending:**

> Blank, Park, Rhee, Petruccione — *"Quantum classifier with tailored quantum kernel"*
> *npj Quantum Information* **6**, 41 (2020).
> Paper: <https://www.nature.com/articles/s41534-020-0272-6>
> Supplemental code: <https://github.com/carstenblank/Quantum-classifier-with-tailored-quantum-kernels---Supplemental>

This repository contains a full, paper-faithful implementation of the classifier, run on both a noise-model simulator (AerSimulator) and a real IBM Quantum computer (`ibm_kingston`, 127-qubit QPU), plus an original novelty contribution called **Virtual Copy Extrapolation (VCE)**.

---

## The Research Paper (Summary)

The paper introduces a distance-based quantum classifier that uses the **quantum swap test** to compute a kernel (similarity measure) between quantum states. The key idea is that the kernel sharpens as the number of data copies `n` increases:

```
K_n(x̃) = Σ_m (-1)^{y_m} w_m |⟨x̃|x_m⟩|^{2n}
```

- As `n → ∞` the kernel approaches a Dirac-delta localised at the training states — perfect classification.
- The paper proves this is equivalent to the Helstrom measurement (optimal quantum decision rule).
- The circuit requires only `H → CSWAP → H` on an ancilla qubit and `2n+3` qubits total.

**Compared baseline:** The Hadamard classifier (prior work) uses `Re⟨x̃|x_m⟩`. The paper's toy problem is designed so that this real part is exactly 0 for all test angles, demonstrating a case the Hadamard classifier cannot solve but the swap-test can.

---

## Novelty Contribution — Virtual Copy Extrapolation (VCE)

Running `n` copies on a NISQ device multiplies the qubit count and circuit depth by `n`, amplifying hardware errors. The novelty asks:

> *Can we estimate the high-n kernel from low-n runs, using fewer qubits?*

**Method (Richardson denoising + analytical kernel map):**

1. Run `n=1` (3 qubits) and `n=2` (5 qubits) circuits on hardware.
2. Richardson-denoise the K₁ measurement: `K₁* = 2·K₁ − K₂`
3. Recover the underlying fidelity: `p = K₁* + 0.5`
4. Map to any target `n`: `K_n = ½·(p^n − (1−p)^n)`

**Results** (vs physical `n=3` run, 1024 shots, `ibm_kingston`):

| Method | Mean abs error | RMSE |
|---|---|---|
| Physical n=3 (actual hardware run) | 0.0818 | 0.0983 |
| **Virtual n=3 via VCE (our method)** | **0.0717** | **0.0860** |

VCE improves mean error by ~12% and RMSE by ~13% on real hardware, using one fewer qubit-copy register. This validates the "software fix for a hardware problem" hypothesis in NISQ-era quantum computing.

---

## Project Structure

```text
quantum/
├── main.py                          # CLI entry point for all experiments
├── requirements.txt                 # Base NumPy/matplotlib deps
├── requirements-qiskit.txt          # Qiskit + IBM runtime deps
├── .env.example                     # Template for IBM credentials
├── novelty.md                       # VCE novelty design document
│
├── core/
│   ├── kernel.py                    # Classical kernel math (Eqs. 6, 9, 16-17)
│   └── quantum_gates.py             # NumPy gate primitives (H, CSWAP, tensor products)
│
├── circuits/
│   ├── hadamard_classifier.py       # Hadamard circuit (paper baseline)
│   ├── swap_test_classifier.py      # Swap-test circuit (paper main contribution)
│   └── quantum_forking.py           # Quantum forking utilities
│
├── experiments/
│   ├── toy_problem.py               # Training/test states, analytical kernels
│   └── noise_simulation.py          # Monte-Carlo shot noise + amplitude damping
│
├── visualization/
│   └── plots.py                     # All figure generators (01–07, 09, 12, 15, 18)
│
├── qiskit_layer/
│   ├── circuits.py                  # Qiskit circuit builders (swap-test + n-copy)
│   ├── backends.py                  # AerSimulator + IBM backend helpers
│   ├── runner.py                    # Shot sweep executor (simulator & hardware)
│   ├── noise.py                     # Depolarising noise model builders
│   └── mitigation.py                # VCE estimators (Richardson + closed-form map)
│
├── scripts/
│   ├── hardware_suite.py            # Resilient consolidated IBM hardware runner
│   └── cross_comparison.py          # Sim-vs-hardware comparison builder
│
└── results/                         # All generated output artifacts (41 files)
    └── RESULTS_INDEX.md             # Full catalogue with metrics and job IDs
```

---

## Setup

**Base (NumPy analytical layer):**

```bash
pip install -r requirements.txt
```

**Full (Qiskit + IBM Runtime):**

```bash
pip install -r requirements-qiskit.txt
```

**IBM credentials** (only needed for hardware runs):

```bash
cp .env.example .env
# Edit .env: fill QISKIT_IBM_TOKEN and QISKIT_IBM_INSTANCE
```

---

## Running Experiments

### NumPy analytical suite (reproduces figures 01–07)

```bash
python main.py --quick          # 30 theta points, fast
python main.py                  # 63 theta points, full
```

### Verification only

```bash
python main.py --verify
```

Runs 8 property checks: normalization, Hadamard kernel = 0, swap kernel range,
2π periodicity, Helstrom equivalence to machine precision (~1e-16), PSD kernel
matrix, circuit-vs-analytical agreement, and 100% classification accuracy.

### Qiskit simulator suite (shots comparison + VCE novelty)

```bash
python main.py --compare-suite --quick --paper-backend simulator \
    --shots 1024 --vce-physical-copies 1,2,3 --vce-target-copies 5
```

Produces files 10–16 in `results/`.

### IBM hardware suite

```bash
# Requires .env with a valid QISKIT_IBM_TOKEN
python scripts/hardware_suite.py --backend ibm_kingston
```

Produces the hardware counterparts of files 10–16. Uses a single
`QiskitRuntimeService` for all five sweeps to avoid IBM Global Catalog timeouts.

### Simulator vs hardware comparison (files 17–18)

```bash
python scripts/cross_comparison.py
```

Reads both `14_*_summary.json` files and writes the `17_*` JSON + `18_*` PNG.

---

## Results Summary

All 41 output artifacts are catalogued in [results/RESULTS_INDEX.md](results/RESULTS_INDEX.md).

### NumPy analytical figures (01–07)

| File | What it shows |
|---|---|
| `01_n_copies_effect.png` | Kernel sharpens as n increases (paper Fig. 3) |
| `02_theory_vs_noisy.png` | Shot noise band around theory curve (paper Fig. 5 style) |
| `03_hadamard_vs_swaptest.png` | Hadamard = 0 everywhere; swap-test classifies correctly |
| `04_bloch_sphere.png` | 3D Bloch sphere: training states, test-state trajectory, class colours |
| `05_kernel_matrix.png` | Gram matrix showing PSD kernel for n=1,2,5 |
| `06_helstrom_equivalence.png` | Swap-test == Helstrom operator to ~1e-16 (Eqs. 16-17 verification) |
| `07_circuit_verification.png` | Circuit vs analytical, log-scale error panel |

### Qiskit shots comparison (10–12)

| Mode | Shots | Sign agreement | Mean abs err | RMSE |
|---|---|---|---|---|
| Simulator | 256 | 96.67% | 0.0591 | 0.0706 |
| Simulator | 1024 | **100.00%** | 0.0422 | 0.0529 |
| Hardware | 256 | 90.00% | 0.0664 | 0.0795 |
| Hardware (`ibm_kingston`) | 1024 | **100.00%** | 0.0692 | 0.0782 |

### VCE novelty — pre vs post comparison (13–18)

Hardware (`ibm_kingston`, 1024 shots, target n=3):

| Curve | Mean abs error | RMSE | Sign agreement |
|---|---|---|---|
| Physical n=1 | 0.0612 | 0.0727 | 96.67% |
| Physical n=2 | 0.0758 | 0.0885 | 100.00% |
| Physical n=3 (pre-novelty) | 0.0818 | 0.0983 | 100.00% |
| **Virtual n=3 via VCE (post-novelty)** | **0.0717** | **0.0860** | 96.67% |

VCE achieves lower mean error and RMSE than the actual n=3 hardware run,
while using one fewer copy register and shorter circuits.

---

## Implementation Notes

- **No legacy API**: uses `SamplerV2` from `qiskit-ibm-runtime`, not the removed `backend.run()`.
- **Channel migration**: automatically maps deprecated `ibm_quantum` channel to `ibm_quantum_platform`.
- **Theta grid parity**: all Qiskit runs use `--quick` (30 points over [0, 2π]) to ensure simulator and hardware grids match for cross-comparison.
- **Noise model**: simulator uses a simple per-gate depolarising model (single-qubit error `p=0.001`, two-qubit error `p=0.01`).
- **IBM hardware jobs** were executed on `ibm_kingston` (127-qubit Eagle QPU, open-plan access).
  - Job IDs are embedded in each result JSON for audit and reproduction.
