# Quantum Classifier with Tailored Quantum Kernel (Pure NumPy)

This project is a precise, reproducible implementation of the core ideas from:

Blank, Park, Rhee, Petruccione — *npj Quantum Information* **6**, 41 (2020)
DOI: [10.1038/s41534-020-0272-6](https://doi.org/10.1038/s41534-020-0272-6)

It compares:

- **Hadamard classifier** (baseline, uses $\Re\langle\tilde{x}|x_m\rangle$)
- **Swap-test classifier** (paper contribution, uses $|\langle\tilde{x}|x_m\rangle|^{2n}$)

and reproduces key theoretical/experimental-style figures.

---

## Project layout

```text
quantum/
├── main.py
├── requirements.txt
├── core/
│   ├── quantum_gates.py
│   └── kernel.py
├── circuits/
│   ├── hadamard_classifier.py
│   ├── swap_test_classifier.py
│   └── quantum_forking.py
├── experiments/
│   ├── toy_problem.py
│   └── noise_simulation.py
├── visualization/
│   └── plots.py
└── results/
```

---

## Install

```bash
pip install -r requirements.txt
```

---

## Run

### Full run (recommended)

```bash
python main.py
```

This performs:
1. mathematical verification,
2. all figure generation,
3. summary report generation.

### Faster run

```bash
python main.py --quick
```

### Verification only

```bash
python main.py --verify
```

### Summary report only

```bash
python main.py --report
```

---

## Generated outputs

Files are written to `results/`:

1. `01_n_copies_effect.png`
2. `02_theory_vs_noisy.png`
3. `03_hadamard_vs_swaptest.png`
4. `04_bloch_sphere.png`
5. `05_kernel_matrix.png`
6. `06_helstrom_equivalence.png`
7. `07_circuit_verification.png`
8. `summary_report.txt`

---

## How to interpret results

### 1) `01_n_copies_effect.png`
Shows kernel sharpening as $n$ increases:
- $n=1$: smooth response
- large $n$: sharper transitions (more localized kernel behavior)

### 2) `02_theory_vs_noisy.png`
Compares noiseless theory with shot-based noisy simulation and an experiment-like trace.
Main insight: amplitude shrinks with noise, but decision sign mostly remains stable.

### 3) `03_hadamard_vs_swaptest.png`
Demonstrates why Hadamard fails for the toy states:
$\Re\langle\tilde{x}|x_m\rangle\approx0$ for all $\theta$, while swap-test remains informative.

### 4) `04_bloch_sphere.png`
Geometric picture of train/test states on the Bloch sphere and decision trajectory.

### 5) `05_kernel_matrix.png`
Gram matrix $K_{ij}=|\langle x_i|x_j\rangle|^{2n}$.
Diagonal entries are 1 (self-fidelity), and higher $n$ makes matrix more diagonal.

### 6) `06_helstrom_equivalence.png`
Numerically verifies swap-test expectation equals Helstrom-operator expectation.
Near-zero difference confirms theoretical equivalence.

### 7) `07_circuit_verification.png`
Compares circuit simulation against analytical formulas; errors should be near floating-point precision.

### 8) `summary_report.txt`
Compact numeric summary of all key checks and outcomes.

---

## Notes

- This implementation is intentionally pure NumPy for full transparency.
- Optional Qiskit dependencies are listed if you later want real backend execution.
