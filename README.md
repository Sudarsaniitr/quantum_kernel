# Quantum Classifier with Tailored Quantum Kernel

This repository now has **two execution layers**:

1. **NumPy analytical layer** for transparent equation-level validation.
2. **Qiskit paper-mode layer** for shot-based simulation, noise modeling, and IBM hardware runs.

Target paper:

Blank, Park, Rhee, Petruccione — *npj Quantum Information* **6**, 41 (2020)
DOI: [10.1038/s41534-020-0272-6](https://doi.org/10.1038/s41534-020-0272-6)

The paper reports both realistic-noise simulation and IBM quantum cloud experiments. The project is structured to support both paths.

---

## Project layout

```text
quantum/
├── main.py
├── requirements.txt
├── requirements-qiskit.txt
├── .env.example
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
├── qiskit_layer/
│   ├── circuits.py
│   ├── noise.py
│   ├── backends.py
│   └── runner.py
└── results/
```

---

## Install

Base (NumPy mode):

```bash
pip install -r requirements.txt
```

Paper-faithful Qiskit mode:

```bash
pip install -r requirements-qiskit.txt
```

---

## Run

### Full NumPy suite

```bash
python main.py
```

### Quick NumPy suite

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

### Qiskit paper-mode (simulator)

```bash
python main.py --paper-mode --paper-backend simulator --paper-circuit swap_test --shots 8192
```

Product-state n-copy run (Qiskit-heavy path):

```bash
python main.py --paper-mode --paper-backend simulator --paper-circuit product_state --paper-copies 5 --shots 8192
```

### Qiskit paper-mode (hardware submit)

1. Copy `.env.example` to `.env` and fill values (`QISKIT_IBM_TOKEN`, optional backend/instance values).
2. Submit hardware job:

```bash
python main.py --paper-mode --paper-only --paper-backend hardware --paper-circuit swap_test --ibm-backend ibm_kyoto --shots 8192
```

To block until completion (may take a long time in queue):

```bash
python main.py --paper-mode --paper-only --paper-backend hardware --paper-circuit swap_test --ibm-backend ibm_kyoto --shots 8192 --hardware-wait
```

---

## Outputs

NumPy outputs in `results/`:

1. `01_n_copies_effect.png`
2. `02_theory_vs_noisy.png`
3. `03_hadamard_vs_swaptest.png`
4. `04_bloch_sphere.png`
5. `05_kernel_matrix.png`
6. `06_helstrom_equivalence.png`
7. `07_circuit_verification.png`
8. `summary_report.txt`

Qiskit paper-mode outputs:

9. `08_qiskit_<circuit_family>_<backend_mode>_results.json`
10. `09_qiskit_<circuit_family>_<backend_mode>_vs_theory.png` (if expectations are available)

---

## Notes

- NumPy mode is ideal for fast theory debugging.
- Qiskit mode is the path that most closely matches the paper's experimental methodology.
- Moving from Qiskit-ready to IBM hardware is **moderate** difficulty: token setup and queue/transpilation constraints are the main practical hurdles.
