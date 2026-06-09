# ECSS Verification Suite

Supplementary materials for:

- **Paper 1:** *"Structural Inheritance in Sensitivity Systems of Differential-Algebraic Equations"*
- **Paper 2:** *"Automatic Generation of Sensitivity Systems for Differential-Algebraic Equations via Multivariate Taylor Series"*

by Bo Cao (McMaster University, 2026).

## Contents

This repository provides an independent Python verification suite that reproduces the cross-solver validation results reported in both companion papers, plus additional validation at indices 4--5 beyond what the papers cover.

## Quick Start

```bash
pip install numpy scipy
python run_all.py
```

## What This Validates

| Test | What | Paper Reference |
|------|------|-----------------|
| `test_solver_independence.py` | SciPy Radau / RK45 / BDF vs. DAETS reference on simple pendulum ECSS | Paper 2, Section 7 |
| `test_high_index.py` | Hessenberg chains at indices 4--5 | Paper 1, numerical confirmation |
| `test_consistent_init.py` | Consistent-point inheritance (Lemma 3, Paper 1) | Paper 1, Lemma 3 |
| `test_npendulum.py` | Coupled N-pendulum scaling | Paper 2, Section 8 |
| `test_vanderpol.py` | Van der Pol oscillator sensitivities | Additional validation |
| `test_runtime_profile.py` | Runtime composition breakdown | Paper 2, Section 8 |
| `test_ad_comparison.py` | Comparison with FADBAD++ / CoDiPack | Paper 2, Section 1 |

## Structure

```
├── ecss_utils.py              # Core utilities (MVTS helpers, DAE definitions)
├── pryce_sa.py                # Independent Python implementation of Pryce SA
├── run_all.py                 # Master test runner
├── requirements.txt           # Python dependencies
├── results/                   # JSON output files
├── verify_cross_solver_high_index.py
├── verify_structure.py        # Σ-matrix and Jacobian structure checks
├── cross_solver_pendulum.py
├── cross_solver_ecss_pendulum.json   # Cross-solver results
└── ...
```

## Limitations

This Python suite **complements** the C++ ECSS builder implementation:

- ECSS equations are manually derived for each test system (the C++ builder handles auto-generation via MVTS arithmetic)
- SciPy integration uses analytically index-reduced ODE forms; the DAETS solver handles the original DAE structure directly
- Structural inheritance is verified by inspecting system structure, not by a full SA implementation

## Citation

If you use this code, please cite:

> B. Cao. Structural Inheritance in Sensitivity Systems of Differential-Algebraic Equations. Submitted, 2026.
>
> B. Cao. Automatic Generation of Sensitivity Systems for Differential-Algebraic Equations via Multivariate Taylor Series. Submitted, 2026.
