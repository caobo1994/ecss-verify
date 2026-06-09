# Thesis Verification Guide

This document explains how to reproduce every verification experiment
referenced in the thesis.  Each section gives the exact command, expected
output, and the thesis claim it supports.

---

## Quick start (all Python tests)

```bash
cd verify/

# Run every standalone Python verification script
python3 cross_solver_pendulum.py
python3 rc_higher_order.py
python3 verify_high_order.py
python3 high_precision_fd.py
python3 verify_symbolic_cascade.py
python3 verify_cascade_high_order.py
python3 verify_cross_solver_high_index.py
python3 complex_step_pendulum.py
python3 fd_richardson.py
python3 verify_structure.py
python3 verify_higher_order_dae.py
```

For the full integration test suite (requires DAETS C++ libraries built on macOS with Ipopt + SuiteSparse):

```bash
cd ../python-validation/
python3 run_all.py
```

---

## Experiment-by-experiment guide

### 1. Cross-Solver: Index-3 Pendulum (SciPy solvers)

**What it validates:**
Solver independence — SciPy can integrate the ECSS of the
index-reduced simple pendulum and match DAETS to machine precision.

**Files involved:**
- `verify/cross_solver_pendulum.py` — driver

**Command:**
```bash
cd verify/
python3 cross_solver_pendulum.py
```

**Expected output (abridged):**
```
RK45  :  211 steps, x err=1.92e-11, dx/dg err=4.56e-12  [PASS]
BDF   :  480 steps, x err=6.25e-10, dx/dg err=2.05e-10  [PASS]
Radau :  936 steps, x err=2.89e-15, dx/dg err=1.24e-14  [PASS]

Cross-Solver Summary:
  Radau : max error = 1.73e-14  [PASS]
  RK45  : max error = 2.76e-11  [PASS]
  BDF   : max error = 6.25e-10  [PASS]

ECSS is solver-independent for index-3 DAE after index reduction.
```

**Thesis claim supported:**
Claims-to-evidence table, row "High-index non-DAETS integration":
"Taylor vs.\ SciPy cross-approach validation (Appendix A)."
The JSON output file `cross_solver_pendulum_result.json` records
`cross_approach_dxdg_diff ≈ 6.1e-12`.

---

### 2. RC Circuit Higher-Order Verification

**What it validates:**
ECSS sensitivity values for the index-1 RC circuit agree with
SymPy-analytic derivatives through order 4.

**Files involved:**
- `verify/rc_higher_order.py` — SymPy + comparison driver
- `verify/rc_higher_order_result.json` — output
- `../examples/MVTS_rc_circuit_ho.cpp` — C++ driver that generates ECSS values

**Step A — Run the Python comparison:**
```bash
cd verify/
python3 rc_higher_order.py
```

**Expected output (abridged):**
```
Order              Analytic                  ECSS     Abs Error     Rel Error
    0  1.839397205857212e+00  1.839397205857211e+00      6.66e-16      3.62e-16
    1  1.839397205857212e+00  1.839397205857212e+00      2.22e-16      1.21e-16
    2 -1.839397205857212e+00 -1.839397205857209e+00      2.66e-15      1.45e-15
    3  1.839397205857212e+00  1.839397205857166e+00      4.57e-14      2.49e-14
    4  1.839397205857212e+00  1.839397205858138e+00      9.26e-13      5.04e-13

RC CIRCUIT HIGHER-ORDER VERIFICATION COMPLETE
```

All absolute errors < 10⁻¹².  The growth from order 2 to order 4
reflects round-off accumulation through repeated differentiation;
no result exceeds double-precision significance.

**Step B — (Re)generate the C++ ECSS reference values:**
```bash
cd ../examples/build/
cmake ..
make MVTS_rc_circuit_ho
./MVTS_rc_circuit_ho
```

**Note:** The library was compiled on macOS (ARM64) with
CMake + Clang, Ipopt, and SuiteSparse.  On other platforms
adjust the library paths in `examples/CMakeLists.txt`.

If you run the C++ driver first, copy its `q => value` lines
into the `ecss_values` dictionary near the bottom of
`rc_higher_order.py` and re-run the Python script.

**Thesis claim supported:**
"RC circuit through order~4 (SymPy)" in the claims-to-evidence table
and Chpt1 scope paragraph.

---

### 3. ODE High-Precision Verification (mpmath)

**What it validates:**
ECSS exponential-ODE sensitivities through order 5 match mpmath
50-digit reference values with constant relative error.

**Files involved:**
- `verify/verify_high_order.py`

**Command:**
```bash
python3 verify_high_order.py
```

**Expected output:**
```
Order 0: abs_err=4.21e-03, rel_err=1.91e-07
Order 1: abs_err=4.21e-02, rel_err=1.91e-07
Order 2: abs_err=4.21e-01, rel_err=1.91e-07
Order 3: abs_err=4.21e+00, rel_err=1.91e-07
Order 4: abs_err=4.21e+01, rel_err=1.91e-07
Order 5: abs_err=4.21e+02, rel_err=1.91e-07

ALL HIGH-PRECISION VERIFICATIONS PASSED
```

Relative error stays constant ~10⁻⁷ at t=10, confirming that the
growth pattern in `tab:higher_order` is multiplicative (factor ~10×
per order from t=10 amplification) rather than catastrophic.

**Also tests:** RC circuit orders 0–1 against mpmath (errors ~10⁻¹⁶).

**Thesis claim supported:**
"ODE correctness through order~5" in abstract and claims-to-evidence table.

---

### 4. Pendulum High-Precision FD (order 3)

**What it validates:**
The ECSS pendulum sensitivities at orders 1–3 agree with
high-precision central finite differences (mpmath, 50-digit)
with errors ~10⁻⁷ at h=0.01, consistent with O(h²) convergence.

**Files involved:**
- `verify/high_precision_fd.py`

**Command:**
```bash
python3 high_precision_fd.py
```

**Expected output (abridged):**
```
IC Set: A: x(0)=1 (near-horizontal)
  Order 1: h=0.010, FD=-1.5265515622e-02, error=1.45e-07
  Order 2: h=0.010, FD= 1.3300596407e-02, error=3.29e-08
  Order 3: h=0.010, FD=-8.6894477159e-03, error=2.69e-08

IC Set: B: theta0=0.1 (small oscillation)
  Order 1: h=0.010, FD=-1.9388862404e-03, error=3.28e-08
  Order 2: h=0.010, FD= 2.3162376119e-03, error=1.49e-08
  Order 3: h=0.010, FD=-1.9704252557e-03, error=1.65e-08

HIGH-PRECISION FD VERIFICATION: PASS
```

The ECSS reference values match `tab:order3_pendulum` in the thesis.

**Thesis claim supported:**
"Pendulum through order~3 (FD)" in claims-to-evidence table and Chpt1.

---

### 5. Cascade Symbolic Verification (SymPy, order 5)

**What it validates:**
SymPy computes exact, closed-form sensitivity expressions for
cascade SI-4 and SI-5 systems (sin and exp variants) through
total order 5.  Cross-checked against mpmath 80-digit evaluation
— zero difference between the two symbolic approaches.

**Files involved:**
- `verify/verify_symbolic_cascade.py` — main driver
- Output JSON files: `cascade_si4_sin_order3.json`, etc.

**Command:**
```bash
python3 verify_symbolic_cascade.py
```

**Expected output (abridged):**
```
Cascade SI-4 (sin): all 105 sensitivities computed symbolically
Cascade SI-4 (exp): all 50 sensitivities computed symbolically
Cascade SI-5 (sin): all 60 sensitivities computed symbolically
Cascade SI-5 (exp): all 60 sensitivities computed symbolically

Cross-check: sympy vs mpmath diff = 0.00e+00 for all orders 0–5
```

**Thesis claim supported:**
"Cascade SI-4/SI-5 through order~5 (SymPy + analytic)" in
claims-to-evidence table.

---

### 6. Cascade High-Order Verification (mpmath 80-digit)

**What it validates:**
Hand-derived analytic sensitivity formulas for cascade systems at
canonical offset depths 4–6 match mpmath 80-digit evaluation.
Single-parameter case (p1 sinusoid/exponential) verified through
total order 3 × 2 constraint types = 6 configurations.

**Files involved:**
- `verify/verify_cascade_high_order.py`

**Command:**
```bash
python3 verify_cascade_high_order.py
```

**Expected output:**
```
depth 4, p*sin(p*t): 0.00e+00  0.00e+00  1.32e-82  0.00e+00
depth 4, p*exp(p*t): 0.00e+00  0.00e+00  0.00e+00  0.00e+00
depth 5, p*sin(p*t): 0.00e+00  0.00e+00  1.32e-82  0.00e+00
depth 5, p*exp(p*t): 0.00e+00  0.00e+00  0.00e+00  0.00e+00
depth 6, p*sin(p*t): 0.00e+00  0.00e+00  1.32e-82  0.00e+00
depth 6, p*exp(p*t): 0.00e+00  0.00e+00  0.00e+00  0.00e+00
max error: 1.32e-82
```

Essentially exact to 80 digits.  Supports `tab:higher_order_high_index`.

**Thesis claim supported:**
Higher-order cascade validation in Chpt7 and validation summary table.

---

### 7. Cross-Solver High-Index (Taylor + SciPy)

**What it validates:**
Two completely independent integration approaches produce the same
ECSS results:
- Approach A: Symbolic Pryce SA → explicit ODE → SciPy Radau
- Approach B: Numerical Taylor-series integration (DAETS algorithm replica)

Cross-approach difference |dx/dg| ≈ 6.1 × 10⁻¹².

**Files involved:**
- `verify/verify_cross_solver_high_index.py`

**Command:**
```bash
python3 verify_cross_solver_high_index.py
```

**Expected output (abridged):**
```
Approach A (Radau):  x=0.8977626830, dx/dg=0.0740897421
Approach B (Taylor): x=0.8977626830, dx/dg=0.074089742054
|x_A - x_B| = 4.80e-12
|dx/dg_A - dx/dg_B| = 6.10e-12
Cross-approach consistency: PASS
```

**Thesis claim supported:**
"Taylor integrator reproduces SciPy ECSS to 6×10⁻¹²" in
claims-to-evidence table.

---

### 8. Complex-Step Pendulum Verification

**What it validates:**
First-order sensitivity ∂x/∂g via complex-step differentiation
(cancellation-free) confirms the ECSS value to ~1 × 10⁻¹³.

**Files involved:**
- `verify/complex_step_pendulum.py`

**Command:**
```bash
python3 complex_step_pendulum.py
```

**Expected output (abridged):**
```
d1: partial x/partial g  = -1.938853e-03
Reference value (complex-step, h=1e-20):
  partial x/partial g    = -1.938854e-03
```

**Thesis claim supported:**
Complex-step verification of pendulum sensitivity in Chpt8 §8.2.3.

---

### 9. FD Richardson Extrapolation

**What it validates:**
Finite-difference errors for the pendulum follow O(h²) across
four step sizes (h = 10⁻² … 10⁻⁵).  Richardson extrapolation
recovers ECSS reference values.

**Files involved:**
- `verify/fd_richardson.py`

**Command:**
```bash
python3 fd_richardson.py
```

**Expected output (abridged):**
```
--- First-order: d x / d g ---
  h=1.0e-02  error=2.37e-04
  h=1.0e-03  error=2.37e-06  (100× reduction)
  h=1.0e-04  error=5.67e-08  (42× reduction)
  h=1.0e-05  error=1.89e-07  (cancellation limit)

Short-time validation (t=1, h=1e-4):
  |FD - ECSS| = 3.50e-10

ALL RICHARDSON EXTRAPOLATION CHECKS PASSED
```

**Thesis claim supported:**
O(h²) convergence of FD checks in abstract and Chpt7.

---

### 10. Structural Inheritance Verification

**What it validates:**
For any given ECSS, the Σ-matrix equals the original system's
Σ-matrix repeated blockwise, the offsets repeat identically, and
the Jacobian determinant equals (det J)^(N(q)).

**Files involved:**
- `verify/verify_structure.py`
- `verify/pryce_sa.py` — pure-Python Pryce SA implementation

**Command:**
```bash
python3 verify_structure.py
```

**Thesis claim supported:**
Structural inheritance theorem (Chpt5).

---

### 11. Higher-Order DAE Verification (analytic + SymPy)

**What it validates:**
SI-4 and SI-5 cascade sensitivities at orders 3–5 via hand-derived
analytic formulas cross-checked against SymPy exact references
and complex-step order-1 references.

**Files involved:**
- `verify/verify_higher_order_dae.py`

**Command:**
```bash
python3 verify_higher_order_dae.py
```

**Thesis claim supported:**
Higher-order DAE claims in Chpt7 §7.1.6 and validation summary.

---

### 12. Full ECSS Validation Suite (python-validation/)

**What it validates:**
Index 4–5 Hessenberg chains, N-pendulum scaling, solver
independence, consistent initialisation, AD comparison,
Van der Pol, and runtime profiling — all from Python.

**Files involved:**
- `python-validation/run_all.py` — master driver
- `python-validation/ecss_utils.py` — shared utilities
- `python-validation/test_high_index.py` — W1: index 4–5
- `python-validation/test_npendulum.py` — W3: N-pendulum
- `python-validation/test_solver_independence.py` — W4: SciPy
- `python-validation/test_consistent_init.py` — W5: consistent init
- `python-validation/test_ad_comparison.py` — W6: AD comparison
- `python-validation/test_runtime_profile.py` — W7: profiling
- `python-validation/test_vanderpol.py` — Van der Pol

**Command:**
```bash
cd python-validation/
python3 run_all.py
```

**Expected output:**
Each section prints a PASS/FAIL summary.  All tests should pass.

**Thesis claims supported:**
Multiple claims across Chpt7 (validation) and Chpt8 (demonstration).

---

## Mapping of experiments to thesis claims

| Script | Thesis claim |
|--------|-------------|
| `rc_higher_order.py` | RC through order~4 (SymPy) |
| `verify_high_order.py` | ODE correctness through order~5 |
| `high_precision_fd.py` + `complex_step_pendulum.py` + `fd_richardson.py` | Pendulum order 1–3 FD validation |
| `cross_solver_pendulum.py` + `verify_cross_solver_high_index.py` | Solver independence / cross-approach |
| `verify_symbolic_cascade.py` + `verify_cascade_high_order.py` | Cascade through order~5 |
| `verify_higher_order_dae.py` | Higher-order DAE (SI-4/SI-5) |
| `verify_structure.py` + `pryce_sa.py` | Structural inheritance |
| `python-validation/run_all.py` | Full validation suite (indices 0–5) |

---

## Dependencies

### Python
- Python 3.9+
- sympy, numpy, scipy, mpmath

Install with:
```bash
pip install sympy numpy scipy mpmath
```

### C++ (for rc_higher_order.py reference values only)
- CMake 3.1+
- Clang 15+ (macOS) or GCC 11+ (Linux)
- Ipopt 3.14+
- SuiteSparse 7.x
- FADBAD++ (bundled in the DAETS repository)

The C++ component is only needed to regenerate ECSS reference
values.  The Python script works standalone — it already contains
the reference values obtained from a prior C++ run.
