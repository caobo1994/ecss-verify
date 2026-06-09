#!/usr/bin/env python3
"""
RC Circuit higher-order sensitivity verification using SymPy.

Computes analytic sensitivities dv/dR through order 4 for the
index-1 RC circuit C v' + v/R = 0 with v(0) = V0.
Analytic solution: v(t) = V0 * exp(-t/(R*C)).

This provides independent verification of ECSS-computed sensitivities
at orders 1-4, closing the "Theory only" gap for index-1 systems
at orders >= 2 in the index x order coverage table.
"""

import json
from pathlib import Path

import numpy as np
import sympy as sp

# ---------------------------------------------------------------------------
# Symbolic computation
# ---------------------------------------------------------------------------

V0, R, C, t = sp.symbols('V0 R C t', positive=True, real=True)
v = V0 * sp.exp(-t / (R * C))

print("=" * 72)
print("RC Circuit Higher-Order Sensitivity Verification")
print("Using SymPy for analytic reference values")
print("=" * 72)
print()
print(f"System: C v' + v/R = 0,  v(0) = V0")
print(f"Solution: v(t) = V0 * exp(-t/(R*C))")
print()

# Parameters matching thesis
params = {V0: 5, R: 1, C: 1, t: 1}
print(f"Parameters: V0={params[V0]}, R={params[R]}, C={params[C]}, t={params[t]}")
print()

# Compute analytic sensitivities through order 4
results = []
for q in range(5):
    dv_dRq = sp.diff(v, R, q)
    dv_dRq_simplified = sp.simplify(dv_dRq)
    analytic_val = float(dv_dRq_simplified.subs(params).evalf(50))
    results.append({
        "order": q,
        "analytic": analytic_val,
        "expression": str(dv_dRq_simplified),
    })

# ECSS values from the thesis (tab:rc_circuit_results for order 1)
# Order 0-1 from existing table; orders 2-4 need ECSS run
# For now, compute analytic-only and show the expected structure
print(f"{'Order':>5s}  {'Analytic (SymPy, 50-digit)':>28s}")
print("-" * 50)
for r in results:
    print(f"{r['order']:5d}  {r['analytic']:28.15e}")

print()
print("Analytic expressions:")
for r in results:
    print(f"  Order {r['order']}: {r['expression']}")

# ---------------------------------------------------------------------------
# Write results for inclusion in thesis tables
# ---------------------------------------------------------------------------

out = {
    "system": "RC circuit",
    "equation": "C v' + v/R = 0",
    "parameters": {"V0": params[V0], "R": params[R], "C": params[C], "t": params[t]},
    "sensitivities": [
        {"order": r["order"], "analytic": r["analytic"]}
        for r in results
    ],
}

out_path = Path(__file__).with_name("rc_higher_order_result.json")
out_path.write_text(json.dumps(out, indent=2) + "\n")
print(f"\nResults written to {out_path}")

# ---------------------------------------------------------------------------
# If ECSS values are available, compare
# ---------------------------------------------------------------------------

# Placeholder: ECSS values to be filled from C++ driver output
# Format: ecss_values[q] = ecss_computed_value
# Values from MVTS_rc_circuit_ho (order 4, h=1, R=1, C=1, v0=5, t=1)
ecss_values = {
    0: 1.839397205857211,       # v(t=1)
    1: 1.839397205857212,       # dv/dR (analytic: same as v at t=R=C=1)
    2: -1.839397205857209,      # d^2v/dR^2
    3: 1.839397205857166,       # d^3v/dR^3
    4: 1.839397205858138,       # d^4v/dR^4
}

print("\n" + "=" * 72)
if all(q in ecss_values for q in range(5)):
    print(f"{'Order':>5s}  {'Analytic':>20s}  {'ECSS':>20s}  {'Abs Error':>12s}  {'Rel Error':>12s}")
    print("-" * 80)
    for r in results:
        q = r["order"]
        if q in ecss_values:
            ecss = ecss_values[q]
            abs_err = abs(r["analytic"] - ecss)
            rel_err = abs_err / abs(r["analytic"]) if abs(r["analytic"]) > 0 else 0
            print(f"{q:5d}  {r['analytic']:20.15e}  {ecss:20.15e}  {abs_err:12.2e}  {rel_err:12.2e}")
    print()
    print("RC CIRCUIT HIGHER-ORDER VERIFICATION COMPLETE")
else:
    print("ECSS values for orders 2-4 not yet available.")
    print("Run MVTS_rc_circuit.cpp with SOV including orders 2-4,")
    print("then update ecss_values dict in this script.")
    print("=" * 72)
