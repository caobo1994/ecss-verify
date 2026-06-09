#!/usr/bin/env python3
"""
Numerical integration of ECSS — validates that ECSS-generated sensitivity
equations produce correct sensitivity trajectories.

System: Exponential ODE  x' = p·x  with sensitivity parameter p.
  Analytic:  x(t) = e^{pt},  ∂x/∂p = t·e^{pt},  ∂²x/∂p² = t²·e^{pt}

The ECSS for SOV={(0),(1),(2)} has n=3 equations:
  f₀: x' - p·x = 0
  f₁: x_p' - x - p·x_p = 0
  f₂: x_pp' - 2·x_p - p·x_pp = 0

We integrate this as an ODE and compare against the analytic solution.

Usage: python integrate_ecss.py
"""

import numpy as np
from scipy.integrate import solve_ivp
import math


def ecss_ode(t, y, p):
    """ECSS for x'=p·x with sensitivities up to order 2.
    
    y = [x, x_p, x_pp] where x_p = ∂x/∂p, x_pp = ∂²x/∂p²
    """
    x, xp, xpp = y
    return [
        p * x,                  # f₀: x' = p·x
        x + p * xp,             # f₁: x_p' = x + p·x_p
        2 * xp + p * xpp,       # f₂: x_pp' = 2·x_p + p·x_pp
    ]


def analytic(t, p):
    """Analytic solution and sensitivities."""
    x = math.exp(p * t)
    xp = t * math.exp(p * t)
    xpp = t * t * math.exp(p * t)
    return x, xp, xpp


def run():
    p = 1.0
    t_span = (0.0, 5.0)
    y0 = [1.0, 0.0, 0.0]  # x(0)=1, ∂x/∂p(0)=0, ∂²x/∂p²(0)=0
    
    sol = solve_ivp(
        lambda t, y: ecss_ode(t, y, p),
        t_span, y0,
        method='Radau',
        rtol=1e-12, atol=1e-14,
        dense_output=True,
    )
    
    print("=" * 72)
    print("  ECSS Numerical Integration — Exponential ODE")
    print("  x' = p·x,  p = 1,  x(0) = 1")
    print("  ECSS orders: 0, 1, 2  (n=3)")
    print("=" * 72)
    
    # Compare at several time points
    t_check = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    
    print(f"\n  {'t':>6s}  {'x (num)':>12s}  {'x (ana)':>12s}  {'err':>10s}  "
          f"{'x_p (num)':>12s}  {'x_p (ana)':>12s}  {'err':>10s}  "
          f"{'x_pp (num)':>12s}  {'x_pp (ana)':>12s}  {'err':>10s}")
    print("  " + "-" * 130)
    
    max_err = 0.0
    for t in t_check:
        y_num = sol.sol(t)  # dense output interpolation
        x_num, xp_num, xpp_num = y_num
        x_ana, xp_ana, xpp_ana = analytic(t, p)
        
        e0 = abs(x_num - x_ana)
        e1 = abs(xp_num - xp_ana)
        e2 = abs(xpp_num - xpp_ana)
        max_err = max(max_err, e0, e1, e2)
        
        print(f"  {t:6.1f}  {x_num:12.8f}  {x_ana:12.8f}  {e0:10.2e}  "
              f"{xp_num:12.8f}  {xp_ana:12.8f}  {e1:10.2e}  "
              f"{xpp_num:12.8f}  {xpp_ana:12.8f}  {e2:10.2e}")
    
    print(f"\n  Maximum error across all orders: {max_err:.2e}")
    print(f"  {'PASS' if max_err < 1e-8 else 'FAIL'}: ECSS matches analytic")

    # Also verify with finite differences
    print(f"\n  ── Finite Difference Verification ──")
    h = 1e-6
    t_test = 2.0
    
    # Nominal
    sol_nom = solve_ivp(lambda t, y: [p*y[0]], [0, t_test], [1.0],
                         method='RK45', rtol=1e-12, atol=1e-14, dense_output=True)
    x_nom = sol_nom.y[0, -1]
    
    # Perturbed
    sol_pert = solve_ivp(lambda t, y: [(p+h)*y[0]], [0, t_test], [1.0],
                          method='RK45', rtol=1e-12, atol=1e-14, dense_output=True)
    x_pert = sol_pert.y[0, -1]
    
    fd_xp = (x_pert - x_nom) / h
    ecss_xp = sol.sol(t_test)[1]
    ana_xp = analytic(t_test, p)[1]
    
    print(f"  At t = {t_test}:")
    print(f"    ECSS  ∂x/∂p = {ecss_xp:.10f}")
    print(f"    FD    ∂x/∂p = {fd_xp:.10f}")
    print(f"    Ana   ∂x/∂p = {ana_xp:.10f}")
    print(f"    ECSS vs FD  error: {abs(ecss_xp - fd_xp):.2e}")
    print(f"    ECSS vs Ana error: {abs(ecss_xp - ana_xp):.2e}")

    return max_err < 1e-8


if __name__ == "__main__":
    import sys
    ok = run()
    sys.exit(0 if ok else 1)
