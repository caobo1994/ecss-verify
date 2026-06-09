#!/usr/bin/env python3
"""
Cross-solver validation: Index-3 simple pendulum ECSS integrated with SciPy Radau/BDF.

Reduces the index-3 pendulum DAE to index-0 ODE (eliminate lambda algebraically),
constructs the ECSS for the reduced ODE by differentiating the ODE rhs w.r.t. g,
then integrates with scipy.integrate.solve_ivp.

This demonstrates that the ECSS, after the same index reduction applied
to the original DAE, is a standard ODE integrable by any ODE solver.
"""
import numpy as np
from scipy.integrate import solve_ivp
import time

# ── Index-reduced simple pendulum (index-0 ODE, 4 states) ──
# x' = u, y' = v, u' = -lambda*x, v' = g - lambda*y
# lambda = (u^2 + v^2 + g*y) / L^2   (from differentiating constraint twice)
# With L=1: lambda = u^2 + v^2 + g*y

def pendulum_ode(t, y, g):
    x, u, yp, v = y
    lam = u*u + v*v + g*yp
    return [u, -lam*x, v, g - lam*yp]

# ── ECSS for the reduced ODE (order 1, parameter g) ──
# 4 original states + 4 sensitivity states = 8 total
# Let s_x = ∂x/∂g, s_u = ∂u/∂g, s_y = ∂y/∂g, s_v = ∂v/∂g
#
# Differentiate ODE wrt g:
# s_x' = s_u
# s_y' = s_v
# lam = u^2 + v^2 + g*yp
# ∂lam/∂g = 2u*s_u + 2v*s_v + yp + g*s_y
# s_u' = -lam*s_x - x*(∂lam/∂g)
# s_v' = 1 - lam*s_y - yp*(∂lam/∂g)

def ecss_rhs(t, y, g):
    x, u, yp, v, sx, su, sy, sv = y
    lam = u*u + v*v + g*yp
    dlam = 2*u*su + 2*v*sv + yp + g*sy
    return [
        u, -lam*x, v, g - lam*yp,
        su, -lam*sx - x*dlam,
        sv, 1 - lam*sy - yp*dlam
    ]


def main():
    g = 9.8
    theta0 = 0.1
    # IC: x=cos(theta0), u=-sin(theta0), y=sin(theta0), v=cos(theta0)
    # All sensitivities = 0 at t=0 (initial state independent of g)
    y0 = [np.cos(theta0), -np.sin(theta0),
          np.sin(theta0),  np.cos(theta0),
          0.0, 0.0, 0.0, 0.0]
    t_span = (0.0, 1.0)

    print("=" * 70)
    print("Cross-Solver: Index-3 Pendulum ECSS (reduced to ODE) with SciPy")
    print("=" * 70)
    print(f"g={g}, L=1, theta0={theta0}")
    print()

    methods = ['RK45', 'BDF', 'Radau']
    results = {}

    for method in methods:
        start = time.time()
        sol = solve_ivp(ecss_rhs, t_span, y0, args=(g,),
                        method=method, rtol=1e-10, atol=1e-12,
                        dense_output=False)
        elapsed = time.time() - start

        if not sol.success:
            print(f"  {method}: FAILED - {sol.message}")
            continue

        tf = sol.t[-1]
        yf = sol.y[:, -1]
        xf, ypf = yf[0], yf[2]
        sxf, syf = yf[4], yf[6]

        # ECSS reference values (from C++ driver, previous run)
        ref_x = -0.9973812220738063
        ref_y = 0.0723235636190628
        ref_sx = -0.0019388534001821
        ref_sy = -0.0267378414022992

        err_x = abs(xf - ref_x)
        err_y = abs(ypf - ref_y)
        err_sx = abs(sxf - ref_sx)
        err_sy = abs(syf - ref_sy)

        print(f"  {method:6s}: {len(sol.t):4d} steps, {sol.nfev:5d} f-evals, "
              f"{elapsed:.3f}s")
        print(f"    x err={err_x:.2e}, y err={err_y:.2e}, "
              f"dx/dg err={err_sx:.2e}, dy/dg err={err_sy:.2e}")
        print(f"    r^2 = {xf**2 + ypf**2:.10f}")

        results[method] = max(err_x, err_y, err_sx, err_sy)

    print()
    print("-" * 70)
    print("Cross-Solver Summary (errors vs DAETS ECSS reference):")
    for method, max_err in sorted(results.items(), key=lambda x: x[1]):
        status = "PASS" if max_err < 1e-7 else "CHECK"
        print(f"  {method:6s}: max error = {max_err:.2e}  [{status}]")
    
    # Verify all methods agree with each other
    errors = list(results.values())
    if errors and max(errors) < 1e-6:
        print("\nAll solvers agree with DAETS ECSS to < 1e-6.")
        print("ECSS is solver-independent for index-3 DAE after index reduction.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
