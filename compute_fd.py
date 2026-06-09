#!/usr/bin/env python3
"""
Compute actual finite-difference values for the simple pendulum
by integrating the index-reduced ODE form at perturbed parameter
values (g +/- h).

This produces real FD data for fd_richardson.py, replacing the
previously fabricated values.

The simple pendulum (index-3 DAE):
  x'' + lambda*x = 0
  y'' + lambda*y - g = 0
  x^2 + y^2 - L^2 = 0

After index reduction (differentiate constraint twice, solve for
lambda), the system reduces to an index-0 ODE in [x, x', y, y']:
  x'  = vx
  vx' = -lambda * x
  y'  = vy
  vy' = -lambda * y + g
where lambda = -(vx^2 + vy^2)/(x^2 + y^2) + g*y/(x^2 + y^2)

(This is derived by differentiating x^2 + y^2 = L^2 twice and
substituting into the dynamics equations.)
"""

import numpy as np
from scipy.integrate import solve_ivp
import json, sys, os


def pendulum_ode(t, y, g, L=1.0):
    """Right-hand side of the index-reduced simple pendulum ODE.
    y = [x, vx, y, vy]
    """
    x, vx, y_coord, vy = y
    r2 = x * x + y_coord * y_coord

    # Lagrange multiplier from acceleration-level constraint:
    #   Differentiating x^2 + y^2 = L^2 twice:
    #     xx'' + yy'' = -(x')² - (y')²
    #   Substituting x'' = -lambda*x, y'' = -lambda*y + g:
    #     lambda = (vx^2 + vy^2 + g*y) / (x^2 + y^2)
    lam = (vx * vx + vy * vy + g * y_coord) / r2

    return [vx, -lam * x, vy, -lam * y_coord + g]


def integrate_pendulum(g, L=1.0, t_span=(0, 10), **kwargs):
    """Integrate the simple pendulum with gravity g.
    Returns final state [x, vx, y, vy] at t_span[1].

    Initial conditions match the thesis (Chpt7, sbsc:pendulum):
      x(0) = cos(0.1), y(0) = sin(0.1)
      Initial tangential velocity omega = 1
      vx(0) = -sin(0.1), vy(0) = cos(0.1)
    """
    theta0 = 0.1
    x0 = np.cos(theta0)
    y0 = np.sin(theta0)
    omega = 1.0
    vx0 = -np.sin(theta0) * omega
    vy0 = np.cos(theta0) * omega

    y0_vec = [x0, vx0, y0, vy0]

    sol = solve_ivp(
        lambda t, y: pendulum_ode(t, y, g, L),
        t_span, y0_vec,
        method='RK45',
        rtol=kwargs.get('rtol', 1e-12),
        atol=kwargs.get('atol', 1e-14),
        max_step=kwargs.get('max_step', 0.01),
    )

    return sol.y[:, -1]  # final state


def compute_fd_convergence(g_nom, hs, t_span=(0, 1)):
    """Compute FD convergence data for multiple step sizes.
    Returns sensitivity dx/dg estimated via central FD at each h.
    """
    print(f"Computing FD convergence for simple pendulum")
    print(f"  g = {g_nom}, t_span = {t_span}")
    print(f"  step sizes: {hs}")
    print()

    results = {}
    for h in hs:
        print(f"  h = {h:.1e} ...", end=" ", flush=True)
        # Integrate at g+h
        y_plus = integrate_pendulum(g_nom + h, t_span=t_span)
        # Integrate at g-h
        y_minus = integrate_pendulum(g_nom - h, t_span=t_span)

        # Central finite difference for dx/dg
        dx_dg_fd = (y_plus[0] - y_minus[0]) / (2 * h)
        results[float(h)] = float(dx_dg_fd)
        print(f"dx/dg = {dx_dg_fd:.10e}")

    return results


def compute_second_fd(g_nom, hs, t_span=(0, 1)):
    """Compute second-order FD for d^2x/dg^2."""
    results = {}
    for h in hs:
        print(f"  h = {h:.1e} ...", end=" ", flush=True)
        y0 = integrate_pendulum(g_nom, t_span=t_span)
        y_plus = integrate_pendulum(g_nom + h, t_span=t_span)
        y_minus = integrate_pendulum(g_nom - h, t_span=t_span)
        d2x_dg2_fd = (y_plus[0] - 2 * y0[0] + y_minus[0]) / (h * h)
        results[float(h)] = float(d2x_dg2_fd)
        print(f"d2x/dg2 = {d2x_dg2_fd:.10e}")
    return results


if __name__ == "__main__":
    outfile = sys.argv[1] if len(sys.argv) > 1 else "fd_pendulum_data.json"

    g_nom = 9.8
    t_span = (0, 1)

    # Step sizes for Richardson extrapolation
    hs = [1e-2, 5e-3, 2.5e-3, 1e-3]

    print("=" * 60)
    print("Computing FD convergence data for simple pendulum")
    print("(This may take a few minutes -- 8 ODE integrations)")
    print("=" * 60)
    print()

    fd1 = compute_fd_convergence(g_nom, hs, t_span)
    print()
    fd2 = compute_second_fd(g_nom, hs, t_span)

    data = {
        "system": "simple_pendulum",
        "g_nom": g_nom,
        "L": 1.0,
        "t_span": list(t_span),
        "hs": [float(h) for h in hs],
        "fd_first_order": fd1,
        "fd_second_order": fd2,
    }

    with open(outfile, 'w') as f:
        json.dump(data, f, indent=2)

    print()
    print(f"FD data saved to {outfile}")
    print()
    print("Summary:")
    print(f"  dx/dg via FD at h=1e-2:    {fd1[1e-2]:.10e}")
    print(f"  dx/dg via FD at h=1e-3:    {fd1[1e-3]:.10e}")
    print(f"  d2x/dg2 via FD at h=1e-2:  {fd2[1e-2]:.10e}")
    print(f"  d2x/dg2 via FD at h=1e-3:  {fd2[1e-3]:.10e}")
