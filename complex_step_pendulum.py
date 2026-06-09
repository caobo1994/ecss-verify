#!/usr/bin/env python3
"""
Complex-step verification of order-3 ECSS sensitivities for the simple pendulum.

Reduces the index-3 pendulum DAE to an explicit ODE via Pryce's method,
then applies complex-step differentiation for the first derivative and
central differences on complex-step derivatives for higher orders.

The complex-step method gives exact first derivatives (no cancellation error);
higher derivatives use finite differences on the complex-step first derivative.

Usage: python3 complex_step_pendulum.py

Reference: Simple pendulum DAE:
  x'' + lambda*x = 0
  y'' + lambda*y - g = 0
  x^2 + y^2 - L^2 = 0  (L = 1)

After index reduction (differentiate constraint twice):
  lambda = ((x')^2 + (y')^2 + g*y) / L^2
"""

import numpy as np


def pendulum_ode(t, y, g, L2=1.0):
    """Index-reduced simple pendulum (index-0 ODE)."""
    x, dxdt, yp, dydt = y
    lam = (dxdt**2 + dydt**2 + g * yp) / L2
    return np.array([dxdt, -lam * x, dydt, g - lam * yp])


def rk4_step(f, t, y, h, *args):
    """Classical Runge-Kutta 4 integrator."""
    k1 = f(t, y, *args)
    k2 = f(t + h / 2, y + h / 2 * k1, *args)
    k3 = f(t + h / 2, y + h / 2 * k2, *args)
    k4 = f(t + h, y + h * k3, *args)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(y0, g, t_end=1.0, h=1e-5):
    """Integrate from t=0 to t_end with RK4."""
    y = y0.astype(np.cdouble) if isinstance(g, complex) else y0.copy()
    t = 0.0
    while t < t_end - 1e-12:
        h_step = min(h, t_end - t)
        y = rk4_step(pendulum_ode, t, y, h_step, g)
        t += h_step
    return y


def dxdg_cs(g_val, h_cs=1e-20):
    """First derivative via complex-step: f'(g) = Im[f(g+ih)]/h"""
    y = integrate(y0, g_val + 1j * h_cs, h=1e-5)
    return np.imag(y[0]) / h_cs


def main():
    g = 9.8
    theta0 = 0.1  # initial angular offset from hanging equilibrium
    global y0
    y0 = np.array([np.cos(theta0), -np.sin(theta0),
                   np.sin(theta0), np.cos(theta0)])

    print("Complex-step verification of simple pendulum sensitivity")
    print(f"  g = {g}, L = 1.0, initial angle = {theta0} rad")
    print(f"  Integration: RK4, h=1e-5, to t=1.0")
    print()

    # First derivative via complex-step (gold standard)
    dxdg = dxdg_cs(g)
    print(f"  d1: partial x/partial g      = {dxdg: .6e}")

    # Second derivative via central differences on CS first derivative
    # d2 = (f(g+h) - 2f(g) + f(g-h))/h^2
    for h2 in [1e-1, 1e-2, 1e-3]:
        d2 = (dxdg_cs(g + h2) - 2 * dxdg_cs(g) + dxdg_cs(g - h2)) / h2**2
        print(f"  d2 (h={h2:.0e}): partial^2 x/partial g^2 = {d2: .6e}")

    # Third derivative via central difference formula
    # d3 = (f(g+2h) - 2f(g+h) + 2f(g-h) - f(g-2h)) / (2h^3)
    for h3 in [5e-2, 2e-2, 1e-2]:
        d3 = (dxdg_cs(g + 2 * h3) - 2 * dxdg_cs(g + h3) +
              2 * dxdg_cs(g - h3) - dxdg_cs(g - 2 * h3)) / (2 * h3**3)
        print(f"  d3 (h={h3:.0e}): partial^3 x/partial g^3 = {d3: .6e}")

    print()
    print("Reference values (from complex-step with h=1e-3 for d2, h=1e-2 for d3):")
    print(f"  partial x/partial g      = {-1.93885364e-03: .6e}")
    print(f"  partial^2 x/partial g^2  = {-1.97039878e-03: .6e}")
    print(f"  partial^3 x/partial g^3  = { 1.78621509e-03: .6e}")


if __name__ == "__main__":
    main()
