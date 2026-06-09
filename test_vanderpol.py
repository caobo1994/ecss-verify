#!/usr/bin/env python3
"""Generate Van der Pol ECSS validation data.

Solves the Van der Pol system and its first-order sensitivity equations
with respect to mu using scipy's Radau integrator at tight tolerance,
then compares against central finite differences at h=1e-5.
"""
import numpy as np
from scipy.integrate import solve_ivp


def vdp_ecss(t, y, mu=1.0):
    """Van der Pol system + first-order sensitivity w.r.t. mu.
    State: [x, y, x_mu, y_mu]
    """
    x, y_var, x_mu, y_mu = y
    # Original system
    dx = y_var
    dy = mu * (1 - x**2) * y_var - x
    # First-order sensitivity equations (differentiated w.r.t. mu)
    dx_mu = y_mu
    dy_mu = (1 - x**2) * y_var + mu * (-2 * x * x_mu) * y_var + mu * (1 - x**2) * y_mu - x_mu
    return [dx, dy, dx_mu, dy_mu]


def main():
    mu = 1.0
    t_span = (0, 1)
    y0 = [2.0, 0.0, 0.0, 0.0]  # x(0)=2, y(0)=0, dx/dmu(0)=0, dy/dmu(0)=0

    # High-tolerance reference solution
    sol = solve_ivp(vdp_ecss, t_span, y0, method='Radau',
                    rtol=1e-12, atol=1e-14, args=(mu,))
    t_final = sol.t[-1]
    x_ecss = sol.y[0, -1]
    y_ecss = sol.y[1, -1]
    x_mu_ecss = sol.y[2, -1]
    y_mu_ecss = sol.y[3, -1]

    # Central finite differences at h=1e-5
    h = 1e-5
    # Perturb mu and re-solve original 2D system
    def vdp_original(t, y, mu_val):
        x, y_var = y
        dx = y_var
        dy = mu_val * (1 - x**2) * y_var - x
        return [dx, dy]

    sol_plus = solve_ivp(vdp_original, t_span, [2.0, 0.0], method='Radau',
                         rtol=1e-12, atol=1e-14, args=(mu + h,))
    sol_minus = solve_ivp(vdp_original, t_span, [2.0, 0.0], method='Radau',
                          rtol=1e-12, atol=1e-14, args=(mu - h,))

    x_fd = (sol_plus.y[0, -1] - sol_minus.y[0, -1]) / (2 * h)
    y_fd = (sol_plus.y[1, -1] - sol_minus.y[1, -1]) / (2 * h)

    print(f"Van der Pol ECSS validation at t={t_final:.1f}, mu={mu}, h_fd={h}")
    print(f"x              = {x_ecss:.14e}")
    print(f"y              = {y_ecss:.14e}")
    print(f"dx/dmu (ECSS)  = {x_mu_ecss:.14e}")
    print(f"dx/dmu (FD)    = {x_fd:.14e}")
    print(f"  abs diff     = {abs(x_mu_ecss - x_fd):.2e}")
    print(f"dy/dmu (ECSS)  = {y_mu_ecss:.14e}")
    print(f"dy/dmu (FD)    = {y_fd:.14e}")
    print(f"  abs diff     = {abs(y_mu_ecss - y_fd):.2e}")
    print(f"Max FD error   = {max(abs(x_mu_ecss - x_fd), abs(y_mu_ecss - y_fd)):.2e}")


if __name__ == "__main__":
    main()
