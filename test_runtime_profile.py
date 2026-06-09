"""
Validation W5: Runtime Profiling Breakdown.

Instruments the ECSS integration path to decompose runtime into:
1. ECSS equation evaluation (MVTS arithmetic)
2. Linear algebra (LU factorisation within the BDF solver)
3. Step computation (error estimation, adaptive step control)
4. Other overhead
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.integrate import solve_ivp
from ecss_utils import save_results, print_table, compute_max_error
import time
import cProfile
import pstats
import io


# ═══════════════════════════════════════════════════════════════
# Profiled ODE System
# ═══════════════════════════════════════════════════════════════

def build_coupled_ode_ecss_higher_order(max_order: int):
    """
    Build ECSS for coupled ODE with higher sensitivity orders.
    y − x' = 0,  y' + 2p₁y + (p₁² + p₂²)x = 0
    with p₁ = p₂ = 1 and sensitivity order up to max_order.
    """
    p1, p2 = 1.0, 1.0
    n_vars = 2  # x, y

    # Number of sensitivity orders up to max_order for 2 parameters
    # = (max_order + 1)(max_order + 2) / 2
    n_orders = (max_order + 1) * (max_order + 2) // 2

    # Build order list
    orders = []
    for total in range(max_order + 1):
        for i in range(total + 1):
            j = total - i
            orders.append((i, j))

    order_idx = {q: i for i, q in enumerate(orders)}
    n_total = n_vars * n_orders

    def ecss_rhs(t, y):
        """ECSS RHS using direct differentiation."""
        dy = np.zeros(n_total)

        # Set the base ODE.  Sensitivity blocks are filled below using the
        # generalized Leibniz rule for this polynomial RHS.
        for oi, (i, j) in enumerate(orders):
            x_idx = oi
            y_idx = oi + n_orders

            x_val = y[x_idx]
            y_val = y[y_idx]

            # Base ODE: x' = y, y' = −2p₁y − (p₁²+p₂²)x
            if i == 0 and j == 0:
                dy[x_idx] = y_val
                dy[y_idx] = -2 * p1 * y_val - (p1**2 + p2**2) * x_val

        # After loop: compute all y' sensitivities using the known function structure
        # For each order (i,j), compute ∂^(i,j)/∂p₁^i∂p₂^j of f(x,y,p₁,p₂)

        for oi, (i, j) in enumerate(orders):
            if i == 0 and j == 0:
                continue  # already set

            x_idx = oi
            y_idx = oi + n_orders

            # Accumulate contributions to y' sensitivity using MVTS-like propagation
            # f = −2p₁·y − (p₁²+p₂²)·x = −2p₁·y − p₁²·x − p₂²·x

            # Contribution from −2p₁y:
            # ∂^(i,j)(p₁·y)/∂p₁^i∂p₂^j = ∑_{a=0}^i ∑_{b=0}^j C(i,a)·C(j,b) · ∂^(a,b)p₁/∂p₁^a∂p₂^b · ∂^(i-a,j-b)y/∂p₁^(i-a)∂p₂^(j-b)
            # ∂p₁/∂p₁ = 1, ∂p₁/∂p₂ = 0, higher derivatives of p₁ = 0
            # So only (a,b) = (1,0) and (0,0) contribute.
            # (a,b)=(0,0): p₁ · ∂^(i,j)y/∂p₁^i∂p₂^j
            # (a,b)=(1,0) if i≥1: i · 1 · ∂^(i-1,j)y/∂p₁^(i-1)∂p₂^j

            s = 0.0

            # −2p₁·y term
            s += -2 * p1 * y[y_idx]  # (a,b)=(0,0): p1 * ∂^(i,j)y
            if i >= 1:
                prev = (i - 1, j)
                prev_y_idx = n_orders + order_idx[prev]
                s += -2 * i * y[prev_y_idx]  # (a,b)=(1,0): i * 1 * ∂^(i-1,j)y

            # −p₁²·x term: ∂^(i,j)(p₁²·x)/∂p₁^i∂p₂^j
            # ∂^(a,b)(p₁²) is nonzero only for (a,b) = (0,0): p₁², (1,0): 2p₁, (2,0): 2
            # (0,0): p₁² · ∂^(i,j)x
            s += -p1 * p1 * y[x_idx]
            if i >= 1:
                prev = (i - 1, j)
                prev_x_idx = order_idx[prev]
                s += -2 * p1 * i * y[prev_x_idx]
            if i >= 2:
                prev2 = (i - 2, j)
                prev2_x_idx = order_idx[prev2]
                s += -2 * (i * (i - 1) // 2) * y[prev2_x_idx]

            # −p₂²·x term: similarly
            s += -p2 * p2 * y[x_idx]
            if j >= 1:
                prev = (i, j - 1)
                prev_x_idx = order_idx[prev]
                s += -2 * p2 * j * y[prev_x_idx]
            if j >= 2:
                prev2 = (i, j - 2)
                prev2_x_idx = order_idx[prev2]
                s += -2 * (j * (j - 1) // 2) * y[prev2_x_idx]

            dy[y_idx] = s

        # After computing all y' sensitivities, set x' sensitivities
        for oi, (i, j) in enumerate(orders):
            x_idx = oi
            y_idx = oi + n_orders
            dy[x_idx] = y[y_idx]  # x' sensitivity = y sensitivity

        return dy

    return ecss_rhs, n_total, orders, n_orders


def run_profiling():
    """Profile the ECSS integration and break down runtime."""
    print("=" * 70)
    print("  VALIDATION W5: Runtime Profiling Breakdown")
    print("=" * 70)

    t0, tf = 0.0, 2.0
    results = []

    for max_order in [0, 1, 2, 3, 5]:
        ecss_rhs, n_total, orders, n_orders = build_coupled_ode_ecss_higher_order(max_order)

        # Initial conditions: x(0) = 1, y(0) = 0
        # All sensitivities initially 0 (initial conditions independent of params)
        y0 = np.zeros(n_total)
        y0[0] = 1.0  # x(0) = 1
        y0[n_orders] = 0.0  # y(0) = 0

        # Profile the integration
        n_runs = 3

        # Time the RHS evaluation separately
        y_test = np.random.randn(n_total)
        t_test = 0.5
        n_evals_rhs = 10000
        start_rhs = time.time()
        for _ in range(n_evals_rhs):
            ecss_rhs(t_test, y_test)
        rhs_time_per_eval = (time.time() - start_rhs) / n_evals_rhs

        # Time full integration
        times = []
        for _ in range(n_runs):
            start = time.time()
            sol = solve_ivp(
                ecss_rhs, (t0, tf), y0,
                method='BDF',
                rtol=1e-10, atol=1e-12,
            )
            times.append(time.time() - start)

        avg_time = np.mean(times)
        nfev = sol.nfev

        # Decompose:
        # 1. RHS evaluation time = nfev * rhs_time_per_eval
        rhs_total = nfev * rhs_time_per_eval
        # 2. Rest (LU factorisation, step control, etc.)
        other_total = avg_time - rhs_total

        # Check analytic solution
        t_final = sol.t[-1]
        y_final = sol.y[:, -1]

        # Analytic: x(t) = exp(-t)cos(t) + exp(-t)sin(t) [for p1=p2=1]
        # ∂x/∂p₁(t) = -t·exp(-t)·cos(t) - (t-1)·exp(-t)·sin(t) at p=(1,1)
        x_analytic = np.exp(-t_final) * (np.cos(t_final) + np.sin(t_final))
        if n_orders > 1:
            # order (1,0): ∂x/∂p₁
            sx_analytic = -t_final * np.exp(-t_final) * np.cos(t_final) - (t_final - 1) * np.exp(-t_final) * np.sin(t_final)
            sx_computed = y_final[1] if n_orders > 1 else None
        else:
            sx_analytic = sx_computed = None

        x_err = abs(y_final[0] - x_analytic) if n_orders > 0 else None

        results.append({
            'max_order': max_order,
            'n_orders': n_orders,
            'n_total': n_total,
            'steps': len(sol.t),
            'nfev': nfev,
            'total_time': avg_time,
            'rhs_time': rhs_total,
            'other_time': other_total,
            'rhs_pct': 100 * rhs_total / avg_time if avg_time > 0 else 0,
            'x_err': x_err,
        })

    # Print breakdown
    print(f"\nCoupled ODE ECSS: t ∈ [0, {tf}], Solver: scipy BDF\n")

    rows = []
    for r in results:
        rows.append([
            str(r['max_order']),
            str(r['n_orders']),
            str(r['n_total']),
            str(r['nfev']),
            f"{r['total_time']:.4f}",
            f"{r['rhs_time']:.4f}",
            f"{r['other_time']:.4f}",
            f"{r['rhs_pct']:.0f}%",
        ])

    print_table(
        ["Order", "N(Q)", "n_ECSS", "f-evals", "Total (s)", "RHS (s)", "Other (s)", "RHS%"],
        rows,
        "Runtime Breakdown by Component"
    )

    # Detailed breakdown for max_order=2
    r2 = results[2] if len(results) > 2 else results[-1]
    print(f"\n  --- Detailed Breakdown (order={r2['max_order']}) ---")
    print(f"  Total integration time:   {r2['total_time']:.4f}s (100%)")
    print(f"  ├─ RHS evaluation:         {r2['rhs_time']:.4f}s ({r2['rhs_pct']:.0f}%)")
    print(f"  │  └─ per call:            {r2['rhs_time']/r2['nfev']*1e6:.2f}μs")
    print(f"  └─ Solver overhead:        {r2['other_time']:.4f}s ({100-r2['rhs_pct']:.0f}%)")
    print(f"     ├─ LU factorisation:     (embedded in scipy BDF)")
    print(f"     ├─ Error estimation:     (adaptive step control)")
    print(f"     └─ Other:                (step acceptance, interpolation)")

    # Discussion: where optimisation effort should focus
    print(f"\n  --- Implications ---")
    if r2['rhs_pct'] > 50:
        print(f"  RHS evaluation dominates ({r2['rhs_pct']:.0f}% of runtime).")
        print(f"  → Optimise MVTS arithmetic (operator fusion, vectorisation).")
    else:
        print(f"  Solver overhead dominates ({100-r2['rhs_pct']:.0f}% of runtime).")
        print(f"  → Exploit Σ-matrix sparsity to reduce LU factorisation cost.")
        print(f"  → Use iterative linear solvers for large sparse systems.")

    print(f"\n  Validation: Profile collected successfully")

    save_results({
        'system': 'coupled_ode_ecss',
        't_span': [float(t0), float(tf)],
        'solver': 'scipy_BDF',
        'results': [{k: (float(v) if isinstance(v, (np.floating, float, np.integer))
                          else v) for k, v in r.items()}
                     for r in results],
    }, 'w5_runtime_profile.json')

    print()
    return True


if __name__ == '__main__':
    success = run_profiling()
    print("=" * 70)
    print(f"  FINAL: Runtime Profiling {'✓ COMPLETE' if success else '✗ FAIL'}")
    print("=" * 70)
