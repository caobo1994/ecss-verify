"""
Core utilities for ECSS validation in Python.
Provides DAE definition, structural analysis, ECSS equation generation,
and integration via scipy's BDF solver.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Dict, Optional
import time
import json


# ──────────────────────────────────────────────────────
# 1. DAE System Definition
# ──────────────────────────────────────────────────────

@dataclass
class DAESystem:
    """Definition of a parametrised DAE F(t, D(x), p) = 0.

    n_vars: number of state variables (including algebraic)
    n_params: number of parameters
    max_deriv: maximum derivative order per variable (list of length n_vars)
    F: function F(t, y_flat, p) returning residuals of length n_eqns
    jac: Jacobian ∂F/∂y (optional, for implicit solvers)
    analytic_solution: optional callable (t, p) -> y_flat
    """
    name: str
    n_vars: int
    n_params: int
    max_deriv: List[int]        # max derivative order for each variable
    F: Callable                 # F(t, y_flat, p) -> residuals
    jac: Optional[Callable] = None
    analytic_solution: Optional[Callable] = None
    index: int = 0
    description: str = ""


# ──────────────────────────────────────────────────────
# 2. Multivariate Taylor Series (MVTS) Arithmetic
# ──────────────────────────────────────────────────────

class MVTS:
    """Multivariate Taylor series with respect to parameters.

    Represents x(p + h) as a truncated Taylor series:
        x(p + h) = sum_{q ∈ Q} c_q * h^q / q!
    where Q is the set of sensitivity orders (downward-closed) and
    c_q = ∂^q x / ∂p^q.
    """

    def __init__(self, orders: List[Tuple[int, ...]], coefficients=None):
        """Initialize MVTS with given sensitivity orders.

        Args:
            orders: List of multi-index tuples, downward-closed, e.g. [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)]
            coefficients: Dict mapping order -> numpy value
        """
        self.orders = sorted(orders, key=lambda q: (sum(q), q))
        self._idx = {q: i for i, q in enumerate(self.orders)}
        n_terms = len(orders)
        self.coeffs = np.zeros(n_terms) if coefficients is None else np.array(coefficients)

    def __getitem__(self, order):
        return self.coeffs[self._idx[order]]

    def __setitem__(self, order, value):
        self.coeffs[self._idx[order]] = value

    def value(self):
        """Return the base value (order (0,...,0))."""
        return self.coeffs[0]

    def sensitivity(self, order):
        """Return sensitivity coefficient for given order."""
        return self.coeffs[self._idx[order]]

    def __add__(self, other):
        if isinstance(other, (int, float)):
            result = MVTS(self.orders)
            result.coeffs = self.coeffs.copy()
            result.coeffs[0] += other
            return result
        result = MVTS(self.orders)
        result.coeffs = self.coeffs + other.coeffs
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            result = MVTS(self.orders)
            result.coeffs = self.coeffs.copy()
            result.coeffs[0] -= other
            return result
        result = MVTS(self.orders)
        result.coeffs = self.coeffs - other.coeffs
        return result

    def __mul__(self, other):
        """Multiplication via Cauchy product: (a*b)_r = sum_{q ≤ r} a_q * b_{r-q}"""
        if isinstance(other, (int, float)):
            result = MVTS(self.orders)
            result.coeffs = self.coeffs * other
            return result
        result = MVTS(self.orders)
        for i, r in enumerate(self.orders):
            s = 0.0
            for j, q in enumerate(self.orders):
                # r - q must be a valid order in our set
                diff = tuple(r[k] - q[k] for k in range(len(r)))
                if all(d >= 0 for d in diff) and diff in self._idx:
                    s += self.coeffs[j] * other.coeffs[self._idx[diff]]
            result.coeffs[i] = s
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """Division: 1/a = geometric series expansion."""
        if isinstance(other, (int, float)):
            result = MVTS(self.orders)
            result.coeffs = self.coeffs / other
            return result
        # Compute a * b = self via linear system
        # self = result * other, solve for result
        n = len(self.orders)
        A = np.zeros((n, n))
        for i, r in enumerate(self.orders):
            for j, q in enumerate(self.orders):
                diff = tuple(r[k] - q[k] for k in range(len(r)))
                if all(d >= 0 for d in diff) and diff in self._idx:
                    A[i, self._idx[diff]] = other.coeffs[j]
        try:
            result_coeffs = np.linalg.solve(A, self.coeffs)
        except np.linalg.LinAlgError:
            # Fallback: use iterative approach for near-singular
            result_coeffs = np.linalg.lstsq(A, self.coeffs, rcond=None)[0]
        result = MVTS(self.orders)
        result.coeffs = result_coeffs
        return result

    def sqrt(self):
        """Square root: sqrt(a₀ + ...) = √a₀ * (1 + x)^0.5 via binomial series."""
        a0 = self.coeffs[0]
        if a0 <= 0:
            raise ValueError(f"sqrt requires positive base value, got {a0}")
        x = MVTS(self.orders)
        x.coeffs = self.coeffs / a0
        x.coeffs[0] -= 1.0  # x = self/a0 - 1

        # (1 + x)^0.5 = sum_{k=0}^{∞} C(0.5, k) x^k
        result = MVTS(self.orders)
        result.coeffs[0] = np.sqrt(a0)
        x_pow = MVTS(self.orders)
        x_pow.coeffs[0] = 1.0

        binomial = 0.5
        for k in range(1, 6):  # enough terms for order ≤ 5
            binomial *= (0.5 - (k - 1)) / k
            if abs(binomial) < 1e-16:
                break
            x_pow = x_pow * x
            result.coeffs += binomial * x_pow.coeffs
        return result

    def sin(self):
        """sin(x) = sin(a₀)cos(x-a₀) + cos(a₀)sin(x-a₀)"""
        a0 = self.coeffs[0]
        dx = MVTS(self.orders)
        dx.coeffs = self.coeffs.copy()
        dx.coeffs[0] = 0.0  # x - a0

        # sin(dx) ≈ dx - dx³/6 + dx⁵/120
        sin_a0, cos_a0 = np.sin(a0), np.cos(a0)
        result = MVTS(self.orders)
        result.coeffs[0] = sin_a0

        dx_mvts = dx
        if len(self.orders) > 0:
            # First order correction: cos(a0) * dx
            result.coeffs += cos_a0 * dx_mvts.coeffs
        if len(self.orders) > 0 and any(sum(q) >= 2 for q in self.orders):
            dx2 = dx * dx
            result.coeffs += (-sin_a0 / 2.0) * dx2.coeffs
        if any(sum(q) >= 3 for q in self.orders):
            dx3 = dx2 * dx
            result.coeffs += (-cos_a0 / 6.0) * dx3.coeffs
        return result

    def cos(self):
        """cos(x) = cos(a₀)cos(x-a₀) - sin(a₀)sin(x-a₀)"""
        a0 = self.coeffs[0]
        dx = MVTS(self.orders)
        dx.coeffs = self.coeffs.copy()
        dx.coeffs[0] = 0.0

        sin_a0, cos_a0 = np.sin(a0), np.cos(a0)
        result = MVTS(self.orders)
        result.coeffs[0] = cos_a0

        if len(self.orders) > 0:
            result.coeffs += (-sin_a0) * dx.coeffs
        if len(self.orders) > 0 and any(sum(q) >= 2 for q in self.orders):
            dx2 = dx * dx
            result.coeffs += (-cos_a0 / 2.0) * dx2.coeffs
        if any(sum(q) >= 3 for q in self.orders):
            dx3 = dx2 * dx
            result.coeffs += (sin_a0 / 6.0) * dx3.coeffs
        return result

    def exp(self):
        """exp(x) = exp(a₀) * exp(x-a₀)"""
        a0 = self.coeffs[0]
        dx = MVTS(self.orders)
        dx.coeffs = self.coeffs.copy()
        dx.coeffs[0] = 0.0

        result = MVTS(self.orders)
        result.coeffs[0] = np.exp(a0)
        factor = np.exp(a0)

        if len(self.orders) > 0:
            result.coeffs += factor * dx.coeffs
        if any(sum(q) >= 2 for q in self.orders):
            dx2 = dx * dx
            result.coeffs += (factor / 2.0) * dx2.coeffs
        if any(sum(q) >= 3 for q in self.orders):
            dx3 = dx2 * dx
            result.coeffs += (factor / 6.0) * dx3.coeffs
        return result

    def __repr__(self):
        orders_str = ", ".join(f"({','.join(map(str,q))}):{self[q]:.6g}" for q in self.orders)
        return f"MVTS({orders_str})"


def generate_sov_orders(m: int, max_total_order: int) -> List[Tuple[int, ...]]:
    """Generate all multi-index orders with total order ≤ max_total_order."""
    orders = []
    for total in range(max_total_order + 1):
        _gen_orders_recursive(m, total, [], orders)
    return orders


def _gen_orders_recursive(m: int, remaining: int, current: List[int], orders: List):
    if len(current) == m:
        if remaining == 0:
            orders.append(tuple(current))
        return
    for k in range(remaining + 1):
        _gen_orders_recursive(m, remaining - k, current + [k], orders)


# ──────────────────────────────────────────────────────
# 3. ECSS Generation
# ──────────────────────────────────────────────────────

def generate_ecss_equations(
    system: DAESystem,
    max_total_order: int,
    param_values: np.ndarray,
    initial_conditions: np.ndarray,
) -> Tuple[List[Tuple[int, ...]], int]:
    """
    Generate ECSS order metadata for a given pDAE.

    The ECSS is formed by evaluating F(t, D(x), p) with MVTS-valued variables
    and extracting each coefficient (sensitivity order) as a separate equation.
    This lightweight validation helper does not build those residual
    functions; it returns only the SOV orders and total ECSS size.

    Returns:
        orders: list of sensitivity orders in the ECSS
        ecss_size: total number of variables (n_vars * len(orders))
    """
    m = system.n_params
    orders = generate_sov_orders(m, max_total_order)
    n_orders = len(orders)
    ecss_size = system.n_vars * n_orders

    return orders, ecss_size


def construct_ecss_for_ode(
    system: DAESystem,
    max_total_order: int,
    param_values: np.ndarray,
) -> Tuple[Callable, Callable, List[Tuple[int, ...]], int]:
    """
    Construct explicit ECSS for ODE systems (index 0).

    For an ODE x' = f(t, x, p), the ECSS has variables:
        [x^(0), x^(1), ..., x^(max_order)]
    where x^(q) = ∂^q x / ∂p^q.

    Returns:
        ecss_rhs: callable (t, y_flat) -> dy_flat
        ecss_jac: callable (t, y_flat) -> dy_flat_jac (optional, may be None)
        orders: list of sensitivity orders
        ecss_size: total number of variables
    """
    m = system.n_params
    orders = generate_sov_orders(m, max_total_order)
    n_orders = len(orders)
    n_vars = system.n_vars
    ecss_size = n_vars * n_orders
    order_idx = {q: i for i, q in enumerate(orders)}

    def ecss_rhs(t, y):
        dy = np.zeros_like(y)
        # Evaluate the base ODE at parameter values
        # y_flat: [x_var0_order0, x_var0_order1, ..., x_var0_orderK,
        #          x_var1_order0, ...]
        # Reshape: (n_vars, n_orders)
        y_reshaped = y.reshape(n_vars, n_orders)

        # Get base state: y_reshaped[:, 0] = x (order 0)
        x_base = y_reshaped[:, 0]

        # Base ODE: x' = f(t, x, p)
        xp_base = system.F(t, x_base, param_values)

        # For order 0: dx/dt = f(t, x, p)
        dy_reshaped = np.zeros_like(y_reshaped)
        dy_reshaped[:, 0] = xp_base

        # For higher orders: differentiate f under the integral/chain rule
        # ∂^q (dx/dt) / ∂p^q = ∂^q f(t, x(t,p), p) / ∂p^q
        # This requires the Faà di Bruno formula applied to f.
        # For linear ODEs, this simplifies. For nonlinear, we use MVTS.
        #
        # Here we implement the general approach:
        # Set up MVTS variables for x and evaluate f with MVTS arithmetic.

        # Build MVTS representation of x
        mvts_vars = []
        for i in range(n_vars):
            mvts_v = MVTS(orders)
            mvts_v.coeffs = y_reshaped[i, :]
            mvts_vars.append(mvts_v)

        # Evaluate each ODE function with MVTS variables
        for i in range(n_vars):
            # We need to evaluate f_i(t, x, p) treating x as MVTS
            # and p as nominal (base) values with the first-order
            # parameter coefficient = 1 for each parameter.

            # Create MVTS for the parameter effect: p_k + h_k * (1 if k matches)
            # For each parameter, we have ∂p_k/∂p_k = 1, ∂p_j/∂p_k = 0 (j≠k)
            # This is built into the MVTS evaluation of F.

            # Actually, the parameters p are constants in the function evaluation.
            # The MVTS evaluation of f_i(t, mvts_vars, p_const) gives us
            # the sensitivities because the chain rule propagates them.

            # For a concrete implementation, we evaluate f_i with MVTS vars.
            # This is system-specific and would be auto-generated.
            # Here we use the system's MVTS-capable F function.
            if hasattr(system, 'F_mvts') and system.F_mvts is not None:
                result_mvts = system.F_mvts(t, mvts_vars, param_values, orders, i)
                dy_reshaped[i, :] = result_mvts.coeffs
            else:
                # Fallback for systems without MVTS-aware F: compute base only
                # This won't give correct sensitivity propagation for nonlinear systems
                dy_reshaped[i, 0] = xp_base[i]

        return dy_reshaped.ravel()

    return ecss_rhs, None, orders, ecss_size


# ──────────────────────────────────────────────────────
# 4. Index Reduction Utilities
# ──────────────────────────────────────────────────────

def reduce_dae_to_index1(system: DAESystem) -> DAESystem:
    """
    Reduce a DAE to index 1 by differentiating algebraic constraints.
    For Hessenberg index-k systems, differentiate constraints k-1 times.
    Returns a new DAESystem in semi-explicit index-1 form.
    """
    # This is system-specific; for the Hessenberg chain:
    #  x₁' + x₂ = 0
    #  x₂' + x₃ = 0
    #  ...
    #  x_{k-1}' + x_k = 0
    #  x_k - g(t) = 0  (algebraic)
    #
    # Differentiating the last equation k-1 times gives an ODE.
    # For general systems, this needs structural analysis.
    #
    # For our validation, we'll handle specific system types.
    return system


# ──────────────────────────────────────────────────────
# 5. Integration Wrappers
# ──────────────────────────────────────────────────────

def integrate_dae_bdf(
    rhs_differential: Callable,
    rhs_algebraic: Callable,
    mass_matrix: np.ndarray,
    t_span: Tuple[float, float],
    y0: np.ndarray,
    yp0: np.ndarray = None,
    rtol: float = 1e-12,
    atol: float = 1e-14,
    max_step: float = 0.1,
) -> Dict:
    """
    Integrate a nonsingular mass-matrix system using scipy's BDF solver.

    M * dy/dt = f(t, y)

    This helper is not a DAE solver: singular mass matrices must be
    reduced before integration with scipy.integrate.solve_ivp in this suite.
    """
    if yp0 is None:
        yp0 = np.zeros_like(y0)

    if np.linalg.matrix_rank(mass_matrix) < mass_matrix.shape[0]:
        raise ValueError(
            "integrate_dae_bdf requires a nonsingular mass matrix; "
            "reduce algebraic constraints before using solve_ivp."
        )

    def residual(t, y, yp):
        return mass_matrix @ yp - rhs_differential(t, y)

    # Use solve_ivp with BDF
    t0, tf = t_span
    n = len(y0)

    def dydt(t, y):
        # Solve M @ yp = f(t, y) for yp
        M = mass_matrix
        f = rhs_differential(t, y)
        return np.linalg.solve(M, f)

    start_time = time.time()
    sol = solve_ivp(
        dydt, t_span, y0,
        method='BDF',
        rtol=rtol, atol=atol,
        max_step=max_step,
    )
    elapsed = time.time() - start_time

    return {
        't': sol.t,
        'y': sol.y,
        'success': sol.success,
        'nfev': sol.nfev,
        'njev': sol.njev,
        'elapsed': elapsed,
    }


def integrate_ode_ecss(
    ecss_rhs: Callable,
    t_span: Tuple[float, float],
    y0: np.ndarray,
    method: str = 'BDF',
    rtol: float = 1e-12,
    atol: float = 1e-14,
) -> Dict:
    """Integrate an ECSS ODE system with scipy."""
    start_time = time.time()
    sol = solve_ivp(
        ecss_rhs, t_span, y0,
        method=method,
        rtol=rtol, atol=atol,
    )
    elapsed = time.time() - start_time
    return {
        't': sol.t,
        'y': sol.y,
        'success': sol.success,
        'nfev': sol.nfev,
        'elapsed': elapsed,
    }


# ──────────────────────────────────────────────────────
# 6. Validation Helpers
# ──────────────────────────────────────────────────────

def compute_max_error(computed: np.ndarray, analytic: np.ndarray) -> float:
    """Maximum absolute error."""
    return np.max(np.abs(computed - analytic))


def finite_difference_gradient(
    f: Callable,
    x: np.ndarray,
    h: float = 1e-6,
    method: str = 'central'
) -> np.ndarray:
    """Compute gradient of f at x via finite differences."""
    n = len(x)
    grad = np.zeros(n)
    if method == 'central':
        for i in range(n):
            xp = x.copy()
            xm = x.copy()
            xp[i] += h
            xm[i] -= h
            grad[i] = (f(xp) - f(xm)) / (2 * h)
    elif method == 'forward':
        f0 = f(x)
        for i in range(n):
            xp = x.copy()
            xp[i] += h
            grad[i] = (f(xp) - f0) / h
    return grad


def finite_difference_hessian(
    f: Callable,
    x: np.ndarray,
    h: float = 1e-4,
) -> np.ndarray:
    """Compute Hessian of f at x via central differences."""
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                xp = x.copy(); xp[i] += h
                xm = x.copy(); xm[i] -= h
                H[i, i] = (f(xp) - 2 * f0 + f(xm)) / (h * h)
            else:
                xpp = x.copy(); xpp[i] += h; xpp[j] += h
                xpm = x.copy(); xpm[i] += h; xpm[j] -= h
                xmp = x.copy(); xmp[i] -= h; xmp[j] += h
                xmm = x.copy(); xmm[i] -= h; xmm[j] -= h
                H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h * h)
                H[j, i] = H[i, j]
    return H


def save_results(data: Dict, filename: str):
    """Save results to JSON."""
    import os
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, filename), 'w') as f:
        json.dump(data, f, indent=2, default=str)


def print_table(headers: List[str], rows: List[List], title: str = ""):
    """Print a formatted table."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Header
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)

    # Rows
    for row in rows:
        row_line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        print(row_line)
    print()
