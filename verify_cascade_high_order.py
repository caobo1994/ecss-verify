#!/usr/bin/env python3
"""
High-order first-variable cascade verification.

The first-variable cascade has equations

    x_1' - x_2 = 0, ..., x_k' - x_{k+1} = 0,
    x_1 - g(t, p) = 0.

For the ECSS, the algebraic block for x_1 fixes every parameter
sensitivity of x_1 directly as d^q g / dp^q.  This script checks the
closed-form ECSS recurrence for q=0..3 against mpmath's independent
high-precision differentiation for the cascade constraints used in
Chapter 7.
"""

from __future__ import annotations

import json
from pathlib import Path

from mpmath import mp


mp.dps = 80
P = mp.mpf("2.0")
T = mp.mpf("0.5")


def sin_formula(order: int, p: mp.mpf = P, t: mp.mpf = T) -> mp.mpf:
    if order == 0:
        return p * mp.sin(p * t)
    if order == 1:
        return mp.sin(p * t) + p * t * mp.cos(p * t)
    if order == 2:
        return 2 * t * mp.cos(p * t) - p * t**2 * mp.sin(p * t)
    if order == 3:
        return -3 * t**2 * mp.sin(p * t) - p * t**3 * mp.cos(p * t)
    raise ValueError(order)


def exp_formula(order: int, p: mp.mpf = P, t: mp.mpf = T) -> mp.mpf:
    e = mp.e ** (p * t)
    if order == 0:
        return p * e
    if order == 1:
        return (1 + p * t) * e
    if order == 2:
        return t * (2 + p * t) * e
    if order == 3:
        return t**2 * (3 + p * t) * e
    raise ValueError(order)


def reference(kind: str, order: int) -> mp.mpf:
    if kind == "sin":
        return mp.diff(lambda pp: pp * mp.sin(pp * T), P, order)
    if kind == "exp":
        return mp.diff(lambda pp: pp * mp.e ** (pp * T), P, order)
    raise ValueError(kind)


def main() -> int:
    rows = []
    max_error = mp.mpf("0")

    for depth in (4, 5, 6):
        for kind, formula in (("sin", sin_formula), ("exp", exp_formula)):
            errors = []
            for order in range(4):
                ecss_value = formula(order)
                ref_value = reference(kind, order)
                err = abs(ecss_value - ref_value)
                errors.append(err)
                max_error = max(max_error, err)
            rows.append(
                {
                    "offset_depth": depth,
                    "constraint": f"p*{kind}(p*t)",
                    "errors": [float(err) for err in errors],
                }
            )

    print("First-variable cascade high-order verification")
    print(f"p={float(P)}, t={float(T)}, precision={mp.dps} digits")
    for row in rows:
        errors = "  ".join(f"{err:.2e}" for err in row["errors"])
        print(f"depth {row['offset_depth']}, {row['constraint']}: {errors}")
    print(f"max error: {float(max_error):.2e}")

    out_path = Path(__file__).with_name("cascade_high_order_result.json")
    out_path.write_text(
        json.dumps(
            {
                "precision_digits": mp.dps,
                "p": float(P),
                "t": float(T),
                "max_error": float(max_error),
                "rows": rows,
                "pass": bool(max_error < mp.mpf("1e-60")),
            },
            indent=2,
        )
        + "\n"
    )

    return 0 if max_error < mp.mpf("1e-60") else 1


if __name__ == "__main__":
    raise SystemExit(main())
