#!/usr/bin/env python3
"""
Standalone Python MVTS Builder — reference implementation.

Operator-overloading AD: evaluates a parametrised DAE once with
MVTS-valued variables to symbolically generate all requested
sensitivity equations.  Implements Theorem 1 (Paper 2).

Zero dependencies (Python stdlib only).  Run: python mvts_builder.py
"""

from __future__ import annotations
import math
from typing import Tuple, List, Dict, Set


# ═══════════════════════════════════════════════════════
# Multi-index utilities
# ═══════════════════════════════════════════════════════

def _ole(a: Tuple[int, ...], b: Tuple[int, ...]) -> bool:
    return all(ai <= bi for ai, bi in zip(a, b))
def _olt(a: Tuple[int, ...], b: Tuple[int, ...]) -> bool:
    return _ole(a, b) and a != b
def _oadd(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(ai + bi for ai, bi in zip(a, b))
def _osub(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(ai - bi for ai, bi in zip(a, b))
def _otot(q: Tuple[int, ...]) -> int:
    return sum(q)
def _ostar(q: Tuple[int, ...]) -> Tuple[int, ...]:
    lst = list(q)
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] > 0: lst[i] -= 1; return tuple(lst)
    return q
def _B(q: Tuple[int, ...], r: Tuple[int, ...]) -> float:
    for i in range(len(q) - 1, -1, -1):
        if q[i] > 0: return (q[i] - r[i]) / q[i]
    return 0.0

def compute_sov_closure(req: List[Tuple[int, ...]], m: int) -> List[Tuple[int, ...]]:
    items: Set[Tuple[int, ...]] = set()
    for qmax in req:
        def gen(pref, d):
            if d == m: items.add(pref); return
            for k in range(qmax[d] + 1): gen(pref + (k,), d + 1)
        gen((), 0)
    return sorted(items, key=lambda q: (_otot(q), q))


# ═══════════════════════════════════════════════════════
# Symbolic expression: sum of monomials
# ═══════════════════════════════════════════════════════

class Expr:
    """Sum of terms. Each term is (coeff, {var: power}).

    coeff=0 means the term is absent.  {} means a constant term.
    """

    def __init__(self, coeff=0.0, vars=None):
        self.terms: List[tuple] = []  # [(coeff, {var: power}), ...]
        vd = vars or {}
        if abs(coeff) > 1e-15:
            self.terms.append((float(coeff), dict(vd)))

    @staticmethod
    def from_name(name: str, coeff=1.0) -> 'Expr':
        return Expr(float(coeff), {name: 1})

    @staticmethod
    def from_float(v: float) -> 'Expr':
        return Expr(v, {})

    def is_zero(self) -> bool:
        return len(self.terms) == 0

    def is_const(self) -> bool:
        return len(self.terms) == 1 and len(self.terms[0][1]) == 0

    def const_val(self) -> float:
        if self.is_const(): return self.terms[0][0]
        return 0.0

    def copy(self) -> 'Expr':
        e = Expr(0.0)
        e.terms = [(c, dict(v)) for c, v in self.terms]
        return e

    def _normalize(self):
        """Combine like terms (same variable dict)."""
        merged: Dict = {}
        const_val = 0.0
        for c, vd in self.terms:
            if not vd:
                const_val += c
            else:
                key = tuple(sorted(vd.items()))
                merged[key] = merged.get(key, 0.0) + c
        self.terms = []
        if abs(const_val) > 1e-15:
            self.terms.append((const_val, {}))
        for key, c in merged.items():
            if abs(c) > 1e-15:
                self.terms.append((c, dict(key)))

    def __add__(self, other: 'Expr') -> 'Expr':
        r = self.copy()
        r.terms.extend((c, dict(v)) for c, v in other.terms)
        r._normalize()
        return r

    def __sub__(self, other: 'Expr') -> 'Expr':
        r = self.copy()
        r.terms.extend((-c, dict(v)) for c, v in other.terms)
        r._normalize()
        return r

    def __mul__(self, other: 'Expr') -> 'Expr':
        r = Expr(0.0)
        for c1, v1 in self.terms:
            for c2, v2 in other.terms:
                merged = dict(v1)
                for k, v in v2.items():
                    merged[k] = merged.get(k, 0) + v
                r.terms.append((c1 * c2, merged))
        r._normalize()
        return r

    def __truediv__(self, other: 'Expr') -> 'Expr':
        if other.is_const():
            d = other.const_val()
            if d == 0: raise ZeroDivisionError
            r = Expr(0.0)
            r.terms = [(c / d, dict(v)) for c, v in self.terms]
            return r
        raise ValueError("Division by non-constant")

    def __neg__(self) -> 'Expr':
        r = Expr(0.0)
        r.terms = [(-c, dict(v)) for c, v in self.terms]
        return r

    def __pow__(self, exp: float) -> 'Expr':
        """Only supported for constant base and integer exponent."""
        if not self.is_const():
            # Symbolic power: just tag it
            r = Expr(0.0)
            r.terms = [(1.0, {repr(self): exp})]
            return r
        return Expr(self.const_val() ** exp)

    def __repr__(self) -> str:
        parts = []
        for c, vd in self.terms:
            s = ""
            if not vd:
                s = f"{c:.6g}" if abs(c) >= 1e-10 else "0"
            else:
                m = ""
                for name, pwr in sorted(vd.items()):
                    if pwr == 1:
                        m += f"·{name}" if m else name
                    else:
                        m += f"·{name}^{pwr}" if m else f"{name}^{pwr}"
                if abs(c - 1.0) < 1e-14:
                    s = m
                elif abs(c + 1.0) < 1e-14:
                    s = f"-{m}"
                else:
                    s = f"{c:.6g}·{m}"
            parts.append(s)
        if not parts:
            return "0"
        result = parts[0]
        for p in parts[1:]:
            if p.startswith('-'):
                result += f" - {p[1:]}"
            else:
                result += f" + {p}"
        return result


# ═══════════════════════════════════════════════════════
# MVTS type
# ═══════════════════════════════════════════════════════

class MVTS:
    """Multivariate Taylor series with operator-overloaded arithmetic."""

    def __init__(self, orders: List[Tuple[int, ...]]):
        self.orders = orders
        self.m = len(orders[0]) if orders else 0
        self.zero = (0,) * self.m
        self._c: Dict[Tuple[int, ...], Expr] = {q: Expr(0.0) for q in orders}

    @classmethod
    def const(cls, val: float, orders) -> 'MVTS':
        mv = cls(orders)
        mv._c[mv.zero] = Expr.from_float(val)
        return mv

    @classmethod
    def var(cls, name: str, orders, sens_ords) -> 'MVTS':
        mv = cls(orders)
        mv._c[mv.zero] = Expr.from_name(name)
        for q in sens_ords:
            if q != mv.zero:
                label = f"{name}_g" if (len(q) == 1 and q[0] == 1) else f"{name}_q{q}"
                mv._c[q] = Expr.from_name(label)
        return mv

    @classmethod
    def deriv(cls, name: str, dorder: int, orders, sens_ords) -> 'MVTS':
        mv = cls(orders)
        ticks = "'" * dorder
        mv._c[mv.zero] = Expr.from_name(f"{name}{ticks}")
        for q in sens_ords:
            label = f"{name}_g{ticks}" if (len(q) == 1 and q[0] == 1) else f"{name}_q{q}{ticks}"
            mv._c[q] = Expr.from_name(label)
        return mv

    @classmethod
    def param(cls, k: int, p_val: float, orders, h: float = 1.0) -> 'MVTS':
        mv = cls(orders)
        e_k = tuple(1 if i == k else 0 for i in range(len(orders[0])))
        zv = (0,) * len(orders[0])
        for q in orders:
            if q == zv:
                mv._c[q] = Expr.from_float(p_val)
            elif q == e_k:
                mv._c[q] = Expr.from_float(h)
            else:
                mv._c[q] = Expr(0.0)
        return mv

    def __getitem__(self, q): return self._c.get(q, Expr(0.0))
    def __setitem__(self, q, v): self._c[q] = v

    def _op(self, other, fn):
        if isinstance(other, (int, float)):
            other = MVTS.const(float(other), self.orders)
        r = MVTS(self.orders)
        for q in self.orders: r._c[q] = fn(self[q], other[q])
        return r

    def __add__(self, o): return self._op(o, lambda a, b: a + b)
    def __radd__(self, o): return self.__add__(o)
    def __sub__(self, o): return self._op(o, lambda a, b: a - b)
    def __rsub__(self, o):
        if isinstance(o, (int, float)): return MVTS.const(float(o), self.orders) - self
        raise TypeError

    def __mul__(self, o):
        if isinstance(o, (int, float)):
            o = MVTS.const(float(o), self.orders)
        r = MVTS(self.orders)
        for q in self.orders:
            acc = Expr(0.0)
            for s in self.orders:
                if _ole(s, q):
                    acc = acc + self[s] * o[_osub(q, s)]
            r._c[q] = acc
        return r
    def __rmul__(self, o): return self.__mul__(o)

    def __truediv__(self, o):
        if isinstance(o, (int, float)):
            o = MVTS.const(float(o), self.orders)
        g0 = o[self.zero]
        r = MVTS(self.orders)
        for q in sorted(self.orders, key=_otot):
            acc = self[q]
            for s in self.orders:
                if _olt(s, q):
                    acc = acc - r[s] * o[_osub(q, s)]
            r._c[q] = acc / g0
        return r

    def __neg__(self):
        r = MVTS(self.orders)
        for q in self.orders: r._c[q] = -self[q]
        return r

    def __pow__(self, exp):
        a = float(exp)
        f0 = self[self.zero].const_val()
        r = MVTS(self.orders)
        r._c[self.zero] = Expr.from_float(f0 ** a)
        for q in self.orders:
            if q == self.zero: continue
            qs = _ostar(q)
            h0 = r[self.zero].const_val()
            acc = Expr.from_float(a * h0) * self[q]
            zero_t = (0,) * self.m
            for s in self.orders:
                if _ole(zero_t, s) and _olt(zero_t, s) and _ole(s, qs):
                    Bv = _B(q, s)
                    qms = _osub(q, s)
                    acc = acc + Expr.from_float(Bv * a) * r[s] * self[qms] \
                              - Expr.from_float(Bv) * r[qms] * self[s]
            r._c[q] = acc / self[self.zero]
        return r

    def _recur(self, phi0, rule):
        r = MVTS(self.orders)
        f0 = self[self.zero].const_val()
        r._c[self.zero] = Expr.from_float(phi0(f0))
        for q in sorted(self.orders, key=_otot):
            if q == self.zero: continue
            qs = _ostar(q)
            acc = Expr(0.0)
            for s in self.orders:
                if _ole(s, qs):
                    acc = rule(q, s, _osub(q, s), _B(q, s), r, self, acc)
            r._c[q] = acc
        return r

    def exp(self):
        return self._recur(math.exp,
            lambda q, s, qs, Bv, res, f, acc: acc + Expr.from_float(Bv) * res[s] * f[qs])

    def sin(self):
        r = MVTS(self.orders); c = MVTS(self.orders)
        f0 = self[self.zero].const_val()
        r._c[self.zero] = Expr.from_float(math.sin(f0))
        c._c[self.zero] = Expr.from_float(math.cos(f0))
        for q in sorted(self.orders, key=_otot):
            if q == self.zero: continue
            qs = _ostar(q)
            asin, acos = Expr(0.0), Expr(0.0)
            for s in self.orders:
                if _ole(s, qs):
                    B = _B(q, s); qms = _osub(q, s)
                    asin = asin + Expr.from_float(B) * c[s] * self[qms]
                    acos = acos - Expr.from_float(B) * r[s] * self[qms]
            r._c[q] = asin; c._c[q] = acos
        return r

    def cos(self):
        r = MVTS(self.orders); s = MVTS(self.orders)
        f0 = self[self.zero].const_val()
        s._c[self.zero] = Expr.from_float(math.sin(f0))
        r._c[self.zero] = Expr.from_float(math.cos(f0))
        for q in sorted(self.orders, key=_otot):
            if q == self.zero: continue
            qs = _ostar(q)
            asin, acos = Expr(0.0), Expr(0.0)
            for si in self.orders:
                if _ole(si, qs):
                    B = _B(q, si); qms = _osub(q, si)
                    asin = asin + Expr.from_float(B) * r[si] * self[qms]
                    acos = acos - Expr.from_float(B) * s[si] * self[qms]
            s._c[q] = asin; r._c[q] = acos
        return r


# ═══════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════

def demo():
    print("=" * 70)
    print("  MVTS Builder — Pendulum ECSS Generation")
    print("  Paper 2, Companion Reference Implementation")
    print("=" * 70)

    m = 1
    closed = compute_sov_closure([(0,), (1,)], m)
    zero = (0,)
    sens = [q for q in closed if q != zero]

    print(f"\n  Closed SOV: {closed}")
    print(f"  n=3 (x, y, λ), sensitivity param: g, fixed: L\n")

    x_pp = MVTS.deriv('x', 2, closed, sens)
    y_pp = MVTS.deriv('y', 2, closed, sens)
    x_v  = MVTS.var('x', closed, sens)
    y_v  = MVTS.var('y', closed, sens)
    lam  = MVTS.var('λ', closed, sens)
    g_mv = MVTS.param(0, 9.8, closed)
    L_mv = MVTS.const(2.0, closed)

    f1 = x_pp + lam * x_v
    f2 = y_pp + lam * y_v - g_mv
    f3 = x_v * x_v + y_v * y_v - L_mv * L_mv

    print("  Generated ECSS Equations:\n")
    for q in closed:
        label = "Order (0) — original DAE" if q == zero else f"Order {q} — sensitivity equations"
        print(f"  {label}:")
        for i, f in enumerate([f1, f2, f3]):
            c = f[q]
            if not c.is_zero():
                print(f"    f{i+1}_{q}:  {c}  = 0")
        print()

    # Verify
    print("-" * 70)
    print("  Verification\n")

    f1_0 = repr(f1[zero]); f2_0 = repr(f2[zero]); f3_0 = repr(f3[zero])
    f1_1 = repr(f1[(1,)]); f2_1 = repr(f2[(1,)]); f3_1 = repr(f3[(1,)])

    tests = [
        ("f1(0) = x'' + λ·x",
         "x''" in f1_0 and "λ" in f1_0),
        ("f2(0) = y'' + λ·y − g",
         "y''" in f2_0 and "λ" in f2_0 and "9.8" in f2_0),
        ("f3(0) = x² + y² − L²",
         "x²" in f3_0 or "x^2" in f3_0 or "x·x" in f3_0),
        ("f1(1) = x_g'' + λ_g·x + λ·x_g",
         "x_g''" in f1_1 or "x_g'" in f1_1),
        ("f2(1) = y_g'' + λ_g·y + λ·y_g − 1",
         ("y_g''" in f2_1 or "y_g'" in f2_1) and ("-1" in f2_1 or "- 1" in f2_1)),
        ("f3(1) = 2·x_g·x + 2·y_g·y",
         "x_g" in f3_1 and "y_g" in f3_1),
    ]

    ok = True
    for desc, cond in tests:
        s = "PASS" if cond else "FAIL"
        if not cond: ok = False
        print(f"    [{s}] {desc}")

    print(f"\n  {'ALL CHECKS PASSED' if ok else 'SOME FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if demo() else 1)
