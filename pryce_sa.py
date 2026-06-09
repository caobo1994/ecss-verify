#!/usr/bin/env python3
"""
Standalone Pryce structural analysis for DAEs.

Given an n x n Sigma-matrix (with -inf for absent derivatives),
computes the highest-value transversal, canonical offsets, system
Jacobian structure, and structural index.

Reference: J. D. Pryce, "A Simple Structural Analysis Method for DAEs",
            BIT Numerical Mathematics, 41(2):364-394, 2001.
"""

from typing import List, Tuple

NEG_INF = float("-inf")


def find_hvt(sigma: List[List[float]]) -> Tuple[List[Tuple[int, int]], float]:
    """Find a highest-value transversal of the Sigma-matrix.

    Uses the Hungarian algorithm (linear sum assignment) from scipy.
    Returns list of (i,j) pairs and the sum of sigma values.
    """
    from scipy.optimize import linear_sum_assignment

    n = len(sigma)
    import numpy as np

    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if sigma[i][j] == NEG_INF:
                cost[i, j] = 1e6  # large penalty to exclude -inf
            else:
                cost[i, j] = -float(sigma[i][j])
    row_ind, col_ind = linear_sum_assignment(cost)
    hvt = [(int(row_ind[k]), int(col_ind[k])) for k in range(n)]
    hvt_sum = sum(sigma[i][j] for i, j in hvt)
    return hvt, hvt_sum


def compute_canonical_offsets(
    sigma: List[List[float]], hvt: List[Tuple[int, int]]
) -> Tuple[List[int], List[int]]:
    """Compute canonical offsets via explicit constraint propagation.

    The canonical offsets satisfy (Pryce 2001, Sec 3):
      d_j - c_i == sigma_{ij}  for HVT pairs
      d_j - c_i >= sigma_{ij}  for all (i,j)
      min c = 0

    Starting from c=0, we propagate along the HVT equations
    to determine all unknowns, then enforce feasibility by
    raising d_j or c_i as needed.
    """
    n = len(sigma)

    # Build system from HVT
    hvt_set = set(hvt)

    # Iterative relaxation: start from c=0, d=0, then
    # (a) enforce HVT equalities (propagate)
    # (b) enforce inequality constraints
    # Repeat until convergence.
    c = [0.0] * n
    d = [0.0] * n

    for _ in range(10 * n):
        changed = False

        # HVT tightness: raise d_j to satisfy d_j >= c_i + sigma_{ij}
        for i, j in hvt:
            needed = c[i] + sigma[i][j]
            if d[j] < needed:
                d[j] = needed
                changed = True

        # Reverse HVT: raise c_i to satisfy c_i >= d_j - sigma_{ij}
        # (derived from d_j - c_i = sigma_{ij} => c_i = d_j - sigma_{ij})
        for i, j in hvt:
            needed = d[j] - sigma[i][j]
            if c[i] < needed:
                c[i] = needed
                changed = True

        # Feasibility: d_j >= c_i + sigma_{ij} for all pairs
        for i in range(n):
            for j in range(n):
                if sigma[i][j] != NEG_INF:
                    needed = c[i] + sigma[i][j]
                    if d[j] < needed:
                        d[j] = needed
                        changed = True

        if not changed:
            break

    # Normalize: shift so min c = 0
    min_c = min(c)
    if min_c != 0:
        shift = min_c
        for i in range(n):
            c[i] -= shift
            d[i] -= shift

    return [round(v) for v in c], [round(v) for v in d]


def system_jacobian_structure(
    sigma: List[List[float]], c: List[int], d: List[int]
) -> List[Tuple[int, int]]:
    """Positions of structural nonzeros in the system Jacobian."""
    n = len(sigma)
    entries = []
    for i in range(n):
        for j in range(n):
            if sigma[i][j] != NEG_INF and sigma[i][j] == d[j] - c[i]:
                entries.append((i, j))
    return entries


def structural_index(c: List[int], d: List[int]) -> int:
    """DAETS-reported structural index."""
    if min(d) == 0:
        return max(c) + 1
    return max(c)


def analyse(sigma: List[List[float]]) -> dict:
    """Run full Pryce structural analysis on a Sigma-matrix."""
    hvt, hvt_sum = find_hvt(sigma)
    c, d = compute_canonical_offsets(sigma, hvt)
    jac = system_jacobian_structure(sigma, c, d)

    n = len(sigma)
    feasible = True
    for i in range(n):
        for j in range(n):
            if sigma[i][j] != NEG_INF:
                if d[j] - c[i] < sigma[i][j]:
                    feasible = False

    return {
        "sigma": sigma,
        "hvt": hvt,
        "hvt_sum": hvt_sum,
        "c": c,
        "d": d,
        "jacobian": jac,
        "s_index": structural_index(c, d),
        "feasible": feasible,
    }


def sigma_to_string(sigma: List[List[float]]) -> str:
    """Format Sigma-matrix for display."""
    lines = []
    for row in sigma:
        cells = []
        for v in row:
            if v == NEG_INF:
                cells.append("-inf")
            else:
                cells.append(f"{int(v):>4d}")
        lines.append("[" + " ".join(cells) + "]")
    return "\n".join(lines)


def format_result(result: dict) -> str:
    """Format analysis result for display."""
    c = result["c"]
    d = result["d"]
    hvt = result["hvt"]
    lines = []
    lines.append(f"Sigma-matrix:\n{sigma_to_string(result['sigma'])}")
    lines.append(f"\nHVT sum = {result['hvt_sum']}")
    lines.append(f"HVT positions: {hvt}")
    lines.append(f"Offsets  c = {c}")
    lines.append(f"         d = {d}")
    lines.append(f"Structural index (DAETS convention) = {result['s_index']}")
    lines.append(
        f"System Jacobian structurally nonzero entries: {result['jacobian']}"
    )
    lines.append(f"Feasible: {result['feasible']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test systems from the thesis
# ---------------------------------------------------------------------------

def controlled_pendulum_sigma():
    return [
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, NEG_INF],
    ]


def cascade_si4_sigma():
    n = 5
    sigma = [[NEG_INF] * n for _ in range(n)]
    for i in range(4):
        sigma[i][i] = 1
        sigma[i][i + 1] = 0
    sigma[4][0] = 0
    return sigma


if __name__ == "__main__":
    print("=" * 60)
    print("Pryce Structural Analysis -- Independent Python Implementation")
    print("=" * 60)

    test_systems = {
        "Controlled pendulum (n=3)": controlled_pendulum_sigma(),
        "Cascade SI-4 (n=5)": cascade_si4_sigma(),
    }

    for name, sigma in test_systems.items():
        print(f"\n--- {name} ---")
        result = analyse(sigma)
        print(format_result(result))

    # Verify against known results from the thesis
    print("\n" + "=" * 60)
    print("Verification against thesis results")
    print("=" * 60)

    r = analyse(controlled_pendulum_sigma())
    assert r["c"] == [0, 0, 2], f"c mismatch: got {r['c']}, expected [0,0,2]"
    assert r["d"] == [2, 2, 0], f"d mismatch: got {r['d']}, expected [2,2,0]"
    assert r["s_index"] == 3, f"index mismatch: got {r['s_index']}, expected 3"
    print("Controlled pendulum: PASS (c=(0,0,2), d=(2,2,0), s_index=3)")

    r = analyse(cascade_si4_sigma())
    assert r["c"] == [3, 2, 1, 0, 4], f"c mismatch: got {r['c']}, expected [3,2,1,0,4]"
    assert r["d"] == [4, 3, 2, 1, 0], f"d mismatch: got {r['d']}, expected [4,3,2,1,0]"
    assert r["s_index"] == 5, f"index mismatch: got {r['s_index']}, expected 5"
    print("Cascade SI-4: PASS (c=(3,2,1,0,4), d=(4,3,2,1,0), s_index=5)")

    print("\nAll verifications passed.")
