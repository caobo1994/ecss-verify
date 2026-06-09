#!/usr/bin/env python3
"""
Fully implemented Richardson extrapolation FD via actual DAETS solver runs.

Strategy: Temporarily overwrite the source of an existing cmake-registered
target (MVTS_simple_pendulum_blocked), rebuild only that target, run it,
parse stdout, and restore the original source.

This avoids:
- ASAN link-flag mismatches (cmake handles all flags automatically)
- CMake reconfigure overhead (target already registered)
- Manual library-path hunting

Estimated runtime (quick mode, 2 step sizes): ~2.5 minutes
Full mode (4 step sizes): ~5 minutes
"""

import subprocess, os, sys, json, re
from pathlib import Path

VERIFY_DIR = Path(__file__).resolve().parent
BC_DIR = VERIFY_DIR.parent.parent            # .../Development/BC/
ROOT_DIR = BC_DIR.parent.parent               # .../daets-1.1/
EXAMPLES_DIR = BC_DIR / "examples"
BUILD_DIR = BC_DIR / "build"

TARGET = "MVTS_simple_pendulum_blocked"
TARGET_SRC = EXAMPLES_DIR / f"{TARGET}.cpp"

# Templates pre-written for fast substitution
# Template A: ECSS mode (sens_order >= 1) — uses integrate_sensitivity
TEMPLATE_ECSS = """#include "DAEsolver.h"
#include <iostream>
#include <iomanip>
#include "fcn_simppend.hpp"
#include "libmvts.hpp"
#undef Int
FCN_ORIGINAL_INIT
using namespace daets;
using namespace std;
using namespace mvts;
int main()
{{
  const int num_vars = 3;
  const int sens_order = {sens_order};
  Vector<double> param_values = {{{g}, {L_sq}}};
  Vector<double> scaling = {{1.0, 1.0}};
  mvts::Vector<InitialValue<double>> initial_values = {{{{0, 0, 1.0}}}};
  auto [ dp, mvts_order, solver_ptr, x_ptr ] = integrate_sensitivity(
      num_vars, param_values, sens_order, scaling,
      initial_values, 0.0, {t_end},
      "{outfile}");
  using namespace std;
  cout << fixed << setprecision(16);
  cout << endl << "== ECSS sensitivity values at t=" << {t_end} << " ==" << endl;
  cout << "x      = " << x_ptr->getX(0,0) << endl;
  cout << "y      = " << x_ptr->getX(1,0) << endl;
  cout << "lambda = " << x_ptr->getX(2,0) << endl;
  if (sens_order >= 1) {{
    cout << "dx/dp  = " << x_ptr->getX(3,0) << endl;
    cout << "dy/dp  = " << x_ptr->getX(4,0) << endl;
    cout << "dlam/dp= " << x_ptr->getX(5,0) << endl;
  }}
  return 0;
}}
"""


def build_and_run(g, L_sq=1.0, sens_order=1, t_end=10.0):
    """Overwrite source, cmake build, run, parse, restore."""

    # Backup original source
    backup = TARGET_SRC.read_text()

    try:
        # Write modified source using template
        outfile = f"_fd_temp_sens{sens_order}.txt"
        src = TEMPLATE_ECSS.format(
            sens_order=sens_order, g=g, L_sq=L_sq, t_end=t_end,
            outfile=outfile)
        TARGET_SRC.write_text(src)

        # Build via cmake (no reconfigure needed — target exists)
        r = subprocess.run(
            ["cmake", "--build", str(BUILD_DIR), "--target", TARGET],
            capture_output=True, text=True, cwd=str(BC_DIR), timeout=180
        )
        if r.returncode != 0:
            err = r.stderr[-600:] or r.stdout[-600:]
            return None, f"Build failed:\n{err}"

        # Find binary
        exe = BUILD_DIR / "examples" / TARGET
        if not exe.exists():
            found = list(BUILD_DIR.rglob(TARGET))
            exe = found[0] if found else None
        if not exe or not exe.exists():
            return None, f"Binary not found in {BUILD_DIR}"

        # Run
        r = subprocess.run(
            [str(exe)], capture_output=True, text=True,
            cwd=str(EXAMPLES_DIR), timeout=120
        )
        output = r.stdout + r.stderr

        # Check for ASAN crash
        if "ABORT" in output and "== ECSS" not in output:
            return None, f"Runtime crash (ASAN):\n{output[-500:]}"

        # Parse values from output (multi-line, real newlines after endl fix)
        values = {}
        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue
            for key in ['x', 'y', 'lambda', 'dx/dp', 'dy/dp', 'dlam/dp']:
                m = re.match(re.escape(key) + r'\s*=\s*([\-\d\.e\+]+)', line)
                if m:
                    values[key] = float(m.group(1))
                    break

        return values, output

    finally:
        # Restore original source
        TARGET_SRC.write_text(backup)
        # Clean temp output files
        for f in EXAMPLES_DIR.glob("_fd_temp_*.txt"):
            try: f.unlink()
            except OSError: pass


def compute_richardson(g_nom=9.8, L_sq=1.0, t_end=1.0, hs=None):
    """Full Richardson extrapolation via DAETS solver runs."""

    if hs is None:
        hs = [1e-2, 5e-3, 2.5e-3, 1e-3]

    print("=" * 64)
    print("Richardson FD via Actual DAETS Solver Runs")
    print(f"Pendulum: g={g_nom}, L^2={L_sq}, t_end={t_end}")
    print(f"Step sizes: {hs}")
    print("=" * 64)

    # Step 1: ECSS at nominal g
    print("\n[1] ECSS reference at nominal g (sens_order=1)...", flush=True)
    ecss, err = build_and_run(g_nom, L_sq, sens_order=1, t_end=t_end)
    if ecss is None:
        print(f"    FAILED: {err}")
        return None
    ecss_dx = ecss.get("dx/dp")
    ecss_dy = ecss.get("dy/dp")
    if ecss_dx is None:
        print(f"    FAILED: could not parse dx/dp from output")
        return None
    print(f"    dx/dg = {ecss_dx:.15e}")
    print(f"    dy/dg = {ecss_dy:.15e}")

    # Step 2: state-only runs at g +/- h
    fd_dx = {}
    fd_dy = {}
    print("\n[2] State-only runs at g +/- h (sens_order=0)...")
    for h in sorted(hs, reverse=True):
        sys.stdout.write(f"    h={h:.1e} ...")
        sys.stdout.flush()

        vp, err_p = build_and_run(g_nom + h, L_sq, sens_order=0, t_end=t_end)
        if vp is None:
            print(f" FAILED (g+h): {err_p[:100]}")
            continue
        vm, err_m = build_and_run(g_nom - h, L_sq, sens_order=0, t_end=t_end)
        if vm is None:
            print(f" FAILED (g-h): {err_m[:100]}")
            continue

        if "x" not in vp or "x" not in vm:
            print(f" FAILED: missing x in state output")
            continue

        fd_x = (vp["x"] - vm["x"]) / (2 * h)
        fd_y = (vp["y"] - vm["y"]) / (2 * h)
        fd_dx[float(h)] = float(fd_x)
        fd_dy[float(h)] = float(fd_y)

        err_x = abs(fd_x - ecss_dx)
        err_y = abs(fd_y - ecss_dy)
        print(f" fd_x={fd_x:.15e} err={err_x:.2e}  fd_y={fd_y:.15e} err={err_y:.2e}")

    if len(fd_dx) < 2:
        print("    FAILED: need >= 2 successful FD values for Richardson")
        return None

    # Step 3: Richardson extrapolation
    def richardson(fd_dict, ecss_val):
        hs_s = sorted(fd_dict.keys(), reverse=True)
        vals = [fd_dict[h] for h in hs_s]
        rels = []
        row = list(vals)
        for k in range(1, len(vals)):
            nxt = []
            p = 4 ** k
            for i in range(len(row) - 1):
                nxt.append((p * row[i+1] - row[i]) / (p - 1))
            row = nxt
            rels.append(abs(row[-1] - ecss_val) / abs(ecss_val))
        return row[-1], abs(row[-1] - ecss_val), rels

    print("\n[3] Richardson extrapolation (dx/dg)...")
    final_x, err_x, rels_x = richardson(fd_dx, ecss_dx)
    for k, rel in enumerate(rels_x):
        print(f"    O(h^{2*(k+2)}): rel_err={rel:.2e}")
    print(f"    Final estimate: {final_x:.15e}")
    print(f"    Error vs ECSS:  {err_x:.2e}")

    print("\n[4] Richardson extrapolation (dy/dg)...")
    final_y, err_y, rels_y = richardson(fd_dy, ecss_dy)
    for k, rel in enumerate(rels_y):
        print(f"    O(h^{2*(k+2)}): rel_err={rel:.2e}")
    print(f"    Final estimate: {final_y:.15e}")
    print(f"    Error vs ECSS:  {err_y:.2e}")

    return {
        "system": "simple_pendulum",
        "g_nom": g_nom, "L_sq": L_sq, "t_end": t_end,
        "hs": sorted(fd_dx.keys(), reverse=True),
        "fd_dx_dg": fd_dx,
        "fd_dy_dg": fd_dy,
        "ecss_dx_dg": ecss_dx,
        "ecss_dy_dg": ecss_dy,
        "richardson_dx": final_x,
        "richardson_dy": final_y,
        "abs_error_dx": err_x,
        "abs_error_dy": err_y,
        "rel_errors_dx": rels_x,
        "rel_errors_dy": rels_y,
        "verified": err_x < 1e-10 * abs(ecss_dx) and err_y < 1e-10 * abs(ecss_dy),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Richardson FD via actual DAETS solver (cmake rebuild)")
    p.add_argument("--g", type=float, default=9.8)
    p.add_argument("--L2", type=float, default=1.0)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("-o", "--output", default="fd_richardson_result.json")
    p.add_argument("--quick", action="store_true",
                   help="Use only 2 step sizes (faster)")
    args = p.parse_args()

    hs = [1e-2, 5e-3] if args.quick else [1e-2, 5e-3, 2.5e-3, 1e-3]

    result = compute_richardson(args.g, args.L2, args.t, hs)
    if result is None:
        sys.exit(1)

    outpath = VERIFY_DIR / args.output
    serializable = {}
    for k, v in result.items():
        if isinstance(v, dict):
            serializable[k] = {str(k2): v2 for k2, v2 in v.items()}
        elif isinstance(v, list):
            serializable[k] = [float(x) for x in v]
        elif isinstance(v, float):
            serializable[k] = float(v)
        else:
            serializable[k] = v
    with open(outpath, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\n{'='*64}")
    print(f"Results saved to {outpath}")
    print(f"{'='*64}")
    print(f"ECSS dx/dg:         {result['ecss_dx_dg']:.15e}")
    print(f"Richardson dx/dg:    {result['richardson_dx']:.15e}")
    print(f"Abs error dx/dg:    {result['abs_error_dx']:.2e}")
    print(f"")
    print(f"ECSS dy/dg:         {result['ecss_dy_dg']:.15e}")
    print(f"Richardson dy/dg:    {result['richardson_dy']:.15e}")
    print(f"Abs error dy/dg:    {result['abs_error_dy']:.2e}")
    if result['verified']:
        print(f"\nVERIFIED: Richardson-extrapolated FD matches ECSS")
        print(f"via actual DAETS solver runs at perturbed parameters.")
    print(f"{'='*64}")
