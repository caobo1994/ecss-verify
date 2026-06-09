#!/usr/bin/env python3
"""
Master runner for ECSS Python validation suite.

Companion verification for:
  Paper 1: Structural Inheritance in Sensitivity Systems of DAEs
  Paper 2: Automatic Generation of Sensitivity Systems for DAEs
           via Multivariate Taylor Series
  by Bo Cao (McMaster University, 2026)

Usage:
    python run_all.py              # Run all tests
    python run_all.py --quick      # Run only must-do tests
    python run_all.py --test W1    # Run specific test
"""
import sys
import os
import time
import argparse
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

# ──────────────────────────────────────────────────────────
# Test Registry
# ──────────────────────────────────────────────────────────

TESTS = {
    'W1': {
        'name': 'Index 4-5 Validation',
        'module': 'test_high_index',
        'function': 'run_all_high_index',
        'priority': 'MUST',
        'weakness': 'No validation beyond index 3',
        'time_est': '~10s',
    },
    'W2': {
        'name': 'Solver Independence (DAE)',
        'module': 'test_solver_independence',
        'function': 'run_all_solver_indep',
        'priority': 'MUST',
        'weakness': 'Solver independence only shown for ODE',
        'time_est': '~15s',
    },
    'W3': {
        'name': 'Coupled N-Pendulum Check',
        'module': 'test_npendulum',
        'function': 'run_coupled_chain_validation',
        'priority': 'SHOULD',
        'weakness': 'Need coupled multibody sanity check beyond single pendulum',
        'time_est': '~60s',
    },
    'W4': {
        'name': 'Consistent-Point Inheritance',
        'module': 'test_consistent_init',
        'function': 'run_w4',
        'priority': 'MUST',
        'weakness': 'No formal consistent initialisation lemma',
        'time_est': '~2s',
    },
    'W5': {
        'name': 'Runtime Profile Breakdown',
        'module': 'test_runtime_profile',
        'function': 'run_profiling',
        'priority': 'NICE',
        'weakness': 'No runtime decomposition',
        'time_est': '~30s',
    },
    'W7': {
        'name': 'AD Framework Comparison',
        'module': 'test_ad_comparison',
        'function': 'run_w7',
        'priority': 'SHOULD',
        'weakness': 'No comparison with modern AD tools',
        'time_est': '~1s',
    },
}

# ──────────────────────────────────────────────────────────
# Test Functions (wrappers for individual module entry points)
# ──────────────────────────────────────────────────────────


def run_all_high_index():
    from test_high_index import _validate_chain, structural_inheritance_summary
    ok1 = _validate_chain("Index-4", 5, "w1_index4_results.json")
    ok2 = _validate_chain("Index-5", 6, "w1_index5_results.json")
    structural_inheritance_summary()
    return ok1 and ok2


def run_all_solver_indep():
    from test_solver_independence import run_rc_circuit_scipy, run_coupled_ode_cross_solver
    ok1 = run_rc_circuit_scipy()
    ok2 = run_coupled_ode_cross_solver()
    return ok1 and ok2


def run_coupled_chain_validation():
    from test_npendulum import run_coupled_chain_validation as _run_coupled_chain_validation
    return _run_coupled_chain_validation()


def run_profiling():
    from test_runtime_profile import run_profiling
    return run_profiling()


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='ECSS Python Validation Suite — Defence Weakness Remediation'
    )
    parser.add_argument('--quick', action='store_true',
                        help='Run only MUST-DO tests (W1, W2)')
    parser.add_argument('--test', type=str, nargs='+',
                        help=f'Run specific tests: {list(TESTS.keys())}')
    parser.add_argument('--list', action='store_true',
                        help='List available tests')
    args = parser.parse_args()

    if args.list:
        print("\nAvailable Tests:")
        print("-" * 60)
        for key, info in TESTS.items():
            print(f"  {key}: {info['name']} [{info['priority']}] — {info['weakness']}")
            print(f"       Est. time: {info['time_est']}")
        return

    # Determine which tests to run
    if args.test:
        selected = {k: TESTS[k] for k in args.test if k in TESTS}
        if not selected:
            print(f"Error: No valid tests in {args.test}")
            print(f"Valid keys: {list(TESTS.keys())}")
            sys.exit(1)
    elif args.quick:
        selected = {k: v for k, v in TESTS.items() if v['priority'] == 'MUST'}
    else:
        selected = TESTS

    # Header
    print("=" * 70)
    print("  ECSS Python Validation Suite")
    print("  Companion verification for:")
    print("    Paper 1: Structural Inheritance in Sensitivity Systems of DAEs")
    print("    Paper 2: ECSS Builder via Multivariate Taylor Series")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Tests: {', '.join(selected.keys())}")
    print("=" * 70)

    results = {}
    start_total = time.time()

    for key, info in selected.items():
        print(f"\n{'─' * 70}")
        print(f"  Running {key}: {info['name']}")
        print(f"  Priority: {info['priority']} | Est. time: {info['time_est']}")
        print(f"{'─' * 70}")

        start_test = time.time()
        try:
            func = globals().get(info['function'])
            if func is None:
                # Try importing from module
                mod = __import__(info['module'])
                func = getattr(mod, info['function'].replace('run_all_', 'run_'))
            ok = func()
            elapsed = time.time() - start_test
            results[key] = {'pass': ok, 'elapsed': elapsed, 'error': None}
            print(f"\n  {key}: {'✓ PASS' if ok else '✗ FAIL'} ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - start_test
            results[key] = {'pass': False, 'elapsed': elapsed, 'error': str(e)}
            print(f"\n  {key}: ✗ ERROR ({elapsed:.1f}s)")
            print(f"  {traceback.format_exc()}")

    total_elapsed = time.time() - start_total

    # Summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    for key, info in selected.items():
        r = results.get(key, {})
        status = '✓ PASS' if r.get('pass') else ('✗ FAIL' if r.get('error') is None else '✗ ERROR')
        print(f"  {key} [{info['priority']:6s}] {info['name']:<30s} {status} ({r.get('elapsed', 0):.1f}s)")

    n_pass = sum(1 for r in results.values() if r.get('pass'))
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} tests passed in {total_elapsed:.1f}s")

    # Summary
    print(f"\n{'─' * 70}")
    print("  PAPER EVIDENCE")
    print(f"{'─' * 70}")
    if results.get('W1', {}).get('pass'):
        print("  ✓ W1 passed → high-index validation (Paper 1, index 4--5)")
    if results.get('W2', {}).get('pass'):
        print("  ✓ W2 passed → solver-independence check (Paper 2, Section 7)")
    if results.get('W3', {}).get('pass'):
        print("  ✓ W3 passed → coupled N-pendulum validation (Paper 2, Section 8)")
    if results.get('W4', {}).get('pass'):
        print("  ✓ W4 passed → consistent-point inheritance (Paper 1, Lemma 3)")
    if results.get('W5', {}).get('pass'):
        print("  ✓ W5 passed → runtime composition (Paper 2, Section 8)")
    if results.get('W7', {}).get('pass'):
        print("  ✓ W7 passed → AD framework comparison (Paper 2, Section 1)")

    print(f"\n  Results saved to: {os.path.join(os.path.dirname(__file__), 'results/')}")

    # Exit code
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == '__main__':
    main()
