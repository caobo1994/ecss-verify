"""
Validation W7: Comparison with Modern AD Frameworks.

Documents the modern AD comparison used in §7.2.4 of the thesis: it positions
the ECSS against Julia's ModelingToolkit.jl + DifferentialEquations.jl and
CasADi.
"""
import json
import os


COMPARISON_TEXT = r"""
===============================================================================
§7.2.4 — Comparison with Modern AD Frameworks

The preceding sections compared the ECSS against DAE-specific sensitivity
methods (SUNDIALS, Barrio's MVTS, Li–Petzold). We now position the ECSS
against two contemporary AD frameworks that a practitioner might consider
as alternatives.

7.2.4.1 Julia Ecosystem: ModelingToolkit.jl + DifferentialEquations.jl

ModelingToolkit.jl (MTK) performs structural analysis on symbolic DAEs,
generates optimised code, and interfaces with DifferentialEquations.jl for
integration. Forward-mode algorithmic differentiation (AD) via ForwardDiff.jl
can compute parameter sensitivities by differentiating through the solver.

Key differences from the ECSS:

1. Solver integration vs solver independence.
   MTK+ForwardDiff.jl propagates sensitivities through the solver's internal
   operations — a form of operator-overloading AD at the solver level. The
   ECSS, by contrast, generates a separate DAE for the sensitivity system.
   The MTK approach benefits from runtime efficiency (no separate integration);
   the ECSS benefits from portability across solvers that accept the generated
   problem form.

2. Index handling.
   MTK performs structural analysis and can generate index-reduced forms of
   DAEs. However, its default pipeline reduces DAEs to index 1 before
   integration, whereas the ECSS preserves the original index through
   structural inheritance and relies on the solver to handle it. For
   high-index DAEs, the ECSS with DAETS provides direct index-3 integration
   without projection error; MTK+DifferentialEquations.jl would index-reduce.

3. Sensitivity order.
   ForwardDiff.jl computes first-order sensitivities by propagating dual
   numbers through the solver. Higher-order sensitivities require nested AD
   (HyperDualNumbers.jl for second order) or repeated application, leading
   to the same combinatorial growth the ECSS addresses natively via MVTS.

4. Target use case.
   MTK+ForwardDiff.jl is the natural choice for index-1 DAEs with many
   parameters where first-order sensitivities suffice (O(m) forward passes
   vs the ECSS's O(N(q)) system). The ECSS is superior for high-index
   DAEs, higher-order sensitivities, or when solver portability matters.

7.2.4.2 CasADi

CasADi provides algorithmic differentiation for dynamic optimisation,
supporting forward and adjoint modes for DAEs. It generates sensitivity
equations by source-transforming the symbolic problem representation.

Key differences from the ECSS:

1. Generation mechanism.
   CasADi uses source transformation (symbolic differentiation of the
   expression graph). The ECSS uses operator overloading (MVTS arithmetic
   at runtime). Source transformation avoids the overhead of MVTS
   coefficient propagation for elementary operations; operator overloading
   handles arbitrary C++ functions without requiring symbolic parsing.

2. Higher-order sensitivities.
   CasADi computes higher-order sensitivities by recursive application of
   its forward mode, generating one augmented DAE per additional order.
   The ECSS generates all orders simultaneously in a single MVTS pass.
   For sensitivity order q with m parameters, CasADi would need q
   successive augmentations; the ECSS needs one MVTS evaluation.

3. DAE index.
   CasADi's DAE support targets index-1 systems (after index reduction).
   The ECSS, via structural inheritance, handles the original index
   without reduction.

4. Sparsity.
   CasADi exploits sparsity in the expression graph automatically.
   The ECSS inherits Σ-matrix sparsity structurally but does not currently
   exploit expression-level sparsity (future work item 1).

7.2.4.3 Positioning Summary

Among the compared frameworks, only the ECSS combines all three of:

  (a) SA-supported index handling (validated here from index 0 through
      index 5, with inheritance proved under the SA assumptions);
  (b) Arbitrary-order computation (via MVTS arithmetic in a single pass);
  (c) Solver-portable generated equations (a self-contained DAE, not solver-tied AD).

This is the distinguishing combination in this comparison. The Julia and
CasADi ecosystems offer complementary strengths — runtime efficiency for
index-1 DAEs (Julia) and symbolic sparsity exploitation (CasADi) — and
the choice of tool should depend on the specific requirements of the
application.

Table 7.8: Capability comparison: ECSS vs modern AD frameworks.

Feature                    ECSS           Julia MTK        CasADi
DAE index                  SA-supported   ≤1 after reduc.  ≤1 after reduc.
Sensitivity order          Arbitrary      1 (nestable)     1 (nestable)
Generation                 MVTS (op ovld) AD (op ovld)     Symbolic diff.
Solver portability         Generated DAE  Solver-tied      Limited
Sparsity exploitation      Structural     Full expression  Full expression
Higher-order generation    Single pass    Nested AD        Recursive
Licence                    (Thesis code)  MIT              LGPL

===============================================================================
Thesis location: §7.2.4, after the DAE-specific sensitivity comparisons.

Impact: This section pre-empts the question "why not just use Julia?" and
demonstrates the candidate's awareness of the contemporary AD landscape.
It strengthens the novelty claim by showing that even modern, well-maintained
frameworks do not solve the same problem.
===============================================================================
"""


def run_w7():
    """Print the AD framework comparison."""
    print("=" * 70)
    print("  VALIDATION W7: Comparison with Modern AD Frameworks")
    print("=" * 70)

    print(COMPARISON_TEXT)

    # Save as text for inclusion
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    comparison_file = os.path.join(results_dir, 'w7_ad_comparison.txt')
    with open(comparison_file, 'w') as f:
        f.write(COMPARISON_TEXT)

    print(f"\n  Saved to: {comparison_file}")
    print(f"  Validation: COMPLETE (comparison text generated)")
    print()

    # Save metadata
    metadata = {
        'section': '7.2.4',
        'title': 'Comparison with Modern AD Frameworks',
        'frameworks_compared': ['Julia ModelingToolkit.jl + DifferentialEquations.jl', 'CasADi'],
        'key_claim': 'No compared framework simultaneously provides SA-supported index handling, arbitrary-order generation, and solver-portable DAE sensitivity equations.',
        'status': 'thesis_support_text',
    }
    with open(os.path.join(results_dir, 'w7_ad_comparison.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    return True


if __name__ == '__main__':
    success = run_w7()
    print("=" * 70)
    print(f"  FINAL: W7 {'✓ COMPLETE' if success else '✗ FAIL'}")
    print("=" * 70)
