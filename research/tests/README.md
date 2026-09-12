# tests — the executable audit

32 files, 325 tests. Run from the package root:

```bash
./.venv/bin/python -m pytest tests/ -q
```

Roughly 26 minutes for the full suite. It was 12 before the mixture M-step's
inner-sweep cap went from 4 to 16 on 18 Aug 2026 — that default was
under-converging the innovation shape, and the cost is accepted rather than
tuned away.

`tests/test_backend_parity.py` needs a GPU and skips without one.

## What these are actually for

Not coverage. Each one pins a claim that a document makes, and several exist
because the claim was once wrong:

| Test | The claim it guards |
|---|---|
| `test_gaussian_bp_equivalence` | grid BP reproduces the Gaussian closed form |
| `test_em_bp::test_xi_matches_brute_force_enumeration` | Ξ against independent enumeration, sharing no code |
| `test_em_bp::test_fisher_identity_matches_finite_differences` | the gradient really is the marginal likelihood's |
| `test_em_bp::test_em_is_monotone` | EM ascends the exact evidence |
| `test_em_bp::test_returned_kernel_is_the_one_the_trace_ends_on` | the returned kernel is the one whose evidence was reported |
| `test_em_bp::test_a_decrease_is_not_reported_as_convergence` | a censored run cannot look converged |
| `test_ring::test_stopping_rule_does_not_depend_on_sample_size` | the plateau tolerance is per sequence |
| `test_laplace_rho_lattice` | the Laplace M-step's ρ is a grid artefact, and says so |

## Two habits worth keeping

**Write the negative control.** A regression test that passes against the bug it
was written for is worse than no test. `test_stopping_rule_does_not_depend_on_sample_size`
asserts that neither fit reached the iteration cap, because at the default
tolerance both did and the two sizes agreed trivially — the test passed while
proving nothing.

**Let the control overrule you.** `test_stopping_tolerance_is_scale_free_in_both_units`
records a *negative* result: an external audit claimed the EM tolerance had to be
normalised by transition count, the first version of the test asserted the two
rules diverge, and its own vacuity guard caught that they do not. The test now
pins the invariance instead of a difference that was never there.
