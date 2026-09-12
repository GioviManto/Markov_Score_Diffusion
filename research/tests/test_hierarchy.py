"""Tree priors, tree BP, and the two time scales.

The load-bearing checks here are the two independent computations of the same
object: the analytic ultrametric spectrum against `eigh` of the dense
covariance, and information-form tree BP against a dense linear solve. Both are
the tree analogues of the checks that validated the chain code.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import spectral
from src.hierarchy import (
    GaussianTree,
    filtered_tree_bp_gaussian,
    TreeIndex,
    tree_bp_gaussian,
    tree_bp_grid,
    tree_posterior_mean_dense,
    tree_score_gaussian,
)
from src.noising import alpha_delta
from src.priors import GaussianAR1, LaplaceAR1
from src.utils import rng_for


# ----------------------------------------------------------------------------
# Tree bookkeeping and the prior
# ----------------------------------------------------------------------------

def test_tree_index_children_are_consistent():
    ti = TreeIndex(depth=3, branching=2)
    assert ti.n_leaves == 8
    assert ti.n_nodes == 15
    seen = []
    for node in ti.nodes_at(0):
        seen.extend(ti.children(int(node), 0).tolist())
    assert seen == ti.nodes_at(1).tolist()


def test_lca_depth_matrix_matches_definition():
    ti = TreeIndex(depth=3, branching=2)
    d = ti.lca_depth_matrix()
    assert np.all(np.diag(d) == 3)          # a leaf with itself
    assert d[0, 1] == 2                     # siblings
    assert d[0, 2] == 1                     # cousins
    assert d[0, 4] == 0                     # opposite halves
    assert np.array_equal(d, d.T)


def test_sampled_covariance_matches_analytic():
    tree = GaussianTree(depth=3, branching=2, rho=0.8)
    a = tree.sample(rng_for("test-tree-cov"), 200_000)
    emp = np.cov(a, rowvar=False)
    assert np.max(np.abs(emp - tree.leaf_covariance())) < 0.02


@pytest.mark.parametrize("branching,depth,rho", [(2, 3, 0.8), (2, 4, 0.9), (3, 2, 0.7)])
def test_analytic_spectrum_matches_dense_eigendecomposition(branching, depth, rho):
    tree = GaussianTree(depth=depth, branching=branching, rho=rho)
    c = tree.leaf_covariance()

    analytic = []
    for _level, lam, mult in tree.level_eigenvalues():
        analytic.extend([lam] * mult)
    analytic = np.sort(np.asarray(analytic))

    numeric = np.sort(np.linalg.eigvalsh(c))
    assert analytic.size == tree.n_leaves
    assert np.allclose(analytic, numeric, atol=1e-10)
    assert np.isclose(analytic.sum(), tree.n_leaves)   # unit marginal variance


def test_level_basis_is_orthonormal_and_diagonalizes():
    tree = GaussianTree(depth=3, branching=2, rho=0.85)
    v, levels = tree.level_projector_basis()
    n = tree.n_leaves
    assert np.allclose(v.T @ v, np.eye(n), atol=1e-12)

    c = tree.leaf_covariance()
    lam_of_level = {lev: lam for lev, lam, _ in tree.level_eigenvalues()}
    for k in range(n):
        expected = lam_of_level[int(levels[k])]
        assert np.allclose(c @ v[:, k], expected * v[:, k], atol=1e-12)


def test_eigenvalues_are_ordered_coarse_to_fine():
    """The uniform mode is the largest and each finer contrast is smaller.

    This ordering is what makes the speciation cascade well posed: distinct,
    monotone eigenvalues give distinct, monotone transition times.
    """
    tree = GaussianTree(depth=4, branching=2, rho=0.9)
    spec = sorted(tree.level_eigenvalues(), key=lambda r: r[0])  # -1, 0, 1, ...
    levels = [lev for lev, _lam, _m in spec]
    lams = [lam for _lev, lam, _m in spec]
    assert levels == [-1, 0, 1, 2, 3]
    assert lams == sorted(lams, reverse=True)
    assert lams[0] / lams[-1] > 5.0


# ----------------------------------------------------------------------------
# Tree BP
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("t", [0.05, 0.3, 1.0, 2.5])
def test_information_form_tree_bp_is_exact(t):
    tree = GaussianTree(depth=4, branching=2, rho=0.85)
    rng = rng_for("test-tree-bp", t)
    a = tree.sample(rng, 16)
    alpha, delta = alpha_delta(t)
    x = alpha * a + np.sqrt(delta) * rng.standard_normal(a.shape)

    bp = tree_bp_gaussian(tree, x, alpha, delta)
    dense = tree_posterior_mean_dense(tree, x, alpha, delta)
    assert np.max(np.abs(bp - dense)) < 1e-10


def test_tree_bp_handles_branching_three():
    tree = GaussianTree(depth=2, branching=3, rho=0.7)
    rng = rng_for("test-tree-bp3")
    a = tree.sample(rng, 8)
    alpha, delta = alpha_delta(0.4)
    x = alpha * a + np.sqrt(delta) * rng.standard_normal(a.shape)
    assert np.max(
        np.abs(tree_bp_gaussian(tree, x, alpha, delta)
               - tree_posterior_mean_dense(tree, x, alpha, delta))
    ) < 1e-10


def test_tree_score_satisfies_the_identity():
    tree = GaussianTree(depth=3, branching=2, rho=0.8)
    rng = rng_for("test-tree-score")
    a = tree.sample(rng, 8)
    t = 0.6
    alpha, delta = alpha_delta(t)
    x = alpha * a + np.sqrt(delta) * rng.standard_normal(a.shape)
    s = tree_score_gaussian(tree, x, t)
    m = tree_bp_gaussian(tree, x, alpha, delta)
    assert np.max(np.abs(s + (x - alpha * m) / delta)) < 1e-12


@pytest.mark.parametrize("t", [0.2, 0.8])
def test_grid_tree_bp_matches_information_form(t):
    """Grid messages reproduce the exact Gaussian answer.

    Same cross-check as the chain: the grid code is the one that will be run on
    non-Gaussian innovations, where no closed form exists, so it has to be
    pinned against the case where one does.
    """
    tree = GaussianTree(depth=3, branching=2, rho=0.8)
    rng = rng_for("test-tree-grid", t)
    a = tree.sample(rng, 6)
    alpha, delta = alpha_delta(t)
    x = alpha * a + np.sqrt(delta) * rng.standard_normal(a.shape)

    grid = np.linspace(-8.0, 8.0, 601)
    prior = GaussianAR1(rho=0.8)
    log_k = prior.log_transition_matrix(grid)
    log_root = -0.5 * grid**2

    grid_mean = tree_bp_grid(
        log_k, grid, log_root, x, alpha, delta, branching=2, depth=3
    )
    exact = tree_bp_gaussian(tree, x, alpha, delta)
    assert np.max(np.abs(grid_mean - exact)) < 2e-6


def test_grid_tree_bp_runs_on_a_nongaussian_innovation():
    """A Laplace-innovation tree has no information form; BP still returns a
    posterior mean, and it must differ from the Gaussian one."""
    grid = np.linspace(-8.0, 8.0, 401)
    lap = LaplaceAR1(rho=0.8)
    gauss = GaussianAR1(rho=0.8)
    log_root = -0.5 * grid**2

    rng = rng_for("test-tree-grid-laplace")
    tree = GaussianTree(depth=3, branching=2, rho=0.8)
    a = tree.sample(rng, 6)
    alpha, delta = alpha_delta(0.3)
    x = alpha * a + np.sqrt(delta) * rng.standard_normal(a.shape)

    m_lap = tree_bp_grid(
        lap.log_transition_matrix(grid), grid, log_root, x, alpha, delta, 2, 3
    )
    m_gauss = tree_bp_grid(
        gauss.log_transition_matrix(grid), grid, log_root, x, alpha, delta, 2, 3
    )
    assert np.all(np.isfinite(m_lap))
    assert np.max(np.abs(m_lap - m_gauss)) > 1e-3


# ----------------------------------------------------------------------------
# Hierarchical filtering (Garnier-Brun et al., arXiv:2408.15138 §2.2)
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("depth,k", [(3, 0), (3, 1), (3, 2), (3, 3), (4, 2)])
def test_filtered_sampler_matches_the_analytic_covariance(depth, k):
    """The filtering must keep the marginals and only flatten the top.

    This is the check that the construction is the paper's and not the easy
    misreading of it. Drawing the depth-k nodes conditionally independently
    *given the root* leaves every cross-block pair correlated at rho^{2L};
    making the blocks outright independent -- which is what "truncate the
    hierarchy" sounds like -- would put a 0 there instead.
    """
    tree = GaussianTree(depth=depth, branching=2, rho=0.85, filter_level=k)
    a = tree.sample(rng_for("test-filter-cov", depth, k), 200_000)
    assert np.max(np.abs(np.cov(a, rowvar=False) - tree.leaf_covariance())) < 0.02
    assert abs(np.var(a) - 1.0) < 0.02              # marginals untouched
    if 0 < k < depth:
        cov = tree.leaf_covariance()
        m = tree.block_size
        assert np.all(cov[:m, m: 2 * m] == tree.cross_block_covariance)
        assert cov[0, 1] > tree.cross_block_covariance   # within-block survives


@pytest.mark.parametrize("depth,k", [(3, 0), (3, 1), (3, 2), (3, 3), (4, 1), (4, 2)])
def test_filtered_spectrum_matches_dense_eigendecomposition(depth, k):
    tree = GaussianTree(depth=depth, branching=2, rho=0.9, filter_level=k)
    analytic = []
    for _lev, lam, mult in tree.level_eigenvalues():
        analytic.extend([lam] * mult)
    analytic = np.sort(np.asarray(analytic))
    assert analytic.size == tree.n_leaves
    assert np.allclose(analytic, np.sort(np.linalg.eigvalsh(tree.leaf_covariance())))
    assert np.isclose(analytic.sum(), tree.n_leaves)


def test_filtering_removes_rungs_from_the_speciation_ladder():
    """The headline consequence of composing the two papers' devices.

    Filtering at level k merges the top k rungs into one, so the number of
    distinct speciation times falls from L+1 to L-k+2. The k=0 and k=1 cases
    coincide, independently reproducing the paper's remark that those two share
    a tree topology and differ only in the top transition probabilities.
    """
    depth = 4
    counts = {
        k: len(GaussianTree(depth=depth, branching=2, rho=0.9, filter_level=k)
               .level_eigenvalues())
        for k in range(depth + 1)
    }
    assert counts == {0: 5, 1: 5, 2: 4, 3: 3, 4: 2}
    for k in range(1, depth + 1):
        assert counts[k] == max(depth - k + 2, 2)

    # At k = L the covariance is equicorrelated: leaves conditionally i.i.d.
    full = GaussianTree(depth=depth, branching=2, rho=0.9, filter_level=depth)
    cov = full.leaf_covariance()
    off = cov[~np.eye(full.n_leaves, dtype=bool)]
    assert np.allclose(off, off[0])


@pytest.mark.parametrize("depth,k,t", [(3, 0, 0.5), (3, 1, 0.3), (4, 2, 0.6),
                                       (3, 3, 1.0), (4, 1, 0.2)])
def test_filtered_bp_is_exact(depth, k, t):
    """`BP_k` is exact on its own model -- the mismatched-oracle comparison
    is only meaningful if each oracle is right about its own hierarchy."""
    tree = GaussianTree(depth=depth, branching=2, rho=0.85, filter_level=k)
    rng = rng_for("test-filtered-bp", depth, k, t)
    a = tree.sample(rng, 12)
    alpha, delta = alpha_delta(t)
    x = alpha * a + np.sqrt(delta) * rng.standard_normal(a.shape)
    assert np.max(
        np.abs(filtered_tree_bp_gaussian(tree, x, alpha, delta)
               - tree_posterior_mean_dense(tree, x, alpha, delta))
    ) < 1e-10


def test_filtered_basis_is_orthonormal_and_diagonalizes():
    tree = GaussianTree(depth=4, branching=2, rho=0.9, filter_level=2)
    v, levels = tree.level_projector_basis()
    n = tree.n_leaves
    assert np.allclose(v.T @ v, np.eye(n), atol=1e-12)
    lam_of = {lev: lam for lev, lam, _ in tree.level_eigenvalues()}
    c = tree.leaf_covariance()
    for j in range(n):
        assert np.allclose(c @ v[:, j], lam_of[int(levels[j])] * v[:, j], atol=1e-12)


# ----------------------------------------------------------------------------
# The bimodal root: modality without touching the covariance
# ----------------------------------------------------------------------------

def test_mixture_root_leaves_the_covariance_and_the_ladder_untouched():
    """The controlled-experiment property the speciation part depends on.

    A symmetric two-component root with unit variance changes the modality and
    nothing else, so every eigenvalue -- and therefore every predicted
    speciation time -- must be bit-identical to the Gaussian-root tree. If this
    ever fails, the speciation experiment stops being a controlled comparison.
    """
    g = GaussianTree(depth=3, branching=2, rho=0.85)
    m = GaussianTree(depth=3, branching=2, rho=0.85, root_separation=0.9)
    assert np.array_equal(g.leaf_covariance(), m.leaf_covariance())
    assert g.level_eigenvalues() == m.level_eigenvalues()

    a = m.sample(rng_for("test-mixture-root"), 200_000)
    assert abs(np.var(a[:, 0]) - 1.0) < 0.02          # still unit variance
    assert abs(np.mean(a)) < 0.02                     # still symmetric
    emp = np.cov(a, rowvar=False)
    assert np.max(np.abs(emp - m.leaf_covariance())) < 0.02


def test_mixture_root_density_normalizes_and_is_bimodal():
    m = GaussianTree(depth=2, branching=2, rho=0.8, root_separation=0.9)
    grid = np.linspace(-10.0, 10.0, 4001)
    dens = np.exp(m.log_root_density(grid))
    assert abs(np.trapezoid(dens, grid) - 1.0) < 1e-6
    assert dens[np.argmin(np.abs(grid - 0.9))] > 2.0 * dens[np.argmin(np.abs(grid))]

    g = GaussianTree(depth=2, branching=2, rho=0.8)
    assert np.allclose(
        np.exp(g.log_root_density(grid)),
        np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi),
    )


def test_information_form_refuses_a_mixture_root():
    """Better a loud error than a silently Gaussian answer."""
    m = GaussianTree(depth=2, branching=2, rho=0.8, root_separation=0.9)
    with pytest.raises(ValueError, match="information form"):
        tree_bp_gaussian(m, np.zeros((1, m.n_leaves)), 0.9, 0.2)


def test_root_belief_is_the_exact_posterior():
    """Checked against the closed form for a Gaussian root.

    `tree_root_belief` is the order parameter of the speciation experiment, so
    it needs its own check rather than inheriting confidence from the leaf
    posterior. For a Gaussian tree the root's posterior mean is available by
    dense linear algebra: Cov(root, leaves) = rho^depth for every leaf.
    """
    from src.hierarchy import tree_root_belief

    tree = GaussianTree(depth=3, branching=2, rho=0.8)
    grid, _w = _grid_and_weights(m=801, half_width=9.0)
    rng = rng_for("test-root-belief")
    a = tree.sample(rng, 8)
    alpha, delta = alpha_delta(0.4)
    x = alpha * a + np.sqrt(delta) * rng.standard_normal(a.shape)

    belief = tree_root_belief(
        GaussianAR1(0.8).log_transition_matrix(grid), grid,
        tree.log_root_density(grid), x, alpha, delta, 2, 3,
    )
    mean_bp = belief @ grid

    n = tree.n_leaves
    cov_rl = np.full(n, 0.8**3)                       # Cov(root, each leaf)
    gram = alpha**2 * tree.leaf_covariance() + delta * np.eye(n)
    mean_dense = x @ np.linalg.solve(gram, alpha * cov_rl)
    assert np.max(np.abs(mean_bp - mean_dense)) < 1e-5


# ----------------------------------------------------------------------------
# EM on a tree
# ----------------------------------------------------------------------------

def _grid_and_weights(m: int = 401, half_width: float = 8.0):
    grid = np.linspace(-half_width, half_width, m)
    w = np.full(m, grid[1] - grid[0])
    w[0] *= 0.5
    w[-1] *= 0.5
    return grid, w


def test_tree_xi_conserves_mass():
    from src.hierarchy import tree_e_step

    grid, w = _grid_and_weights()
    tree = GaussianTree(depth=3, branching=2, rho=0.8)
    rng = rng_for("test-tree-xi")
    a = tree.sample(rng, 12)
    alpha, delta = alpha_delta(0.4)
    x = alpha * a + np.sqrt(delta) * rng.standard_normal(a.shape)

    stats = tree_e_step(
        grid, w, GaussianAR1(0.8).log_transition_matrix(grid),
        -0.5 * grid**2 - 0.5 * np.log(2 * np.pi), x, alpha, delta, 2, 3,
    )
    assert stats.n_edges == 12 * (tree.index.n_nodes - 1)
    assert np.isclose(stats.xi.sum(), stats.n_edges, rtol=1e-8)


@pytest.mark.parametrize("t", [0.3, 1.0])
def test_tree_log_evidence_matches_the_closed_form(t):
    """The evidence a tree E-step reports is the true marginal likelihood.

    For a Gaussian tree the noisy leaves are exactly N(0, alpha^2 C + Delta I),
    so there is a closed form to check against -- and the evidence is what the
    monotone-ascent test relies on, so it has to be right, not merely stable.
    """
    from src.hierarchy import tree_e_step

    grid, w = _grid_and_weights(m=601, half_width=9.0)
    tree = GaussianTree(depth=3, branching=2, rho=0.8)
    rng = rng_for("test-tree-evidence", t)
    a = tree.sample(rng, 8)
    alpha, delta = alpha_delta(t)
    x = alpha * a + np.sqrt(delta) * rng.standard_normal(a.shape)

    stats = tree_e_step(
        grid, w, GaussianAR1(0.8).log_transition_matrix(grid),
        -0.5 * grid**2 - 0.5 * np.log(2 * np.pi), x, alpha, delta, 2, 3,
    )

    n = tree.n_leaves
    s = alpha**2 * tree.leaf_covariance() + delta * np.eye(n)
    _sign, logdet = np.linalg.slogdet(s)
    quad = np.einsum("ci,ij,cj->c", x, np.linalg.inv(s), x)
    closed = float(np.sum(-0.5 * (n * np.log(2 * np.pi) + logdet + quad)))
    assert abs(stats.log_evidence - closed) / abs(closed) < 1e-6


def test_em_on_a_tree_ascends_and_converges_to_the_truth():
    """The headline claim of Layer 5, transplanted onto a tree.

    Two budgets rather than one, because the interesting property here is the
    *rate*. On a chain every site is observed and EM is done in a few tens of
    iterations; on a tree the internal nodes are never observed, the missing
    information fraction is far larger, and EM's linear convergence is
    correspondingly slower -- at depth 4 an estimate taken at 40 iterations is
    still moving. A test that fixed one budget would read as a broken M-step
    when it is really an unconverged fit, so the assertion is that the error
    shrinks with budget and that the converged value is right.
    """
    from src.hierarchy import fit_em_tree
    from src.kernels import GaussianAR1Kernel

    grid, w = _grid_and_weights(m=201)
    true_rho = 0.75
    tree = GaussianTree(depth=3, branching=2, rho=true_rho)
    rng = rng_for("test-tree-em")
    a = tree.sample(rng, 128)
    groups = []
    for t in (0.2, 0.5):
        alpha, delta = alpha_delta(t)
        x = alpha * a + np.sqrt(delta) * rng.standard_normal(a.shape)
        groups.append((x, alpha, delta))

    start = GaussianAR1Kernel(rho=0.2, q=0.9)
    early, trace_early = fit_em_tree(start, grid, w, groups, 2, 3, n_iters=10, tol=0.0)
    late, trace_late = fit_em_tree(start, grid, w, groups, 2, 3, n_iters=150, tol=1e-11)

    assert trace_early.monotone_violation == 0.0
    assert trace_late.monotone_violation == 0.0
    assert abs(late.rho - true_rho) < abs(early.rho - true_rho)
    assert abs(late.rho - true_rho) < 0.06
    assert abs(late.q - (1 - true_rho**2)) < 0.06


# ----------------------------------------------------------------------------
# Speciation and collapse time scales
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("lam", [0.3, 1.0, 4.0, 12.33])
def test_commitment_crosses_one_over_sqrt_two_at_the_speciation_time(lam):
    t_s = spectral.speciation_time(lam)
    assert np.isclose(spectral.commitment(t_s, lam), 1.0 / np.sqrt(2.0))
    assert spectral.commitment(t_s * 0.5, lam) > 1.0 / np.sqrt(2.0)
    assert spectral.commitment(t_s * 2.0, lam) < 1.0 / np.sqrt(2.0)


def test_commitment_matches_the_forward_process_empirically():
    """`commitment` is a claim about the OU joint law; check it by sampling."""
    rho, n, t = 0.85, 64, 0.7
    cov = spectral.chain_covariance(n, rho)
    w, v = np.linalg.eigh(cov)
    top = v[:, -1]
    lam = w[-1]

    rng = rng_for("test-commitment")
    chol = np.linalg.cholesky(cov + 1e-12 * np.eye(n))
    a = rng.standard_normal((40_000, n)) @ chol.T
    alpha, delta = alpha_delta(t)
    x = alpha * a + np.sqrt(delta) * rng.standard_normal(a.shape)

    p0, pt = a @ top, x @ top
    measured = np.corrcoef(p0, pt)[0, 1]
    assert abs(measured - spectral.commitment(t, lam)) < 0.02


def test_chain_top_eigenvalue_saturates_at_the_limit():
    rho = 0.85
    limit = spectral.chain_top_eigenvalue_limit(rho)
    tops = [spectral.chain_spectrum(n, rho).max() for n in (16, 64, 256, 1024)]
    assert all(t < limit for t in tops)
    assert tops == sorted(tops)
    assert tops[-1] > 0.9 * limit           # saturating, not diverging
    assert tops[-1] / tops[0] < 2.0         # 64x the length, under 2x the top mode


def test_excess_entropy_matches_the_gaussian_determinant():
    n, rho = 32, 0.85
    cov = spectral.chain_covariance(n, rho)
    sign, logdet = np.linalg.slogdet(cov)
    assert sign > 0
    # H(noise) - H(chain) = -1/2 log det C for unit-variance reference.
    assert np.isclose(spectral.gaussian_chain_excess_entropy(n, rho), -0.5 * logdet)
    rate = spectral.gaussian_chain_excess_entropy_rate(rho)
    assert np.isclose(
        spectral.gaussian_chain_excess_entropy(n, rho) / n, rate, rtol=0.05
    )


def test_collapse_dataset_size_is_exponential_in_length():
    rho = 0.85
    sizes = [spectral.collapse_dataset_size(n, rho) for n in (8, 16, 24)]
    ratios = [sizes[1] / sizes[0], sizes[2] / sizes[1]]
    assert np.isclose(ratios[0], ratios[1], rtol=1e-6)   # constant per-site cost
    assert spectral.collapse_dataset_size(33, rho) > 1e8
