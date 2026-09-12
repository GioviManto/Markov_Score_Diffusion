"""BP-informed diffusion scores for (approximately) Markovian sequence data.

Modules
-------
noising       : OU forward process, likelihood construction (log-domain safe).
priors        : clean-chain priors (Gaussian/Laplace/Student-t/mixture AR(1),
                Gaussian AR(1) + global latent, Gaussian long-range).
bp_grid       : reference grid BP on a chain (numerically exact up to grid error),
                including the model evidence log p_t(x).
bp_gaussian   : analytic information-form Gaussian chain BP (exact for Gaussian AR(1);
                equals moment-matched ADF for any linear-transition chain), plus the
                legacy grid-projected variant kept for comparison.
exact_scores  : closed-form Gaussian scores/means for arbitrary prior covariance.
markov_approx : best chain (Chow-Liu style) Gaussian Markov approximations and
                residual-operator analysis for approximately Markov models.
nnet          : minimal pure-numpy MLP + Adam for score/residual regression.
reverse       : reverse-time SDE and probability-flow ODE samplers for the OU process.
metrics       : shared error metrics.
utils         : seeding, CSV/JSON writers.

Images (the wavelet-tree extension)
-----------------------------------
wavelet       : orthonormal 2-D Haar transform and the quadtree it induces. The
                transform is orthonormal so that coordinatewise OU noising in
                pixel space *is* coordinatewise OU noising in wavelet space, with
                the same alpha and Delta -- which is what carries the exactness
                of the chain results onto images.
wavelet_bp    : exact BP on that quadtree, with an observation at every node, a
                per-scale noise level, and one kernel per scale.
wavelet_model : assembly -- per-subband standardisation, per-level EM, exact
                held-out likelihood in pixel coordinates, denoising.
image_data    : CIFAR-10 luminance loading and the normalisation the OU
                convention requires.
"""
