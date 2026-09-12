"""Sequence architectures that can, in principle, represent chain inference.

WHY THIS EXISTS (round-two review, sections 3 and 10.2)
-------------------------------------------------------
The first draft compared exact BP against a fully connected MLP and reported a
7--20x error ratio. Most of that turned out to be architectural: the MLP has no
locality, no weight sharing and no notion that the sequence has an order, so it
spends its capacity rediscovering the chain. Adding a shared-window head closed
most of the gap, which is the honest headline and also the reason a stronger
baseline is now required rather than optional.

A shared window of radius r is one convolution followed by pointwise layers. It
cannot propagate information further than r sites no matter how much data it
sees, so on a chain of length 32 it is structurally incapable of the thing BP
does -- and beating it says little. Both architectures here can represent
long-range propagation:

  DilatedConv1d       receptive field grows geometrically with depth, so a
                      4-block stack at dilations 1,2,4,8 already spans the chain
  BiMessagePassing    forward and backward recurrences with a shared local
                      update; this is literally the shape of forward-backward,
                      so it is the architecture with the best claim to being
                      able to learn what BP computes

Everything is pure numpy with hand-written gradients, matching src/nnet.py.
That is a real hazard: a subtly wrong backward pass yields a network that
trains badly and *looks like evidence for the paper's thesis*. Given the whole
review turns on having understated a baseline, an undetected gradient bug would
manufacture exactly the artefact under dispute. So every parameter of both
architectures is finite-difference checked in tests/test_seq_nets_gradients.py,
and that test is not optional.

Layout convention: activations are (B, C, L) -- batch, channels, sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .nnet import time_features


def _ein(subscripts: str, *operands: np.ndarray) -> np.ndarray:
    """`np.einsum` with `optimize=True`, which here is an 18x speedup, not a tweak.

    Every channel mix in this file has the shape (O,C) x (B,C,L) -> (B,O,L).
    Plain `np.einsum` runs that through its own nested-loop kernel instead of
    handing it to BLAS: measured at 3.05 ms per call against 0.16 ms with
    `optimize=True`, at B=C=O=64, L=32. Over a 64,000-step run that is the
    difference between roughly 3.5 hours and 11 minutes for one candidate --
    the difference between the architecture sweep being affordable and not.
    """
    return np.einsum(subscripts, *operands, optimize=True)


def _shift(a: np.ndarray, s: int) -> np.ndarray:
    """Shift along the last axis by `s`, zero-filling: out[..., i] = a[..., i-s]."""
    if s == 0:
        return a
    out = np.zeros_like(a)
    if s > 0:
        out[..., s:] = a[..., :-s]
    else:
        out[..., :s] = a[..., -s:]
    return out


def _site_features(X: np.ndarray, t: float) -> np.ndarray:
    """(B, L) observations at one noise level -> (B, C_in, L) input channels.

    The time embedding is broadcast across sites rather than concatenated once,
    so a convolution sees it in every receptive field. A time feature supplied
    only at the sequence ends would be invisible to a local kernel.
    """
    B, L = X.shape
    tf = time_features(float(t)).reshape(-1)          # (3,)
    chans = [X[:, None, :]]
    chans.append(np.broadcast_to(tf[None, :, None], (B, tf.size, L)))
    return np.concatenate(chans, axis=1)


N_IN_CHANNELS = 1 + 3


def _orthogonal(n: int, rng: np.random.Generator, gain: float = 1.0) -> np.ndarray:
    """Orthogonal recurrent init, so a message survives the length of the chain.

    NOT cosmetic, and not a default copied from elsewhere. With the usual
    scaled-Gaussian init the recurrence has spectral radius well below one, and
    on a 32-site chain a perturbation at site 0 arrives at site 31 attenuated to
    ~1e-13 -- the network is then structurally incapable of end-to-end
    propagation no matter how long it trains. It would have lost to EM-BP for a
    reason that has nothing to do with statistical efficiency, and the loss
    would have looked like evidence.

    An orthogonal F is norm-preserving, and tanh is near-identity for small
    activations, so the signal decays gently instead of geometrically.
    tests/test_seq_nets_gradients.py pins this end to end.
    """
    a = rng.standard_normal((n, n))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))          # make the decomposition unique
    return gain * q


# ---------------------------------------------------------------------------
# Dilated residual convolution
# ---------------------------------------------------------------------------
@dataclass
class DilatedConv1d:
    """Residual stack of 3-tap dilated convolutions.

    Receptive radius is sum(dilations) over the stack, so (1,2,4,8) reaches 15
    sites either way -- the full chain at L=32. Contrast the shared-window head,
    whose radius is fixed by construction.
    """

    hidden: int
    dilations: tuple[int, ...]
    params: list[np.ndarray] = field(default_factory=list)

    @classmethod
    def init(cls, hidden: int, dilations, rng: np.random.Generator) -> "DilatedConv1d":
        dil = tuple(int(d) for d in dilations)
        p: list[np.ndarray] = []
        # stem: 1x1 from input channels to hidden
        p += [rng.standard_normal((hidden, N_IN_CHANNELS)) * np.sqrt(2.0 / N_IN_CHANNELS),
              np.zeros(hidden)]
        for _ in dil:
            # 3-tap conv (hidden -> hidden), then 1x1 projection back
            p += [rng.standard_normal((3, hidden, hidden)) * np.sqrt(2.0 / (3 * hidden)),
                  np.zeros(hidden),
                  rng.standard_normal((hidden, hidden)) * np.sqrt(2.0 / hidden),
                  np.zeros(hidden)]
        # head: 1x1 to a single output channel
        p += [rng.standard_normal((1, hidden)) * np.sqrt(1.0 / hidden), np.zeros(1)]
        return cls(hidden=hidden, dilations=dil, params=p)

    @property
    def n_params(self) -> int:
        return int(sum(p.size for p in self.params))

    def forward(self, feat: np.ndarray):
        """feat: (B, C_in, L) -> out (B, L), plus cache."""
        p = self.params
        cache = {"feat": feat, "blocks": []}
        h = _ein("hc,bcl->bhl", p[0], feat) + p[1][None, :, None]
        h = np.tanh(h)
        cache["stem"] = h
        i = 2
        for d in self.dilations:
            W, b, P, pb = p[i], p[i + 1], p[i + 2], p[i + 3]
            taps = [_shift(h, d), h, _shift(h, -d)]      # sites i-d, i, i+d
            z = sum(_ein("oc,bcl->bol", W[k], taps[k]) for k in range(3))
            z = z + b[None, :, None]
            a = np.tanh(z)
            proj = _ein("oc,bcl->bol", P, a) + pb[None, :, None]
            cache["blocks"].append({"h_in": h, "taps": taps, "a": a})
            h = h + proj                                  # residual
            i += 4
        out = _ein("oc,bcl->bol", p[i], h) + p[i + 1][None, :, None]
        cache["h_final"] = h
        return out[:, 0, :], cache

    def backward(self, cache, grad_out: np.ndarray) -> list[np.ndarray]:
        """grad_out: (B, L) gradient of the loss wrt the output."""
        p = self.params
        grads: list[np.ndarray] = [np.zeros_like(q) for q in p]
        g = grad_out[:, None, :]                          # (B,1,L)

        i = 2 + 4 * len(self.dilations)
        grads[i] = _ein("bol,bcl->oc", g, cache["h_final"])
        grads[i + 1] = g.sum(axis=(0, 2))
        gh = _ein("oc,bol->bcl", p[i], g)

        for d, blk in zip(reversed(self.dilations), reversed(cache["blocks"])):
            i -= 4
            W, P = p[i], p[i + 2]
            # residual: gradient flows both to the projection and straight past
            gproj = gh
            grads[i + 2] = _ein("bol,bcl->oc", gproj, blk["a"])
            grads[i + 3] = gproj.sum(axis=(0, 2))
            ga = _ein("oc,bol->bcl", P, gproj)
            gz = ga * (1.0 - blk["a"] ** 2)
            grads[i + 1] = gz.sum(axis=(0, 2))
            gtaps = np.zeros_like(blk["h_in"])
            for k, s in enumerate((d, 0, -d)):
                grads[i][k] = _ein("bol,bcl->oc", gz, blk["taps"][k])
                # y = _shift(x, s)  =>  dL/dx = _shift(g, -s)
                gtaps += _shift(_ein("oc,bol->bcl", W[k], gz), -s)
            gh = gh + gtaps

        gstem = gh * (1.0 - cache["stem"] ** 2)
        grads[0] = _ein("bhl,bcl->hc", gstem, cache["feat"])
        grads[1] = gstem.sum(axis=(0, 2))
        return grads


# ---------------------------------------------------------------------------
# Bidirectional message passing
# ---------------------------------------------------------------------------
@dataclass
class BiMessagePassing:
    """Forward and backward recurrences with shared local updates, then a readout.

    h_f[i] = tanh(A x[i] + F h_f[i-1] + a)
    h_b[i] = tanh(B x[i] + G h_b[i+1] + b)
    y[i]   = C [h_f[i], h_b[i], x[i]] + c

    The same F is applied at every site, so the model has one local rule and a
    topology -- the two structural facts BP is given. It is the strongest
    baseline in this family: if a network can learn what sum-product does on a
    chain, this is the one that should.
    """

    hidden: int
    params: list[np.ndarray] = field(default_factory=list)

    @classmethod
    def init(cls, hidden: int, rng: np.random.Generator) -> "BiMessagePassing":
        h, c = hidden, N_IN_CHANNELS
        s = lambda a, b: rng.standard_normal((a, b)) * np.sqrt(1.0 / b)  # noqa: E731
        p = [
            s(h, c), _orthogonal(h, rng), np.zeros(h),   # A, F, a  (forward)
            s(h, c), _orthogonal(h, rng), np.zeros(h),   # B, G, b  (backward)
            s(1, 2 * h + c), np.zeros(1),                # C, c     (readout)
        ]
        return cls(hidden=hidden, params=p)

    @property
    def n_params(self) -> int:
        return int(sum(p.size for p in self.params))

    def forward(self, feat: np.ndarray):
        A, F, a, Bm, G, b, C, c = self.params
        B, _, L = feat.shape
        hf = np.zeros((B, self.hidden, L))
        hb = np.zeros((B, self.hidden, L))
        xa = _ein("hc,bcl->bhl", A, feat)
        xb = _ein("hc,bcl->bhl", Bm, feat)

        prev = np.zeros((B, self.hidden))
        for i in range(L):
            hf[:, :, i] = np.tanh(xa[:, :, i] + prev @ F.T + a)
            prev = hf[:, :, i]
        nxt = np.zeros((B, self.hidden))
        for i in reversed(range(L)):
            hb[:, :, i] = np.tanh(xb[:, :, i] + nxt @ G.T + b)
            nxt = hb[:, :, i]

        cat = np.concatenate([hf, hb, feat], axis=1)
        out = _ein("oc,bcl->bol", C, cat) + c[None, :, None]
        return out[:, 0, :], {"feat": feat, "hf": hf, "hb": hb, "cat": cat}

    def backward(self, cache, grad_out: np.ndarray) -> list[np.ndarray]:
        A, F, a, Bm, G, b, C, c = self.params
        feat, hf, hb = cache["feat"], cache["hf"], cache["hb"]
        Bn, _, L = feat.shape
        g = grad_out[:, None, :]

        gC = _ein("bol,bcl->oc", g, cache["cat"])
        gc = g.sum(axis=(0, 2))
        gcat = _ein("oc,bol->bcl", C, g)
        H = self.hidden
        ghf, ghb, gfeat = gcat[:, :H], gcat[:, H:2 * H], gcat[:, 2 * H:].copy()

        gA = np.zeros_like(A); gF = np.zeros_like(F); ga = np.zeros_like(a)
        gB = np.zeros_like(Bm); gG = np.zeros_like(G); gb = np.zeros_like(b)

        # forward recurrence: walk backwards in time
        carry = np.zeros((Bn, H))
        for i in reversed(range(L)):
            gz = (ghf[:, :, i] + carry) * (1.0 - hf[:, :, i] ** 2)
            gA += gz.T @ feat[:, :, i]
            ga += gz.sum(axis=0)
            if i > 0:
                gF += gz.T @ hf[:, :, i - 1]
            gfeat[:, :, i] += gz @ A
            carry = gz @ F

        # backward recurrence: walk forwards in time
        carry = np.zeros((Bn, H))
        for i in range(L):
            gz = (ghb[:, :, i] + carry) * (1.0 - hb[:, :, i] ** 2)
            gB += gz.T @ feat[:, :, i]
            gb += gz.sum(axis=0)
            if i < L - 1:
                gG += gz.T @ hb[:, :, i + 1]
            gfeat[:, :, i] += gz @ Bm
            carry = gz @ G

        return [gA, gF, ga, gB, gG, gb, gC, gc]


# ---------------------------------------------------------------------------
# Shared training loop with checkpointing
# ---------------------------------------------------------------------------
def train_sequence_net(
    net,
    A: np.ndarray,
    t_values,
    rng: np.random.Generator,
    *,
    checkpoints,
    parameterization: str = "eps",
    batch_size: int = 64,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    sample_fn=None,
    exact_target_fn=None,
):
    """Denoising score matching by Adam, snapshotting at each checkpoint.

    Returning a LADDER of snapshots rather than one final net is the point.
    Training for a fixed 8,000 steps regardless of dataset size means the
    expected number of presentations of each chain falls by 32x between n=32
    and n=2048, so a curve that widens with n confounds optimisation budget with
    statistical efficiency -- the confound that sank the previous result. With
    checkpoints the caller can select on validation and demonstrate saturation.

    `A` is the CLEAN training set; noise is redrawn every step, which is what
    DSM means. Two optional hooks exist for the diagnostic that separates the
    possible causes of a widening gap:

      sample_fn(rng, k)      draw k fresh clean chains instead of resampling A,
                             which removes finite-data estimation error
      exact_target_fn(X, t)  supply the exact posterior mean instead of the
                             single-sample DSM target, which removes target
                             variance

    With both off this is ordinary DSM on a fixed finite set.
    """
    ckpts = sorted(int(c) for c in checkpoints)
    n_data = A.shape[0]
    m = [np.zeros_like(p) for p in net.params]
    v = [np.zeros_like(p) for p in net.params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    snaps: dict[int, list[np.ndarray]] = {}

    for step in range(1, ckpts[-1] + 1):
        k = min(batch_size, n_data)
        if sample_fn is not None:
            clean = sample_fn(rng, k)
        else:
            clean = A[rng.integers(0, n_data, size=k)]
        t = float(t_values[rng.integers(0, len(t_values))])
        alpha = np.exp(-t)
        delta = 1.0 - np.exp(-2.0 * t)
        noise = rng.standard_normal(clean.shape)
        xb = alpha * clean + np.sqrt(delta) * noise

        if exact_target_fn is not None:
            mean = exact_target_fn(xb, t)
            yb = mean if parameterization == "x0" else (xb - alpha * mean) / np.sqrt(delta)
        else:
            yb = clean if parameterization == "x0" else noise

        feat = _site_features(xb, t)
        out, cache = net.forward(feat)
        gout = 2.0 * (out - yb) / xb.shape[0]
        grads = net.backward(cache, gout)

        total = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
        scale = min(1.0, grad_clip / (total + 1e-12))
        for j, g in enumerate(grads):
            g = g * scale
            m[j] = b1 * m[j] + (1 - b1) * g
            v[j] = b2 * v[j] + (1 - b2) * g ** 2
            mh = m[j] / (1 - b1 ** step)
            vh = v[j] / (1 - b2 ** step)
            net.params[j] -= lr * mh / (np.sqrt(vh) + eps)

        if step in ckpts:
            snaps[step] = [p.copy() for p in net.params]
    return snaps


def predict(net, params, X: np.ndarray, t: float) -> np.ndarray:
    """Run `net` with a checkpoint's parameters, without disturbing its state."""
    saved = net.params
    net.params = params
    try:
        out, _ = net.forward(_site_features(X, t))
    finally:
        net.params = saved
    return out


def posterior_mean(net, params, X: np.ndarray, t: float, parameterization: str):
    """Checkpoint prediction converted to a posterior mean E[a | x_t]."""
    out = predict(net, params, X, t)
    if parameterization == "x0":
        return out
    alpha = np.exp(-t)
    delta = 1.0 - np.exp(-2.0 * t)
    return (X - np.sqrt(delta) * out) / alpha
