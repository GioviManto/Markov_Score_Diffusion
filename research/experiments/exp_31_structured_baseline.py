"""Experiment 31 -- the structured-baseline comparison, done properly.

WHY THIS REPLACES exp_12's EFFICIENCY PARTS
-------------------------------------------
exp_12 produced the revised 1.8--5.5x headline. The round-two review found four
defects in it, and each on its own is enough to void the number:

1. CENTRE SITE ONLY. `interior_slice(33, 16)` returns `slice(16, 17)`. Every
   arm was scored on one coordinate of a 33-site chain, and the table was read
   as a whole-sequence denoising comparison.

2. FIXED BUDGET. Both network arms trained for exactly 8,000 steps at batch 64
   regardless of dataset size, so expected presentations per chain fall 8000 ->
   250 as n goes 32 -> 2048. A gap that widens with n therefore confounds
   optimisation budget with statistical efficiency, and "the convolution fails
   to acquire the structure as data grows" is a mechanism read into that
   confound. Its error does fall, 0.1157 -> 0.0520.

3. PROTOCOL DRIFT. 33 sites not 32, five noise levels not twelve, C=4 not C=8 --
   so its ratio could not legitimately be compared with the headline ratio as
   though only the architecture differed.

4. UNREPRODUCIBLE SOURCE. The outputs name a commit whose experiment file does
   not define the override the recorded command passes.

WHAT THIS DOES INSTEAD
----------------------
Headline protocol throughout (32 sites, the twelve-level schedule, C=8). Every
arm inside a seed sees the SAME training chains, the SAME validation bundle and
the SAME test bundle. Both the neural and the EM arms are checkpointed and
selected on validation, and saturation is checked rather than assumed. Errors
are reported over all sites, over a predeclared fixed bulk, and at the centre
site as a diagnostic only -- with the same bulk slice for every radius, so no
candidate is handed an easier set of interior sites than another.

The baselines can actually represent chain inference. A shared window of radius
r cannot propagate past r sites however much data it sees, so beating it says
little; `src/seq_nets.py` adds a dilated residual convolution whose receptive
field spans the chain and a bidirectional message-passing network whose shape is
forward-backward's.

Architecture and optimiser are chosen on four DEVELOPMENT seeds and then frozen.
The sixteen confirmatory seeds do not influence selection, so the reported
interval is not a selection artefact.

Rows carry the summed squared error and the summed squared reference norm per
region, NOT a pre-divided ratio, so the aggregation question the review raised
is settled downstream instead of being baked in here.
"""

from __future__ import annotations

import json

import numpy as np

from common import (
    apply_overrides,
    experiment_parser,
    provenance,
    resolved_config_hash,
    select_parts,
)
from frozen_config import FROZEN
from src.bp_grid import grid_bp_batch, make_grid
from src.em import fit_em
from src.kernels import MixtureInnovationKernel
from src.local_head import local_posterior_mean, train_local_head
from src.noising import alpha_delta
from src.priors import LaplaceAR1
from src.seq_nets import (
    BiMessagePassing,
    DilatedConv1d,
    posterior_mean,
    train_sequence_net,
)
from src.utils import ensure_dir, rng_for, write_csv, write_json

# Headline protocol. These are FROZEN's values, restated so a drift is visible
# in a diff rather than hidden behind an import.
N_SITES = FROZEN.n_sites            # 32
RHO = FROZEN.rho                    # 0.85
GRID_M = FROZEN.n_grid              # 401
GRID_A = FROZEN.half_width          # 8.0
T_SCHEDULE = FROZEN.t_grid          # twelve log-spaced levels

# The bulk is predeclared and identical for every arm. Four sites at each end
# are dropped because a window head's padding contaminates them; choosing it
# per-candidate from its own radius would give wider heads an easier target.
BULK = slice(4, N_SITES - 4)
REGIONS = {
    "all": slice(0, N_SITES),
    "bulk": BULK,
    "centre": slice(N_SITES // 2, N_SITES // 2 + 1),
}

SETTINGS = dict(
    rho=RHO,
    grid_size=GRID_M,
    sizes=(32, 128, 512, 2048),
    n_val=1024,
    n_test=2048,
    # Confirmatory seeds; dev seeds are disjoint by construction (below).
    seeds=16,
    seed0=0,
    dev_seeds=4,
    # --- EM arm ---
    em_components=8,
    em_inner=16,
    em_init_rho=0.3,
    em_checkpoints=(10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200),
    # --- neural arms ---
    net_checkpoints=(500, 1000, 2000, 4000, 8000, 16000, 32000, 64000),
    batch_size=64,
    lrs=(3e-4, 1e-3, 3e-3),
    parameterizations=("eps", "x0"),
    window_radii=(2, 4, 8, 16),
    window_widths=(64, 128),
    conv_channels=(64, 128),
    conv_dilations=(1, 2, 4, 8),
    bimp_hidden=(64, 128),
    # Architectures to screen. The confirmatory stage runs the winner plus the
    # shared window, which is kept regardless as the review's reference point.
    screen_arch=("window", "conv", "bimp"),
    # Filled in from part_screen's output before the confirmatory run, as
    # {arch: hyperparameters}. Declared here (rather than accepted as a new
    # key at the command line) because `apply_overrides` refuses unknown keys
    # -- the check that revealed the exp_12 provenance defect.
    winners=None,
    grad_clip=1.0,
    # Saturation: a selected checkpoint that is the last one on the ladder is
    # flagged, because then validation error was still falling when we stopped.
    saturation_rel=0.01,
)

QUICK = dict(
    sizes=(32, 128),
    n_val=128,
    n_test=128,
    seeds=2,
    dev_seeds=1,
    em_checkpoints=(5, 10, 20),
    net_checkpoints=(100, 200, 400),
    lrs=(1e-3,),
    window_radii=(4,),
    window_widths=(64,),
    conv_channels=(32,),
    bimp_hidden=(32,),
    grid_size=201,
)


# ---------------------------------------------------------------------------
# Data: one bundle per seed, shared by every arm
# ---------------------------------------------------------------------------
def _bundle(prior, grid, weights, tag, seed, n_chains, t_values):
    """Noised chains and their exact-BP posterior means under the TRUE kernel.

    Every random operation draws from its own named stream via `rng_for`, so the
    training chains, the noise, the network initialisation and the minibatch
    order are independent. That is not tidiness: with one seed driving several
    logically distinct draws, reordering code changes the dataset AND the
    initialisation AND the training noise together, and a difference between two
    runs cannot be attributed.
    """
    rng = rng_for(tag, seed)
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])
    log_k = prior.log_transition_matrix(grid)
    out = {}
    for t in t_values:
        alpha, delta = alpha_delta(t)
        X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
        m_ref, _ = grid_bp_batch(grid, weights, log_k, X, alpha, delta)
        out[t] = (X, m_ref)
    return A, out


def _bundle_hash(bundle) -> str:
    """A digest of a validation/test bundle, so "same bundle" is checkable.

    The design says every arm inside a seed is scored on identical data. Saying
    so in a caption is not the same as it being true, and a bundle rebuilt from
    a drifting seed would produce a comparison that looks paired and is not.
    Hashing the actual arrays turns the claim into something the output file
    can be audited against.
    """
    import hashlib

    h = hashlib.sha256()
    for t in sorted(bundle):
        X, m = bundle[t]
        h.update(np.ascontiguousarray(X, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(m, dtype=np.float64).tobytes())
    return h.hexdigest()[:12]


def _sq(m_hat, m_ref, X, t, sl):
    """Summed squared score error and summed squared reference norm on `sl`.

    Returned unreduced. A ratio computed here would fix the estimand at
    per-cell level, and the review's section 5 is precisely that the choice of
    where to divide changes the headline by a factor of 1.4.
    """
    _, delta = alpha_delta(t)
    alpha = np.exp(-t)
    s_hat = -(X[:, sl] - alpha * m_hat[:, sl]) / delta
    s_ref = -(X[:, sl] - alpha * m_ref[:, sl]) / delta
    return float(((s_hat - s_ref) ** 2).sum()), float((s_ref ** 2).sum())


def _schedule_risk(pairs) -> float:
    """sqrt(sum_t err_t / sum_t ref_t): one risk per arm per seed.

    Aggregating numerator and denominator BEFORE the square root is the
    review's recommended estimand. A mean of per-level ratios lets a level with
    a small reference norm dominate, which is how the original headline came to
    be the largest of the five summaries that could reasonably have been quoted.
    """
    num = sum(e for e, _ in pairs)
    den = sum(r for _, r in pairs)
    return float(np.sqrt(num / den)) if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
def _em_arm(cfg, grid, weights, A, val, test, seed):
    """EM-BP with a checkpoint ladder, selected on validation."""
    rng = rng_for("exp31-em-init", seed)
    ck = tuple(int(c) for c in cfg["em_checkpoints"])
    groups = []
    parts_idx = np.array_split(
        rng_for("exp31-em-split", seed).permutation(len(A)), len(T_SCHEDULE)
    )
    noise = rng_for("exp31-em-noise", seed)
    for t, idx in zip(T_SCHEDULE, parts_idx):
        al, de = alpha_delta(t)
        sub = A[idx]
        groups.append((al * sub + np.sqrt(de) * noise.standard_normal(sub.shape), al, de))

    _, trace, saved = fit_em(
        MixtureInnovationKernel.init(
            cfg["em_components"], rho=cfg["em_init_rho"], var=0.8, rng=rng
        ),
        grid, weights, groups,
        n_iters=max(ck), tol=FROZEN.em_loglik_tol, checkpoints=set(ck),
    )

    def _eval(kernel, bundle, region):
        log_k = kernel.log_transition_matrix(grid)
        pairs = []
        for t in T_SCHEDULE:
            X, m_ref = bundle[t]
            al, de = alpha_delta(t)
            m, _ = grid_bp_batch(grid, weights, log_k, X, al, de)
            pairs.append(_sq(m, m_ref, X, t, REGIONS[region]))
        return pairs

    best, best_risk = None, np.inf
    for it in sorted(saved):
        r = _schedule_risk(_eval(saved[it], val, "bulk"))
        if r < best_risk:
            best, best_risk = it, r
    return {
        "checkpoint": best,
        "at_cap": best == max(saved),
        "eval": lambda region: _eval(saved[best], test, region),
        "n_params": int(len(saved[best].theta)),
        "stop_reason": trace.stop_reason,
    }


def _net_arm(cfg, arch, hp, A, val, test, seed):
    """One neural candidate: train, checkpoint, select on validation."""
    rng = rng_for("exp31-net", arch, seed, *sorted(map(str, hp.items())))
    ck = tuple(int(c) for c in cfg["net_checkpoints"])
    mode = hp["parameterization"]

    if arch == "window":
        # The shared-window head keeps its own trainer, which already does DSM
        # with a fixed radius. It is checkpointed by retraining to each rung:
        # its trainer has no snapshot hook and rewriting it would change the
        # baseline the review asked us to keep as a reference point.
        snaps, net = {}, None
        for c in ck:
            h = train_local_head(
                A, T_SCHEDULE, hp["radius"],
                rng_for("exp31-win", seed, hp["radius"], mode, c),
                hidden=(hp["width"], hp["width"]), n_steps=c,
                parameterization=mode,
            )
            snaps[c] = h
        predict_at = lambda s, X, t: local_posterior_mean(snaps[s], X, t)  # noqa: E731
        n_params = snaps[ck[0]].n_params
    else:
        if arch == "conv":
            net = DilatedConv1d.init(hp["channels"], cfg["conv_dilations"], rng)
        elif arch == "bimp":
            net = BiMessagePassing.init(hp["hidden"], rng)
        else:
            raise SystemExit(f"unknown architecture {arch!r}")
        snaps = train_sequence_net(
            net, A, T_SCHEDULE, rng, checkpoints=ck, parameterization=mode,
            batch_size=cfg["batch_size"], lr=hp["lr"], grad_clip=cfg["grad_clip"],
        )
        predict_at = lambda s, X, t: posterior_mean(net, snaps[s], X, t, mode)  # noqa: E731
        n_params = net.n_params

    def _eval(step, bundle, region):
        pairs = []
        for t in T_SCHEDULE:
            X, m_ref = bundle[t]
            pairs.append(_sq(predict_at(step, X, t), m_ref, X, t, REGIONS[region]))
        return pairs

    val_curve = {s: _schedule_risk(_eval(s, val, "bulk")) for s in sorted(snaps)}
    best = min(val_curve, key=val_curve.get)
    return {
        "checkpoint": best,
        "at_cap": best == max(snaps),
        "val_risk": val_curve[best],
        "val_curve": val_curve,
        "eval": lambda region: _eval(best, test, region),
        "n_params": n_params,
    }


def _candidates(cfg, arch):
    """The hyperparameter grid screened for one architecture."""
    out = []
    for mode in cfg["parameterizations"]:
        if arch == "window":
            for r in cfg["window_radii"]:
                for w in cfg["window_widths"]:
                    out.append({"parameterization": mode, "radius": r, "width": w,
                                "lr": cfg["lrs"][len(cfg["lrs"]) // 2]})
        elif arch == "conv":
            for c in cfg["conv_channels"]:
                for lr in cfg["lrs"]:
                    out.append({"parameterization": mode, "channels": c, "lr": lr})
        elif arch == "bimp":
            for h in cfg["bimp_hidden"]:
                for lr in cfg["lrs"]:
                    out.append({"parameterization": mode, "hidden": h, "lr": lr})
    return out


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------
def part_screen(cfg, out):
    """Choose architecture and optimiser on development seeds only."""
    grid, weights = make_grid(GRID_A, cfg["grid_size"])
    prior = LaplaceAR1(cfg["rho"])
    # Development seeds live above the confirmatory block so the two cannot
    # overlap however the confirmatory range is later sharded.
    dev = range(10_000, 10_000 + cfg["dev_seeds"])

    # Resume, per candidate. Screening is the longer of the two stages and
    # writes nothing until it ends, so a wall-clock timeout at hour 70 would
    # discard all of it AND leave the dependent confirmatory job permanently
    # unsatisfiable, since `afterok` never fires on TIMEOUT. Flushing per
    # candidate makes a resubmission continue. The unit is the candidate rather
    # than the cell because a single candidate is at most ~36 minutes.
    dest = out / "screening.csv"
    rows, done = [], set()
    if dest.exists():
        import csv as _csv
        rows = list(_csv.DictReader(dest.open()))
        done = {(int(r["seed"]), int(r["n_chains"]), r["arch"], r["hp"]) for r in rows}
        print(f"resuming: {len(done)} candidate(s) already on disk", flush=True)

    for seed in dev:
        todo = [
            (n, arch, hp)
            for n in cfg["sizes"] for arch in cfg["screen_arch"]
            for hp in _candidates(cfg, arch)
            if (seed, n, arch, json.dumps(hp, sort_keys=True)) not in done
        ]
        if not todo:
            continue
        _, val = _bundle(prior, grid, weights, "exp31-val", seed, cfg["n_val"], T_SCHEDULE)
        _, test = _bundle(prior, grid, weights, "exp31-test", seed, cfg["n_test"], T_SCHEDULE)
        for n_chains in sorted({n for n, _, _ in todo}):
            A, _ = _bundle(prior, grid, weights, "exp31-train",
                           (seed, n_chains), n_chains, ())
            for n, arch, hp in [x for x in todo if x[0] == n_chains]:
                    res = _net_arm(cfg, arch, hp, A, val, test, seed)
                    for region in REGIONS:
                        err, ref = map(sum, zip(*res["eval"](region)))
                        rows.append({
                            "stage": "screen", "seed": seed, "n_chains": n_chains,
                            "arch": arch, "region": region,
                            # One column, because the grids differ per
                            # architecture and a per-key column would make the
                            # CSV's shape depend on which arms were screened.
                            "hp": json.dumps(hp, sort_keys=True),
                            "checkpoint": res["checkpoint"], "at_cap": res["at_cap"],
                            "n_params": res["n_params"],
                            "sq_err": err, "sq_ref": ref,
                            "risk": float(np.sqrt(err / ref)),
                        })
                    print(f"  seed={seed} n={n_chains} {arch} {hp} "
                          f"ck={res['checkpoint']}{' CAP' if res['at_cap'] else ''} "
                          f"val={res['val_risk']:.4f}", flush=True)
                    write_csv(dest, rows)
    write_csv(dest, rows)
    return rows


def _winners_from_screening(path) -> dict:
    """Pick the architecture and hyperparameters, from development seeds only.

    Selection is on the BULK region, pooled across development seeds and sizes
    by the schedule-level risk -- numerator and denominator summed before the
    square root, the same estimand the confirmatory stage reports. Two
    architectures are carried forward: the winner, and the shared-window head
    regardless of whether it wins, because it is the review's reference point
    and dropping it when it loses would be selective.

    A tie is broken toward the cheaper model. This runs inside the confirmatory
    job so the choice is reproducible from the screening CSV rather than
    transcribed by hand into a job script.
    """
    import csv
    from collections import defaultdict

    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"screening file {path} is empty")
    agg = defaultdict(lambda: [0.0, 0.0, 0])
    for r in rows:
        if r["region"] != "bulk":
            continue
        k = (r["arch"], r["hp"])
        a = agg[k]
        a[0] += float(r["sq_err"])
        a[1] += float(r["sq_ref"])
        a[2] += int(r["n_params"])
    if not agg:
        raise SystemExit(f"screening file {path} has no bulk-region rows")

    scored = sorted(
        ((float(np.sqrt(e / d)), params, arch, hp)
         for (arch, hp), (e, d, params) in agg.items()),
        key=lambda x: (x[0], x[1]),
    )
    for risk, _, arch, hp in scored[:6]:
        print(f"  screen {arch:8} {hp}  bulk risk {risk:.5f}", flush=True)

    best_arch, best_hp = scored[0][2], scored[0][3]
    winners = {best_arch: json.loads(best_hp)}
    if "window" not in winners:
        win = min((s for s in scored if s[2] == "window"), default=None)
        if win is not None:
            winners["window"] = json.loads(win[3])
    return winners


def part_confirm(cfg, out):
    """Confirmatory run on the sixteen seeds, with selection already frozen."""
    grid, weights = make_grid(GRID_A, cfg["grid_size"])
    prior = LaplaceAR1(cfg["rho"])
    winners = cfg.get("winners")
    if isinstance(winners, str) and winners.startswith("auto:"):
        winners = _winners_from_screening(winners[5:])
        print(f"selected from screening: {json.dumps(winners, sort_keys=True)}",
              flush=True)
    if not winners:
        raise SystemExit(
            "part_confirm needs --set winners=... , the architecture and "
            "hyperparameters chosen by part_screen. Selecting them here would "
            "let the confirmatory seeds influence the choice, which is the "
            "thing the two-stage design exists to prevent."
        )
    # Resume support. A three-day job that dies at hour 70 with nothing on disk
    # is a three-day job wasted, and the wall clock here is a real constraint
    # rather than a formality. Completed (seed, size) cells are appended as they
    # finish and skipped on restart, so resubmitting continues instead of
    # starting over. The unit is the cell because that is what shares a bundle.
    dest = out / "confirm.csv"
    done: set = set()
    rows = []
    if dest.exists():
        import csv as _csv
        rows = list(_csv.DictReader(dest.open()))
        done = {(int(r["seed"]), int(r["n_chains"])) for r in rows}
        print(f"resuming: {len(done)} cell(s) already on disk", flush=True)

    for seed in range(cfg["seed0"], cfg["seed0"] + cfg["seeds"]):
        pending = [n for n in cfg["sizes"] if (seed, n) not in done]
        if not pending:
            continue
        _, val = _bundle(prior, grid, weights, "exp31-val", seed, cfg["n_val"], T_SCHEDULE)
        _, test = _bundle(prior, grid, weights, "exp31-test", seed, cfg["n_test"], T_SCHEDULE)
        for n_chains in pending:
            A, _ = _bundle(prior, grid, weights, "exp31-train",
                           (seed, n_chains), n_chains, ())
            arms = {"em_bp": _em_arm(cfg, grid, weights, A, val, test, seed)}
            for arch, hp in winners.items():
                arms[arch] = _net_arm(cfg, arch, dict(hp), A, val, test, seed)
            vh, th = _bundle_hash(val), _bundle_hash(test)
            for name, res in arms.items():
                for region in REGIONS:
                    err, ref = map(sum, zip(*res["eval"](region)))
                    rows.append({
                        "stage": "confirm", "seed": seed, "n_chains": n_chains,
                        "method": name, "region": region,
                        "checkpoint": res["checkpoint"], "at_cap": res["at_cap"],
                        "n_params": res["n_params"],
                        "sq_err": err, "sq_ref": ref,
                        "risk": float(np.sqrt(err / ref)),
                        # Identical within a seed by construction; recorded so a
                        # merge step can verify the pairing rather than trust it.
                        "val_bundle": vh, "test_bundle": th,
                    })
            print(f"seed={seed} n={n_chains} " + " ".join(
                f"{k}={np.sqrt(sum(e for e, _ in v['eval']('bulk')) / sum(r for _, r in v['eval']('bulk'))):.4f}"
                for k, v in arms.items()), flush=True)
            # Flush after every cell, not at the end: see the resume note above.
            write_csv(dest, rows)
    write_csv(dest, rows)
    return rows


def main() -> None:
    parser = experiment_parser(
        "exp_31_structured_baseline",
        "Structured-baseline comparison at the headline protocol.",
    )
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)

    parts = {
        "screen": ("select architecture on development seeds", part_screen),
        "confirm": ("confirmatory run on the frozen selection", part_confirm),
    }
    if args.list_parts:
        print("\n".join(parts))
        return

    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    tag = "_".join(selected) if args.only else "all"
    resolved = {
        "n_sites": N_SITES, "t_schedule": list(T_SCHEDULE),
        "grid_half_width": GRID_A, "bulk": [BULK.start, BULK.stop],
        "quick": args.quick, "parts": list(selected), "overrides": args.set,
        **cfg,
    }
    write_json(out / f"params_{tag}.json", {
        **resolved,
        "resolved_config_hash": resolved_config_hash(resolved),
        **provenance(resolved),
    })
    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(cfg, out)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
