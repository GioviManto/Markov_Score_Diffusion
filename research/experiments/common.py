"""Shared experiment scaffolding: paths, argument parsing, provenance."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

OUTPUT_ROOT = PACKAGE_ROOT / "outputs"


def experiment_parser(name: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=name, description=description)
    parser.add_argument(
        "--quick", action="store_true",
        help="Reduced settings for a fast smoke run.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_ROOT / name,
        help="Directory for CSV/JSON/PNG outputs.",
    )
    parser.add_argument(
        "--only", default=None,
        help=(
            "Comma-separated part names to run (default: all). Parts are "
            "independent and write disjoint CSVs, so a scheduler can run them "
            "as parallel array tasks into one --output-dir with no merge step. "
            "Use --list-parts to see the names."
        ),
    )
    parser.add_argument(
        "--list-parts", action="store_true",
        help="Print the part names this experiment defines, then exit.",
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help=(
            "Override a setting, repeatable, e.g. --set n_rep=64 "
            "--set sizes='(64,256,1024,4096)'. VALUE is parsed as a Python "
            "literal. Overrides land in params.json, so a scaled-up run stays "
            "self-describing."
        ),
    )
    return parser


def apply_overrides(settings: dict, assignments) -> dict:
    """Apply `--set KEY=VALUE` assignments to a settings dict.

    Unknown keys are rejected rather than silently added: a typo in a job
    script would otherwise run the default configuration on the cluster and
    look like a completed experiment.
    """
    import ast

    updated = dict(settings)
    for item in assignments:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in updated:
            raise SystemExit(
                f"--set: unknown key {key!r}. Known keys: {sorted(updated)}"
            )
        try:
            updated[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError) as exc:
            raise SystemExit(f"--set {key}: cannot parse {raw!r} ({exc})") from exc
    return updated


def select_parts(parts: dict, only: str | None) -> dict:
    """Filter an ordered {name: callable} mapping by a --only specification."""
    if not only:
        return parts
    wanted = [p.strip() for p in only.split(",") if p.strip()]
    unknown = [p for p in wanted if p not in parts]
    if unknown:
        raise SystemExit(
            f"--only: unknown part(s) {unknown}. Known parts: {list(parts)}"
        )
    return {name: parts[name] for name in wanted}


_REVISION_FILE = Path(__file__).resolve().parents[1] / "REVISION"


def resolved_config_hash(payload) -> str:
    """A digest of the settings a run ACTUALLY used, overrides included.

    WHY THIS EXISTS (round-two review item 4). `frozen_config.config_hash`
    digests the global `FrozenConfig`, which is the right object for the
    experiments that import it -- and the wrong one for every experiment that
    carries its own local settings. `exp_12` is the case that broke: it defines
    `N_SITES = 33`, its own noise levels, its own radii, and then takes
    `--set` overrides on top. None of that reaches `FrozenConfig`, so two runs
    with materially different protocols produced the same `frozen_hash` and a
    merge step had no way to tell them apart.

    This hashes the resolved dict instead -- what the code read after defaults,
    local settings and command-line overrides were all applied. Sorted keys and
    `repr`, so the digest depends on values and not on insertion order, and
    lists/tuples that compare equal still hash differently because `repr` keeps
    the distinction that `==` drops.
    """
    import hashlib

    def _norm(v):
        if isinstance(v, (list, tuple)):
            return tuple(_norm(x) for x in v)
        if isinstance(v, dict):
            return tuple(sorted((k, _norm(x)) for k, x in v.items()))
        return v

    body = "\n".join(
        f"{k}={_norm(v)!r}" for k, v in sorted(dict(payload).items())
    )
    return hashlib.sha256(body.encode()).hexdigest()[:12]


def _revision_stamp() -> dict[str, str]:
    """Commit and dirty state recorded at deploy time, for hosts without .git.

    The graceful degradation below did exactly what it was designed to do and
    that turned out to be the problem: every cluster run has an EMPTY git_commit,
    because the code is rsynced without .git. The runs that most need a revision
    are the ones that never had one.

    So the deploy step writes a REVISION file next to the code, and it is read
    here when git itself is unavailable. NGBP_GIT_COMMIT and NGBP_GIT_DIRTY
    override both, for a scheduler that would rather pass them in the
    environment.

    THE STAMP WAS NOT ENOUGH (round-two review item 4). A stamp records the
    tree at the moment `stamp_revision.sh` ran; `sync_to_cluster.sh` then
    rsynced the working tree as a SEPARATE step. Edit a file between the two
    and the stamp is silently stale -- which is exactly what happened to
    `exp_12`: the outputs name commit 286b305, whose `exp_12_receptive_field.py`
    does not define `eff_seed0`, while the recorded command passes
    `--set eff_seed0=0` and `apply_overrides` exits on an unknown key. That
    command cannot have run at that commit. The deployed source is not
    recoverable from anything the run wrote down.

    A commit alone cannot detect this, because the stamp and the code travel
    independently. So the new format additionally carries the SHA-256 of the
    `git archive` that was actually shipped, and `hpc/deploy_clean.sh` refuses
    to build one from a dirty tree. The archive digest is a property of the
    bytes that ran, not of what git believed at stamp time.

    Format is `key=value` lines; a bare 40-hex first line is read as the legacy
    commit-then-porcelain form so old outputs still parse.
    """
    env = os.environ.get("NGBP_GIT_COMMIT", "").strip()
    if env:
        return {"commit": env, "dirty": os.environ.get("NGBP_GIT_DIRTY", "").strip(),
                "archive_sha256": os.environ.get("NGBP_SOURCE_SHA256", "").strip(),
                "source": "env"}
    try:
        text = _REVISION_FILE.read_text()
        lines = text.splitlines()
        if lines and lines[0].startswith("commit="):
            kv = dict(
                ln.split("=", 1) for ln in lines if "=" in ln and not ln.startswith(" ")
            )
            return {"commit": kv.get("commit", "").strip(),
                    "dirty": kv.get("dirty", "").strip(),
                    "archive_sha256": kv.get("archive_sha256", "").strip(),
                    "source": "REVISION file"}
        return {"commit": lines[0].strip() if lines else "",
                "dirty": "\n".join(lines[1:]).strip(),
                "archive_sha256": "",
                "source": "REVISION file (legacy format, no archive digest)"}
    except Exception:
        return {"commit": "", "dirty": "", "archive_sha256": "",
                "source": "unavailable"}


def _git(*args: str) -> str:
    """A git query that degrades to "" rather than killing an experiment.

    Runs on compute nodes and from exported tarballs where .git may be absent,
    so every failure mode -- no repo, no git binary, timeout -- returns the empty
    string. Callers that need the commit should go through `provenance`, which
    falls back to `_revision_stamp` rather than recording nothing.
    """
    import subprocess

    try:
        return subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), *args],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception:
        return ""


def provenance(config=None) -> dict[str, str]:
    """What to write into every params.json.

    Pass `config` -- the fully resolved settings dict the run actually used --
    so the output carries a digest of its own protocol. Without it a merge step
    can only compare commits, and two runs of the same commit under different
    `--set` overrides are indistinguishable.

    Set NGBP_REQUIRE_CLEAN=1 on any run whose output is meant to be citable.
    The stamp then has to name a commit, carry an archive digest, and report a
    clean tree, or the run refuses to start. Warning-and-continuing is how the
    unreproducible `exp_12` outputs and the dirty n=8192 endpoint were produced:
    by the time anyone reads the warning the compute is spent, and the only
    honest options left are to rerun or to withdraw the number.
    """
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    stamp = {"source": "git", "archive_sha256": ""}
    if not commit:
        stamp = _revision_stamp()
        commit, dirty = stamp["commit"], stamp["dirty"]

    if os.environ.get("NGBP_REQUIRE_CLEAN", "").strip() not in ("", "0"):
        problems = []
        if not commit:
            problems.append("no commit could be determined")
        if dirty:
            n = len([ln for ln in dirty.splitlines() if ln.strip()])
            problems.append(f"tree is dirty ({n} path(s))")
        if not stamp.get("archive_sha256") and stamp["source"] != "git":
            problems.append("no source-archive digest (deployed without deploy_clean.sh)")
        if problems:
            raise SystemExit(
                "NGBP_REQUIRE_CLEAN: refusing to produce citable output.\n  - "
                + "\n  - ".join(problems)
                + "\nBuild the deployment with hpc/deploy_clean.sh from a clean "
                  "committed tree, or unset NGBP_REQUIRE_CLEAN for a scratch run."
            )

    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": commit,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # Non-empty means the tree carried uncommitted changes when this ran, so
        # git_commit alone does not reproduce the result.
        "git_dirty": dirty,
        # SHA-256 of the `git archive` that was actually shipped. This is the
        # field that would have caught the exp_12 defect: the commit and the
        # dirty list both looked plausible, and neither described the code.
        "source_archive_sha256": stamp.get("archive_sha256", ""),
        "source_is_clean": bool(commit) and not dirty,
        # Which of git / the REVISION file / the environment supplied the commit,
        # so "" is distinguishable from "recorded, from a deploy stamp".
        "revision_source": stamp["source"],
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "cpu_count": str(os.cpu_count()),
        "blas_threads": os.environ.get("OMP_NUM_THREADS", ""),
    }
    if config is not None:
        out["resolved_config_hash"] = resolved_config_hash(config)
    return out
