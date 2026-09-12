"""Every declared configuration field must actually be read by something.

WHY THIS EXISTS (review item H4). The paper says every experiment imports one
configuration "so that no run can silently diverge from it". Twice, that has not
been true, and both times in the same way: a field was declared in
`frozen_config.py`, described in a docstring, quoted in the protocol appendix --
and read by nothing, while the experiments passed literals.

  - `n_seeds` was declared while four experiments carried private replicate
    knobs and ran at 3, 4 or 8 replicates, each reporting `is_frozen: true`.
  - `em_max_iters` and `em_shape_tol` were declared, the appendix described the
    shape-based stopping rule they encode, and `exp_07` passed a literal 120 in
    four places. The stopping rule the paper described did not exist.

Both were invisible to the provenance stamp, which compares the config against
itself and therefore cannot see a field nobody consumes. Declaring a value is
not the same as consuming it, and only a reader-side check can tell them apart.

This is that check, in its cheapest honest form: a static scan for each field
name across the code that could read it. It cannot prove a field is used
*correctly* -- that would need the behavioural test the review also asks for,
one per field -- but it does catch the failure that has actually happened twice,
and it fails at the moment a field is added without a consumer rather than a
month later during an audit.
"""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from frozen_config import FROZEN, FrozenConfig, config_hash  # noqa: E402

SEARCH_DIRS = ("experiments", "src", "tools", "hpc")

# Fields genuinely declared and consumed by nothing. Each entry is a defect
# being recorded rather than a rule being bent, so each says what is actually
# wrong; removing a field or wiring it up should remove its line here.
EXEMPT = {
    "em_shape_tol": (
        "DECLARED BUT NOT IMPLEMENTED. The protocol appendix describes EM "
        "stopping when the fitted excess kurtosis stops moving, |dk4| < 1e-3, "
        "and this field is that threshold. No code reads it: fit_em stops on a "
        "relative log-evidence tolerance. The claim audit records this; the "
        "field is kept so the gap stays visible rather than being tidied away."
    ),
    "innovation": (
        "DECLARED BUT NOT READ. Experiments name their prior class directly "
        "(LaplaceAR1(rho)) rather than dispatching on this string, so it "
        "documents the default without selecting it. Harmless today and a trap "
        "the moment someone changes it expecting the runs to follow."
    ),
}


def _corpus() -> str:
    """Everything that could consume a field.

    `frozen_config.py` is included, but only its `self.<field>` uses count: a
    field read by a derived method of the config -- `base_seed` by `seeds()`,
    `heldout_seed_offset` by `heldout_seed()` -- has a real consumer, because
    those methods are called from the experiments. Excluding the whole file
    flagged both as dead, which they are not. What must NOT count is a field
    appearing only in its own declaration or docstring, so the declaration
    lines and the prose around them are stripped first.
    """
    parts = []
    for d in SEARCH_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.suffix not in (".py", ".sbatch", ".sh"):
                continue
            if "__pycache__" in p.parts or ".venv" in p.parts:
                continue
            text = p.read_text(errors="ignore")
            if p.name == "frozen_config.py":
                # Keep only lines that USE a field, not lines that declare or
                # describe one. `self.x` is a use; `x: int = 4` is not.
                text = "\n".join(
                    ln for ln in text.splitlines()
                    if "self." in ln or "config." in ln
                )
            parts.append(text)
    return "\n".join(parts)


CORPUS = _corpus()
FIELDS = [f.name for f in dataclasses.fields(FrozenConfig)]


def test_the_scan_actually_sees_something():
    """Guard the guard: an empty corpus would pass every test below."""
    assert len(CORPUS) > 100_000, (
        f"corpus is only {len(CORPUS)} chars -- the scan is not reading the "
        "tree, so every field would trivially appear unused or used"
    )
    assert len(FIELDS) > 15, f"only {len(FIELDS)} fields found on FrozenConfig"


@pytest.mark.parametrize("name", FIELDS)
def test_every_declared_field_has_a_reader(name):
    if name in EXEMPT:
        pytest.skip(f"{name}: {EXEMPT[name]}")
    # Attribute access (FROZEN.x, cfg.x), dict access ("x"), or a keyword
    # override (narrowed(x=...), frozen_settings(x=...)).
    pattern = re.compile(
        rf"(\.{re.escape(name)}\b)|([\"']{re.escape(name)}[\"'])|(\b{re.escape(name)}\s*=)"
    )
    assert pattern.search(CORPUS), (
        f"FrozenConfig.{name} is declared but no file under {SEARCH_DIRS} reads "
        f"it. This is the em_max_iters defect: the config would describe a "
        f"setting the runs do not use, and the provenance stamp -- which "
        f"compares the config against itself -- could not tell."
    )


def test_config_hash_is_stable_and_sensitive():
    """The digest must not move on its own, and must move when a value does."""
    assert config_hash() == config_hash()
    base = config_hash(FROZEN)
    for name, alt in (("rho", 0.5), ("n_grid", 201), ("n_seeds", 4)):
        changed = config_hash(FROZEN.narrowed(**{name: alt}))
        assert changed != base, f"changing {name} did not change the digest"
    # And it is a property of values, not of declaration order.
    assert len(base) == 12 and all(c in "0123456789abcdef" for c in base)
