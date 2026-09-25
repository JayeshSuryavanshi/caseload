"""The Elliptic Bitcoin transaction graph, as a held-out investigation episode.

The data set is from Weber et al., "Anti-Money Laundering in Bitcoin: Experimenting
with Graph Convolutional Networks for Financial Forensics", KDD '19 Workshop on
Anomaly Detection in Finance (arXiv:1908.02591): 203,769 transactions over 49 time
steps, each with 166 features, the first of which is the time step itself. The
archive keeps the other 165 as ``x``.

Weber et al. also document the regime break this environment is built around. A dark
market closed at time step 43, and every model they tried, including one retrained
after each step, performed poorly on the illicit transactions that followed.
``scripts/measure_collapse.py`` reproduces that failure at an investigation budget
and writes the per-step counts to ``results/collapse.log``: a detector frozen on
steps 1 to 34 finds a large share of the illicit cases per step up to step 42 and
almost none from 43 on, while the same detector refitted on labels up to step 44
finds many times more of the fraud in steps 45 to 49, so the information needed to
recover exists. It is only obtainable by spending budget on cases the broken detector
ranks low.

Most transactions carry no label (the exact share is in ``results/collapse.log``).
Unlabelled cases are real cases whose status nobody established; investigating one
costs budget and reveals nothing. That is left in deliberately, because it is what
the operational problem actually looks like.

The data is CC BY-NC-ND 4.0 and is not shipped here. ``scripts/fetch_elliptic.py``
says where to get it and converts the Kaggle CSVs into ``elliptic_parsed.npz`` with
arrays ``x``, ``node_time``, ``labels``, ``src``, ``dst`` and ``txid``. The archive
is looked for in ``~/.cache/caseload``. Two older locations are still read if that
is empty: ``~/.cache/auditgym`` (this package's former name) and ``~/.cache/graphspot``
(the graphspot library, which writes the same arrays in the same row order). The
graphspot parser rounds some features differently in the last bit, and the post-break
counts are small enough for that to move them, so convert with
``scripts/fetch_elliptic.py`` to reproduce the committed results exactly.
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np

from ..mdp import Episode, Round

CANONICAL_DIR = pathlib.Path.home() / ".cache" / "caseload"
LEGACY_DIRS = (
    pathlib.Path.home() / ".cache" / "auditgym",
    pathlib.Path.home() / ".cache" / "graphspot",
)
ARCHIVE = "elliptic_parsed.npz"
# the dark-market shutdown reported by Weber et al., in the dataset's own numbering
BREAK_TIME = 43
DEFAULT_WARM_UNTIL = 34


def find_archive(path: str | pathlib.Path | None = None) -> pathlib.Path:
    if path is not None:
        p = pathlib.Path(path).expanduser()
        if p.is_dir():
            p = p / ARCHIVE
        if p.exists():
            return p
        raise FileNotFoundError(p)
    for base in (CANONICAL_DIR, *LEGACY_DIRS):
        p = base / ARCHIVE
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{ARCHIVE} not found in {CANONICAL_DIR}. "
        "Run scripts/fetch_elliptic.py, which prints where to get the data."
    )


def load_raw(path: str | pathlib.Path | None = None) -> dict[str, np.ndarray]:
    d = np.load(find_archive(path), allow_pickle=True)
    return {k: d[k] for k in d.files}


def fingerprint(x: np.ndarray, t: np.ndarray, y: np.ndarray) -> str:
    """sha256 over the feature, time-step and label arrays, to tell archives apart."""
    h = hashlib.sha256()
    for a in (x, t, y):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def archive_fingerprint(path: str | pathlib.Path | None = None) -> str:
    raw = load_raw(path)
    return fingerprint(
        raw["x"], raw["node_time"].astype(np.int64), raw["labels"].astype(np.int64)
    )


def load_episode(
    path: str | pathlib.Path | None = None,
    warm_until: int = DEFAULT_WARM_UNTIL,
    drop_unlabelled: bool = False,
    labelled_only_rewards: bool = True,
) -> tuple[Episode, int]:
    """Build the Elliptic episode.

    Returns the episode and the index of the break within its rounds.

    ``drop_unlabelled`` removes cases nobody adjudicated. That makes the problem
    easier and less faithful; it exists so the cost of the unlabelled majority can
    be measured rather than assumed. ``labelled_only_rewards`` maps the unlabelled
    class to 0, meaning investigating one wastes budget, which is the honest
    reading: you looked and learned nothing.
    """
    raw = load_raw(path)
    x, t, y = raw["x"], raw["node_time"].astype(int), raw["labels"].astype(int)
    if drop_unlabelled:
        keep = y >= 0
        x, t, y = x[keep], t[keep], y[keep]
    y_eff = np.where(y == 1, 1, 0)
    if not labelled_only_rewards:
        y_eff = np.where(y < 0, 0, y_eff)

    warm = t <= warm_until
    # the detector warm-starts only from cases that were actually adjudicated
    warm_labelled = warm & (y >= 0)
    steps = sorted(int(s) for s in np.unique(t[t > warm_until]))
    rounds = [Round(x=x[t == s], y=y_eff[t == s]) for s in steps]
    break_index = steps.index(BREAK_TIME) if BREAK_TIME in steps else -1
    ep = Episode(
        rounds=rounds,
        warm_x=x[warm_labelled],
        warm_y=y_eff[warm_labelled],
        name=f"elliptic(t>{warm_until})",
    )
    return ep, break_index
