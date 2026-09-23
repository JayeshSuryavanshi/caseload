"""The Elliptic Bitcoin transaction graph, as a held-out investigation episode.

203,769 transactions over 49 time steps, 166 features, and a documented regime
break: partway through the series a darknet market was shut down, and a detector
fitted before it stops working afterwards. Measured here, a 30-iteration gradient
boosting detector frozen on steps 1 to 34 finds 33 to 88 percent of the labelled
illicit transactions per step at a 2 percent budget up to step 42, and then finds
exactly zero for seven consecutive steps from step 43 on.

That collapse is what the environment is for. The same detector, refitted with
labels from steps 43 and 44, recovers to 56.2 percent recall on steps 45 to 49 at
a 10 percent budget, so the information needed to recover is obtainable. It is only
obtainable by spending budget on cases the broken detector ranks low.

Only 22.9 percent of transactions carry a label. Unlabelled cases are real cases
whose status nobody established; investigating one costs budget and reveals
nothing. That is left in deliberately, because it is what the operational problem
actually looks like.

The parsed archive is expected at ``~/.cache/graphspot/elliptic_parsed.npz`` or
``~/.cache/caseload/elliptic_parsed.npz`` with arrays ``x``, ``node_time``,
``labels``, ``src``, ``dst``. See ``scripts/fetch_elliptic.py``.
"""

from __future__ import annotations

import pathlib

import numpy as np

from ..mdp import Episode, Round

# "auditgym" is this package's former name; keep reading archives placed there
CACHE_NAMES = ("caseload", "auditgym", "graphspot")
ARCHIVE = "elliptic_parsed.npz"
# the documented regime break, as a time step in the dataset's own numbering
BREAK_TIME = 43
DEFAULT_WARM_UNTIL = 34


def _find_archive(path: str | pathlib.Path | None = None) -> pathlib.Path:
    if path is not None:
        p = pathlib.Path(path).expanduser()
        if p.is_dir():
            p = p / ARCHIVE
        if p.exists():
            return p
        raise FileNotFoundError(p)
    for name in CACHE_NAMES:
        p = pathlib.Path.home() / ".cache" / name / ARCHIVE
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{ARCHIVE} not found under ~/.cache/{{{','.join(CACHE_NAMES)}}}/. "
        "Run scripts/fetch_elliptic.py, which prints where to get the data."
    )


def load_raw(path: str | pathlib.Path | None = None) -> dict[str, np.ndarray]:
    d = np.load(_find_archive(path), allow_pickle=True)
    return {k: d[k] for k in d.files}


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
