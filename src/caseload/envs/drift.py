"""A procedural fraud-drift simulator, used for training.

The real dataset in this package, the Elliptic transaction graph, contains exactly
one regime break. One break is one trajectory, and you cannot fit a policy to one
trajectory without simply memorising it. So policies are trained here, on randomly
generated drift scenarios, and evaluated on the real break they have never seen.

Fraud is a mixture of modes in feature space. At a break, some modes stop emitting
and new ones start. Nothing about the feature vectors announces this: the only
visible symptom is that the detector's yield falls, and the only way to recover is
to spend budget on cases the detector does not rank highly, which costs recall now
in exchange for a detector that still works later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mdp import Episode, Round


@dataclass
class DriftConfig:
    n_rounds: int = 20
    warm_rounds: int = 8
    pool_lo: int = 400
    pool_hi: int = 1200
    dim: int = 24
    base_rate_lo: float = 0.02
    base_rate_hi: float = 0.10
    n_modes: int = 3
    mode_scale: float = 1.15
    mode_spread: float = 0.55
    break_lo_frac: float = 0.45
    break_hi_frac: float = 0.75
    n_breaks: int = 1
    mode_turnover: float = 1.0
    label_noise: float = 0.0
    adversarial_break: bool = True
    adversarial_candidates: int = 40


class DriftScenario:
    """One randomly drawn drift world. Call :meth:`episode` to materialise it."""

    def __init__(self, cfg: DriftConfig, seed: int) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        c = cfg
        self.base_rate = float(self.rng.uniform(c.base_rate_lo, c.base_rate_hi))
        total = c.warm_rounds + c.n_rounds
        lo = c.warm_rounds + int(c.break_lo_frac * c.n_rounds)
        hi = c.warm_rounds + int(c.break_hi_frac * c.n_rounds)
        self.breaks = sorted(
            int(self.rng.integers(lo, max(lo + 1, hi + 1))) for _ in range(max(c.n_breaks, 0))
        )
        # a bank of fraud modes; regimes pick disjoint-ish subsets of them
        n_bank = c.n_modes * (len(self.breaks) + 1)
        self.modes = self.rng.normal(0.0, c.mode_spread, size=(n_bank, c.dim)) * c.mode_scale
        if c.adversarial_break and len(self.breaks) > 0:
            self._hide_later_modes(n_bank)
        self.regimes: list[np.ndarray] = []
        for r in range(len(self.breaks) + 1):
            if c.mode_turnover >= 1.0:
                idx = np.arange(r * c.n_modes, (r + 1) * c.n_modes) % n_bank
            else:
                keep = int(round((1 - c.mode_turnover) * c.n_modes))
                prev = self.regimes[-1] if self.regimes else np.arange(c.n_modes)
                fresh = (np.arange(c.n_modes - keep) + (r + 1) * c.n_modes) % n_bank
                idx = np.concatenate([prev[:keep], fresh])
            self.regimes.append(np.asarray(idx, dtype=int))
        self.total_rounds = total

    def _hide_later_modes(self, n_bank: int) -> None:
        """Move post-break modes to where the incumbent detector scores lowest.

        Without this the new fraud mode lands somewhere the old detector still gives
        a middling score, so a pure top-k policy stumbles into a few cases every
        round, feeds them to the refit, and quietly heals itself. That recovers a
        large share of post-break fraud with no exploration at all, which leaves
        exploration nothing to contribute and makes the environment a poor model of
        the real thing: on Elliptic the frozen detector finds 2 of the 169 illicit
        transactions in steps 43-49 (``results/collapse.log``).

        Screening candidate centres against a detector fitted on pre-break traffic
        reproduces that. It is also the more realistic story, since an adversary who
        adapts moves to where the current model is not looking.
        """
        from sklearn.ensemble import HistGradientBoostingClassifier

        c = self.cfg
        pre = self.modes[: c.n_modes]
        # a small pre-break sample, enough to fit a screening detector
        xs, ys = [], []
        for _ in range(6):
            k = int(self.rng.binomial(600, self.base_rate))
            x = self.rng.normal(0.0, 1.0, size=(600, c.dim))
            y = np.zeros(600, dtype=int)
            if k:
                pick = self.rng.choice(len(pre), size=k, replace=True)
                x[:k] = pre[pick] + self.rng.normal(0.0, 0.42, size=(k, c.dim))
                y[:k] = 1
            xs.append(x)
            ys.append(y)
        x_all, y_all = np.concatenate(xs), np.concatenate(ys)
        if int((y_all == 1).sum()) < 2:
            return
        screen = HistGradientBoostingClassifier(
            max_iter=20, max_leaf_nodes=8, random_state=0, early_stopping=False
        ).fit(x_all, y_all)

        n_new = n_bank - c.n_modes
        if n_new <= 0:
            return
        cand = (
            self.rng.normal(0.0, c.mode_spread, size=(c.adversarial_candidates, c.dim))
            * c.mode_scale
        )
        # score each candidate centre by how suspicious the old detector finds it
        s = screen.predict_proba(cand)[:, 1]
        # also require distance from the pre-break modes, so "hidden" does not mean
        # "sitting on top of an old mode the detector happens to score low"
        far = np.sqrt(((cand[:, None, :] - pre[None]) ** 2).sum(-1)).min(1)
        rank = np.argsort(s - 0.05 * far)
        self.modes[c.n_modes :] = cand[rank[:n_new]]

    def _regime_at(self, t: int) -> int:
        return int(np.searchsorted(self.breaks, t, side="right"))

    def _draw(self, t: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        k = int(self.rng.binomial(n, self.base_rate))
        x = self.rng.normal(0.0, 1.0, size=(n, c.dim))
        y = np.zeros(n, dtype=int)
        if k > 0:
            modes = self.regimes[self._regime_at(t)]
            pick = self.rng.choice(modes, size=k, replace=True)
            x[:k] = self.modes[pick] + self.rng.normal(0.0, 0.42, size=(k, c.dim))
            y[:k] = 1
        if c.label_noise > 0:
            flip = self.rng.random(n) < c.label_noise
            y[flip] = 1 - y[flip]
        order = self.rng.permutation(n)
        return x[order], y[order]

    def episode(self) -> Episode:
        c = self.cfg
        sizes = self.rng.integers(c.pool_lo, c.pool_hi + 1, size=self.total_rounds)
        warm_x, warm_y = [], []
        for t in range(c.warm_rounds):
            x, y = self._draw(t, int(sizes[t]))
            warm_x.append(x)
            warm_y.append(y)
        rounds = []
        for t in range(c.warm_rounds, self.total_rounds):
            x, y = self._draw(t, int(sizes[t]))
            rounds.append(Round(x=x, y=y))
        return Episode(
            rounds=rounds,
            warm_x=np.concatenate(warm_x),
            warm_y=np.concatenate(warm_y),
            name=f"drift-{self.seed}",
        )

    @property
    def break_rounds(self) -> list[int]:
        """Break positions expressed as indices into the episode's rounds."""
        return [b - self.cfg.warm_rounds for b in self.breaks]


def make_episode(seed: int, cfg: DriftConfig | None = None) -> tuple[Episode, list[int]]:
    """Draw one drift episode and report where its breaks fall."""
    sc = DriftScenario(cfg or DriftConfig(), seed)
    return sc.episode(), sc.break_rounds
