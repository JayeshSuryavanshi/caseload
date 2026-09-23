"""Baseline policies, including the one that is hard to beat.

``TopK`` is the policy real systems run: score everything, investigate the highest,
spend the budget evenly over time. It is strong. On Elliptic before the regime break
it finds 33 to 88 percent of labelled illicit transactions per step at a 2 percent
budget. Any learned policy that cannot beat it is not worth shipping, and saying so
is the point of keeping it here as the headline opponent rather than comparing
against random.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .mdp import EXPLORE_GRID, PACE_GRID, InvestigationMDP, RolloutResult

UNIFORM_PACE = PACE_GRID.index(1.0)
NO_EXPLORE = EXPLORE_GRID.index(0.0)


def _nearest(grid: tuple[float, ...], v: float) -> int:
    return int(np.argmin([abs(g - v) for g in grid]))


class Policy(Protocol):
    name: str

    def act(self, obs: np.ndarray) -> tuple[int, int]: ...

    def reset(self) -> None: ...


class TopK:
    """Uniform spend, no exploration. The industry default."""

    name = "top-k"

    def act(self, obs: np.ndarray) -> tuple[int, int]:  # noqa: ARG002
        return UNIFORM_PACE, NO_EXPLORE

    def reset(self) -> None:
        return None


class EpsilonExplore:
    """Uniform spend, a fixed slice always aimed at cases the detector ranks low."""

    def __init__(self, eps: float = 0.15) -> None:
        self.eps = eps
        self.name = f"eps-explore({eps:g})"
        self._idx = _nearest(EXPLORE_GRID, eps)

    def act(self, obs: np.ndarray) -> tuple[int, int]:  # noqa: ARG002
        return UNIFORM_PACE, self._idx

    def reset(self) -> None:
        return None


class RandomPolicy:
    """Uniform spend, everything chosen at random. The floor."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> tuple[int, int]:  # noqa: ARG002
        return UNIFORM_PACE, EXPLORE_GRID.index(1.0)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)


class RandomGrid:
    """Picks uniformly from the action grid. Shows the grid is not itself the answer."""

    name = "random-action"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> tuple[int, int]:  # noqa: ARG002
        return int(self._rng.integers(len(PACE_GRID))), int(
            self._rng.integers(len(EXPLORE_GRID))
        )

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)


class YieldTriggered:
    """A hand-written adaptive rule, and the real bar for the learned agent.

    Watches its own recent hit rate. When yield falls far enough below what it had
    been getting, it switches budget into exploration until yield recovers. This is
    what a competent engineer writes in an afternoon without any RL, so a learned
    policy has to beat it to justify itself.
    """

    def __init__(self, drop: float = 0.5, explore: float = 0.6, patience: int = 1) -> None:
        self.drop = drop
        self.explore = explore
        self.patience = patience
        self.name = f"yield-triggered(drop={drop:g})"
        self._explore_idx = _nearest(EXPLORE_GRID, explore)
        self.reset()

    def reset(self) -> None:
        self._best = 0.0
        self._bad = 0

    def act(self, obs: np.ndarray) -> tuple[int, int]:
        # obs layout is fixed by auditgym.mdp.OBS_NAMES
        yield_last, yield_ma3 = float(obs[8]), float(obs[9])
        self._best = max(self._best, yield_ma3)
        collapsed = self._best > 0 and yield_last < self.drop * self._best
        self._bad = self._bad + 1 if collapsed else 0
        if self._bad >= self.patience:
            return UNIFORM_PACE, self._explore_idx
        return UNIFORM_PACE, NO_EXPLORE

    def reset_best(self) -> None:
        self._best = 0.0


def rollout(mdp: InvestigationMDP, policy: Policy, seed: int = 0) -> RolloutResult:
    """Run one episode under one policy."""
    obs = mdp.reset(seed=seed)
    policy.reset()
    while not mdp.done:
        pace, explore = policy.act(obs)
        obs, _, _, _ = mdp.step(pace, explore)
    return mdp.result()


def oracle_ceiling(mdp: InvestigationMDP, budget_frac: float | None = None) -> RolloutResult:
    """The best any policy could do with this budget if it knew every label.

    Spends uniformly and takes only true positives, so it reports the recall a
    perfect ranker would achieve under the same per-round budget. Recall below this
    is the gap a policy actually left on the table.
    """
    ep = mdp.episode
    bf = budget_frac if budget_frac is not None else mdp.budget_frac
    total = max(1, int(round(bf * ep.n_cases)))
    left = total
    found = spent = avail = 0
    steps = []
    from .mdp import StepRecord

    for i, r in enumerate(ep.rounds):
        rounds_left = ep.n_rounds - i
        b = int(min(round(left / max(rounds_left, 1)), len(r), left))
        pos = int((r.y == 1).sum())
        got = min(b, pos)
        found += got
        spent += b
        avail += pos
        left -= b
        steps.append(
            StepRecord(
                round_index=i,
                pool_size=len(r),
                budget_before=left + b,
                spend=b,
                exploit_spend=b,
                explore_spend=0,
                found=got,
                found_by_exploit=got,
                found_by_explore=0,
                available=pos,
                refit=False,
                pace=1.0,
                explore_frac=0.0,
            )
        )
    return RolloutResult(found=found, available=avail, spent=spent, budget=total, steps=steps)
