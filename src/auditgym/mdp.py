"""The budgeted investigation MDP.

An operator screens a stream of cases in rounds. Each round presents a pool of
cases whose labels are hidden. The operator holds a budget, measured in cases it
may investigate over the whole episode, and every round decides two things:

    how much of the remaining budget to spend now, and
    how much of that spend to aim at cases the current detector already suspects
    versus cases it knows nothing about.

Investigating a case reveals its label. Nothing else does. That single rule is
what makes this a decision problem rather than a classification problem: the
detector can only ever be retrained on cases somebody chose to look at, so a
policy that always spends on the highest scores stops learning the moment the
fraud population moves, and never finds out that it has.

This is the selective-labels setting (Lakkaraju et al., KDD 2017) written as a
sequential problem, and it is an instance of a Monitored MDP (Parisi et al.,
AAMAS 2024): the reward exists whether or not you observe it, and observing it
costs the same budget you need for exploiting.

The environment is deliberately small. Episodes are tens of rounds over pools of
a few thousand cases, so a study with the 30 to 50 seeds needed for honest
confidence intervals runs on a laptop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

# action grids. the agent picks one index into each. pace multiplies the budget
# a uniform spender would use this round, so 1.0 reproduces uniform spending
PACE_GRID: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.5)
EXPLORE_GRID: tuple[float, ...] = (0.0, 0.15, 0.35, 0.6, 1.0)

OBS_NAMES: tuple[str, ...] = (
    "round_frac",
    "budget_frac_left",
    "pace_available",
    "pool_size_rel",
    "score_mean",
    "score_p90",
    "score_max",
    "score_std",
    "yield_last",
    "yield_ma3",
    "yield_lifetime",
    "rounds_since_fit",
    "labelled_log",
    "positives_log",
    "score_shift",
)
OBS_DIM = len(OBS_NAMES)


class Scorer(Protocol):
    """A detector the environment retrains on whatever labels it has been given."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> None: ...

    def score(self, x: np.ndarray) -> np.ndarray:
        """Suspicion in [0, 1], higher meaning more likely positive."""
        ...

    @property
    def fitted(self) -> bool: ...


@dataclass
class Round:
    """One round's pool: features and the hidden labels the operator cannot see."""

    x: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        if len(self.x) != len(self.y):
            raise ValueError(f"x has {len(self.x)} rows but y has {len(self.y)}")

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class Episode:
    """A full trajectory of rounds, plus the warm-start pool the detector begins with."""

    rounds: list[Round]
    warm_x: np.ndarray
    warm_y: np.ndarray
    name: str = "episode"

    @property
    def n_rounds(self) -> int:
        return len(self.rounds)

    @property
    def n_cases(self) -> int:
        return sum(len(r) for r in self.rounds)

    @property
    def n_positives(self) -> int:
        return int(sum(int((r.y == 1).sum()) for r in self.rounds))


@dataclass
class StepRecord:
    """What happened in one round. The audit trail is the point, so keep all of it."""

    round_index: int
    pool_size: int
    budget_before: int
    spend: int
    exploit_spend: int
    explore_spend: int
    found: int
    found_by_exploit: int
    found_by_explore: int
    available: int
    refit: bool
    pace: float
    explore_frac: float


@dataclass
class RolloutResult:
    """Outcome of one episode under one policy."""

    found: int
    available: int
    spent: int
    budget: int
    steps: list[StepRecord] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.found / self.available if self.available else 0.0

    @property
    def precision(self) -> float:
        return self.found / self.spent if self.spent else 0.0

    def recall_over(self, lo: int, hi: int) -> float:
        """Recall restricted to rounds in [lo, hi], for reporting a post-break window."""
        f = sum(s.found for s in self.steps if lo <= s.round_index <= hi)
        a = sum(s.available for s in self.steps if lo <= s.round_index <= hi)
        return f / a if a else 0.0

    def spent_over(self, lo: int, hi: int) -> int:
        return sum(s.spend for s in self.steps if lo <= s.round_index <= hi)


def _safe_std(a: np.ndarray) -> float:
    return float(a.std()) if len(a) > 1 else 0.0


class InvestigationMDP:
    """The environment core, free of any RL framework.

    ``gymnasium`` wrappers live in :mod:`auditgym.envs`; keeping the mechanics here
    means the baselines and the tests do not need gymnasium at all.
    """

    def __init__(
        self,
        episode: Episode,
        scorer: Scorer,
        budget_frac: float = 0.10,
        refit_every: int = 1,
        min_positives_to_fit: int = 2,
        reward_scale: str = "found",
    ) -> None:
        if not 0.0 < budget_frac <= 1.0:
            raise ValueError(f"budget_frac must be in (0, 1], got {budget_frac}")
        if reward_scale not in ("found", "recall"):
            raise ValueError("reward_scale must be 'found' or 'recall'")
        self.episode = episode
        self.scorer = scorer
        self.budget_frac = budget_frac
        self.refit_every = refit_every
        self.min_positives_to_fit = min_positives_to_fit
        self.reward_scale = reward_scale
        self.total_budget = max(1, int(round(budget_frac * episode.n_cases)))
        self._rng = np.random.default_rng(0)

    # ---------------------------------------------------------------- lifecycle

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.t = 0
        self.budget_left = self.total_budget
        self.pool_x: list[np.ndarray] = [self.episode.warm_x]
        self.pool_y: list[np.ndarray] = [self.episode.warm_y]
        self.rounds_since_fit = 0
        self.found_total = 0
        self.spent_total = 0
        self.available_total = 0
        self.yields: list[float] = []
        self.records: list[StepRecord] = []
        self._train_score_mean: float | None = None
        self._fit()
        return self._observe()

    @property
    def done(self) -> bool:
        return self.t >= self.episode.n_rounds

    # ------------------------------------------------------------------ helpers

    def _labelled(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate(self.pool_x) if self.pool_x else np.empty((0, 1))
        y = np.concatenate(self.pool_y) if self.pool_y else np.empty((0,), dtype=int)
        return x, y

    def _fit(self) -> bool:
        x, y = self._labelled()
        if len(y) == 0 or int((y == 1).sum()) < self.min_positives_to_fit:
            return False
        self.scorer.fit(x, y)
        self.rounds_since_fit = 0
        # remember what the detector's own scores looked like on its training pool,
        # so the agent can see when the incoming pool stops resembling it
        self._train_score_mean = float(self.scorer.score(x).mean())
        return True

    def _scores(self, x: np.ndarray) -> np.ndarray:
        if not self.scorer.fitted:
            return self._rng.random(len(x))
        return self.scorer.score(x)

    def _observe(self) -> np.ndarray:
        if self.done:
            return np.zeros(OBS_DIM, dtype=np.float32)
        pool = self.episode.rounds[self.t]
        s = self._scores(pool.x)
        mean_pool = self.episode.n_cases / max(self.episode.n_rounds, 1)
        rounds_left = self.episode.n_rounds - self.t
        uniform = self.budget_left / max(rounds_left, 1)
        _, y = self._labelled()
        shift = 0.0
        if self._train_score_mean is not None:
            shift = float(np.clip(s.mean() - self._train_score_mean, -1.0, 1.0))
        obs = np.array(
            [
                self.t / max(self.episode.n_rounds, 1),
                self.budget_left / max(self.total_budget, 1),
                min(uniform / max(len(pool), 1), 1.0),
                min(len(pool) / max(mean_pool, 1.0), 4.0) / 4.0,
                float(s.mean()),
                float(np.quantile(s, 0.90)),
                float(s.max()),
                _safe_std(s),
                self.yields[-1] if self.yields else 0.0,
                float(np.mean(self.yields[-3:])) if self.yields else 0.0,
                self.found_total / max(self.spent_total, 1),
                min(self.rounds_since_fit / 5.0, 1.0),
                float(np.log1p(len(y)) / 12.0),
                float(np.log1p(int((y == 1).sum())) / 8.0),
                shift,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    # --------------------------------------------------------------------- step

    def step(
        self, pace_idx: int, explore_idx: int
    ) -> tuple[np.ndarray, float, bool, StepRecord]:
        """Spend part of the budget on this round's pool.

        ``pace_idx`` and ``explore_idx`` index :data:`PACE_GRID` and
        :data:`EXPLORE_GRID`. Pace is relative to what a uniform spender would use,
        so index 2 (1.0) reproduces uniform spending.
        """
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        pace = PACE_GRID[int(pace_idx)]
        explore_frac = EXPLORE_GRID[int(explore_idx)]

        pool = self.episode.rounds[self.t]
        n = len(pool)
        rounds_left = self.episode.n_rounds - self.t
        uniform = self.budget_left / max(rounds_left, 1)
        # never spend more than the pool holds, the budget allows, or what would
        # leave nothing at all for the rounds still to come
        spend = int(min(round(pace * uniform), n, self.budget_left))
        spend = max(spend, 0)

        scores = self._scores(pool.x)
        n_explore = int(round(explore_frac * spend))
        n_exploit = spend - n_explore

        order = np.argsort(-scores)
        exploit_idx = order[:n_exploit]
        rest = order[n_exploit:]
        if n_explore > 0 and len(rest) > 0:
            explore_idx = self._rng.choice(rest, size=min(n_explore, len(rest)), replace=False)
        else:
            explore_idx = np.empty(0, dtype=int)
        picked = np.concatenate([exploit_idx, explore_idx]).astype(int)

        got_exploit = int((pool.y[exploit_idx] == 1).sum()) if len(exploit_idx) else 0
        got_explore = int((pool.y[explore_idx] == 1).sum()) if len(explore_idx) else 0
        found = got_exploit + got_explore
        available = int((pool.y == 1).sum())

        # only investigated cases become training data. this is the whole point
        if len(picked):
            self.pool_x.append(pool.x[picked])
            self.pool_y.append(pool.y[picked])

        budget_before = self.budget_left
        self.budget_left -= spend
        self.spent_total += spend
        self.found_total += found
        self.available_total += available
        self.yields.append(found / spend if spend else 0.0)
        self.rounds_since_fit += 1

        refit = False
        if self.rounds_since_fit >= self.refit_every:
            refit = self._fit()

        rec = StepRecord(
            round_index=self.t,
            pool_size=n,
            budget_before=budget_before,
            spend=spend,
            exploit_spend=int(n_exploit),
            explore_spend=int(len(explore_idx)),
            found=found,
            found_by_exploit=got_exploit,
            found_by_explore=got_explore,
            available=available,
            refit=refit,
            pace=pace,
            explore_frac=explore_frac,
        )
        self.records.append(rec)
        self.t += 1

        reward = float(found) if self.reward_scale == "found" else found / max(available, 1)
        return self._observe(), reward, self.done, rec

    def result(self) -> RolloutResult:
        return RolloutResult(
            found=self.found_total,
            available=self.available_total,
            spent=self.spent_total,
            budget=self.total_budget,
            steps=list(self.records),
        )
