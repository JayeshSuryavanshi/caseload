"""Gymnasium wrapper.

Kept separate from :mod:`caseload.mdp` so the baselines, tests and the Elliptic
measurement scripts never import gymnasium. The environment core is a plain object
with ``reset`` and ``step``; this only adds spaces and the five-tuple contract.
"""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..mdp import EXPLORE_GRID, OBS_DIM, PACE_GRID, Episode, InvestigationMDP, Scorer
from ..scorers import GradientBoostScorer


class InvestigationEnv(gym.Env):
    """One investigation episode per ``reset``.

    ``episode_fn`` is called with the reset seed and returns an episode, so a fresh
    drift scenario is drawn every reset. That is what stops a policy from memorising
    a single trajectory, which matters because the real dataset only has one.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_fn: Callable[[int], Episode],
        scorer_fn: Callable[[], Scorer] = GradientBoostScorer,
        budget_frac: float = 0.10,
        refit_every: int = 1,
        reward_scale: str = "found",
        normalise_reward: bool = True,
    ) -> None:
        super().__init__()
        self.episode_fn = episode_fn
        self.scorer_fn = scorer_fn
        self.budget_frac = budget_frac
        self.refit_every = refit_every
        self.reward_scale = reward_scale
        self.normalise_reward = normalise_reward
        self.observation_space = spaces.Box(-4.0, 4.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([len(PACE_GRID), len(EXPLORE_GRID)])
        self._mdp: InvestigationMDP | None = None

    def reset(self, *, seed: int | None = None, options=None):  # noqa: ARG002
        super().reset(seed=seed)
        s = 0 if seed is None else int(seed)
        episode = self.episode_fn(s)
        self._mdp = InvestigationMDP(
            episode,
            self.scorer_fn(),
            budget_frac=self.budget_frac,
            refit_every=self.refit_every,
            reward_scale=self.reward_scale,
        )
        obs = self._mdp.reset(seed=s)
        self._scale = max(episode.n_positives, 1) if self.normalise_reward else 1.0
        return obs, {}

    def step(self, action):
        if self._mdp is None:
            raise RuntimeError("call reset() first")
        pace, explore = int(action[0]), int(action[1])
        obs, reward, done, rec = self._mdp.step(pace, explore)
        info = {
            "found": rec.found,
            "available": rec.available,
            "spend": rec.spend,
            "found_by_explore": rec.found_by_explore,
            "round": rec.round_index,
        }
        if done:
            r = self._mdp.result()
            info["episode_recall"] = r.recall
            info["episode_precision"] = r.precision
            info["episode_found"] = r.found
        return obs, reward / self._scale, done, False, info

    @property
    def mdp(self) -> InvestigationMDP:
        if self._mdp is None:
            raise RuntimeError("call reset() first")
        return self._mdp
