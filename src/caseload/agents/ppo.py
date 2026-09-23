"""A compact PPO for the investigation MDP, in one file.

Deliberately small. The action space is two grids of five, the observation is
fifteen numbers and an episode is a few tens of rounds, so the policy is a
two-layer MLP and the whole thing trains on a laptop CPU. Measured on an M1 Pro,
CPU beats MPS by 2.7 to 4.7x at these batch sizes; MPS only wins above roughly a
thousand rows per update, which this never reaches. There is no device flag
because there is no decision to make.

The only non-obvious part is that the environment is expensive relative to the
policy: a round refits a gradient boosting detector, so almost all wall-clock is
environment, not gradients. Updates are therefore large and infrequent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..mdp import EXPLORE_GRID, OBS_DIM, PACE_GRID


@dataclass
class PPOConfig:
    hidden: int = 64
    lr: float = 1e-3
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 10
    minibatch: int = 32
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    episodes_per_update: int = 8
    updates: int = 60
    # the environment refits a detector every round, so it dominates wall-clock and
    # a batch is only a few hundred transitions. small minibatches and more epochs
    # are what turn that into enough gradient steps to actually move the policy:
    # at minibatch 256 a batch is one minibatch, which is ~4 steps per update

    seed: int = 0


class ActorCritic(nn.Module):
    """Two independent categorical heads, one per action grid, and a value head."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pace_head = nn.Linear(hidden, len(PACE_GRID))
        self.explore_head = nn.Linear(hidden, len(EXPLORE_GRID))
        self.value_head = nn.Linear(hidden, 1)
        # small final-layer init keeps the initial policy close to uniform, which
        # matters here because a confidently wrong opening policy wastes the budget
        for head in (self.pace_head, self.explore_head):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor) -> tuple[Categorical, Categorical, torch.Tensor]:
        h = self.body(obs)
        return (
            Categorical(logits=self.pace_head(h)),
            Categorical(logits=self.explore_head(h)),
            self.value_head(h).squeeze(-1),
        )

    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = False) -> tuple[int, int, float, float]:
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        dp, de, v = self(o)
        if greedy:
            a_p = int(dp.probs.argmax())
            a_e = int(de.probs.argmax())
        else:
            a_p = int(dp.sample())
            a_e = int(de.sample())
        lp = float(dp.log_prob(torch.tensor([a_p])) + de.log_prob(torch.tensor([a_e])))
        return a_p, a_e, lp, float(v)


@dataclass
class Batch:
    obs: list[np.ndarray] = field(default_factory=list)
    pace: list[int] = field(default_factory=list)
    explore: list[int] = field(default_factory=list)
    logp: list[float] = field(default_factory=list)
    value: list[float] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    done: list[bool] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.obs)

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            "obs": torch.as_tensor(np.asarray(self.obs), dtype=torch.float32),
            "pace": torch.as_tensor(self.pace, dtype=torch.long),
            "explore": torch.as_tensor(self.explore, dtype=torch.long),
            "logp": torch.as_tensor(self.logp, dtype=torch.float32),
            "value": torch.as_tensor(self.value, dtype=torch.float32),
        }


def gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation over a batch of concatenated episodes."""
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    last = 0.0
    for i in reversed(range(n)):
        nonterminal = 0.0 if dones[i] else 1.0
        next_value = values[i + 1] if i + 1 < n else 0.0
        delta = rewards[i] + gamma * next_value * nonterminal - values[i]
        last = delta + gamma * lam * nonterminal * last
        adv[i] = last
    return adv, adv + np.asarray(values, dtype=np.float32)


class PPO:
    def __init__(self, cfg: PPOConfig | None = None) -> None:
        self.cfg = cfg or PPOConfig()
        torch.manual_seed(self.cfg.seed)
        self.net = ActorCritic(hidden=self.cfg.hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.history: list[dict[str, float]] = []

    def update(self, batch: Batch) -> dict[str, float]:
        c = self.cfg
        adv, ret = gae(batch.reward, batch.value, batch.done, c.gamma, c.gae_lambda)
        t = batch.tensors()
        adv_t = torch.as_tensor(adv)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(ret)

        n = len(batch)
        idx = np.arange(n)
        stats: dict[str, float] = {}
        for _ in range(c.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, c.minibatch):
                mb = idx[start : start + c.minibatch]
                dp, de, v = self.net(t["obs"][mb])
                logp = dp.log_prob(t["pace"][mb]) + de.log_prob(t["explore"][mb])
                ratio = torch.exp(logp - t["logp"][mb])
                a = adv_t[mb]
                unclipped = ratio * a
                clipped = torch.clamp(ratio, 1 - c.clip, 1 + c.clip) * a
                pg_loss = -torch.min(unclipped, clipped).mean()
                v_loss = ((v - ret_t[mb]) ** 2).mean()
                ent = (dp.entropy() + de.entropy()).mean()
                loss = pg_loss + c.value_coef * v_loss - c.entropy_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    stats = {
                        "pg_loss": pg_loss.item(),
                        "value_loss": v_loss.item(),
                        "entropy": ent.item(),
                        "approx_kl": (t["logp"][mb] - logp).mean().abs().item(),
                    }
        self.history.append(stats)
        return stats


class PPOPolicy:
    """Wraps a trained network so it plugs into ``caseload.policies.rollout``."""

    def __init__(self, net: ActorCritic, greedy: bool = True, name: str = "ppo") -> None:
        self.net = net
        self.greedy = greedy
        self.name = name

    def act(self, obs: np.ndarray) -> tuple[int, int]:
        p, e, _, _ = self.net.act(obs, greedy=self.greedy)
        return p, e

    def reset(self) -> None:
        return None
