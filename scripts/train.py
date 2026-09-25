"""Train a PPO policy on procedurally generated drift, never on Elliptic.

Elliptic contains one regime break. A policy tuned on it has memorised one
trajectory, and no amount of seeding fixes that. So training happens here, on
random breaks, and Elliptic is only ever used for evaluation by
``scripts/evaluate.py``.

The defaults are the ones behind the committed fleet, 200 updates x 8 episodes =
1,600 episodes per seed:

    python scripts/train.py --seed 0 --out results/fleet/ppo_s0.pt

The checkpoint holds only tensors and plain values, so it loads with
``torch.load(..., weights_only=True)``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from caseload import GradientBoostScorer, InvestigationMDP, TopK, YieldTriggered, rollout
from caseload.agents.ppo import PPO, Batch, PPOConfig, PPOPolicy
from caseload.envs import DriftConfig, make_episode


def collect(
    agent: PPO, cfg: DriftConfig, budget: float, seeds: list[int]
) -> tuple[Batch, list[float]]:
    batch = Batch()
    recalls = []
    for s in seeds:
        ep, _ = make_episode(seed=s, cfg=cfg)
        mdp = InvestigationMDP(ep, GradientBoostScorer(), budget_frac=budget)
        obs = mdp.reset(seed=s)
        scale = max(ep.n_positives, 1)
        while not mdp.done:
            p, e, lp, v = agent.net.act(obs)
            nxt, reward, done, _ = mdp.step(p, e)
            batch.obs.append(obs)
            batch.pace.append(p)
            batch.explore.append(e)
            batch.logp.append(lp)
            batch.value.append(v)
            batch.reward.append(reward / scale)
            batch.done.append(done)
            obs = nxt
        recalls.append(mdp.result().recall)
    return batch, recalls


def checkpoint(state_dict: dict, args: dict, cfg: DriftConfig) -> dict:
    """What gets saved: weights plus a config of str/int/float/bool values only."""

    def plain(v: object) -> str | int | float | bool:
        return v if isinstance(v, (str, int, float, bool)) else str(v)

    config = {k: plain(v) for k, v in args.items()}
    config |= {f"drift.{k}": plain(v) for k, v in vars(cfg).items()}
    return {"state_dict": state_dict, "config": config}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--episodes-per-update", type=int, default=8)
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--rounds", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refit-every", type=int, default=1)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/ppo.pt"))
    a = ap.parse_args()

    # training seeds are drawn from a disjoint range from the evaluation seeds in
    # scripts/evaluate.py, so a reported transfer number is never a memorised one
    cfg = DriftConfig(n_rounds=a.rounds, adversarial_break=True)
    agent = PPO(PPOConfig(seed=a.seed, episodes_per_update=a.episodes_per_update))
    rng = np.random.default_rng(a.seed)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []
    t0 = time.perf_counter()
    for u in range(a.updates):
        seeds = [
            int(rng.integers(0, 100_000)) + 1_000_000 for _ in range(a.episodes_per_update)
        ]
        batch, recalls = collect(agent, cfg, a.budget, seeds)
        stats = agent.update(batch)
        row = {
            "update": u,
            "recall_mean": float(np.mean(recalls)),
            "recall_std": float(np.std(recalls)),
            "elapsed_s": round(time.perf_counter() - t0, 1),
            **{k: round(v, 5) for k, v in stats.items()},
        }
        log.append(row)
        print(
            f"update {u:>3}  recall {row['recall_mean']:.3f}  "
            f"entropy {stats.get('entropy', 0):.3f}  kl {stats.get('approx_kl', 0):.4f}  "
            f"{row['elapsed_s']:.0f}s",
            flush=True,
        )

    torch.save(checkpoint(agent.net.state_dict(), vars(a), cfg), a.out)
    pathlib.Path(str(a.out).replace(".pt", "_log.json")).write_text(json.dumps(log, indent=1))
    print(f"\nsaved {a.out}")

    # a quick sanity comparison on held-out drift seeds, not on Elliptic
    print("\nheld-out drift seeds (not used in training):")
    pol = PPOPolicy(agent.net, greedy=True)
    for p in (pol, TopK(), YieldTriggered(drop=0.5, explore=0.6)):
        vals = []
        for s in range(200, 210):
            ep, _ = make_episode(seed=s, cfg=cfg)
            mdp = InvestigationMDP(ep, GradientBoostScorer(), budget_frac=a.budget)
            vals.append(rollout(mdp, p, seed=s).recall)
        print(f"  {p.name:26} recall {np.mean(vals):.3f} +/- {np.std(vals):.3f}")


if __name__ == "__main__":
    main()
