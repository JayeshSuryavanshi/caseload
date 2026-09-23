"""Baselines with seeds and intervals, which is the whole point.

Reports the interquartile mean and a 95% bootstrap interval over seeds, plus a
paired comparison against top-k, because episode difficulty varies far more than
the gap between policies and an unpaired comparison hides real effects.

    python scripts/run_baselines.py --seeds 30
    python scripts/run_baselines.py --elliptic
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from auditgym import (
    EpsilonExplore,
    GradientBoostScorer,
    InvestigationMDP,
    RandomGrid,
    RandomPolicy,
    TopK,
    YieldTriggered,
    oracle_ceiling,
    rollout,
)
from auditgym.envs import DriftConfig, make_episode
from auditgym.evaluation import table


def policies() -> list:
    return [
        TopK(),
        EpsilonExplore(0.15),
        EpsilonExplore(0.35),
        EpsilonExplore(0.6),
        YieldTriggered(drop=0.5, explore=0.6),
        RandomPolicy(seed=0),
        RandomGrid(seed=0),
    ]


def run_drift(seeds: int, budget: float, rounds: int, out: pathlib.Path | None) -> None:
    cfg = DriftConfig(n_rounds=rounds, adversarial_break=True)
    overall: dict[str, list[float]] = {}
    post: dict[str, list[float]] = {}
    pre: dict[str, list[float]] = {}
    t0 = time.perf_counter()
    for s in range(seeds):
        ep, breaks = make_episode(seed=s, cfg=cfg)
        b = breaks[0]
        for p in policies():
            mdp = InvestigationMDP(ep, GradientBoostScorer(), budget_frac=budget)
            r = rollout(mdp, p, seed=s)
            overall.setdefault(p.name, []).append(r.recall)
            pre.setdefault(p.name, []).append(r.recall_over(0, b - 1))
            post.setdefault(p.name, []).append(r.recall_over(b, 10**9))
        oc = oracle_ceiling(InvestigationMDP(ep, GradientBoostScorer(), budget_frac=budget))
        overall.setdefault("oracle-ceiling", []).append(oc.recall)
        pre.setdefault("oracle-ceiling", []).append(oc.recall_over(0, b - 1))
        post.setdefault("oracle-ceiling", []).append(oc.recall_over(b, 10**9))
        if (s + 1) % 5 == 0:
            print(f"  ...{s + 1}/{seeds} seeds  ({time.perf_counter() - t0:.0f}s)", flush=True)

    print(f"\ndrift simulator, adversarial break, {budget:.0%} budget, {seeds} seeds")
    for title, data in (("overall recall", overall), ("pre-break", pre), ("post-break", post)):
        print(f"\n--- {title} ---")
        print(table({k: np.asarray(v) for k, v in data.items()}, reference="top-k"))
    if out:
        out.write_text(json.dumps({"overall": overall, "pre": pre, "post": post}, indent=1))
        print(f"\nwrote {out}")


def run_elliptic(budget: float, out: pathlib.Path | None) -> None:
    from auditgym.envs.elliptic import load_episode

    ep, b = load_episode()
    print(
        f"elliptic: {ep.n_rounds} rounds, {ep.n_cases:,} cases, "
        f"{ep.n_positives:,} illicit, break at round {b}"
    )
    rows: dict[str, list[float]] = {}
    # one trajectory, so seeds only vary the policy's own randomness. reported as
    # a spread, never as a confidence interval over episodes, because there is one
    for p in policies():
        vals = []
        for s in range(5):
            mdp = InvestigationMDP(ep, GradientBoostScorer(), budget_frac=budget)
            r = rollout(mdp, p, seed=s)
            vals.append((r.recall, r.recall_over(0, b - 1), r.recall_over(b, 10**9)))
        a = np.asarray(vals)
        rows[p.name] = a[:, 0].tolist()
        print(
            f"  {p.name:26} overall {a[:, 0].mean():6.1%}  pre {a[:, 1].mean():6.1%}  "
            f"post {a[:, 2].mean():6.1%}  (spread over 5 policy seeds "
            f"{a[:, 0].min():.1%} to {a[:, 0].max():.1%})"
        )
    oc = oracle_ceiling(InvestigationMDP(ep, GradientBoostScorer(), budget_frac=budget))
    print(
        f"  {'oracle-ceiling':26} overall {oc.recall:6.1%}  "
        f"pre {oc.recall_over(0, b - 1):6.1%}  post {oc.recall_over(b, 10**9):6.1%}"
    )
    print("\nElliptic is one trajectory. These are not confidence intervals over")
    print("episodes and must not be read as such; the drift results are.")
    if out:
        out.write_text(json.dumps(rows, indent=1))
        print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--rounds", type=int, default=18)
    ap.add_argument("--elliptic", action="store_true")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    a = ap.parse_args()
    if a.elliptic:
        run_elliptic(a.budget, a.out)
    else:
        run_drift(a.seeds, a.budget, a.rounds, a.out)


if __name__ == "__main__":
    main()
