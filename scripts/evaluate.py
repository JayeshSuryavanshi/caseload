"""Evaluate a trained policy, first on held-out drift seeds, then on the real break.

The Elliptic evaluation is zero-shot: the policy trained only on procedurally
generated drift and has never seen this trajectory. That is the whole reason the
simulator exists, since one real break is one episode and a policy tuned on it has
memorised it.

Elliptic is a single trajectory, so the spread reported for it is over the policy's
own randomness, not over episodes. It is not a confidence interval and is not
labelled as one. The drift numbers are the ones that carry intervals.

The defaults reproduce the committed ``results/fleet/eval_s0.json``, and the output
path follows the model's, so ``ppo_s3.pt`` writes ``eval_s3.json`` beside it:

    python scripts/evaluate.py --model results/fleet/ppo_s0.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from caseload import (
    EpsilonExplore,
    GradientBoostScorer,
    InvestigationMDP,
    TopK,
    YieldTriggered,
    oracle_ceiling,
    rollout,
)
from caseload.agents.ppo import ActorCritic, PPOPolicy
from caseload.envs import DriftConfig, make_episode
from caseload.evaluation import (
    bootstrap_interval,
    iqm,
    paired_difference,
    probability_of_improvement,
)

# disjoint from the 1,000,000+ range scripts/train.py samples from
EVAL_SEEDS = range(500, 520)  # overridden by --seeds


def load_policy(path: pathlib.Path) -> PPOPolicy:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    net = ActorCritic()
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return PPOPolicy(net, greedy=True)


def opponents() -> list:
    return [
        TopK(),
        EpsilonExplore(0.15),
        EpsilonExplore(0.35),
        EpsilonExplore(0.6),
        YieldTriggered(drop=0.5, explore=0.6),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", type=pathlib.Path, default=pathlib.Path("results/fleet/ppo_s0.pt")
    )
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--rounds", type=int, default=18)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--elliptic-seeds", type=int, default=3)
    ap.add_argument(
        "--elliptic",
        type=pathlib.Path,
        default=None,
        help="Elliptic archive to use instead of the default cache lookup",
    )
    ap.add_argument("--out", type=pathlib.Path, default=None)
    a = ap.parse_args()
    if not a.model.exists():
        ap.error(f"no model at {a.model}; train one with scripts/train.py")
    if a.out is None:
        a.out = a.model.with_name(a.model.stem.replace("ppo", "eval", 1) + ".json")

    pols: list = [load_policy(a.model), *opponents()]

    global EVAL_SEEDS
    EVAL_SEEDS = range(500, 500 + a.seeds)
    cfg = DriftConfig(n_rounds=a.rounds, adversarial_break=True)
    overall: dict[str, list[float]] = {}
    post: dict[str, list[float]] = {}
    pre: dict[str, list[float]] = {}
    ceil: list[float] = []
    for s in EVAL_SEEDS:
        ep, breaks = make_episode(seed=s, cfg=cfg)
        b = breaks[0]
        for p in pols:
            mdp = InvestigationMDP(ep, GradientBoostScorer(), budget_frac=a.budget)
            r = rollout(mdp, p, seed=s)
            overall.setdefault(p.name, []).append(r.recall)
            pre.setdefault(p.name, []).append(r.recall_over(0, b - 1))
            post.setdefault(p.name, []).append(r.recall_over(b, 10**9))
        ceil.append(
            oracle_ceiling(
                InvestigationMDP(ep, GradientBoostScorer(), budget_frac=a.budget)
            ).recall
        )
        done = s - EVAL_SEEDS.start + 1
        if done % 5 == 0:
            print(f"  ...{done}/{len(list(EVAL_SEEDS))} drift seeds", flush=True)

    print(
        f"held-out drift seeds {EVAL_SEEDS.start}..{EVAL_SEEDS.stop - 1}, "
        f"{a.budget:.0%} budget, {len(list(EVAL_SEEDS))} episodes"
    )
    print(f"{'policy':28} {'overall':>18} {'pre':>8} {'post':>8} {'vs top-k':>22} {'P':>5}")
    ref = np.asarray(overall["top-k"])
    for p in pols:
        v = np.asarray(overall[p.name])
        ci = bootstrap_interval(v, reps=4000)
        d = paired_difference(v, ref, reps=4000)
        pbi = probability_of_improvement(v, ref)
        span = f"{ci.point:.3f} [{ci.lo:.3f},{ci.hi:.3f}]"
        delta = (
            "(reference)" if p.name == "top-k" else f"{d.point:+.3f} [{d.lo:+.3f},{d.hi:+.3f}]"
        )
        print(
            f"{p.name:28} {span:>18} {iqm(np.asarray(pre[p.name])):>8.3f} "
            f"{iqm(np.asarray(post[p.name])):>8.3f} {delta:>22} {pbi:>5.2f}"
        )
    # IQM, like the overall column, so the ceiling reads against the same statistic
    print(f"{'oracle ceiling':28} {iqm(np.asarray(ceil)):>18.3f}")

    result = {"drift": {"overall": overall, "pre": pre, "post": post, "ceiling": ceil}}

    # zero-shot transfer to the real regime break
    try:
        from caseload.envs.elliptic import archive_fingerprint, load_episode

        ep, b = load_episode(a.elliptic)
        sha = archive_fingerprint(a.elliptic)
        print(
            f"\nzero-shot on Elliptic: {ep.n_rounds} rounds, break at round {b}, "
            f"{ep.n_positives:,} illicit"
        )
        print(f"archive sha256 {sha}")
        print(f"{'policy':28} {'overall':>9} {'pre':>8} {'post':>8}  spread over policy seeds")
        ell: dict[str, list] = {}
        for p in pols:
            rows = []
            for s in range(a.elliptic_seeds):
                mdp = InvestigationMDP(ep, GradientBoostScorer(), budget_frac=a.budget)
                r = rollout(mdp, p, seed=s)
                rows.append((r.recall, r.recall_over(0, b - 1), r.recall_over(b, 10**9)))
            m = np.asarray(rows)
            ell[p.name] = m.tolist()
            print(
                f"{p.name:28} {m[:, 0].mean():>9.3f} {m[:, 1].mean():>8.3f} "
                f"{m[:, 2].mean():>8.3f}  {m[:, 0].min():.3f} to {m[:, 0].max():.3f}"
            )
        oc = oracle_ceiling(InvestigationMDP(ep, GradientBoostScorer(), budget_frac=a.budget))
        print(
            f"{'oracle ceiling':28} {oc.recall:>9.3f} {oc.recall_over(0, b - 1):>8.3f} "
            f"{oc.recall_over(b, 10**9):>8.3f}"
        )
        print("\nOne trajectory. The spread is over the policy's own randomness and is")
        print("not a confidence interval over episodes.")
        result["elliptic"] = ell
        result["elliptic_sha256"] = sha
    except FileNotFoundError:
        print("\n(Elliptic not available; run scripts/fetch_elliptic.py for the transfer test)")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
