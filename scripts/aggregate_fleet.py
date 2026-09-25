"""Aggregate a fleet of trained seeds into one table with intervals.

Treats each training seed as one observation, which is the right unit: the question
is whether the method works, not whether one initialisation got lucky.

    python scripts/aggregate_fleet.py results/fleet > results/fleet_aggregate.log
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

from caseload.evaluation import (
    bootstrap_interval,
    iqm,
    paired_difference,
    probability_of_improvement,
)


def main() -> None:
    d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results/fleet")
    files = sorted(d.glob("eval_s*.json"))
    if not files:
        sys.exit(f"no eval_s*.json under {d}; run scripts/evaluate_fleet.sh first")

    per_policy: dict[str, list[float]] = {}
    per_policy_post: dict[str, list[float]] = {}
    ceiling: list[float] = []
    for f in files:
        blob = json.loads(f.read_text())
        drift = blob["drift"]
        for name, vals in drift["overall"].items():
            # one number per training seed: that seed's IQM over its eval episodes
            per_policy.setdefault(name, []).append(iqm(np.asarray(vals)))
        for name, vals in drift["post"].items():
            per_policy_post.setdefault(name, []).append(iqm(np.asarray(vals)))
        ceiling.append(iqm(np.asarray(drift["ceiling"])))

    n = len(files)
    print(f"{n} training seeds, each evaluated on the same held-out drift episodes\n")
    ref = "top-k"
    print(f"{'policy':28} {'overall':>20} {'post-break':>20} {'vs ' + ref:>24} {'P':>5}")
    for name in per_policy:
        ov = np.asarray(per_policy[name])
        po = np.asarray(per_policy_post[name])
        ci, cp = bootstrap_interval(ov, reps=8000), bootstrap_interval(po, reps=8000)
        a = f"{ci.point:.3f} [{ci.lo:.3f},{ci.hi:.3f}]"
        b = f"{cp.point:.3f} [{cp.lo:.3f},{cp.hi:.3f}]"
        if name == ref:
            print(f"{name:28} {a:>20} {b:>20} {'(reference)':>24}")
            continue
        dd = paired_difference(ov, np.asarray(per_policy[ref]), reps=8000)
        p = probability_of_improvement(ov, np.asarray(per_policy[ref]))
        delta = f"{dd.point:+.3f} [{dd.lo:+.3f},{dd.hi:+.3f}]"
        if dd.lo <= 0 <= dd.hi:
            flag = ""
        else:
            flag = "  significant" if dd.lo > 0 else "  worse"
        print(f"{name:28} {a:>20} {b:>20} {delta:>24} {p:>5.2f}{flag}")
    # the same episodes for every seed, so this is one number, with the same statistic
    print(f"{'oracle ceiling':28} {iqm(np.asarray(ceiling)):>20.3f}")

    if "ppo" in per_policy and "yield-triggered(drop=0.5)" in per_policy:
        dd = paired_difference(
            np.asarray(per_policy["ppo"]),
            np.asarray(per_policy["yield-triggered(drop=0.5)"]),
            reps=8000,
        )
        verdict = (
            "CLEARS the hand-written rule"
            if dd.lo > 0
            else "does NOT clear the hand-written rule"
        )
        print(
            f"\nppo vs yield-triggered: {dd.point:+.4f} "
            f"[{dd.lo:+.4f},{dd.hi:+.4f}]  ->  {verdict}"
        )
    print(
        "\nThis says nothing about transfer. Run scripts/evaluate.py "
        "with Elliptic present for that."
    )


if __name__ == "__main__":
    main()
