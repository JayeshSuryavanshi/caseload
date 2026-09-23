"""The headline measurement: when does spending review capacity sequentially pay?

FiFAR ships review capacity equal to 100% of alerts in its training scenarios and
90.9% in its test scenarios. At those levels there is nothing to ration, so a
one-shot assignment is the right tool and the dataset's own constraint-programming
baseline is hard to improve on. The open question is where that stops being true.

This sweeps capacity tightness and asks, at each level, whether any pacing rule
beats allocating capacity in proportion to batch size. Proportional is the reference
because even-per-batch pacing is a straw man here: FiFAR's batches run from 5,000
alerts down to 400, so dividing by the number of remaining batches under-reviews the
big ones. Measured against even pacing, almost anything wins, and that would be a
result about a bad allocation rule rather than about sequential planning.

    python scripts/capacity_sweep.py --scenarios 8
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from caseload.envs import fifar
from caseload.evaluation import bootstrap_interval, paired_difference
from caseload.triage import BAND_GRID, PACE_GRID, TriageConfig, TriageMDP

EVEN = PACE_GRID.index(1.0)


class Fixed:
    """Even pacing with a fixed routing band."""

    def __init__(self, band: str) -> None:
        self.band = band
        self.name = f"even/{band}"
        self._b = BAND_GRID.index(band)

    def act(self, obs: np.ndarray) -> tuple[int, int]:  # noqa: ARG002
        return EVEN, self._b


class FrontLoaded:
    """Spend fast and run dry. What an operator does without a capacity plan."""

    name = "front-loaded/near-threshold"

    def act(self, obs: np.ndarray) -> tuple[int, int]:  # noqa: ARG002
        return PACE_GRID.index(2.5), BAND_GRID.index("near-threshold")


class SizeAware:
    """Spend in proportion to how large this batch is relative to the average.

    The intertemporal rule that needs no learning: a batch ten times the size of the
    next one deserves more than an equal share. If sequential planning is worth
    anything at this capacity level, this should already show it.
    """

    name = "size-aware/near-threshold"

    def act(self, obs: np.ndarray) -> tuple[int, int]:
        rel = float(obs[3]) * 3.0  # batch size relative to the scenario mean
        if rel > 1.25:
            p = PACE_GRID.index(1.5)
        elif rel < 0.6:
            p = PACE_GRID.index(0.5)
        else:
            p = EVEN
        return p, BAND_GRID.index("near-threshold")


class Proportional:
    """Allocate in proportion to batch size. The honest operator default.

    Even-per-batch pacing is a straw man when batch sizes are uneven: FiFAR's
    training batches run from 5,000 alerts down to 400, so dividing capacity by the
    number of remaining batches under-reviews the big ones and hoards slots for the
    small ones. Anything claiming to show the value of sequential planning has to
    beat this, not that.
    """

    name = "proportional/near-threshold"

    def act(self, obs: np.ndarray) -> tuple[int, int]:
        rel = float(obs[3]) * 3.0  # this batch's size over the scenario mean
        # pace is relative to an even share, so the proportional share is just rel
        p = int(np.argmin([abs(g - rel) for g in PACE_GRID]))
        return p, BAND_GRID.index("near-threshold")


def run(data, scenario, policy, cfg: TriageConfig, seed: int = 0):
    m = TriageMDP(data, scenario, cfg)
    obs = m.reset(seed=seed)
    while not m.done:
        p, b = policy.act(obs)
        obs, _, _, _ = m.step(p, b)
    return m.result()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=8)
    ap.add_argument("--fp-cost", type=float, default=0.05)
    ap.add_argument(
        "--out", type=pathlib.Path, default=pathlib.Path("results/capacity_sweep.json")
    )
    a = ap.parse_args()

    data = fifar.load()
    names = [f"shuffle_{i}#team_{t}" for i in (1, 2, 3, 4, 5) for t in (1, 2, 3, 4, 5)]
    names = names[: a.scenarios]
    cfg = TriageConfig(fp_cost=a.fp_cost)
    ratios = [0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.00]
    policies = [
        Proportional(),
        Fixed("near-threshold"),
        Fixed("top-score"),
        Fixed("mixed"),
        SizeAware(),
        FrontLoaded(),
    ]

    print(f"FiFAR, {len(names)} scenarios, fp_cost={a.fp_cost} (fn_cost=1.0)")
    print("saving = fraction of the model-only cost removed by review\n")
    out: dict = {"ratios": ratios, "fp_cost": a.fp_cost, "data": {}}
    for ratio in ratios:
        scen = [
            fifar.load_scenario(data, n, "train_alert", capacity_ratio=ratio) for n in names
        ]
        res: dict[str, list[float]] = {}
        spend: dict[str, list[float]] = {}
        per: dict[str, list[float]] = {}
        for pol in policies:
            rr = [run(data, s, pol, cfg) for s in scen]
            res[pol.name] = [r.saving for r in rr]
            spend[pol.name] = [r.capacity_used for r in rr]
            per[pol.name] = [r.saving_per_review for r in rr]
        out["data"][str(ratio)] = {"saving": res, "capacity_used": spend, "per_review": per}

        ref = "proportional/near-threshold"
        print(f"--- capacity = {ratio:.0%} of alerts ---")
        base = bootstrap_interval(np.asarray(res[ref]), reps=4000)
        print(
            f"  {ref:28} {base.point:7.3f} [{base.lo:.3f}, {base.hi:.3f}]"
            f"  used {np.mean(spend[ref]):6.1%}  per-review {np.mean(per[ref]):7.4f}"
            f"   (reference)"
        )
        for pol in policies:
            if pol.name == ref:
                continue
            ci = bootstrap_interval(np.asarray(res[pol.name]), reps=4000)
            d = paired_difference(np.asarray(res[pol.name]), np.asarray(res[ref]), reps=4000)
            dpi = paired_difference(np.asarray(per[pol.name]), np.asarray(per[ref]), reps=4000)
            # a win only counts if it survives spend-normalisation
            if dpi.lo > 0:
                sig = "  <-- better per review"
            elif dpi.hi < 0:
                sig = "  (worse per review)"
            else:
                sig = "  (same per review)"
            dp = paired_difference(np.asarray(per[pol.name]), np.asarray(per[ref]), reps=4000)
            print(
                f"  {pol.name:28} {ci.point:7.3f} [{ci.lo:.3f}, {ci.hi:.3f}]"
                f"  used {np.mean(spend[pol.name]):6.1%}"
                f"  per-review {np.mean(per[pol.name]):7.4f}"
                f"   d_total {d.point:+.3f}"
                f"  d_per_review {dp.point:+.4f} [{dp.lo:+.4f}, {dp.hi:+.4f}]{sig}"
            )
        print()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")
    print("\nFiFAR ships 1.00 in training scenarios and 0.909 in test scenarios.")
    print("Read the rows near those values to see what the shipped benchmark leaves on")
    print("the table, and the tight rows to see where rationing starts to matter.")


if __name__ == "__main__":
    main()
