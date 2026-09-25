"""When is human review worth doing at all? Sweep the cost ratio, refit the threshold.

FiFAR's analysts were generated at one cost structure, lambda_t = 0.057 (a false
alarm costs 0.057 of a missed fraud, about 17.5:1), and every one of the fifty was
constrained to cost less than rejecting every alert at that value. DeCCaF, the
dataset's own baseline paper, also runs lambda_t/5 and 5*lambda_t and calls them
"not strictly comparable". This script measures review against the model alone over
a grid of ratios that contains all three, and it refits the automated threshold at
every ratio with ``cost_optimal_threshold``. Without the refit, a change in the
ratio moves the model's own operating point away from optimal and review gets the
credit for undoing it.

Two evaluations, both at a fixed review budget of 10% of alerts:

* in-sample: all 25 shipped training scenarios (the same 26,165 alerts from months
  4 to 7, in 5 batch orders times 5 analyst teams), threshold fitted on those alerts
* held-out: the threshold fitted on months 4 to 7 and applied to the month-8 test
  alerts, one batch per team. The 25 test scenarios are 5 teams times 5 capacity
  variants, and rescaling the capacity makes the variants identical, so one per team

Intervals are 95% percentile bootstrap intervals that resample scenarios and alerts
together (the scenarios share their alerts, so resampling scenarios alone would
ignore the uncertainty that matters). The threshold is held at its fitted value
inside the bootstrap.

    python scripts/cost_ratio_sweep.py
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from capacity_sweep import Fixed, FrontLoaded, Proportional

from caseload.envs import fifar
from caseload.triage import (
    ALERT_THRESHOLD,
    LAMBDA_T,
    STRESS_FP_COSTS,
    TriageConfig,
    TriageMDP,
    cost_optimal_threshold,
)

# false-alarm cost with a missed fraud at 1.0. Contains DeCCaF's three values
FP_COSTS: tuple[float, ...] = (
    STRESS_FP_COSTS["lambda_t/5"],
    0.02,
    0.03,
    0.04,
    LAMBDA_T,
    0.10,
    STRESS_FP_COSTS["5*lambda_t"],
    0.50,
)
LABELS = {
    STRESS_FP_COSTS["lambda_t/5"]: "lambda_t/5 (DeCCaF stress case)",
    LAMBDA_T: "lambda_t (FiFAR and DeCCaF stated regime)",
    STRESS_FP_COSTS["5*lambda_t"]: "5*lambda_t (DeCCaF stress case)",
}
HEADLINE = "even/near-threshold"


def policies() -> list:
    return [
        Fixed("near-threshold"),
        Fixed("top-score"),
        Fixed("mixed"),
        Fixed("bottom-score"),
        Proportional(),
        FrontLoaded(),
    ]


def per_alert(data, scenario, policy, cfg: TriageConfig) -> tuple[np.ndarray, np.ndarray]:
    """Cost avoided by review on each alert row, and which rows an analyst decided."""
    m = TriageMDP(data, scenario, cfg)
    obs = m.reset(seed=0)
    while not m.done:
        p, b = policy.act(obs)
        obs, _, _, _ = m.step(p, b)
    y = data.y
    model = (data.score > cfg.threshold).astype(int)

    def cost(dec: np.ndarray) -> np.ndarray:
        return cfg.fn_cost * ((dec == 0) & (y == 1)) + cfg.fp_cost * ((dec == 1) & (y == 0))

    seen = m.final_decision >= 0
    delta = np.where(seen, cost(model) - cost(np.where(seen, m.final_decision, model)), 0.0)
    # the per-alert split must add back up to what the environment reported
    r = m.result()
    assert np.isclose(delta.sum(), r.cost_model_only - r.cost)
    assert int(m.routed_mask.sum()) == r.reviewed
    return delta, m.routed_mask.copy()


def two_way_bootstrap(
    delta: np.ndarray,
    base: np.ndarray,
    routed: np.ndarray,
    rows: np.ndarray,
    reps: int,
    seed: int,
) -> dict:
    """Point estimate and 95% interval for total saving and saving per review.

    ``delta`` and ``routed`` are (scenarios, alerts) over the shared alert ``rows``;
    ``base`` is the model-only cost per alert. Each replicate draws scenarios and
    alerts with replacement, independently.
    """
    s = delta.shape[0]
    d, r, b = delta[:, rows], routed[:, rows].astype(float), base[rows]
    rng = np.random.default_rng(seed)
    sav, per = np.empty(reps), np.empty(reps)
    chunk = 200
    for start in range(0, reps, chunk):
        n = min(chunk, reps - start)
        w = np.stack(
            [
                np.bincount(rng.integers(0, len(rows), len(rows)), minlength=len(rows))
                for _ in range(n)
            ]
        ).astype(float)
        wd, wr, wb = w @ d.T, w @ r.T, w @ b
        pick = rng.integers(0, s, size=(n, s))
        num = np.take_along_axis(wd, pick, axis=1).sum(axis=1)
        rev = np.take_along_axis(wr, pick, axis=1).sum(axis=1)
        sav[start : start + n] = num / (wb * s)
        per[start : start + n] = num / np.maximum(rev, 1.0)

    def ci(x: np.ndarray) -> list[float]:
        return [float(v) for v in np.quantile(x, [0.025, 0.975])]

    return {
        "saving": float(d.sum() / (b.sum() * s)),
        "saving_ci": ci(sav),
        "per_review": float(d.sum() / max(r.sum(), 1.0)),
        "per_review_ci": ci(per),
        "reviewed_mean": float(r.sum() / s),
        "cost_model_only": float(b.sum()),
    }


def analysts_vs_full_rejection(data, fp_cost: float) -> dict:
    """How many of the fifty analysts, deciding every alert alone, beat blocking all.

    FiFAR's generator constrains each analyst to beat full rejection at lambda_t, so
    this is the mechanism behind any sign change: an analyst who loses to "block
    every alert" can only raise cost when a blocked alert is routed to them.
    """
    y = data.y[:, None]
    ec = fp_cost * ((data.expert == 1) & (y == 0)).mean(0) + (
        (data.expert == 0) & (y == 1)
    ).mean(0)
    full_rejection = fp_cost * float((data.y == 0).mean())
    return {
        "full_rejection_cost_per_alert": full_rejection,
        "analysts_beating_full_rejection": int((ec < full_rejection).sum()),
        # P(legit) x (1 - FPR) x fp_cost - P(fraud) x FNR, averaged over the analysts:
        # the saving per review of routing a blocked alert to a random analyst
        "random_review_vs_blocking_all": full_rejection - float(ec.mean()),
        "analyst_cost_per_alert_min_mean_max": [
            float(ec.min()),
            float(ec.mean()),
            float(ec.max()),
        ],
    }


def evaluate(data, scenarios, rows, fp_cost, threshold, reps, seed) -> dict:
    cfg = TriageConfig(fp_cost=fp_cost, threshold=threshold)
    y, model = data.y, (data.score > threshold).astype(int)
    base = cfg.fn_cost * ((model == 0) & (y == 1)) + cfg.fp_cost * ((model == 1) & (y == 0))
    out = {}
    for pol in policies():
        pairs = [per_alert(data, s, pol, cfg) for s in scenarios]
        delta = np.stack([p[0] for p in pairs])
        routed = np.stack([p[1] for p in pairs])
        res = two_way_bootstrap(delta, base.astype(float), routed, rows, reps, seed)
        res["per_scenario_saving"] = [
            float(delta[i, rows].sum() / base[rows].sum()) for i in range(len(scenarios))
        ]
        res["capacity_used"] = float(routed.sum() / sum(sc.total_capacity for sc in scenarios))
        out[pol.name] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.10, help="review slots / alerts")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", type=pathlib.Path, default=pathlib.Path("results/cost_ratio_sweep.json")
    )
    a = ap.parse_args()

    data = fifar.load()
    train_names = [f"shuffle_{i}#team_{t}" for i in range(1, 6) for t in range(1, 6)]
    train = [
        fifar.load_scenario(data, n, "train_alert", capacity_ratio=a.budget)
        for n in train_names
    ]
    test = [
        fifar.load_scenario(data, f"testsize#team_{t}-hom", "test", capacity_ratio=a.budget)
        for t in range(1, 6)
    ]
    train_rows = np.unique(np.concatenate(train[0].batches))
    test_rows = np.unique(np.concatenate(test[0].batches))
    for sc in train[1:]:
        assert np.array_equal(np.unique(np.concatenate(sc.batches)), train_rows)
    for sc in test[1:]:
        assert np.array_equal(np.unique(np.concatenate(sc.batches)), test_rows)

    print(
        f"FiFAR, review budget {a.budget:.0%} of alerts, fn_cost = 1.0, "
        f"{a.reps} bootstrap replicates (scenarios and alerts resampled together)"
    )
    print(
        f"in-sample: {len(train)} training scenarios over {len(train_rows):,} alerts; "
        f"held-out: {len(test)} teams over {len(test_rows):,} month-8 alerts"
    )
    print("saving = share of the model-only cost that review removed (negative = review")
    print("raised cost); per review = cost avoided per alert an analyst decided\n")

    out: dict = {
        "budget": a.budget,
        "reps": a.reps,
        "seed": a.seed,
        "lambda_t": LAMBDA_T,
        "stress_fp_costs": STRESS_FP_COSTS,
        "train_scenarios": [s.name for s in train],
        "test_scenarios": [s.name for s in test],
        "headline_policy": HEADLINE,
        "grid": [],
    }
    for fp in FP_COSTS:
        th = cost_optimal_threshold(data.score[train_rows], data.y[train_rows], 1.0, fp)
        cell = {
            "fp_cost": fp,
            "ratio": 1.0 / fp,
            "label": LABELS.get(fp, ""),
            "threshold_refit": th,
            "analysts": analysts_vs_full_rejection(data, fp),
            "in_sample": evaluate(data, train, train_rows, fp, th, a.reps, a.seed),
            "in_sample_fixed_threshold": evaluate(
                data, train, train_rows, fp, ALERT_THRESHOLD, a.reps, a.seed
            ),
            "held_out": evaluate(data, test, test_rows, fp, th, a.reps, a.seed),
        }
        out["grid"].append(cell)

        tag = f"  [{cell['label']}]" if cell["label"] else ""
        an = cell["analysts"]
        print(f"=== {1 / fp:5.1f}:1  (fp_cost {fp:.4f}){tag}")
        print(
            f"  refit threshold {th:.4f}; analysts who beat blocking every alert: "
            f"{an['analysts_beating_full_rejection']}/50; a blocked alert sent to a random "
            f"analyst saves {an['random_review_vs_blocking_all']:+.4f}"
        )
        for key, title in (
            ("in_sample", "in-sample, refit threshold"),
            ("held_out", "held-out month 8, threshold from months 4-7"),
        ):
            print(f"  {title}")
            for name, r in cell[key].items():
                print(
                    f"    {name:28} saving {r['saving']:+7.2%} "
                    f"[{r['saving_ci'][0]:+7.2%}, {r['saving_ci'][1]:+7.2%}]"
                    f"   per review {r['per_review']:+.4f} "
                    f"[{r['per_review_ci'][0]:+.4f}, {r['per_review_ci'][1]:+.4f}]"
                    f"   reviews {r['reviewed_mean']:.0f}"
                )
        fx = cell["in_sample_fixed_threshold"][HEADLINE]
        print(
            f"  without the refit (threshold 0.051), {HEADLINE}: saving {fx['saving']:+.2%}"
            f" [{fx['saving_ci'][0]:+.2%}, {fx['saving_ci'][1]:+.2%}]"
            f"   per review {fx['per_review']:+.4f}"
        )
        print()

    print(f"summary, {HEADLINE}, in-sample with the refit threshold:")
    print(f"  {'ratio':>7}  {'saving':>8}  {'95% interval':>20}  {'per review':>10}")
    for cell in out["grid"]:
        r = cell["in_sample"][HEADLINE]
        print(
            f"  {cell['ratio']:6.1f}:1  {r['saving']:+8.2%}  "
            f"[{r['saving_ci'][0]:+7.2%}, {r['saving_ci'][1]:+7.2%}]  "
            f"{r['per_review']:+10.4f}  {cell['label']}".rstrip()
        )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
