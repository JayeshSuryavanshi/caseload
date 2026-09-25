"""Reproduce the Elliptic collapse numbers in the README, and write them down.

Every number the README quotes about the Elliptic break comes from this script, and
so does the hero figure: ``scripts/make_figures.py`` reads ``results/collapse.json``.
The run is deterministic, and the JSON records a fingerprint of the archive it read,
so if your copy of the data differs the fingerprint will say so rather than the
README quietly being wrong.

The break at t=43 is the dark-market shutdown documented by Weber et al. (KDD '19
Workshop on Anomaly Detection in Finance, arXiv:1908.02591). This measures what it
does to a detector frozen before it, at an investigation budget.

    python scripts/measure_collapse.py      # writes results/collapse.{log,json}
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import time

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from caseload.envs.elliptic import BREAK_TIME, find_archive, fingerprint, load_raw

PER_STEP_BUDGET = 0.02
REFIT_BUDGETS = (0.02, 0.05, 0.10, 0.20)


def top_k_recall(scores: np.ndarray, y: np.ndarray, budget_frac: float) -> tuple[int, int, int]:
    b = max(1, int(round(budget_frac * len(y))))
    # stable, so ties are broken by row order and the pick is the same everywhere
    pick = np.argsort(-scores, kind="stable")[:b]
    return int((y[pick] == 1).sum()), int((y == 1).sum()), b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path", default=None, help="directory or file holding elliptic_parsed.npz"
    )
    ap.add_argument("--warm-until", type=int, default=34)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--out", default="results", help="directory for collapse.{log,json}")
    ap.add_argument(
        "--force", action="store_true", help="overwrite results made from a different archive"
    )
    a = ap.parse_args()

    start = time.perf_counter()
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    raw = load_raw(a.path)
    x = raw["x"]
    t = raw["node_time"].astype(np.int64)
    y = raw["labels"].astype(np.int64)
    lab = y >= 0
    data = {
        "archive": find_archive(a.path).name,
        "sha256": fingerprint(x, t, y),
        "transactions": int(len(y)),
        "features": int(x.shape[1]),
        "time_steps": int(len(np.unique(t))),
        "labelled": int(lab.sum()),
        "illicit": int((y == 1).sum()),
    }
    say(
        f"elliptic: {len(y):,} transactions, {len(np.unique(t))} time steps, "
        f"{lab.sum():,} labelled ({lab.mean():.1%}), "
        f"illicit share of labelled {y[lab].mean():.1%}"
    )
    say(f"archive sha256 {data['sha256']}")
    prev = pathlib.Path(a.out) / "collapse.json"
    if prev.exists():
        was = json.loads(prev.read_text()).get("data", {}).get("sha256")
        if was and was != data["sha256"]:
            # a different float parser changes the last bit of some features, and the
            # post-break counts are small enough that this can move them
            print(f"  {prev} was made from a different archive ({was[:12]}...).")
            if not a.force:
                raise SystemExit("  refusing to overwrite it; pass --out DIR or --force")

    tr = lab & (t <= a.warm_until)
    te = lab & (t > a.warm_until)
    frozen = HistGradientBoostingClassifier(
        max_iter=a.max_iter, random_state=0, early_stopping=False
    ).fit(x[tr], y[tr])
    p = frozen.predict_proba(x[te])[:, 1]
    auprc = float(average_precision_score(y[te], p))
    auroc = float(roc_auc_score(y[te], p))
    say(f"\ndetector frozen on t<={a.warm_until} ({tr.sum():,} labelled rows)")
    say(f"  held-out AUPRC {auprc:.4f}  AUROC {auroc:.4f}")

    say(
        f"\nper-step recall at a {PER_STEP_BUDGET:.0%} budget, frozen detector "
        f"(break at t={BREAK_TIME})"
    )
    say(
        f"  {'t':>3} {'labelled':>9} {'illicit':>8} {'found':>6} {'budget':>7} "
        f"{'recall':>8} {'precision':>10}"
    )
    per_step = []
    for ts in sorted(int(s) for s in np.unique(t[t > a.warm_until])):
        idx = np.where(t == ts)[0]
        f, av, b = top_k_recall(frozen.predict_proba(x[idx])[:, 1], y[idx], PER_STEP_BUDGET)
        per_step.append(
            {
                "t": ts,
                "labelled": int((y[idx] >= 0).sum()),
                "illicit": av,
                "found": f,
                "budget": b,
                "recall": f / max(av, 1),
                "precision": f / max(b, 1),
            }
        )
        mark = "   <-- break" if ts == BREAK_TIME else ""
        say(
            f"  {ts:>3} {per_step[-1]['labelled']:>9} {av:>8} {f:>6} {b:>7} "
            f"{f / max(av, 1):>7.1%} {f / max(b, 1):>9.1%}{mark}"
        )

    before = [r for r in per_step if r["t"] < BREAK_TIME]
    after = [r for r in per_step if r["t"] >= BREAK_TIME]
    found_after = sum(r["found"] for r in after)
    illicit_after = sum(r["illicit"] for r in after)
    say(
        f"\n  t={before[0]['t']}..{before[-1]['t']}: per-step recall "
        f"{min(r['recall'] for r in before):.1%} to {max(r['recall'] for r in before):.1%}"
    )
    say(
        f"  t={after[0]['t']}..{after[-1]['t']}: {found_after} of {illicit_after} "
        f"illicit found ({found_after / max(illicit_after, 1):.1%}), "
        f"{sum(r['found'] == 0 for r in after)} of {len(after)} steps at zero"
    )

    first_eval = BREAK_TIME + 2
    say(
        f"\nis post-break fraud recoverable? recall on t={first_eval}..{int(t.max())} by budget"
    )
    post = lab & (t < first_eval)
    refit = HistGradientBoostingClassifier(
        max_iter=a.max_iter, random_state=0, early_stopping=False
    ).fit(x[post], y[post])
    rng = np.random.default_rng(0)
    say(
        f"  {'budget':>7} {'frozen t<=' + str(a.warm_until):>13} "
        f"{'refit t<=' + str(first_eval - 1):>12} {'random':>9}"
    )
    recovery = []
    for bf in REFIT_BUDGETS:
        row: dict[str, float | int] = {"budget_frac": bf}
        for scorer in ("frozen", "refit", "random"):
            fo = av = 0
            for ts in range(first_eval, int(t.max()) + 1):
                idx = np.where(t == ts)[0]
                if len(idx) == 0:
                    continue
                if scorer == "random":
                    s = rng.random(len(idx))
                else:
                    m = frozen if scorer == "frozen" else refit
                    s = m.predict_proba(x[idx])[:, 1]
                f, av_i, _ = top_k_recall(s, y[idx], bf)
                fo += f
                av += av_i
            row[f"{scorer}_found"] = fo
            row[f"{scorer}_recall"] = fo / max(av, 1)
            row["illicit"] = av
        recovery.append(row)
        say(
            f"  {bf:>6.0%} {row['frozen_recall']:>13.1%} {row['refit_recall']:>12.1%} "
            f"{row['random_recall']:>9.1%}"
        )
    say("\nThe gap between the first and second columns is what a policy has to close,")
    say("and it can only be closed by investigating cases the frozen detector ranks low.")

    wall = time.perf_counter() - start
    say(f"\nwall time {wall:.1f} s")

    result = {
        "command": "python scripts/measure_collapse.py"
        + (f" --warm-until {a.warm_until}" if a.warm_until != 34 else "")
        + (f" --max-iter {a.max_iter}" if a.max_iter != 300 else ""),
        "config": {
            "warm_until": a.warm_until,
            "max_iter": a.max_iter,
            "break_time": BREAK_TIME,
            "per_step_budget": PER_STEP_BUDGET,
            "refit_on_t_upto": first_eval - 1,
            "recovery_window": [first_eval, int(t.max())],
        },
        "data": data,
        "frozen_detector": {"heldout_auprc": auprc, "heldout_auroc": auroc},
        "per_step": per_step,
        "post_break": {
            "steps": [after[0]["t"], after[-1]["t"]],
            "found": found_after,
            "illicit": illicit_after,
            "steps_at_zero": sum(r["found"] == 0 for r in after),
        },
        "recovery": recovery,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "wall_seconds": round(wall, 1),
    }
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "collapse.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "collapse.log").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'collapse.log'} and {out / 'collapse.json'}")


if __name__ == "__main__":
    main()
