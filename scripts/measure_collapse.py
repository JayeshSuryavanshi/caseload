"""Reproduce the collapse tables in the README, from the Elliptic data.

Every number the README quotes about Elliptic comes from this script. Run it and
compare, and if the data on your machine differs, the tables will say so rather
than the README quietly being wrong.

    python scripts/measure_collapse.py
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from auditgym.envs.elliptic import BREAK_TIME, load_raw


def top_k_recall(scores: np.ndarray, y: np.ndarray, budget_frac: float) -> tuple[int, int, int]:
    b = max(1, int(round(budget_frac * len(y))))
    pick = np.argsort(-scores)[:b]
    return int((y[pick] == 1).sum()), int((y == 1).sum()), b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path", default=None, help="directory or file holding elliptic_parsed.npz"
    )
    ap.add_argument("--warm-until", type=int, default=34)
    ap.add_argument("--max-iter", type=int, default=300)
    a = ap.parse_args()

    raw = load_raw(a.path)
    x, t, y = raw["x"], raw["node_time"].astype(int), raw["labels"].astype(int)
    lab = y >= 0
    print(
        f"elliptic: {len(y):,} transactions, {len(np.unique(t))} time steps, "
        f"{lab.sum():,} labelled ({lab.mean():.1%}), "
        f"illicit share of labelled {y[lab].mean():.1%}"
    )

    tr = lab & (t <= a.warm_until)
    te = lab & (t > a.warm_until)
    frozen = HistGradientBoostingClassifier(
        max_iter=a.max_iter, random_state=0, early_stopping=False
    ).fit(x[tr], y[tr])
    p = frozen.predict_proba(x[te])[:, 1]
    print(f"\ndetector frozen on t<={a.warm_until} ({tr.sum():,} labelled rows)")
    print(
        f"  held-out AUPRC {average_precision_score(y[te], p):.4f}"
        f"  AUROC {roc_auc_score(y[te], p):.4f}"
    )

    print(f"\nper-step recall at a 2% budget, frozen detector (break at t={BREAK_TIME})")
    print(f"  {'t':>3} {'labelled':>9} {'illicit':>8} {'recall':>8} {'precision':>10}")
    for ts in sorted(int(s) for s in np.unique(t[t > a.warm_until])):
        idx = np.where(t == ts)[0]
        if len(idx) == 0:
            continue
        f, av, b = top_k_recall(frozen.predict_proba(x[idx])[:, 1], y[idx], 0.02)
        mark = "   <-- break" if ts == BREAK_TIME else ""
        print(
            f"  {ts:>3} {int((y[idx] >= 0).sum()):>9} {av:>8} "
            f"{f / max(av, 1):>7.1%} {f / max(b, 1):>9.1%}{mark}"
        )

    print("\nis post-break fraud recoverable? recall on t=45..49 by budget")
    post = lab & (t <= 44)
    refit = HistGradientBoostingClassifier(
        max_iter=a.max_iter, random_state=0, early_stopping=False
    ).fit(x[post], y[post])
    rng = np.random.default_rng(0)
    print(f"  {'budget':>7} {'frozen t<=34':>13} {'refit t<=44':>12} {'random':>9}")
    for bf in (0.02, 0.05, 0.10, 0.20):
        out = []
        for scorer in ("frozen", "refit", "random"):
            fo = av = 0
            for ts in range(45, int(t.max()) + 1):
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
            out.append(fo / max(av, 1))
        print(f"  {bf:>6.0%} {out[0]:>13.1%} {out[1]:>12.1%} {out[2]:>9.1%}")
    print("\nThe gap between the first and second columns is what a policy has to close,")
    print("and it can only be closed by investigating cases the frozen detector ranks low.")


if __name__ == "__main__":
    main()
