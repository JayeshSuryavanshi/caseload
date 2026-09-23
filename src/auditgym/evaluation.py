"""Aggregation that does not lie about small samples.

Patterson et al. (JMLR 2024) show that 10 runs is not enough to estimate a mean
reliably on even a small environment, and that several common interval methods miss
the true mean at that sample size. Roughly 20 to 30 runs gives trustworthy bootstrap
intervals. The environments here are cheap precisely so that 30 or more seeds is
routine rather than heroic.

Point estimates use the interquartile mean, and intervals come from a stratified
bootstrap over seeds, following Agarwal et al. (NeurIPS 2021). The interquartile mean
is used instead of the mean because a single episode where the detector never found a
second positive, and so never became rankable, otherwise drags the average somewhere
no policy actually lives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    n: int
    method: str = "iqm/stratified-bootstrap"

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}] (n={self.n})"

    @property
    def width(self) -> float:
        return self.hi - self.lo


def iqm(x: np.ndarray) -> float:
    """Interquartile mean: the mean of the middle half."""
    a = np.sort(np.asarray(x, dtype=float))
    if len(a) == 0:
        return float("nan")
    if len(a) < 4:
        return float(a.mean())
    lo, hi = int(np.floor(0.25 * len(a))), int(np.ceil(0.75 * len(a)))
    mid = a[lo:hi]
    return float(mid.mean()) if len(mid) else float(a.mean())


def bootstrap_interval(
    x: np.ndarray,
    statistic=iqm,
    reps: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for ``statistic`` over runs."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    if len(a) == 1:
        v = float(a[0])
        return Interval(v, v, v, 1, "single-run (no interval)")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(reps, len(a)))
    boot = np.array([statistic(a[i]) for i in idx])
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return Interval(statistic(a), float(lo), float(hi), len(a))


def paired_difference(
    a: np.ndarray,
    b: np.ndarray,
    reps: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Interval on ``a - b`` when both were run on the same seeds.

    Pairing matters here. Episode difficulty varies far more than the gap between
    policies, so an unpaired comparison on 30 seeds can hide a real effect entirely.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(x) != len(y):
        raise ValueError(f"paired comparison needs equal lengths, got {len(x)} and {len(y)}")
    d = x - y
    d = d[np.isfinite(d)]
    if len(d) < 2:
        v = float(d[0]) if len(d) else float("nan")
        return Interval(v, v, v, len(d), "single-run (no interval)")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    boot = np.array([iqm(d[i]) for i in idx])
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return Interval(iqm(d), float(lo), float(hi), len(d), "paired iqm/bootstrap")


def probability_of_improvement(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) over independently drawn runs, ties counted as half.

    Reported alongside the difference because a policy can win on average while
    losing most individual episodes, and an operator cares which.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    wins = (x[:, None] > y[None, :]).sum()
    ties = (x[:, None] == y[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(x) * len(y)))


def summarise(results: dict[str, np.ndarray], reps: int = 10_000) -> dict[str, Interval]:
    return {k: bootstrap_interval(v, reps=reps) for k, v in results.items()}


def table(
    results: dict[str, np.ndarray],
    reference: str | None = None,
    reps: int = 10_000,
) -> str:
    """A text table with intervals, and paired deltas against a reference policy."""
    cis = summarise(results, reps=reps)
    w = max((len(k) for k in results), default=8) + 2
    lines = [f"{'policy':<{w}} {'IQM':>7} {'95% CI':>18} {'n':>4}"]
    if reference and reference in results:
        lines[0] += f" {'vs ' + reference:>22} {'P(better)':>10}"
    lines.append("-" * len(lines[0]))
    for k, v in results.items():
        ci = cis[k]
        span = f"[{ci.lo:.3f}, {ci.hi:.3f}]"
        row = f"{k:<{w}} {ci.point:>7.3f} {span:>18} {ci.n:>4}"
        if reference and reference in results and k != reference:
            d = paired_difference(v, results[reference], reps=reps)
            pbi = probability_of_improvement(v, results[reference])
            delta = f"{d.point:+.3f} [{d.lo:+.3f}, {d.hi:+.3f}]"
            row += f" {delta:>22} {pbi:>10.2f}"
        elif reference and k == reference:
            row += f" {'(reference)':>22} {'':>10}"
        lines.append(row)
    return "\n".join(lines)
