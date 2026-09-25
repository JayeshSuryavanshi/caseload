"""README figures, in light and dark variants so they read on either GitHub theme.

Every plotted value is read from a committed result file; nothing here is a number
typed by hand. The collapse figure reads ``results/collapse.json`` (written by
``scripts/measure_collapse.py``). The tradeoff figure reads
``results/fleet/eval_s*.json`` (written by ``scripts/evaluate_fleet.sh``) and uses the
same statistic as ``scripts/aggregate_fleet.py``: the IQM over each seed's evaluation
episodes, then the IQM over training seeds.

    python scripts/make_figures.py      # writes docs/img/{collapse,tradeoff}-{light,dark}.png

The tradeoff figure's plotted points are also written to ``results/tradeoff_points.json``
and printed, so the numbers quoted about the figure come from a file.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from caseload.evaluation import bootstrap_interval, iqm  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"
COLLAPSE = ROOT / "results" / "collapse.json"
FLEET = ROOT / "results" / "fleet"
POINTS = ROOT / "results" / "tradeoff_points.json"
THEMES = {
    "light": dict(bg="#ffffff", ink="#1f2328", soft="#59636e", grid="#d1d9e0"),
    "dark": dict(bg="#0d1117", ink="#e6edf3", soft="#9198a1", grid="#30363d"),
}
ACCENT = "#2f81f7"
GOOD = "#3fb950"
WARN = "#d29922"
MUTED = "#8b949e"
# matplotlib embeds the build time otherwise, so identical inputs give identical files
SAVE = dict(dpi=200, bbox_inches="tight", pad_inches=0.18, metadata={"Software": None})


def style(t: dict) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": t["bg"],
            "axes.facecolor": t["bg"],
            "savefig.facecolor": t["bg"],
            "text.color": t["ink"],
            "axes.labelcolor": t["ink"],
            "axes.edgecolor": t["soft"],
            "xtick.color": t["soft"],
            "ytick.color": t["soft"],
            "xtick.labelcolor": t["ink"],
            "ytick.labelcolor": t["ink"],
            "font.size": 10,
            "axes.linewidth": 0.8,
        }
    )


def _pct(ax) -> None:
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")


def fig_collapse(name: str, t: dict, res: dict) -> None:
    """Per-step recall of the frozen detector, and what refitting after the break recovers."""
    cfg = res["config"]
    rows = res["per_step"]
    brk = cfg["break_time"]
    steps = [r["t"] for r in rows]
    recall = [r["recall"] for r in rows]
    pb = res["post_break"]

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(9.6, 3.6), gridspec_kw=dict(width_ratios=[2.3, 1], wspace=0.28)
    )
    ax.axvspan(brk - 0.5, steps[-1] + 0.5, color=WARN, alpha=0.10, lw=0)
    ax.plot(steps, recall, "-o", color=ACCENT, ms=5, lw=2, zorder=3)
    for r in rows:
        if r["t"] >= brk:
            # found/illicit under each post-break point, above it where it peaks
            up = r["recall"] > 0.1
            ax.annotate(
                f"{r['found']}/{r['illicit']}",
                (r["t"], r["recall"]),
                textcoords="offset points",
                xytext=(0, 9 if up else -10),
                ha="center",
                va="bottom" if up else "top",
                fontsize=7.5,
                color=t["soft"],
            )
    ax.annotate(
        f"dark-market shutdown at t={brk}\n{pb['found']} of {pb['illicit']} illicit found "
        f"over t={pb['steps'][0]}-{pb['steps'][1]}",
        xy=(brk - 0.3, 0.93),
        fontsize=9,
        color=t["ink"],
        ha="left",
        va="top",
    )
    ax.set_xlabel("Elliptic time step")
    ax.set_ylabel(f"recall at a {cfg['per_step_budget']:.0%} budget")
    ax.set_xlim(steps[0] - 0.6, steps[-1] + 0.8)
    ax.set_ylim(-0.1, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    _pct(ax)
    ax.set_title(f"Detector frozen on t≤{cfg['warm_until']}", loc="left", fontsize=10, pad=8)

    rec = res["recovery"]
    budgets = [r["budget_frac"] for r in rec]
    lo, hi = cfg["recovery_window"]
    for key, label, colour, lw in (
        ("refit", f"refit on t≤{cfg['refit_on_t_upto']}", GOOD, 2),
        ("frozen", f"frozen on t≤{cfg['warm_until']}", ACCENT, 2),
        ("random", "random", MUTED, 1.2),
    ):
        ys = [r[f"{key}_recall"] for r in rec]
        bx.plot(budgets, ys, "-o", color=colour, ms=4, lw=lw, zorder=3)
        bx.annotate(
            label,
            (budgets[-1], ys[-1]),
            textcoords="offset points",
            xytext=(-4, -13 if key == "frozen" else 7),
            ha="right",
            fontsize=8.5,
            color=colour,
        )
    bx.set_xticks(budgets)
    bx.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    bx.set_xlabel("budget")
    bx.set_ylabel(f"recall on t={lo}-{hi}")
    bx.set_ylim(-0.1, 1.0)
    bx.set_yticks(np.arange(0, 1.01, 0.2))
    _pct(bx)
    bx.set_title("Is it recoverable?", loc="left", fontsize=10, pad=8)

    for a in (ax, bx):
        a.grid(axis="y", color=t["grid"], lw=0.7, alpha=0.7)
        a.set_axisbelow(True)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "A detector frozen before the break finds almost nothing after it.",
        x=0.07,
        ha="left",
        fontsize=11,
        color=t["ink"],
        y=1.04,
    )
    fig.savefig(OUT / name, **SAVE)
    plt.close(fig)


def fleet_points() -> tuple[
    dict[str, tuple[float, float]], dict[str, list[float]], dict[str, list[float]]
]:
    """Pre- and post-break recall per policy: IQM over episodes, then IQM over seeds.

    Also returns the per-seed values, from which the interval on ``ppo`` is drawn.
    """
    files = sorted(FLEET.glob("eval_s*.json"))
    if not files:
        raise SystemExit(f"no eval_s*.json under {FLEET}; run scripts/evaluate_fleet.sh")
    pre: dict[str, list[float]] = {}
    post: dict[str, list[float]] = {}
    for f in files:
        d = json.loads(f.read_text())["drift"]
        for n, v in d["pre"].items():
            pre.setdefault(n, []).append(iqm(np.asarray(v)))
        for n, v in d["post"].items():
            post.setdefault(n, []).append(iqm(np.asarray(v)))
    return {n: (iqm(np.asarray(pre[n])), iqm(np.asarray(post[n]))) for n in pre}, pre, post


def write_points() -> None:
    pts, pre, post = fleet_points()
    rows: dict[str, dict] = {}
    print(
        f"{'policy':28} {'pre-break':>10} {'post-break':>11}  (IQM over episodes, then seeds)"
    )
    for n, (x, y) in pts.items():
        row: dict = {"pre": x, "post": y, "seeds": len(pre[n])}
        extra = ""
        if n == "ppo":
            cx = bootstrap_interval(np.asarray(pre[n]), reps=8000)
            cy = bootstrap_interval(np.asarray(post[n]), reps=8000)
            row["pre_ci"], row["post_ci"] = [cx.lo, cx.hi], [cy.lo, cy.hi]
            extra = (
                f"  seed interval pre [{cx.lo:.3f}, {cx.hi:.3f}], "
                f"post [{cy.lo:.3f}, {cy.hi:.3f}]"
            )
        rows[n] = row
        print(f"{n:28} {x:>10.3f} {y:>11.3f}{extra}")
    POINTS.write_text(json.dumps(rows, indent=1) + "\n")


# where each label sits relative to its point, in points; the right-hand cluster is
# too tight for labels beside the markers, so those get a leader line
LABELS = {
    "top-k": ("top-k", (34, -14), "left"),
    "eps-explore(0.15)": ("eps-explore(.15)", (40, -4), "left"),
    "eps-explore(0.35)": ("eps-explore(.35)", (0, -16), "center"),
    "eps-explore(0.6)": ("eps-explore(.6)", (0, -16), "center"),
    "yield-triggered(drop=0.5)": ("yield-triggered", (42, 12), "left"),
}


def fig_tradeoff(name: str, t: dict) -> None:
    """Every fixed rule trades pre-break recall against post-break recall."""
    pts, pre, post = fleet_points()
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for n, (x, y) in pts.items():
        if n == "ppo":
            continue
        label, off, ha = LABELS.get(n, (n, (0, 10), "center"))
        c = WARN if n.startswith("yield") else MUTED
        ax.scatter([x], [y], s=95, color=c, zorder=3, edgecolor=t["bg"], linewidth=1.4)
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=off,
            ha=ha,
            va="center",
            fontsize=9,
            color=t["soft"],
            arrowprops=dict(arrowstyle="-", color=t["grid"], lw=0.8, shrinkA=2, shrinkB=6)
            if abs(off[0]) > 20
            else None,
        )
    if "ppo" in pts:
        x, y = pts["ppo"]
        # the interval is over training seeds: the baselines do not depend on the seed
        cx = bootstrap_interval(np.asarray(pre["ppo"]), reps=8000)
        cy = bootstrap_interval(np.asarray(post["ppo"]), reps=8000)
        ax.errorbar(
            [x],
            [y],
            xerr=[[x - cx.lo], [cx.hi - x]],
            yerr=[[y - cy.lo], [cy.hi - y]],
            fmt="none",
            ecolor=ACCENT,
            elinewidth=1.2,
            capsize=3,
            zorder=3,
        )
        ax.scatter([x], [y], s=220, color=ACCENT, zorder=4, edgecolor=t["bg"], linewidth=1.6)
        ax.annotate(
            f"ppo ({len(pre['ppo'])} seeds)",
            (x, y),
            textcoords="offset points",
            xytext=(0, 18),
            ha="center",
            fontsize=10.5,
            color=ACCENT,
            fontweight="bold",
        )
    eps = sorted(
        (float(n[len("eps-explore(") : -1]), n) for n in pts if n.startswith("eps-explore(")
    )
    if len(eps) >= 2:
        (x0, y0), (x1, y1) = pts[eps[0][1]], pts[eps[-1][1]]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                color=t["grid"],
                lw=1.0,
                connectionstyle="arc3,rad=0.2",
                shrinkA=14,
                shrinkB=14,
            ),
        )
        ax.annotate(
            "more exploration",
            ((x0 + x1) / 2, (y0 + y1) / 2),
            textcoords="offset points",
            xytext=(0, 26),
            ha="center",
            fontsize=8.5,
            color=t["soft"],
        )
    ax.set_xlabel("recall before the break")
    ax.set_ylabel("recall after the break")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _pct(ax)
    ax.grid(color=t["grid"], lw=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0.6, 1.13)
    ax.set_xticks(np.arange(0.6, 1.001, 0.05))
    ax.set_ylim(-0.03, 1.02)
    ax.set_title(
        "No fixed rule is good at both, and the corner is empty.",
        loc="left",
        fontsize=11,
        color=t["ink"],
        pad=10,
    )
    fig.savefig(OUT / name, **SAVE)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    collapse = json.loads(COLLAPSE.read_text())
    for label, t in THEMES.items():
        style(t)
        fig_collapse(f"collapse-{label}.png", t, collapse)
        fig_tradeoff(f"tradeoff-{label}.png", t)
    write_points()
    print("wrote:", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
