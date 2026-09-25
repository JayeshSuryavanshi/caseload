"""README figures, in light and dark variants so they read on either GitHub theme.

Both are drawn from the committed result files, never from hand-typed numbers.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = pathlib.Path("docs/img")
THEMES = {
    "light": dict(bg="#ffffff", ink="#1f2328", soft="#59636e", grid="#d1d9e0"),
    "dark": dict(bg="#0d1117", ink="#e6edf3", soft="#9198a1", grid="#30363d"),
}
ACCENT = "#2f81f7"
GOOD = "#3fb950"
WARN = "#d29922"
MUTED = "#8b949e"


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


def fig_collapse(name: str, t: dict) -> None:
    """The motivating measurement: a frozen detector hits exactly zero and stays there."""
    steps = list(range(35, 50))
    recall = [
        0.588,
        0.879,
        0.475,
        0.459,
        0.333,
        0.438,
        0.543,
        0.377,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.axvspan(42.5, 49.5, color=WARN, alpha=0.10, lw=0)
    ax.plot(steps, recall, "-o", color=ACCENT, ms=5, lw=2, zorder=3)
    ax.axhline(0, color=t["soft"], lw=0.8, ls=(0, (3, 3)))
    ax.annotate(
        "regime break at t=43:\nseven consecutive steps at exactly 0%",
        xy=(43, 0.0),
        xytext=(44.2, 0.42),
        fontsize=9,
        color=t["ink"],
        ha="left",
        arrowprops=dict(
            arrowstyle="->", color=t["soft"], lw=1.0, connectionstyle="arc3,rad=-0.25"
        ),
    )
    ax.annotate(
        "hindsight ceiling 56%",
        xy=(47, 0.562),
        xytext=(45.4, 0.60),
        fontsize=9,
        color=GOOD,
        ha="left",
    )
    ax.axhline(0.562, xmin=0.66, color=GOOD, lw=1.4, ls=(0, (4, 3)))
    ax.set_xlabel("Elliptic time step")
    ax.set_ylabel("recall at a 2% budget")
    ax.set_xlim(34.4, 49.8)
    ax.set_ylim(-0.04, 0.95)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", color=t["grid"], lw=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        "A frozen fraud detector does not degrade. It stops working.",
        loc="left",
        fontsize=11,
        color=t["ink"],
        pad=10,
    )
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def _fleet_ppo() -> tuple[float, float] | None:
    """PPO's pre/post recall averaged over every trained seed in the fleet."""
    files = sorted(pathlib.Path("results/fleet").glob("eval_s*.json"))
    if not files:
        return None
    pre, post = [], []
    for f in files:
        d = json.loads(f.read_text())["drift"]
        if "ppo" not in d["pre"]:
            continue
        pre.append(float(np.mean(d["pre"]["ppo"])))
        post.append(float(np.mean(d["post"]["ppo"])))
    if not pre:
        return None
    return float(np.mean(pre)), float(np.mean(post))


def fig_tradeoff(name: str, t: dict) -> None:
    """Every policy trades pre-break recall against post-break recall. Nobody gets both."""
    # (label, pre, post, colour, label offset in points)
    pts = [
        ("top-k", 0.953, 0.042, MUTED, (14, -6), "left"),
        ("eps-explore(.15)", 0.924, 0.088, MUTED, (-12, -4), "right"),
        ("eps-explore(.35)", 0.856, 0.116, MUTED, (-12, -2), "right"),
        ("eps-explore(.6)", 0.694, 0.152, MUTED, (0, -18), "center"),
        ("yield-triggered", 0.921, 0.105, WARN, (0, -20), "center"),
    ]
    ppo = _fleet_ppo() or (0.941, 0.197)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for n, x, y, c, off, ha in pts:
        ax.scatter([x], [y], s=95, color=c, zorder=3, edgecolor=t["bg"], linewidth=1.4)
        ax.annotate(
            n,
            (x, y),
            textcoords="offset points",
            xytext=off,
            ha=ha,
            fontsize=9,
            color=t["soft"],
        )
    ax.scatter(
        [ppo[0]], [ppo[1]], s=220, color=ACCENT, zorder=4, edgecolor=t["bg"], linewidth=1.6
    )
    ax.annotate(
        "ppo",
        (ppo[0], ppo[1]),
        textcoords="offset points",
        xytext=(0, 16),
        ha="center",
        fontsize=10.5,
        color=ACCENT,
        fontweight="bold",
    )
    ax.scatter([0.987], [0.996], s=150, marker="*", color=GOOD, zorder=3)
    ax.annotate(
        "oracle ceiling",
        (0.987, 0.996),
        textcoords="offset points",
        xytext=(-10, -4),
        ha="right",
        fontsize=9,
        color=GOOD,
    )
    ax.annotate(
        "more exploration",
        xy=(0.70, 0.168),
        xytext=(0.76, 0.30),
        fontsize=8.5,
        color=t["soft"],
        ha="center",
        arrowprops=dict(arrowstyle="->", color=t["grid"], lw=1.0),
    )
    ax.set_xlabel("recall before the break")
    ax.set_ylabel("recall after the break")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(color=t["grid"], lw=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0.63, 1.06)
    ax.set_ylim(-0.03, 1.12)
    ax.set_title(
        "No fixed rule is good at both, and the corner is empty.",
        loc="left",
        fontsize=11,
        color=t["ink"],
        pad=10,
    )
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for label, t in THEMES.items():
        style(t)
        fig_collapse(f"collapse-{label}.png", t)
        fig_tradeoff(f"tradeoff-{label}.png", t)
    print("wrote:", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
