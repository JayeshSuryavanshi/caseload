from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

from caseload.evaluation import bootstrap_interval, iqm

pytest.importorskip("matplotlib")

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def figures():
    spec = importlib.util.spec_from_file_location(
        "make_figures", ROOT / "scripts" / "make_figures.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tradeoff_points_use_the_aggregate_table_statistic(figures):
    """Post-break values must equal what scripts/aggregate_fleet.py prints."""
    pts, _, _ = figures.fleet_points()
    files = sorted((ROOT / "results" / "fleet").glob("eval_s*.json"))
    for name, (_, post) in pts.items():
        per_seed = [
            iqm(np.asarray(json.loads(f.read_text())["drift"]["post"][name])) for f in files
        ]
        assert post == pytest.approx(bootstrap_interval(np.asarray(per_seed), reps=8000).point)


def test_both_figures_render_in_both_themes(figures, tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "OUT", tmp_path)
    collapse = json.loads(figures.COLLAPSE.read_text())
    for label, theme in figures.THEMES.items():
        figures.style(theme)
        figures.fig_collapse(f"collapse-{label}.png", theme, collapse)
        figures.fig_tradeoff(f"tradeoff-{label}.png", theme)
    written = sorted(p.name for p in tmp_path.glob("*.png"))
    assert written == sorted(
        f"{f}-{t}.png" for f in ("collapse", "tradeoff") for t in figures.THEMES
    )
    for p in tmp_path.glob("*.png"):
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
