from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "collapse.json"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def result() -> dict:
    return json.loads(RESULT.read_text())


def test_committed_result_is_internally_consistent(result):
    """The headline counts are sums of the per-step rows, not separately typed."""
    rows = result["per_step"]
    cfg = result["config"]
    assert [r["t"] for r in rows] == list(range(cfg["warm_until"] + 1, rows[-1]["t"] + 1))
    for r in rows:
        assert r["recall"] == pytest.approx(r["found"] / max(r["illicit"], 1))
        assert r["found"] <= min(r["illicit"], r["budget"])
    after = [r for r in rows if r["t"] >= cfg["break_time"]]
    pb = result["post_break"]
    assert pb["found"] == sum(r["found"] for r in after)
    assert pb["illicit"] == sum(r["illicit"] for r in after)
    assert pb["steps_at_zero"] == sum(r["found"] == 0 for r in after)
    for row in result["recovery"]:
        for k in ("frozen", "refit", "random"):
            assert row[f"{k}_recall"] == pytest.approx(row[f"{k}_found"] / row["illicit"])


def test_converter_keeps_file_order_and_ids(tmp_path):
    """Two people converting the same CSVs must get the same archive, row for row."""
    fetch = _script("fetch_elliptic")
    src = tmp_path / "csv"
    src.mkdir()
    # txIds deliberately out of numeric and string order
    (src / "elliptic_txs_features.csv").write_text(
        "30,1,0.1,-0.18466755143291433\n4,1,0.2,0.5\n200,2,0.3,0.25\n"
    )
    (src / "elliptic_txs_classes.csv").write_text("txId,class\n4,1\n30,unknown\n200,2\n")
    (src / "elliptic_txs_edgelist.csv").write_text("txId1,txId2\n30,4\n4,200\n9,30\n")
    out = tmp_path / "out" / "elliptic_parsed.npz"
    fetch.convert(src, out)
    d = np.load(out)
    assert d["txid"].tolist() == [30, 4, 200]
    assert d["node_time"].tolist() == [1, 1, 2]
    assert d["labels"].tolist() == [-1, 1, 0]
    assert d["x"][0, 1] == float("-0.18466755143291433")
    # edges to transactions not in the feature file are dropped
    assert list(zip(d["src"].tolist(), d["dst"].tolist(), strict=False)) == [(0, 1), (1, 2)]


def test_rerun_matches_committed_result(result, tmp_path, monkeypatch):
    """Slow (about half a minute). Runs only against the archive the result came from."""
    from caseload.envs.elliptic import find_archive, load_raw

    try:
        find_archive()
    except FileNotFoundError:
        pytest.skip("Elliptic not converted; run scripts/fetch_elliptic.py")
    measure = _script("measure_collapse")
    raw = load_raw()
    sha = measure.fingerprint(
        raw["x"], raw["node_time"].astype(np.int64), raw["labels"].astype(np.int64)
    )
    if sha != result["data"]["sha256"]:
        pytest.skip("local archive differs from the one results/collapse.json was made from")
    monkeypatch.setattr(sys, "argv", ["measure_collapse.py", "--out", str(tmp_path)])
    measure.main()
    fresh = json.loads((tmp_path / "collapse.json").read_text())
    for k in ("per_step", "post_break", "recovery", "frozen_detector", "data"):
        assert fresh[k] == result[k]
