"""Check for the Elliptic dataset, and explain how to get it if it is missing.

The Elliptic Data Set was released with Weber et al., "Anti-Money Laundering in
Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics",
KDD '19 Workshop on Anomaly Detection in Finance (arXiv:1908.02591). Cite that paper
if you use it.

It is published on Kaggle under CC BY-NC-ND 4.0: non-commercial use only, and no
redistribution of modified versions. This package does not ship it. Download it
from Kaggle yourself, then let this script convert the three CSVs into the single
archive the loader expects. The converted archive is a derivative of the dataset, so
keep it on your machine and do not share it.

The converter keeps rows in the order of ``elliptic_txs_features.csv``, stores each
row's ``txId``, and parses numbers with Python's correctly rounded ``float``, so
everyone who converts the published files gets a bit-identical archive and the same
numbers. ``scripts/measure_collapse.py`` prints the archive's sha256 to check against
``results/collapse.json``.

    python scripts/fetch_elliptic.py              # check
    python scripts/fetch_elliptic.py --convert DIR  # build the archive from CSVs
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

from caseload.envs.elliptic import ARCHIVE, CANONICAL_DIR, find_archive

SOURCE = "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set"
CSVS = (
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
)
TARGET = CANONICAL_DIR / ARCHIVE


def check() -> bool:
    try:
        p = find_archive()
    except FileNotFoundError:
        return False
    d = np.load(p, allow_pickle=True)
    print(f"found {p}")
    print(f"  arrays: {', '.join(d.files)}")
    x, t, y = d["x"], d["node_time"].astype(int), d["labels"].astype(int)
    print(f"  {len(y):,} transactions, {x.shape[1]} features, {len(np.unique(t))} time steps")
    lab, bad = int((y >= 0).sum()), int((y == 1).sum())
    print(f"  labelled {lab:,} ({(y >= 0).mean():.1%}), illicit {bad:,}")
    return True


def convert(src: pathlib.Path, out: pathlib.Path) -> None:
    import csv

    missing = [c for c in CSVS if not (src / c).exists()]
    if missing:
        sys.exit(f"missing in {src}: {', '.join(missing)}")

    ids: list[str] = []
    feats: list[list[float]] = []
    times: list[int] = []
    with open(src / CSVS[0], newline="") as f:
        # no header row; columns are txId, time step, then 165 features
        for row in csv.reader(f):
            ids.append(row[0])
            times.append(int(float(row[1])))
            feats.append([float(v) for v in row[2:]])
    index = {t: i for i, t in enumerate(ids)}
    if len(index) != len(ids):
        sys.exit(f"{CSVS[0]} has duplicate txIds")

    labels = np.full(len(ids), -1, dtype=np.int64)
    with open(src / CSVS[1], newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            # the published encoding is "1" illicit, "2" licit, "unknown" unlabelled
            if row[0] in index:
                labels[index[row[0]]] = {"1": 1, "2": 0}.get(row[1], -1)

    src_i, dst_i = [], []
    with open(src / CSVS[2], newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if row[0] in index and row[1] in index:
                src_i.append(index[row[0]])
                dst_i.append(index[row[1]])

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        x=np.asarray(feats, dtype=np.float64),
        node_time=np.asarray(times, dtype=np.float64),
        labels=labels,
        src=np.asarray(src_i, dtype=np.int64),
        dst=np.asarray(dst_i, dtype=np.int64),
        txid=np.asarray([int(t) for t in ids], dtype=np.int64),
    )
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--convert",
        metavar="DIR",
        default=None,
        help="directory holding the three Elliptic CSVs",
    )
    ap.add_argument(
        "--out", default=str(TARGET), help=f"where to write the archive (default {TARGET})"
    )
    a = ap.parse_args()
    if a.convert:
        convert(pathlib.Path(a.convert).expanduser(), pathlib.Path(a.out).expanduser())
        return
    if check():
        return
    print("Elliptic dataset not found.\n")
    print(f"1. download it from {SOURCE} (CC BY-NC-ND 4.0, non-commercial)")
    print(
        "2. unzip it, then run:\n     python scripts/fetch_elliptic.py --convert <unzipped dir>"
    )
    print(f"\nThe loader looks for {TARGET}")
    print(
        "The drift simulator needs no download, so the rest of the package works without this."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
