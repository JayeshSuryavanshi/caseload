"""Check for the Elliptic dataset, and explain how to get it if it is missing.

The dataset is not redistributed with this package. It is published on Kaggle under
terms that do not permit mirroring, so this script tells you where to download it
and converts the CSVs into the single archive the loader expects.

    python scripts/fetch_elliptic.py              # check
    python scripts/fetch_elliptic.py --convert DIR  # build the archive from CSVs
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

SOURCE = "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set"
CSVS = (
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
)
TARGET = pathlib.Path.home() / ".cache" / "caseload" / "elliptic_parsed.npz"
ALSO = pathlib.Path.home() / ".cache" / "graphspot" / "elliptic_parsed.npz"


def check() -> bool:
    for p in (TARGET, ALSO):
        if p.exists():
            d = np.load(p, allow_pickle=True)
            print(f"found {p}")
            print(f"  arrays: {', '.join(d.files)}")
            x, t, y = d["x"], d["node_time"].astype(int), d["labels"].astype(int)
            print(
                f"  {len(y):,} transactions, {x.shape[1]} features, "
                f"{len(np.unique(t))} time steps"
            )
            print(
                f"  labelled {int((y >= 0).sum()):,} ({(y >= 0).mean():.1%}), "
                f"illicit {int((y == 1).sum()):,}"
            )
            return True
    return False


def convert(src: pathlib.Path) -> None:
    import csv

    missing = [c for c in CSVS if not (src / c).exists()]
    if missing:
        sys.exit(f"missing in {src}: {', '.join(missing)}")

    feats: dict[str, list[float]] = {}
    times: dict[str, int] = {}
    with open(src / CSVS[0], newline="") as f:
        for row in csv.reader(f):
            tid = row[0]
            times[tid] = int(float(row[1]))
            feats[tid] = [float(v) for v in row[2:]]

    cls: dict[str, int] = {}
    with open(src / CSVS[1], newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            # the published encoding is "1" illicit, "2" licit, "unknown" unlabelled
            cls[row[0]] = {"1": 1, "2": 0}.get(row[1], -1)

    ids = sorted(feats)
    index = {t: i for i, t in enumerate(ids)}
    x = np.asarray([feats[t] for t in ids], dtype=np.float64)
    node_time = np.asarray([times[t] for t in ids], dtype=np.float64)
    labels = np.asarray([cls.get(t, -1) for t in ids], dtype=np.int64)

    src_i, dst_i = [], []
    with open(src / CSVS[2], newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if row[0] in index and row[1] in index:
                src_i.append(index[row[0]])
                dst_i.append(index[row[1]])

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        TARGET,
        x=x,
        node_time=node_time,
        labels=labels,
        src=np.asarray(src_i, dtype=np.int64),
        dst=np.asarray(dst_i, dtype=np.int64),
    )
    print(f"wrote {TARGET}  ({TARGET.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--convert",
        metavar="DIR",
        default=None,
        help="directory holding the three Elliptic CSVs",
    )
    a = ap.parse_args()
    if a.convert:
        convert(pathlib.Path(a.convert).expanduser())
        return
    if check():
        return
    print("Elliptic dataset not found.\n")
    print(f"1. download it from {SOURCE}")
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
