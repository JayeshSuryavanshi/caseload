"""Download FiFAR, verify it, and unpack it where ``caseload.envs.fifar`` looks.

FiFAR is Feedzai's Financial Fraud Alert Review dataset (Alves et al., "A
benchmarking framework and dataset for learning to defer in human-AI
decision-making", Scientific Data 12, 506, 2025). It is published on figshare as
article 28351172 (https://doi.org/10.6084/m9.figshare.28351172) under CC BY 4.0, as
a single 209,350,760-byte FiFAR.zip. Cite the paper if you use it.

The download resumes if it is interrupted, is checked against the md5 figshare
publishes, and is unpacked into ``~/.cache/caseload/fifar``, giving
``~/.cache/caseload/fifar/FiFAR/{alert_data,synthetic_experts,testbed}``.

    python scripts/fetch_fifar.py                    # download, verify, unpack
    python scripts/fetch_fifar.py --zip FiFAR.zip    # verify and unpack a copy you have
    python scripts/fetch_fifar.py --check-url        # confirm the file is still served
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import sys
import urllib.request
import zipfile

from caseload.envs.fifar import CACHE

ARTICLE = 28351172
URL = "https://ndownloader.figshare.com/files/52147616"
NAME = "FiFAR.zip"
SIZE = 209_350_760
MD5 = "2255e7b8a391d0ab567f37ec27e52f3d"
DEST = CACHE / "fifar"
# what the loader reads; the archive has more (the raw BAF table, the alert model)
REQUIRED = (
    "FiFAR/alert_data/processed_data/alerts.parquet",
    "FiFAR/synthetic_experts/expert_predictions.parquet",
    "FiFAR/testbed/train_alert",
    "FiFAR/testbed/test",
)
CHUNK = 1 << 20


def md5sum(path: pathlib.Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def check_url() -> bool:
    # figshare redirects to a signed S3 link that refuses HEAD, so ask for one byte
    req = urllib.request.Request(URL, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        total = r.headers.get("Content-Range", "").rpartition("/")[2]
        print(f"{URL}: HTTP {r.status}, Content-Range total {total or 'missing'}")
    ok = total == str(SIZE)
    print("size matches" if ok else f"size does not match the pinned {SIZE:,} bytes")
    return ok


def download(target: pathlib.Path) -> None:
    part = target.with_suffix(target.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > SIZE:
        part.unlink()
        have = 0
    if have < SIZE:
        headers = {"Range": f"bytes={have}-"} if have else {}
        req = urllib.request.Request(URL, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            if have and r.status != 206:
                # the server ignored the range, so start again rather than append
                have = 0
            mode = "ab" if have else "wb"
            print(f"downloading {URL} from byte {have:,} of {SIZE:,}")
            with open(part, mode) as f:
                done, shown = have, -1
                for block in iter(lambda: r.read(CHUNK), b""):
                    f.write(block)
                    done += len(block)
                    pct = int(100 * done / SIZE)
                    if pct != shown and pct % 5 == 0:
                        print(f"  {pct:3d}%  {done / 1e6:7.1f} MB", flush=True)
                        shown = pct
    if part.stat().st_size != SIZE:
        raise SystemExit(f"{part} is {part.stat().st_size:,} bytes, expected {SIZE:,}; rerun")
    part.rename(target)


def verify(zip_path: pathlib.Path) -> None:
    size = zip_path.stat().st_size
    if size != SIZE:
        raise SystemExit(f"{zip_path} is {size:,} bytes, expected {SIZE:,}")
    got = md5sum(zip_path)
    if got != MD5:
        raise SystemExit(f"{zip_path} md5 {got} does not match the published {MD5}")
    print(f"verified {zip_path}: {size:,} bytes, md5 {got}")


def unpack(zip_path: pathlib.Path, dest: pathlib.Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            if not (root / member).resolve().is_relative_to(root):
                raise SystemExit(f"refusing to unpack {member!r} outside {dest}")
        z.extractall(dest)
    missing = [p for p in REQUIRED if not (dest / p).exists()]
    if missing:
        raise SystemExit(f"unpacked, but these are missing: {', '.join(missing)}")
    print(f"unpacked into {dest / 'FiFAR'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", type=pathlib.Path, default=DEST, help=f"default {DEST}")
    ap.add_argument("--zip", type=pathlib.Path, help="use this FiFAR.zip, do not download")
    ap.add_argument("--check-url", action="store_true", help="probe the URL and exit")
    ap.add_argument("--force", action="store_true", help="unpack even if already present")
    a = ap.parse_args()

    if a.check_url:
        sys.exit(0 if check_url() else 1)

    dest = a.dest.expanduser()
    if not a.force and all((dest / p).exists() for p in REQUIRED):
        print(f"FiFAR is already unpacked at {dest / 'FiFAR'}; pass --force to redo it")
        return

    if a.zip is not None:
        zip_path = a.zip.expanduser()
    else:
        dest.mkdir(parents=True, exist_ok=True)
        zip_path = dest / NAME
        if not zip_path.exists():
            free = shutil.disk_usage(dest).free
            # the zip plus its unpacked contents come to about 570 MB
            if free < 600e6:
                raise SystemExit(f"only {free / 1e6:.0f} MB free under {dest}")
            download(zip_path)
    verify(zip_path)
    unpack(zip_path, dest)
    print(f"FiFAR (figshare article {ARTICLE}, CC BY 4.0) is ready. Cite Alves et al. 2025.")


if __name__ == "__main__":
    main()
