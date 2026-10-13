"""Download the raw public datasets used in the study into data/raw/.

Run once:  python -m src.download_data
Files are only parsed (never executed) by src/datasets.py.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

SOURCES = {
    "ca-GrQc.txt.gz": "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
    "email-Eu-core.txt.gz": "https://snap.stanford.edu/data/email-Eu-core.txt.gz",
    "facebook_combined.txt.gz": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
    "collins_yeast.csv.zip": "https://networks.skewed.de/net/collins_yeast/files/collins_yeast.csv.zip",
    "celegansneural.csv.zip": "https://networks.skewed.de/net/celegansneural/files/celegansneural.csv.zip",
    "power.csv.zip": "https://networks.skewed.de/net/power/files/power.csv.zip",
    "airports.dat": "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat",
    "routes.dat": "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat",
}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for fname, url in SOURCES.items():
        out = RAW / fname
        if out.exists() and out.stat().st_size > 0:
            print(f"[skip] {fname} already present")
            continue
        print(f"[get ] {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research script)"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out.write_bytes(r.read())
        print(f"       -> {out.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
