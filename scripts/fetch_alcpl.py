"""Download the AL-CPL prerequisite files into data/alcpl/."""
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/harrylclc/AL-CPL-dataset/master/data/"
DOMAINS = ["data_mining", "geometry", "physics", "precalculus"]

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "alcpl"
    out.mkdir(parents=True, exist_ok=True)
    for d in DOMAINS:
        for ext in ("preqs", "pairs"):
            urllib.request.urlretrieve(f"{BASE}{d}.{ext}", out / f"{d}.{ext}")
            print("fetched", d, ext)
