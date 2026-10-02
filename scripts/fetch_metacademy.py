"""Build data/metacademy/edges.csv (prerequisite -> concept) from the Metacademy content repository.

Source: https://github.com/metacademy/metacademy-content (CC BY-SA 3.0 US). Requires git.
"""
import csv
import os
import re
import subprocess
import tempfile
from pathlib import Path

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "metacademy"
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/metacademy/metacademy-content.git", tmp], check=True)
        root = Path(tmp) / "concepts"
        edges, titles = [], {}
        for c in sorted(os.listdir(root)):
            d = root / c
            if not d.is_dir():
                continue
            if (d / "title.txt").exists():
                titles[c] = (d / "title.txt").read_text().strip()
            if (d / "dependencies.txt").exists():
                for line in (d / "dependencies.txt").read_text(errors="ignore").splitlines():
                    m = re.match(r"\s*tag:\s*(\S+)", line)
                    if m and m.group(1) != c:
                        edges.append((m.group(1), c))
    with open(out / "edges.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["prerequisite", "concept"])
        w.writerows(sorted(set(edges)))
    with open(out / "titles.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "title"])
        w.writerows(sorted(titles.items()))
    print(f"{len(set(edges))} edges, {len(titles)} titled concepts")
