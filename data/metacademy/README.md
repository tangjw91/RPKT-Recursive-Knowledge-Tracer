# Metacademy prerequisite graph

Source: https://github.com/metacademy/metacademy-content, the hand-curated machine-learning and
mathematics concept graph behind metacademy.org. Licence: CC BY-SA 3.0 US. `edges.csv` lists
`prerequisite,concept` pairs parsed from each concept's `dependencies.txt`; `titles.csv` maps
tags to display titles. Rebuild with `python scripts/fetch_metacademy.py`.
