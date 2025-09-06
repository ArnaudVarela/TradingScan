#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean sector_breadth.csv produced by your pipeline:
- Drop NaN sectors
- Replace empty strings with 'Unclassified' (optional)
- Sort by count desc
Writes in place.
"""

import os
import pandas as pd

PUBLIC_DIR = os.path.join("dashboard", "public")
FILE = "sector_breadth.csv"

def main():
    path = os.path.join(PUBLIC_DIR, FILE)
    if not os.path.isfile(path):
        print(f"[breadth] {FILE} not found, skip.")
        return

    df = pd.read_csv(path)

    # Normalize column names expected: 'sector','count'
    # If different, try to detect similar ones
    cols = {c.lower(): c for c in df.columns}
    sector_col = cols.get("sector")
    count_col = cols.get("count") or cols.get("n") or cols.get("total")

    if sector_col is None or count_col is None:
        print("[breadth] unexpected columns; skip.")
        return

    # Drop NaN sector
    df = df.dropna(subset=[sector_col]).copy()

    # Optional: treat empty strings as 'Unclassified'
    df[sector_col] = df[sector_col].astype(str).str.strip()
    df.loc[df[sector_col] == "", sector_col] = "Unclassified"

    # Sort by count desc
    df[count_col] = pd.to_numeric(df[count_col], errors="coerce").fillna(0)
    df = df.sort_values(count_col, ascending=False)

    df.to_csv(path, index=False)
    print("[breadth] cleaned and sorted.")

if __name__ == "__main__":
    main()
