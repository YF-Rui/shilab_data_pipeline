"""
Convert fiji output coordinate file: configurations.txt to clustermap stiching input foramt: grid.csv
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path


# -------------------------
# Parse configurations.txt
# -------------------------
def parse_configurations(cfg_file):
    pat = re.compile(
        r"(Tile_(\d+))\.tif.*\(\s*([-\d.]+)\s*,\s*([-\d.]+)"
    )

    records = []
    with open(cfg_file) as f:
        for line in f:
            m = pat.search(line)
            if m:
                records.append({
                    "tile": m.group(1),
                    "id": int(m.group(2)),
                    "x": float(m.group(3)),
                    "y": float(m.group(4)),
                })

    return pd.DataFrame(records)


# -------------------------
# Infer grid from coordinates
# -------------------------
def infer_grid(df, tol=0.2):
    """
    tol: fraction of tile spacing tolerated for snapping
    """

    # Sort by y then x (top-left → bottom-right)
    df = df.sort_values(["y", "x"]).reset_index(drop=True)

    # Estimate grid spacing
    dx = np.median(np.diff(np.unique(df["x"])))
    dy = np.median(np.diff(np.unique(df["y"])))

    if not np.isfinite(dx) or not np.isfinite(dy):
        raise ValueError("Could not infer grid spacing")

    # Snap x/y to grid indices
    col_index = np.round((df["x"] - df["x"].min()) / dx).astype(int)
    row_index = np.round((df["y"] - df["y"].min()) / dy).astype(int)

    # Normalize so they start at 0
    df["col_count"] = col_index - col_index.min()
    df["row_count"] = row_index - row_index.min()

    return df


# -------------------------
# Generate grid.csv
# -------------------------
def generate_grid_csv(cfg_file, out_csv):
    df = parse_configurations(cfg_file)
    df = infer_grid(df)

    grid_df = df.loc[:, ["col_count", "row_count", "id"]].copy()
    grid_df["grid"] = (
        "R" + grid_df["row_count"].astype(str)
        + "C" + grid_df["col_count"].astype(str)
    )

    # Match expected column order
    grid_df = grid_df[["col_count", "row_count", "id", "grid"]]

    grid_df.to_csv(out_csv)
    print(f"Saved grid.csv → {out_csv}")
    print(grid_df.sort_values(["row_count", "col_count"]))


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    output_root = r"/media/shilab/e1d4624c-bf72-4136-9366-40e20138e615/Yanfang/YJ_AE_16gene/plate1_WT/ff_decon_16bit/output/max3d0.03_zcorrected_voxel332"
    generate_grid_csv(
        cfg_file=rf"{output_root}/images/fused/configurations.txt",
        out_csv=rf"{output_root}/images/fused/grid.csv"
    )
