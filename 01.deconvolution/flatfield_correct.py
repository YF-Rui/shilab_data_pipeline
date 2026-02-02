from pathlib import Path
import re
import random
import numpy as np
from skimage.io import imread, imsave
from basicpy import BaSiC
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import os
import logging
from typing import Optional, Callable, Mapping
import concurrent.futures
import cupy as cp


profiles = {}
def extract_channel(filename: str) -> str:
    """Extract the last number from filename as the channel."""
    numbers = re.findall(r"\d+", filename)
    if numbers:
        return numbers[-1]  # last number in the filename
    return "unknown"
    
    
def compute_flatfields_from_folder(
    root_dir: Path
) -> None:
    """
    Compute flatfields from TIFFs in a folder using BaSiC.
    Saves .npy and .png outputs to folder/flatfields/.
    """
    folder = Path(root_dir)
    output_dir = folder / "flatfields"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by channel
    channel_files = {}
    for round, _, files in os.walk(root_dir):
        folder_path = Path(round)  # convert string folder to Path object
        for file in folder_path.glob("*.tif*"):
            ch = extract_channel(file.stem)
            channel_files.setdefault(ch, []).append(file)

    for ch, files in channel_files.items():
        print(f"Processing channel: {ch} ({len(files)} files)")
        random.shuffle(files)
        files = files[:500]

        print("z projecting")
        # Load image stack
        stack = np.stack([
            np.max(imread(str(p)), axis=0)
            for p in files
        ]).astype(np.float32)

        # stack = np.stack([
        #     imread(str(p))
        #     for p in files
        # ]).astype(np.float32)

        print("calculating")
        # Fit BaSiC
        basic = BaSiC(get_darkfield=False, smoothness_flatfield=3)
        basic.fit(stack)
        ff = basic.flatfield.astype(np.float32)

        # Save .npy
        np.save(output_dir / f"flatfield_{ch}.npy", ff)

        # Save quick preview PNG
        _write_preview_png(ff, output_dir / f"flatfield_{ch}.png")

    print(f"Done. Results saved in: {output_dir}")

def _write_preview_png(ff: np.ndarray, png_path: Path) -> None:
    cmap = cm.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=float(ff.max()))
    rgba = cmap(norm(ff))  # H×W×4 floats in [0, 1]
    rgb = (rgba[..., :3] * 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    ax.imshow(rgb, interpolation="nearest")
    ax.axis("off")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.5)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)



def load_profiles(src: Path) -> Mapping[str, np.ndarray]:
    """Return a channel→flatfield mapping from *src* (manifest or folder)."""
    # if src.is_file() and src.suffix == ".json":
    #     meta = json.loads(src.read_text())
    #     base = src.parent
    #     profiles = {ch: np.load(base / fname) for ch, fname in meta["files"].items()}
    #     return profiles 
    # Directory case
    profiles = {p.stem.split("_")[-1]: np.load(p) for p in src.glob("flatfield_*.npy")}
    # print(profiles)
    return profiles


def process(profiles, img, p: Path, save=False) -> np.ndarray:
    ch = extract_channel(p.stem)
    ff = profiles.get(ch)
    if ff is None:
        logging.warning("No flatfield for channel '%s' (%s) — skipped", ch, p.name)
        return None

    # Convert img and ff to CuPy arrays
    img = cp.asarray(img)  # Convert img (numpy) to cupy
    ff = cp.asarray(ff)  # Convert ff (numpy) to cupy

    # Perform flat-field correction with CuPy arrays
    ff_norm = ff / cp.max(ff)
    ff_gamma = ff_norm ** 1.8
    mean_ff = cp.percentile(ff_gamma, 50) 
    corrected = img / (ff_gamma + 1e-6) * mean_ff

    # Restore original dtype + safe clipping
    orig_dtype = img.dtype
    if cp.issubdtype(orig_dtype, cp.integer):
        info = cp.iinfo(orig_dtype)
    else:
        info = cp.finfo(orig_dtype)
    corrected = cp.clip(corrected, info.min, info.max).astype(orig_dtype)

    if save:
        imsave(p, corrected.get(), check_contrast=False)  # Convert back to numpy before saving
    else:
        return corrected.get()  # Convert back to numpy before returning
