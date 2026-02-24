"""
1). This script using DAPI 16bit 3D raw xxxtiles.tiff as input, to generate 8bit DAPI 3D xxxtiles.tiff (removed the 1st z-step for now). 
2). Also generate initail coordinate for fiji stiching using the coordinates.csv file from raw dataset.
3). Using the 16bit ref_merged files from matlab decoding results to generate 8bit ref_merged file for stitching all reads.
"""

import re
import pandas as pd
import tifffile
from pathlib import Path
import cupy as cp
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set path
coordinates_directory = r"/media/shilab/Expd-NSD/2025-09-10-YJ_AE_16gene/Plate1-C4-inverted-SeqA1-KO/KO-star16-SeqA1-40layers_2025-09-11_13-02-03.660881/0"
output_root = Path("/media/shilab/e1d4624c-bf72-4136-9366-40e20138e615/Yanfang/YJ_AE_16gene/plate1_KO/ff_decon_16bit")
output_dir = output_root / "output/max3d0.03_zcorrected_voxel332/images/fused"
output_dir.mkdir(parents=True, exist_ok=True)
# fov_n = [i for i in range(0,16)]

# ==========================================================
# LOAD & SCALE COORDINATES FOR FIJI STITCHING
# ==========================================================
coord = pd.read_csv(rf"{coordinates_directory}/coordinates.csv")
coord = coord[coord['z_level']==0]

# coord = coord[coord['fov'].isin(fov_n)]
xy_dim = 2720
delta = max(abs(coord.iloc[0]['x (mm)'] - coord.iloc[1]['x (mm)']),abs(coord.iloc[0]['y (mm)'] - coord.iloc[1]['y (mm)']))
b = (xy_dim*0.9)/delta
coord['x'] = coord['x (mm)'] *b
coord['y'] = coord['y (mm)'] *b
coord['z'] = coord['z (um)'] *10

output_lines = []
output_lines.append("# Define the number of dimensions we are working on")
output_lines.append("dim = 3")
output_lines.append("")
output_lines.append("# Define the image coordinates")

for idx, row in coord.iterrows():
    tile_name = f"Tile_{row['fov']:03d}.tif"
    coords_str = f"({row['x']}, {row['y']}, {row['z']})"
    output_lines.append(f"{tile_name}; ; {coords_str}")

# Save to file
output_file = rf"{output_dir}\configurations.txt"
with open(output_file, 'w') as f:
    f.write('\n'.join(output_lines))
print(f"Saved coordinates to {output_file}")

# ==========================================================
# IMAGE PROCESSING FUNCTION
# ==========================================================
def process_file(path, o, flat=False):
    output_folder = output_dir / o
    output_folder.mkdir(parents=True, exist_ok=True)
    filename = path.name
    m = re.search(r"Tile_(\d{3})", filename)
    new_name = f"Tile_{m.group(1)}.tif"
    try:
        img16 = tifffile.imread(path)
        img_gpu = cp.asarray(img16, dtype=cp.float32)
        if flat:
            # empty_layer = cp.zeros((1, img_gpu.shape[1], img_gpu.shape[2]), dtype=cp.uint8)
            # img_gpu = cp.concatenate([empty_layer, img_gpu], axis=0)
            img_gpu = cp.max(img_gpu, axisNewFolder=0)
        img_gpu = img_gpu / 65535 * 255
        img8 = cp.asnumpy(img_gpu.astype(cp.uint8))
        if o == 'dapi':
            tqdm.tqdm.write(" Removing first z step")
            img8 = img8[1:, :, :]
        tifffile.imwrite(output_folder / new_name, img8)
        return f"Processed: {filename} → {new_name}"
    except Exception as e:
        return f"Failed: {filename}, error: {e}"

# ==========================================================
# PROCESS DAPI FILES (16bit to 8bit)
# ==========================================================
dapi_path = output_root / "round1"
files = list(dapi_path.rglob("*ch04*.tif"))
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(process_file, f, "dapi") for f in files]
    for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing dapi files"):
        try:
            result = future.result()
            tqdm.tqdm.write(result)
        except KeyboardInterrupt:
            tqdm.tqdm.write("Interrupted by user!")
            executor.shutdown(wait=False)
            break

# ==========================================================
# PROCESS REF_MERGED FILES (could comment out if stitch for single cell only) (16bit to 8bit)
# ==========================================================

ref_folder = output_root / "output/max3d0.03_zcorrected_voxel332/images/ref_merged"  
ref_files = list(ref_folder.rglob("*.tif"))
with ThreadPoolExecutor(max_workers=6) as executor:
    futures = [executor.submit(process_file, f, "ref_merged", flat=False) for f in ref_files]
    for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing ref_merged files"):
        try:
            result = future.result()
            tqdm.tqdm.write(result)
        except KeyboardInterrupt:
            tqdm.tqdm.write("Interrupted by user!")
            executor.shutdown(wait=False)
            break