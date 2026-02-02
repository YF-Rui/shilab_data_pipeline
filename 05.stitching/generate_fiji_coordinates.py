import re
import pandas as pd
import tifffile
from pathlib import Path
# import cupy as cp
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

directory = r"J:\XG_AE_trial1\Well_D3"
output_root = Path("J:/XG_AE_trial1/Well_D3/ff_decon_16bit")
output_dir = output_root / "output/max3d_minIntens0.1_voxel331/images/fused"
output_dir.mkdir(parents=True, exist_ok=True)
# fov_n = [i for i in range(0,16)]


coord = pd.read_csv(rf"{directory}\coordinates.csv")
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

# def process_file(path, o, flat=False):
#     output_folder = output_dir / o
#     output_folder.mkdir(parents=True, exist_ok=True)
#     filename = path.name
#     m = re.search(r"Tile_(\d{3})", filename)
#     new_name = f"Tile_{m.group(1)}.tif"
#     try:
#         img16 = tifffile.imread(path)
#         img_gpu = cp.asarray(img16, dtype=cp.float32)
#         if flat:
#             # empty_layer = cp.zeros((1, img_gpu.shape[1], img_gpu.shape[2]), dtype=cp.uint8)
#             # img_gpu = cp.concatenate([empty_layer, img_gpu], axis=0)
#             img_gpu = cp.max(img_gpu, axis=0)
#         img_gpu = img_gpu / 65535 * 255
#         img8 = cp.asnumpy(img_gpu.astype(cp.uint8))
#         if o == 'dapi':
#             tqdm.tqdm.write(" Removing first z step")
#             img8 = img8[1:, :, :]
#         tifffile.imwrite(output_folder / new_name, img8)
#         return f"Processed: {filename} → {new_name}"
#     except Exception as e:
#         return f"Failed: {filename}, error: {e}"

# dapi_path = output_root / "round1"
# files = list(dapi_path.rglob("*ch04*.tif"))
# with ThreadPoolExecutor(max_workers=2) as executor:
#     futures = [executor.submit(process_file, f, "dapi") for f in files]
#     for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing dapi files"):
#         try:
#             result = future.result()
#             tqdm.tqdm.write(result)
#         except KeyboardInterrupt:
#             tqdm.tqdm.write("Interrupted by user!")
#             executor.shutdown(wait=False)
#             break

# # Save to file
output_file = rf"{output_dir}\configurations.txt"
with open(output_file, 'w') as f:
    f.write('\n'.join(output_lines))
print(f"Saved coordinates to {output_file}")


# ref_folder = output_root / "output/max3d_minIntens0.03_ref638_z_correction_voxel332/images/ref_merged"  
# ref_files = list(ref_folder.rglob("*.tif"))
# with ThreadPoolExecutor(max_workers=6) as executor:
#     futures = [executor.submit(process_file, f, "ref_merged", flat=False) for f in ref_files]
#     for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing ref_merged files"):
#         try:
#             result = future.result()
#             tqdm.tqdm.write(result)
#         except KeyboardInterrupt:
#             tqdm.tqdm.write("Interrupted by user!")
#             executor.shutdown(wait=False)
#             break