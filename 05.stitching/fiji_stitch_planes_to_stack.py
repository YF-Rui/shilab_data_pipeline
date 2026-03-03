#!/usr/bin/env python3

import os
import re
from glob import glob
import tifffile as tiff

input_folder = "/media/shilab/e1d4624c-bf72-4136-9366-40e20138e615/Yanfang/YJ_AE_16gene/plate1_WT/ff_decon_16bit/output/max3d0.03_zcorrected_voxel332/images/fused/ref_merged_stitched/"
output_file = "ref_merge_3D.tif"

file_list = glob(os.path.join(input_folder, "*_c1"))
print(f"Total matching files found: {len(file_list)}")

if len(file_list) == 0:
    raise RuntimeError("No DAPI (_c1) images found. Check filename pattern.")

def extract_z(filename):
    match = re.search(r'_z(\d+)', filename)
    return int(match.group(1)) if match else -1

file_list = sorted(file_list, key=lambda x: extract_z(os.path.basename(x)))

print("First few files after sorting:")
for f in file_list[:5]:
    print(os.path.basename(f))

output_path = os.path.join(input_folder, output_file)

# Use TiffWriter for incremental saving
with tiff.TiffWriter(output_path, bigtiff=True) as tif:
    for i, f in enumerate(file_list):
        img = tiff.imread(f)
        tif.write(img)  # no imagej=True here
        print(f"Wrote plane {i+1}/{len(file_list)}")

print("Saved 3D stack to:", output_path)