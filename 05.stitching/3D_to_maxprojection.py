import numpy as np
import tifffile as tiff

input_path = "/media/shilab/e1d4624c-bf72-4136-9366-40e20138e615/Yanfang/YJ_AE_16gene/plate1_WT/ff_decon_16bit/output/max3d0.03_zcorrected_voxel332/images/fused/ref_merged_stitched/ref_merge_3D.tif"
output_path = "/media/shilab/e1d4624c-bf72-4136-9366-40e20138e615/Yanfang/YJ_AE_16gene/plate1_WT/ff_decon_16bit/output/max3d0.03_zcorrected_voxel332/images/fused/max_ref_merged.tif"

# Open without loading entire stack
with tiff.TiffFile(input_path) as tif:
    n_pages = len(tif.pages)
    
    # Read first slice to initialize
    max_proj = tif.pages[0].asarray()
    
    # Process slice by slice
    for i in range(1, n_pages):
        slice_img = tif.pages[i].asarray()
        max_proj = np.maximum(max_proj, slice_img)

# Save result
tiff.imwrite(output_path, max_proj)