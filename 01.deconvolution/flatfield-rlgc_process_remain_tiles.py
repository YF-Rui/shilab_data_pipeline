import tqdm
import re
from pathlib import Path
import os
from rlgc_combined_NEW import deconvolve_single_image
from flatfield_correct import compute_flatfields_from_folder, load_profiles, process
import gc
import cupy as cp
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import threading
# import time
# from datetime import datetime
#specify which gpu to use
cp.cuda.Device(0).use()

delete_raw = True
#chaneg base dir (should be a level up from raw)
base_dir = "/media/shilab/L/YJ_AE_16gene/plate1_WT"

##Don't change
input_path = rf"{base_dir}/raw"
output_path = rf"{base_dir}/ff_decon_16bit"
psf_path = r"/media/shilab/ssd2tb/Yanfang/code_Yanfang/01.deconvolution/theoretical_psf"

#only change if your stacks are not saved from 2d-to-stack script
pattern = re.compile(r'_ch0(\d+)_(?:current|Tile)_(\d+)_(\d+)\.tif')

#set tile number to split to 2 gpu
# tile_start= 0
# tile_end = 3

#Select iteration number of deconv run
iter_n = 30

#psf parameter, check to make sure match the imaging acquisition parameter
dz = "0.5" #choose from 0.5, 0.75, 0.1
xy = "60x" #choose from 25x, 40x, 60x
medium = "water"#choose from water, oil

# Configuration
image_param = f"{xy}_{medium}_{dz}" #ex: 60x_water_0.5
psf_dir = rf"{psf_path}/{image_param}"
WAVELENGTH_TO_PSF_FOLDER = {
    405: psf_dir + "/psf_405.tif",
    488: psf_dir + "/psf_488.tif",
    638: psf_dir + "/psf_638.tif",
    561: psf_dir + "/psf_561.tif",
    730: psf_dir + "/psf_730.tif"
}

log = open(f"{base_dir}/raw/deconvolution_log.txt", "a", buffering=1)
# t=0
compute_flatfields_from_folder(Path(input_path))
log.write("loading computed flatfield masks\n")
profiles = load_profiles(Path(input_path+"/flatfields"))
rounds = [f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f)) and "round" in f.lower()]
for r in rounds:
    _ = os.path.join(input_path, r)
    filenames = [f for f in os.listdir(_) if f.lower().endswith((".tif", ".tiff"))]

    for filename in tqdm.tqdm(filenames):
        log.write(f"\nProcessing {filename}\n")
        # if t >= tile_start and t < tile_end:
        image_path = os.path.join(input_path,r,filename)
        round_output = os.path.join(output_path,r)

            # channel = tifffile.imread(input_path).astype(np.float32)
        channel, out_path, messages = deconvolve_single_image(
                            pattern = pattern, 
                            channel = image_path, 
                            filename = filename,
                            psf_json = WAVELENGTH_TO_PSF_FOLDER, 
                            exp = False, 
                            output_dir = round_output,
                            num_iterations = iter_n, 
                            snr_results = None)
        tqdm.tqdm.write(messages[0])
        log.write(messages[0])
        tqdm.tqdm.write(messages[1])
        log.write(messages[1])
        tqdm.tqdm.write("correcting for flatfield")
        log.write("correcting for flatfield\n")
        channel_ff = process(profiles, channel, Path(out_path), save=True)
        if delete_raw:
            tqdm.tqdm.write("deleting raw stack\n")
            try:
                os.remove(image_path)
            except FileNotFoundError:
                pass
        t=0
        if t % 10 == 0:
            tqdm.tqdm.write(f"\n--- MEMORY CLEANUP at iteration {t} ---")
            del channel, channel_ff
            gc.collect()

            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except:
                pass

            tqdm.tqdm.write("Cleanup complete.\n")
    
