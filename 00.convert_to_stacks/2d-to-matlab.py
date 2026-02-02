import os
import re
import tifffile as tiff
import numpy as np
from collections import defaultdict
import tqdm


# Input and output directory roots
dir_list = [r"/media/shilab/Expd-24T/2025-09-16-YJ_AE_1000g_trial2/round6/Plate1-A4-Inverted-WT-SeqA6/Plate1-A4-WT-STAR-SeqA6_2025-09-21_17-14-52.477857/0"]
#replace with tiles needed
# fov_n = [139,140,141,142,159,160,161,162,178,179,180,181,195,196,197,198]
fov_n = range(0,337)
round_num = 6

channel_map = {
    '488': 'ch00',
    '561': 'ch02',
    '730': 'ch01',
    '638': 'ch03',
    '405': 'ch04'
}


t=0
for input_root in dir_list:
    output_root = rf'/media/shilab/ShiLab_SSD1/WT_WellA4_RIBO/raw/round{round_num}'
    os.makedirs(output_root, exist_ok=True)

    file_index = defaultdict(list)
    pattern = re.compile(r"manual_(\d+)_(\d+)_Fluorescence_(\d+)_nm_Ex\.tiff",re.IGNORECASE) #check file name pattern

    all_files = os.listdir(input_root)
    for filename in all_files:
        # print(filename)
        match = pattern.match(filename)
        if match:
            fov = int(match.group(1))
            z = int(match.group(2))
            wl = match.group(3)
            file_index[(fov, wl)].append((z, filename))


    for ch_wavelength, ch_label in channel_map.items():
        for fov in tqdm.tqdm(fov_n):
            # if t >= 1261:
                zfiles = sorted(file_index.get((fov, ch_wavelength), []), key=lambda x: x[0])
                if not zfiles:
                    continue
                volume = np.stack([tiff.imread(os.path.join(input_root, fname)) for _, fname in zfiles], axis=0)
                save_name = f"WT_WellA4_round{round_num:01d}_{ch_label}_Tile_{fov:03d}_{ch_wavelength}.tif" #save output file name with necessary info
                tiff.imwrite(os.path.join(output_root, save_name), volume.astype(np.uint16))
                tqdm.tqdm.write(f"\nSaved: {save_name}")
            # t += 1
    round_num += 1

print('All files renamed and consolidated into fov folders.')

