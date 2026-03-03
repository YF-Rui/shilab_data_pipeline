import os
import re
import tifffile as tiff
import numpy as np
from collections import defaultdict
import tqdm

####################################################################################################
####                                                                                            ####
#### Change: "dir_list, fov_n, round_num, condition, output_dir" according to your tile         ####
####                                                                                            ####
####################################################################################################

# Input and output directory roots
dir_list = [r"/media/shilab/Expd-NSD/2025-09-10-YJ_AE_16gene/Plate1-C4-inverted-SeqA1-KO/KO-star16-SeqA1-40layers_2025-09-11_13-02-03.660881/0"]
#replace with tiles needed
# fov_n = [139,140,141,142,159,160,161,162,178,179,180,181,195,196,197,198]
fov_n = range(0,236) ##the largest tile num+1 (ex:235+1)
round_num = 1 #specify round number
condition = " " #specify details, this will be saved as prefix of filenames
output_dir = r"/media/shilab/e1d4624c-bf72-4136-9366-40e20138e615/Yanfang/YJ_AE_16gene/plate1_KO/ff_decon_16bit/"


channel_map = {
    '488': 'ch00',
    '561': 'ch02',
    '730': 'ch01',
    '638': 'ch03',
    '405': 'ch04'
}
t=0
for input_root in dir_list:
    output_root = rf'{output_dir}/raw/round{round_num}'
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

            #only keep 405 channel files
            if wl != '405':
                 continue

            file_index[(fov, wl)].append((z, filename))


    for ch_wavelength, ch_label in channel_map.items():
        ch_wavelength = '405'
        ch_label = channel_map[ch_wavelength]

        for fov in tqdm.tqdm(fov_n):
            # if t >= 820:
                zfiles = sorted(file_index.get((fov, ch_wavelength), []), key=lambda x: x[0])
                if not zfiles:
                    continue
                volume = np.stack([tiff.imread(os.path.join(input_root, fname)) for _, fname in zfiles], axis=0)
                save_name = f"{condition}_round{round_num:01d}_{ch_label}_Tile_{fov:03d}_{ch_wavelength}.tif" #save output file name with necessary info
                tiff.imwrite(os.path.join(output_root, save_name), volume.astype(np.uint16))
                tqdm.tqdm.write(f"\nSaved: {save_name}")
            # t += 1
    round_num += 1

print('All files renamed and consolidated into fov folders.')

