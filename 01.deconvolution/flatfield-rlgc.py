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
cp.cuda.Device(1).use()

delete_raw = True
#chaneg base dir (should be a level up from raw)
base_dir = "/media/shilab/L/YJ_AE_16gene/plate1_WT"
input_path = rf"{base_dir}/raw"
output_path = rf"{base_dir}/ff_decon_16bit"
#only change if your stacks are not saved from 2d-to-stack script
pattern = re.compile(r'_ch0(\d+)_(?:current|Tile)_(\d+)_(\d+)\.tif')
tile_start= 472
tile_end = 5000

# Configuration
psf_dir = r"/media/shilab/ssd2tb/Xinlin_Gao/theoretical_psf/60x_water"
WAVELENGTH_TO_PSF_FOLDER = {
    405: psf_dir + "/psf_405.tif",
    488: psf_dir + "/psf_488.tif",
    638: psf_dir + "/psf_638.tif",
    561: psf_dir + "/psf_561.tif",
    730: psf_dir + "/psf_730.tif"
}

log = open(f"{base_dir}/raw/deconvolution_log.txt", "a", buffering=1)
t=0
# compute_flatfields_from_folder(Path(input_path))
log.write("loading computed flatfield masks\n")
profiles = load_profiles(Path(input_path+"/flatfields"))
rounds = [f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f)) and "round" in f.lower()]
for r in rounds:
    _ = os.path.join(input_path, r)
    filenames = [f for f in os.listdir(_) if f.lower().endswith((".tif", ".tiff"))]

    for filename in tqdm.tqdm(filenames):
        log.write(f"\nProcessing {filename}\n")
        if t >= 631:
            image_path = os.path.join(input_path,r,filename)
            round_output = os.path.join(output_path,r)

            # channel = tifffile.imread(input_path).astype(np.float32)
            channel, out_path, messages = deconvolve_single_image(
                            pattern = pattern, 
                            channel = image_path, 
                            filename = filename,
                            psf_json = WAVELENGTH_TO_PSF_FOLDER, 
                            exp=False, 
                            output_dir=round_output,
                            num_iterations=30, 
                            snr_results=None)
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
        t +=1


# # GPU Locks - One thread per GPU
# gpu_locks = {0: threading.Lock(), 1: threading.Lock()}
# log_lock = threading.Lock()

# # Setup logging
# log_path = "/media/shilab/XG_Shi_SSD/XG_AE_trial2/plate6/WellA4/deconvolution_log.txt"
# log = open(log_path, "a", buffering=1)

# def safe_log(message, level="INFO", console=True):
#     """
#     Thread-safe logging with levels.
    
#     Args:
#         message: Log message
#         level: INFO, WARN, ERROR, SUCCESS, DEBUG
#         console: Also print to console
#     """
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     formatted = f"[{timestamp}] [{level:7s}] {message}"
    
#     with log_lock:
#         log.write(formatted + "\n")
#         log.flush()
#         if console:
#             print(formatted)

# def check_gpu_memory(gpu_id, min_free_gb=2.0):
#     """Check if GPU has enough free memory"""
#     try:
#         with cp.cuda.Device(gpu_id):
#             mempool = cp.get_default_memory_pool()
#             gpu_free, gpu_total = cp.cuda.runtime.memGetInfo()
#             free_gb = gpu_free / 1e9
            
#             if free_gb < min_free_gb:
#                 safe_log(f"GPU {gpu_id}: Low memory ({free_gb:.1f}GB). Cleaning up...", "WARN", console=False)
#                 mempool.free_all_blocks()
#                 cp.get_default_pinned_memory_pool().free_all_blocks()
#                 time.sleep(1)
#                 gpu_free, gpu_total = cp.cuda.runtime.memGetInfo()
#                 free_gb = gpu_free / 1e9
                
#             return free_gb
#     except Exception as e:
#         safe_log(f"GPU {gpu_id}: Memory check failed - {e}", "ERROR")
#         return 0.0

# def aggressive_cleanup(gpu_id):
#     """Force cleanup of GPU memory"""
#     try:
#         with cp.cuda.Device(gpu_id):
#             cp.get_default_memory_pool().free_all_blocks()
#             cp.get_default_pinned_memory_pool().free_all_blocks()
#             cp.cuda.Stream.null.synchronize()
#             time.sleep(0.5)
#     except Exception as e:
#         safe_log(f"GPU {gpu_id}: Cleanup error - {e}", "WARN", console=False)

# def process_single_file(args):
#     """Process one file on assigned GPU with exclusive lock"""
#     filename, round_dir, round_output, gpu_id, file_idx = args
    
#     # Acquire GPU lock
#     acquired = gpu_locks[gpu_id].acquire(timeout=300)
#     if not acquired:
#         safe_log(f"GPU {gpu_id}: Lock timeout for {filename}", "ERROR")
#         return {
#             'success': False,
#             'filename': filename,
#             'gpu_id': gpu_id,
#             'error': 'GPU lock timeout',
#             'time': 0
#         }
    
#     try:
#         cp.cuda.Device(gpu_id).use()
        
#         # Check memory
#         free_gb = check_gpu_memory(gpu_id, min_free_gb=2.0)
#         if free_gb < 1.5:
#             raise RuntimeError(f"Insufficient GPU memory: {free_gb:.1f}GB")
        
#         image_path = os.path.join(round_dir, filename)
#         start_time = time.time()
        
#         # Log start
#         safe_log(f"GPU {gpu_id}: Processing {filename}", "INFO", console=False)
        
#         # Deconvolve
#         channel, out_path, messages = deconvolve_single_image(
#             pattern=pattern,
#             channel=image_path,
#             filename=filename,
#             psf_json=WAVELENGTH_TO_PSF_FOLDER,
#             exp=False,
#             output_dir=round_output,
#             num_iterations=30,
#             snr_results=None
#         )
        
#         # Log deconvolution messages (early stopping or max iterations)
#         for msg in messages:
#             if msg:  # Only log non-empty messages
#                 safe_log(f"GPU {gpu_id}: {msg}", "INFO", console=False)
        
#         # Flatfield correction
#         safe_log(f"GPU {gpu_id}: Applying flatfield correction", "INFO", console=False)
#         channel_ff = process(profiles, channel, Path(out_path), save=True)
        
#         total_time = time.time() - start_time
#         safe_log(f"GPU {gpu_id}: {filename} complete ({total_time:.1f}s)", "SUCCESS", console=False)
        
#         # Cleanup
#         del channel, channel_ff
#         aggressive_cleanup(gpu_id)
        
#         return {
#             'success': True,
#             'filename': filename,
#             'gpu_id': gpu_id,
#             'time': total_time,
#             'messages': messages
#         }
        
#     except Exception as e:
#         safe_log(f"GPU {gpu_id}: {filename} - ERROR: {str(e)}", "ERROR")
#         aggressive_cleanup(gpu_id)
        
#         return {
#             'success': False,
#             'filename': filename,
#             'gpu_id': gpu_id,
#             'error': str(e),
#             'time': 0
#         }
    
#     finally:
#         gpu_locks[gpu_id].release()

# def verify_gpu_availability():
#     """Check that both GPUs are available"""
#     try:
#         n_gpus = cp.cuda.runtime.getDeviceCount()
#         if n_gpus < 2:
#             safe_log(f"Only {n_gpus} GPU(s) detected. Need 2 GPUs.", "ERROR")
#             return False
        
#         safe_log("=" * 70, "INFO")
#         safe_log("GPU Configuration", "INFO")
#         safe_log("=" * 70, "INFO")
        
#         for gpu_id in [0, 1]:
#             with cp.cuda.Device(gpu_id):
#                 props = cp.cuda.runtime.getDeviceProperties(gpu_id)
#                 free, total = cp.cuda.runtime.memGetInfo()
                
#                 name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
#                 safe_log(f"GPU {gpu_id}: {name} - {total/1e9:.1f}GB total, {free/1e9:.1f}GB free", "INFO")
                
#                 if free / 1e9 < 2.0:
#                     safe_log(f"GPU {gpu_id}: Warning - Low free memory", "WARN")
        
#         safe_log("=" * 70, "INFO")
#         return True
        
#     except Exception as e:
#         safe_log(f"GPU verification failed: {e}", "ERROR")
#         return False

# # ============================================================================
# # Main Processing
# # ============================================================================

# # Header
# safe_log("=" * 70, "INFO")
# safe_log("Dual-GPU Deconvolution Pipeline", "INFO")
# safe_log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "INFO")
# safe_log("=" * 70, "INFO")

# # Verify GPUs
# if not verify_gpu_availability():
#     safe_log("Exiting due to GPU configuration error", "ERROR")
#     log.close()
#     exit(1)

# # Load flatfield profiles
# safe_log("Loading flatfield correction profiles...", "INFO")
# profiles = load_profiles(Path(input_path + "/flatfields"))
# safe_log("Flatfield profiles loaded", "SUCCESS")

# # Get rounds
# rounds = [f for f in os.listdir(input_path) 
#           if os.path.isdir(os.path.join(input_path, f)) and "round" in f.lower()]
# rounds.sort()

# safe_log(f"Found {len(rounds)} round(s) to process", "INFO")

# # Statistics
# total_files = 0
# total_success = 0
# total_failed = 0
# total_time = 0
# failed_files = []

# # Process each round
# for round_idx, round_name in enumerate(rounds, 1):
#     round_dir = os.path.join(input_path, round_name)
#     round_output = os.path.join(output_path, round_name)
#     os.makedirs(round_output, exist_ok=True)
    
#     filenames = [f for f in os.listdir(round_dir) 
#                  if f.lower().endswith((".tif", ".tiff"))]
#     filenames.sort()
    
#     total_files += len(filenames)
#     round_start = time.time()
    
#     safe_log("=" * 70, "INFO")
#     safe_log(f"Round {round_idx}/{len(rounds)}: {round_name} ({len(filenames)} files)", "INFO")
#     safe_log("=" * 70, "INFO")
    
#     # Prepare tasks
#     tasks = [(filenames[i], round_dir, round_output, i % 2, i) 
#              for i in range(len(filenames))]
    
#     completed = 0
#     failed = 0
    
#     # Process with progress bar
#     with ThreadPoolExecutor(max_workers=2) as executor:
#         future_to_task = {executor.submit(process_single_file, task): task 
#                          for task in tasks}
        
#         with tqdm.tqdm(
#             total=len(tasks), 
#             desc=f"{round_name:15s}",
#             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] GPU0:{postfix[0]} GPU1:{postfix[1]} Pass:{postfix[2]} Fail:{postfix[3]}',
#             postfix=['idle', 'idle', 0, 0]
#         ) as pbar:
            
#             for future in as_completed(future_to_task):
#                 result = future.result()
                
#                 if result['success']:
#                     completed += 1
#                     total_success += 1
#                     total_time += result['time']
#                 else:
#                     failed += 1
#                     total_failed += 1
#                     failed_files.append(f"{round_name}/{result['filename']}")
                
#                 # Update progress bar
#                 pbar.postfix = [
#                     'busy' if gpu_locks[0].locked() else 'idle',
#                     'busy' if gpu_locks[1].locked() else 'idle',
#                     completed,
#                     failed
#                 ]
#                 pbar.update(1)
    
#     round_time = time.time() - round_start
#     avg_time = round_time / len(filenames) if len(filenames) > 0 else 0
    
#     safe_log(f"Round {round_name} complete: {completed} succeeded, {failed} failed ({round_time/60:.1f}min, {avg_time:.1f}s/file)", 
#              "SUCCESS" if failed == 0 else "WARN")
    
#     # Cleanup between rounds
#     safe_log("Cleaning up between rounds...", "INFO", console=False)
#     gc.collect()
#     for gpu_id in [0, 1]:
#         aggressive_cleanup(gpu_id)
#     time.sleep(2)

# # Final summary
# safe_log("=" * 70, "INFO")
# safe_log("Processing Complete", "SUCCESS")
# safe_log("=" * 70, "INFO")
# safe_log(f"Total files:     {total_files}", "INFO")
# safe_log(f"Succeeded:       {total_success}", "SUCCESS")
# safe_log(f"Failed:          {total_failed}", "ERROR" if total_failed > 0 else "INFO")
# safe_log(f"Success rate:    {100*total_success/total_files:.1f}%", "INFO")
# safe_log(f"Total time:      {total_time/60:.1f} minutes", "INFO")
# safe_log(f"Average/file:    {total_time/total_files:.1f} seconds", "INFO")

# if failed_files:
#     safe_log("=" * 70, "INFO")
#     safe_log("Failed Files:", "ERROR")
#     for ff in failed_files:
#         safe_log(f"  - {ff}", "ERROR")

# safe_log("=" * 70, "INFO")
# safe_log(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "INFO")
# safe_log(f"Log saved to: {log_path}", "INFO")
# safe_log("=" * 70, "INFO")


# log.close()

