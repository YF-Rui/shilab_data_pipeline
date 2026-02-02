import numpy as np
import tifffile
import os
import timeit
import cupy as cp
import tqdm
from cupyx.scipy.ndimage import grey_opening, center_of_mass, shift

# GPU-specific PSF caches
padded_psf_cache = {0: {}, 1: {}}  # Separate cache per GPU

def get_current_gpu():
    """Get current GPU device ID"""
    return cp.cuda.Device().id

def load_psf_for_wavelength(wavelength, psf_folder_mapping):
    """Loads precomputed PSF from folder based on wavelength."""
    psf_path = psf_folder_mapping.get(wavelength)
    if psf_path is None:
        raise ValueError(f"PSF file for wavelength {wavelength} not found!")
    
    psf = cp.array(tifffile.imread(psf_path), dtype=cp.float32)
    return psf

def pad_psf(image, psf_temp, exp=False, wavelength=None, background=110, sigma=5.0):
    """
    Pads a PSF to match the size of `image` and centers it in all dimensions.
    """
    psf_temp = cp.asarray(psf_temp, dtype=cp.float32)

    if exp:
        psf_temp = psf_temp - background
        psf_temp[psf_temp < 0] = 0.0

        yy = cp.arange(psf_temp.shape[1]) - psf_temp.shape[1] / 2.0
        xx = cp.arange(psf_temp.shape[2]) - psf_temp.shape[2] / 2.0
        xg, yg = cp.meshgrid(yy, xx, indexing='ij')
        r = cp.sqrt(xg**2 + yg**2)
        filt = cp.exp(-(r**2) / (2.0 * sigma**2))
        filt /= filt.max()
        for z in range(psf_temp.shape[0]):
            psf_temp[z] *= filt

    com = center_of_mass(psf_temp)
    geom_center = (cp.array(psf_temp.shape) - 1.0) / 2.0
    shift_vec = geom_center - cp.array(com)
    if cp.linalg.norm(shift_vec) > 0.01:
        psf_temp = shift(psf_temp, shift_vec, order=1, mode='constant', cval=0.0)

    out_psf = cp.zeros_like(image, dtype=cp.float32)

    start = [(o - p)//2 for o, p in zip(image.shape, psf_temp.shape)]
    end = [s + p for s, p in zip(start, psf_temp.shape)]

    for s, e, o in zip(start, end, image.shape):
        if s < 0 or e > o:
            raise ValueError("PSF is larger than image along at least one axis.")

    out_psf[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = psf_temp
    out_psf = cp.fft.ifftshift(out_psf)

    s = out_psf.sum()
    if s <= 0:
        raise ValueError("PSF sum is zero or negative after processing.")
    out_psf /= s

    return out_psf

def compute_otf(psf):
    """Computes the OTF and its conjugate."""
    otf = cp.fft.fftn(psf)
    otfT = cp.conjugate(otf)
    return otf, otfT

def fftconv_gpu(x, H):
    """Perform convolution in Fourier space on GPU."""
    return cp.real(cp.fft.ifftn(cp.fft.fftn(x) * H))

def kldiv_gpu(p, q):
    """Computes the KL divergence on GPU."""
    p = p + 1E-4
    q = q + 1E-4
    p = p / cp.sum(p)
    q = q / cp.sum(q)
    kldiv = p * (cp.log(p) - cp.log(q))
    kldiv[cp.isnan(kldiv)] = 0
    kldiv = cp.sum(kldiv)
    return kldiv

def rlgc_deconvolve(image_gpu, psf_gpu, otf=None, otfT=None, HTones=None, max_iters=50, step_size=1.5):
    """GPU-accelerated Richardson-Lucy deconvolution."""
    if otf is None or otfT is None or HTones is None:
        otf, otfT = compute_otf(psf_gpu)
        HTones = fftconv_gpu(cp.ones_like(image_gpu), otfT)

    recon_gpu = cp.full_like(image_gpu, cp.mean(image_gpu))
    previous_recon_gpu = cp.copy(recon_gpu)

    epsilon = 1e-12
    prev_kld1 = cp.inf
    prev_kld2 = cp.inf

    best_recon = cp.copy(recon_gpu)
    best_kld = cp.inf
    best_iter = 0

    for num_iters in range(max_iters):
        Hu_gpu = fftconv_gpu(recon_gpu, otf)

        rng = cp.random.default_rng()
        split1_gpu = rng.binomial(image_gpu.astype(cp.int64), p=0.5)
        split2_gpu = image_gpu - split1_gpu

        kldim = kldiv_gpu(Hu_gpu, image_gpu)
        kld1 = kldiv_gpu(Hu_gpu, split1_gpu)
        kld2 = kldiv_gpu(Hu_gpu, split2_gpu)

        if (kld1 > prev_kld1) and (kld2 > prev_kld2):
            message = f"Early stopping at iteration {num_iters} due to increasing KL divergence."
            recon_gpu = cp.copy(previous_recon_gpu)
            break

        if kldim < best_kld:
            best_kld = kldim
            best_iter = num_iters + 1
            best_recon = cp.copy(recon_gpu)

        HTratio_gpu = fftconv_gpu(image_gpu / (Hu_gpu + epsilon), otfT) / HTones
        previous_recon_gpu = cp.copy(recon_gpu)
        recon_gpu *= step_size * HTratio_gpu

        prev_kld1 = kld1
        prev_kld2 = kld2

    if num_iters + 1 >= max_iters:
        message = f"Max iterations reached. Using best result from iteration {best_iter} (KLD: {best_kld:.4f})"
        recon_gpu = best_recon

    return cp.asnumpy(recon_gpu), message

def create_blend_weights(tile_shape, overlap, blend_width=None):
    """Create smooth blending weights using cosine taper."""
    if blend_width is None:
        blend_width = overlap
    
    weights = cp.ones(tile_shape, dtype=cp.float32)
    
    for axis in range(3):
        size = tile_shape[axis]
        blend = min(blend_width[axis], size // 4)
        
        if blend > 0:
            taper = 0.5 * (1 - cp.cos(cp.pi * cp.arange(blend) / blend))
            
            slices_start = [slice(None)] * 3
            slices_start[axis] = slice(0, blend)
            weights[tuple(slices_start)] *= taper.reshape(
                [-1 if i == axis else 1 for i in range(3)]
            )
            
            slices_end = [slice(None)] * 3
            slices_end[axis] = slice(size - blend, size)
            weights[tuple(slices_end)] *= taper[::-1].reshape(
                [-1 if i == axis else 1 for i in range(3)]
            )
    
    return weights

def deconvolve_with_tiles(image, psf, exp, wavelength, num_iterations=30, step_size=1.2,
                          tile_size=(16, 512, 512), overlap=(4, 32, 32)):
    """Tiled GPU deconvolution with feathered blending."""
    Z, Y, X = image.shape
    decon_full = cp.zeros_like(image, dtype=cp.float32)
    weight_sum = cp.zeros_like(image, dtype=cp.float32)
    
    z_step = tile_size[0] - overlap[0]
    y_step = tile_size[1] - overlap[1]
    x_step = tile_size[2] - overlap[2]
    
    z_tiles = list(range(0, Z - overlap[0], z_step)) + [max(0, Z - tile_size[0])]
    y_tiles = list(range(0, Y - overlap[1], y_step)) + [max(0, Y - tile_size[1])]
    x_tiles = list(range(0, X - overlap[2], x_step)) + [max(0, X - tile_size[2])]
    
    z_tiles = sorted(set(z_tiles))
    y_tiles = sorted(set(y_tiles))
    x_tiles = sorted(set(x_tiles))
    
    for z0 in z_tiles:
        for y0 in y_tiles:
            for x0 in x_tiles:
                z1 = min(z0 + tile_size[0], Z)
                y1 = min(y0 + tile_size[1], Y)
                x1 = min(x0 + tile_size[2], X)
                
                tile_shape = (z1 - z0, y1 - y0, x1 - x0)
                tile = image[z0:z1, y0:y1, x0:x1]
                tile_psf = psf[:tile_shape[0], :tile_shape[1], :tile_shape[2]]
                
                decon_tile, message = rlgc_deconvolve(
                    tile, tile_psf, 
                    max_iters=num_iterations, 
                    step_size=step_size
                )
                
                if not isinstance(decon_tile, cp.ndarray):
                    decon_tile = cp.asarray(decon_tile)
                
                weights = create_blend_weights(tile_shape, overlap)
                
                decon_full[z0:z1, y0:y1, x0:x1] += decon_tile * weights
                weight_sum[z0:z1, y0:y1, x0:x1] += weights
                
                del decon_tile, weights
                cp.get_default_memory_pool().free_all_blocks()
    
    decon_full /= cp.maximum(weight_sum, 1e-8)
    return decon_full, message

def process_single_channel(channel_data, psf, exp, wavelength, num_iterations=15, step_size=1.5, gpu_id=None):
    """
    Use GPU for Richardson-Lucy deconvolution per channel.
    GPU-specific PSF caching.
    """
    # Determine which GPU we're on
    if gpu_id is None:
        gpu_id = get_current_gpu()
    
    # Check GPU-specific cache
    if wavelength in padded_psf_cache[gpu_id]:
        psf_padded = padded_psf_cache[gpu_id][wavelength]
    else:
        psf_padded = pad_psf(channel_data, psf, exp, wavelength)
        tqdm.tqdm.write(f"[GPU {gpu_id}] Storing calculated PSF for {wavelength}nm")
        padded_psf_cache[gpu_id][wavelength] = psf_padded
    
    channel_gpu = cp.asarray(channel_data, dtype=cp.float32)
    psf_gpu = cp.asarray(psf_padded, dtype=cp.float32)
    
    deconvolved, message = deconvolve_with_tiles(
        channel_gpu, psf_gpu, exp, wavelength,
        num_iterations=num_iterations, 
        step_size=step_size,
        tile_size=(16, 512, 512), 
        overlap=(4, 32, 32)
    )

    return deconvolved, message

def deconvolve_single_image(pattern, channel, filename, psf_json, exp, output_dir='deconvolved_output', 
                            num_iterations=15, snr_results=None, bgrm=False, gpu_id=None):
    """Processes each channel TIFF file individually."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    match = pattern.search(filename)
    if not match:
        print(f"Skipping unrecognized file: {filename}")
        return
    
    channel_index = int(match.group(1))
    wavelength = int(match.group(3))

    tqdm.tqdm.write(f"Processing {filename}: channel {channel_index} (λ={wavelength}nm)...")
    start = timeit.default_timer()

    if snr_results is not None:
        if filename in snr_results:
            snr_data = snr_results[filename]
            snr_value = snr_data["SNR"]
            print(f"SNR: {snr_value:.2f}")
            channel = optimize_channel_for_snr(channel, snr_value, snr_data)
        else:
            print(f"SNR data not found for {filename}, skipping enhancement.")
    
    if isinstance(channel, str):
        channel = tifffile.memmap(channel, dtype=np.float32)
        channel = cp.array(channel)
        if bgrm:
            print("Removing background with radius=10")
            channel = rolling_ball_3d(channel, radius=10)
    
    psf = None
    if wavelength == 405:
        decon = cp.asnumpy(channel)
        message = ""
    else:    
        if wavelength == 730:
            channel = cp.clip(channel * 10, 0, 65535).astype(cp.uint16)

        if wavelength not in padded_psf_cache.get(gpu_id or get_current_gpu(), {}):
            psf = load_psf_for_wavelength(wavelength, psf_json)

        decon, message = process_single_channel(
            channel, psf, exp, wavelength, 
            num_iterations=num_iterations, 
            step_size=1.2,
            gpu_id=gpu_id
        )

    decon_min = np.min(decon)
    decon_max = np.max(decon)

    if decon_max > decon_min:
        decon = ((decon - decon_min) / (decon_max - decon_min) * 65535).astype(np.uint16)
    else:
        decon = np.zeros_like(decon, dtype=np.uint16)

    out_path = os.path.join(output_dir, f'decon_{filename}')

    messages = [message, f"Channel {channel_index} done in {timeit.default_timer() - start:.1f}s"]

    del channel, psf
    cp.get_default_memory_pool().free_all_blocks()
    return decon, out_path, messages

def rolling_ball_3d(image_gpu, radius=10):
    selem = cp.ones((2*radius+1, 2*radius+1, 2*radius+1), dtype=cp.float32)
    background = grey_opening(image_gpu, footprint=selem)
    corrected = image_gpu - background
    corrected = cp.clip(corrected, 0, None)
    return corrected

def optimize_channel_for_snr(channel_data, snr_value, snr_data):
    """Apply adaptive background correction based on SNR."""
    if snr_value >= 40:
        channel_data = channel_data - snr_data["background_mean"]
    elif snr_value >= 20:
        channel_data = channel_data - (0.7 * snr_data["background_mean"])
        print("Moderate SNR: applying partial background correction.")
    else:
        channel_data = sigmoid_contrast_enhancement_3d(
            channel_data, 
            snr_data["background_mean"], 
            snr_data["background_std"],
            snr_data["signal_mean"],
            snr_data["signal_std"], 
            gain_factor=0.2
        )
        print("Low SNR: correcting using sigmoidal stretch.")
    
    channel_data[channel_data < 0] = 0    
    print("Min/Max of channel before decon:", cp.min(channel_data), cp.max(channel_data))
    return channel_data

def sigmoid_contrast_enhancement_3d(image_stack, background_mean, background_std, 
                                    signal_mean, signal_std, gain_factor=0.2):
    enhanced_stack = cp.empty_like(image_stack, dtype=cp.float32)
    cutoff = (background_mean * 0.7 + signal_mean * 0.3)
    snr_separation = abs(signal_mean - background_mean) / (background_std + signal_std + 1e-8)
    gain = gain_factor * snr_separation

    for i in range(image_stack.shape[0]):
        image = image_stack[i, :, :]
        image = (image - image.min()) / (image.max() - image.min()) * 255
        
        x = gain * (image - cutoff)
        log_scaled_x = cp.log1p(cp.abs(x)) * cp.sign(x)
        contrast_enhanced = 1 / (1 + cp.exp(-log_scaled_x))
        
        enhanced_stack[i, :, :] = contrast_enhanced

    return enhanced_stack