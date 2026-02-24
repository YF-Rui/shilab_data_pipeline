import re
from pathlib import Path

import numpy as np
import cupy as cp
import tifffile as tiff
from tqdm import tqdm


# ----------------------------
# Parse ImageJ-style coord file
# ----------------------------
def load_coords(coord_file):
    """Parse tile coordinates from ImageJ registration file.
    
    Supports both 2D format: Tile_X.tif; ; (x, y)
    and 3D format: Tile_X.tif; ; (x, y, z)
    """
    # Try 3D pattern first
    pat_3d = re.compile(r"(Tile_\d+\.tif).*?\(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)")
    # Fallback to 2D pattern
    pat_2d = re.compile(r"(Tile_\d+\.tif).*?\(([-\d.]+),\s*([-\d.]+)\)")
    
    tiles = []
    is_3d = False

    print("📄 Parsing coordinate file...")
    with open(coord_file) as f:
        for line in f:
            # Try 3D first
            m = pat_3d.search(line)
            if m:
                tiles.append((
                    m.group(1),
                    float(m.group(2)),
                    float(m.group(3)),
                    float(m.group(4))
                ))
                is_3d = True
                continue
            
            # Fallback to 2D
            m = pat_2d.search(line)
            if m:
                tiles.append((
                    m.group(1),
                    float(m.group(2)),
                    float(m.group(3)),
                    0.0  # z = 0 for 2D
                ))
    
    return tiles, is_3d


# ----------------------------
# Stitch tiles and save Z-planes separately
# ----------------------------
def stitch_tiles_gpu(tile_dir, coord_file, output_dir, blend_overlap=False, batch_size=10, z_chunk_size=20):
    """Stitch 2D or 3D tiles, saving each Z-plane as a separate 2D TIFF.
    
    Args:
        tile_dir: Directory containing tile TIFF files
        coord_file: ImageJ registration configuration file
        output_dir: Directory to save individual Z-plane TIFFs
        blend_overlap: If True, average overlapping regions (slower)
        batch_size: Number of tiles to process on GPU at once
        z_chunk_size: Number of Z-planes to process at once
    """
    tile_dir = Path(tile_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tiles, is_3d = load_coords(coord_file)
    
    print(f"\n{'='*60}")
    print(f"✓ Found {len(tiles)} tiles ({'3D' if is_3d else '2D'} stitching)")
    print(f"{'='*60}")
    
    # GPU info
    device = cp.cuda.Device()
    free_mem, total_mem = device.mem_info
    print(f"\n🖥️  GPU Information:")
    # print(f"   Device: {device.name}")
    print(f"   Total VRAM: {total_mem / 1e9:.2f} GB")
    print(f"   Free VRAM: {free_mem / 1e9:.2f} GB")
    print(f"   Batch size: {batch_size} tiles")

    # Load one tile to get shape + dtype
    print(f"\n📊 Analyzing tile properties...")
    sample = tiff.imread(tile_dir / tiles[0][0])
    
    if sample.ndim == 2:
        tile_h, tile_w = sample.shape
        tile_d = 1
    elif sample.ndim == 3:
        tile_d, tile_h, tile_w = sample.shape
    else:
        raise ValueError(f"Unexpected tile dimensions: {sample.shape}")
    
    dtype = sample.dtype
    tile_size_mb = np.prod(sample.shape) * np.dtype(dtype).itemsize / 1e6
    print(f"   Tile shape: {sample.shape}")
    print(f"   Data type: {dtype}")
    print(f"   Tile size: {tile_size_mb:.2f} MB")

    # Shift coordinates so all are >= 0
    print(f"\n🔧 Computing mosaic dimensions...")
    xs = [x for _, x, _, _ in tiles]
    ys = [y for _, _, y, _ in tiles]
    zs = [z for _, _, _, z in tiles]
    
    min_x, min_y, min_z = min(xs), min(ys), min(zs)

    tiles = [
        (fname,
         int(round(x - min_x)),
         int(round(y - min_y)),
         int(round(z - min_z)))
        for fname, x, y, z in tiles
    ]

    # Compute canvas size
    W = max(x for _, x, _, _ in tiles) + tile_w
    H = max(y for _, _, y, _ in tiles) + tile_h
    D = max(z for _, _, _, z in tiles) + tile_d

    if is_3d or D > 1:
        print(f"   Mosaic dimensions: {D:,} × {H:,} × {W:,}")
        print(f"   Total voxels: {D*H*W/1e9:.3f} Gvoxels")
        mosaic_bytes = D * H * W * np.dtype(dtype).itemsize
    else:
        print(f"   Mosaic dimensions: {H:,} × {W:,}")
        print(f"   Total pixels: {H*W/1e9:.3f} Gpixels")
        mosaic_bytes = H * W * np.dtype(dtype).itemsize
    
    mosaic_gb = mosaic_bytes / 1e9
    plane_bytes = H * W * np.dtype(dtype).itemsize
    plane_gb = plane_bytes / 1e9
    
    print(f"   Mosaic size: {mosaic_gb:.2f} GB")
    print(f"   Plane size: {plane_gb:.2f} GB each")
    
    # Determine processing strategy
    use_chunking = False
    if is_3d or D > 1:
        required_mem = mosaic_bytes
        if required_mem > free_mem * 0.8:
            use_chunking = True
            max_z_per_chunk = int((free_mem * 0.7) / plane_bytes)
            z_chunk_size = min(max_z_per_chunk, z_chunk_size, D)
            
            print(f"\n⚠️  Using Z-chunked processing")
            print(f"   Chunk size: {z_chunk_size} planes ({z_chunk_size * plane_gb:.2f} GB)")
            print(f"   Number of chunks: {(D + z_chunk_size - 1) // z_chunk_size}")
        else:
            print(f"   ✓ Sufficient GPU memory for full mosaic")
    
    # Convert 16-bit to 8-bit on-the-fly if needed
    if dtype == np.uint16:
        print(f"   Note: Will convert uint16 → uint8 on-the-fly")
        output_dtype = np.uint8
    else:
        output_dtype = dtype
    
    # CHUNKED PROCESSING
    if use_chunking:
        print(f"\n💾 Processing and saving Z-planes in chunks...")
        
        num_chunks = (D + z_chunk_size - 1) // z_chunk_size
        
        for chunk_idx in range(num_chunks):
            z_start = chunk_idx * z_chunk_size
            z_end = min(z_start + z_chunk_size, D)
            chunk_depth = z_end - z_start
            
            print(f"\n{'─'*60}")
            print(f"📦 Chunk {chunk_idx + 1}/{num_chunks}: Z-planes {z_start}-{z_end}")
            print(f"{'─'*60}")
            
            # Allocate chunk
            chunk_shape = (chunk_depth, H, W)
            chunk_gb = chunk_depth * plane_gb
            print(f"   Allocating {chunk_gb:.2f} GB on GPU...")
            
            try:
                mosaic_chunk_gpu = cp.zeros(chunk_shape, dtype=dtype)
            except cp.cuda.memory.OutOfMemoryError:
                print(f"   ✗ Failed! Try reducing z_chunk_size")
                raise
            
            if blend_overlap:
                counts_chunk_gpu = cp.zeros(chunk_shape, dtype=cp.float32)
            
            # Filter tiles for this chunk
            chunk_tiles = []
            for fname, x, y, z in tiles:
                tile_z_end = z + tile_d
                if z < z_end and tile_z_end > z_start:
                    z_rel = z - z_start
                    chunk_tiles.append((fname, x, y, z_rel, z))
            
            print(f"   Processing {len(chunk_tiles)} tiles...")
            
            # Process tiles
            with tqdm(total=len(chunk_tiles), desc=f"Chunk {chunk_idx+1}", unit="tile") as pbar:
                for batch_start in range(0, len(chunk_tiles), batch_size):
                    batch_end = min(batch_start + batch_size, len(chunk_tiles))
                    batch = chunk_tiles[batch_start:batch_end]
                    
                    for fname, x, y, z_rel, z_global in batch:
                        img_cpu = tiff.imread(tile_dir / fname)
                        if img_cpu.ndim == 2:
                            img_cpu = img_cpu[np.newaxis, :, :]
                        
                        # Clip to chunk boundaries
                        z_tile_start = max(0, z_rel)
                        z_tile_end = min(chunk_depth, z_rel + tile_d)
                        z_img_start = max(0, -z_rel)
                        z_img_end = z_img_start + (z_tile_end - z_tile_start)
                        
                        if z_tile_end > z_tile_start:
                            img_slice = img_cpu[z_img_start:z_img_end]
                            img_gpu = cp.asarray(img_slice)
                            
                            if blend_overlap:
                                mosaic_chunk_gpu[z_tile_start:z_tile_end, y:y+tile_h, x:x+tile_w] += img_gpu
                                counts_chunk_gpu[z_tile_start:z_tile_end, y:y+tile_h, x:x+tile_w] += 1
                            else:
                                mosaic_chunk_gpu[z_tile_start:z_tile_end, y:y+tile_h, x:x+tile_w] = img_gpu
                            
                            del img_gpu
                        
                        pbar.update(1)
                    
                    cp.get_default_memory_pool().free_all_blocks()
            
            # Blend overlaps
            if blend_overlap:
                print(f"   Blending overlaps...")
                mask = counts_chunk_gpu > 0
                mosaic_chunk_gpu[mask] = (mosaic_chunk_gpu[mask] / counts_chunk_gpu[mask]).astype(dtype)
                del counts_chunk_gpu
                cp.get_default_memory_pool().free_all_blocks()
            
            # Convert 16→8 bit if needed
            if mosaic_chunk_gpu.dtype == cp.uint16:
                mosaic_chunk_gpu = (mosaic_chunk_gpu / 256).astype(cp.uint8)
            
            # Save individual Z-planes
            print(f"   Saving {chunk_depth} planes...")
            with tqdm(total=chunk_depth, desc="Saving planes", unit="plane") as pbar:
                for z_local in range(chunk_depth):
                    z_global = z_start + z_local
                    plane_cpu = cp.asnumpy(mosaic_chunk_gpu[z_local])
                    
                    plane_path = output_dir / f"z_{z_global:04d}.tif"
                    tiff.imwrite(plane_path, plane_cpu, compression=None)
                    
                    del plane_cpu
                    pbar.update(1)
            
            del mosaic_chunk_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
            print(f"   ✓ Chunk {chunk_idx+1} complete")
    
    # NORMAL PROCESSING (fits in memory)
    else:
        print(f"\n💾 Allocating GPU memory...")
        mosaic_shape = (D, H, W) if (is_3d or D > 1) else (H, W)
        mosaic_gpu = cp.zeros(mosaic_shape, dtype=dtype)
        
        if blend_overlap:
            counts_gpu = cp.zeros(mosaic_shape, dtype=cp.float32)
        
        print(f"\n🔄 Processing tiles...")
        with tqdm(total=len(tiles), desc="Stitching", unit="tile") as pbar:
            for batch_start in range(0, len(tiles), batch_size):
                batch_end = min(batch_start + batch_size, len(tiles))
                batch = tiles[batch_start:batch_end]
                
                for fname, x, y, z in batch:
                    img_cpu = tiff.imread(tile_dir / fname)
                    if img_cpu.ndim == 2:
                        img_cpu = img_cpu[np.newaxis, :, :]
                    
                    img_gpu = cp.asarray(img_cpu)
                    
                    if blend_overlap:
                        if is_3d or D > 1:
                            mosaic_gpu[z:z+tile_d, y:y+tile_h, x:x+tile_w] += img_gpu
                            counts_gpu[z:z+tile_d, y:y+tile_h, x:x+tile_w] += 1
                        else:
                            mosaic_gpu[y:y+tile_h, x:x+tile_w] += img_gpu[0]
                            counts_gpu[y:y+tile_h, x:x+tile_w] += 1
                    else:
                        if is_3d or D > 1:
                            mosaic_gpu[z:z+tile_d, y:y+tile_h, x:x+tile_w] = img_gpu
                        else:
                            mosaic_gpu[y:y+tile_h, x:x+tile_w] = img_gpu[0]
                    
                    del img_gpu
                    pbar.update(1)
                
                cp.get_default_memory_pool().free_all_blocks()
        
        # Blend overlaps
        if blend_overlap:
            print(f"\n🎨 Averaging overlaps...")
            mask = counts_gpu > 0
            mosaic_gpu[mask] = (mosaic_gpu[mask] / counts_gpu[mask]).astype(dtype)
            del counts_gpu
        
        # Convert 16→8 bit
        if mosaic_gpu.dtype == cp.uint16:
            print(f"\n🔄 Converting uint16 to uint8...")
            mosaic_gpu = (mosaic_gpu / 256).astype(cp.uint8)
        
        # Save Z-planes
        print(f"\n💾 Saving {D} Z-planes...")
        with tqdm(total=D, desc="Saving", unit="plane") as pbar:
            for z in range(D):
                plane_cpu = cp.asnumpy(mosaic_gpu[z] if mosaic_gpu.ndim == 3 else mosaic_gpu)
                plane_path = output_dir / f"z_{z:04d}.tif"
                tiff.imwrite(plane_path, plane_cpu, compression=None)
                del plane_cpu
                pbar.update(1)
        
        del mosaic_gpu
        cp.get_default_memory_pool().free_all_blocks()
    
    print(f"\n{'='*60}")
    print(f"✅ Stitching complete!")
    print(f"{'='*60}")
    print(f"   Saved {D} Z-planes to: {output_dir}")
    print(f"{'='*60}\n")

def combine_z_planes_to_stack(planes_dir, output_tif, compression=None, z_project=False):
    """Combine individual Z-plane TIFFs into a single 3D stack or max projection.
    
    Args:
        planes_dir: Directory containing z_XXXX.tif files
        output_tif: Output 3D TIFF path
        compression: 'deflate' (ZIP), 'lzw', or None
        z_project: If True, save maximum projection instead of 3D stack
    """
    planes_dir = Path(planes_dir)
    
    print(f"\n{'='*60}")
    if z_project:
        print(f"📊 Creating Z-projection (maximum intensity)")
    else:
        print(f"📚 Combining Z-planes into 3D stack")
    print(f"{'='*60}")
    
    # Find all plane files
    plane_files = sorted(planes_dir.glob("z_*.tif"))
    num_planes = len(plane_files)
    
    if num_planes == 0:
        raise ValueError(f"No z_*.tif files found in {planes_dir}")
    
    print(f"   Found {num_planes} Z-planes")
    
    # Load first plane to get shape/dtype
    first_plane = tiff.imread(plane_files[0])
    print(f"   Plane shape: {first_plane.shape}")
    print(f"   Data type: {first_plane.dtype}")
    
    if z_project:
        # Z-PROJECTION MODE: Create maximum intensity projection
        print(f"\n💾 Computing maximum projection...")
        print(f"   Processing {num_planes} planes...")
        
        # Initialize max projection with first plane
        max_proj = first_plane.copy()
        
        with tqdm(total=num_planes-1, desc="Max projection", unit="plane") as pbar:
            for plane_file in plane_files[1:]:
                plane = tiff.imread(plane_file)
                max_proj = np.maximum(max_proj, plane)
                pbar.update(1)
        
        print(f"\n💾 Writing maximum projection to disk...")
        print(f"   Output: {output_tif}")
        
        tiff.imwrite(
            output_tif,
            max_proj,
            bigtiff=True,
            compression=compression,
            photometric='minisblack'
        )
        
        file_size_gb = Path(output_tif).stat().st_size / 1e9
        print(f"\n✅ Maximum projection created!")
        print(f"   File size: {file_size_gb:.2f} GB")
        print(f"   Shape: {max_proj.shape}")
        
    else:
        # 3D STACK MODE: Combine all planes
        uncompressed_gb = num_planes * np.prod(first_plane.shape) * first_plane.dtype.itemsize / 1e9
        print(f"   Uncompressed size: {uncompressed_gb:.2f} GB")
        
        if compression:
            print(f"   Using {compression.upper()} compression (Fiji-compatible)")
        
        print(f"\n💾 Writing 3D stack incrementally to disk...")
        print(f"   Output: {output_tif}")
        
        with tiff.TiffWriter(output_tif, bigtiff=True) as tif_writer:
            with tqdm(total=num_planes, desc="Writing stack", unit="plane") as pbar:
                for plane_file in plane_files:
                    plane = tiff.imread(plane_file)
                    tif_writer.write(
                        plane,
                        photometric='minisblack',
                        compression=compression,
                        metadata=None
                    )
                    pbar.update(1)
        
        # Calculate compression ratio
        compressed_size = Path(output_tif).stat().st_size
        compressed_gb = compressed_size / 1e9
        compression_ratio = uncompressed_gb / compressed_gb if compressed_gb > 0 else 1
        
        print(f"\n✅ 3D stack created successfully!")
        print(f"   File size: {compressed_gb:.2f} GB")
        if compression:
            print(f"   Compression ratio: {compression_ratio:.2f}x")
            print(f"   Space saved: {uncompressed_gb - compressed_gb:.2f} GB")
        
        # Verify
        print(f"\n🔍 Verifying...")
        with tiff.TiffFile(output_tif) as tif:
            print(f"   Pages: {len(tif.pages)}")
            print(f"   First page shape: {tif.pages[0].shape}")
            print(f"   First page dtype: {tif.pages[0].dtype}")
            if compression:
                print(f"   Compression: {tif.pages[0].compression}")
            
            # Test loading first plane
            try:
                first = tif.pages[0].asarray()
                print(f"   ✓ First plane reads successfully")
            except Exception as e:
                print(f"   ⚠️  Warning: {e}")
    
    print(f"{'='*60}\n")


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["stitch", "combine", "both"], default="both")
    parser.add_argument("--tile-dir", type=str)
    parser.add_argument("--coord-file", type=str)
    parser.add_argument("--planes-dir", type=str)
    parser.add_argument("--output-tif", type=str)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--z-chunk-size", type=int, default=20)
    parser.add_argument("--compression", type=str, choices=['deflate', 'lzw', 'none'], default='none',
                        help="Compression type: deflate (ZIP), lzw, or none")
    parser.add_argument("--z-project", action='store_true',
                        help="Create maximum intensity projection instead of 3D stack")
    args = parser.parse_args()
    
    # Handle compression argument
    compression = None if args.compression == 'none' else args.compression
    
    try:
        if args.mode in ["stitch", "both"]:
            stitch_tiles_gpu(
                tile_dir=args.tile_dir or r"D:\YJ_AE_1000_full\WT_WellC3_STAR\ff_decon_16bit\output\max3d_minIntens0.03_ref638_z_correction_voxel332\images\fused\DAPI",
                coord_file=args.coord_file or r"/media/shilab/ShiLab_SSD1/Thick_tissue_data/deep_STARmap_mNB/TileConfiguration.registered.txt",
                output_dir=args.planes_dir or r"D:\YJ_AE_1000_full\WT_WellC3_STAR\ff_decon_16bit\output\max3d_minIntens0.03_ref638_z_correction_voxel332\images\fused\dapi_planes",
                blend_overlap=False,
                batch_size=args.batch_size,
                z_chunk_size=args.z_chunk_size
            )
        
        # if args.mode in ["combine", "both"]:
        #     combine_z_planes_to_stack(
        #         planes_dir=args.planes_dir or r"D:\YJ_AE_1000_full\WT_WellC3_STAR\ff_decon_16bit\output\max3d_minIntens0.03_ref638_z_correction_voxel332\images\fused\dapi_planes",
        #         output_tif=args.output_tif or r"D:\YJ_AE_1000_full\WT_WellC3_STAR\ff_decon_16bit\output\max3d_minIntens0.03_ref638_z_correction_voxel332\images\fused\MAX_DAPI.tif",
        #         compression=compression,
        #         z_project=args.z_project
        #     )
            
    except cp.cuda.memory.OutOfMemoryError:
        print(f"\n GPU Out of Memory Error")
        print(f"   Try: reducing z_chunk_size or batch_size")
    except Exception as e:
        print(f"\n Error: {e}")
        raise

    """
    reads stitching: configurations.registered.txt; dapi.tif (stitched); ref_merged.tif (stitched), MAX_ref_merged.tif (z-projection of ref_merged.tif)
    single cell stitching: configurations.registered.txt; configurations.txt; grid.csv; {ouput_dir}/expr/clustermap
    """