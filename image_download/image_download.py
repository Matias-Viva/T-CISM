import os
import glob
import time
import argparse
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import sys

from datetime import datetime
from goes2go import GOES

# Band metadata
BAND_INFO = {
    'C01': ('0.47 µm',  'Blue'),
    'C02': ('0.64 µm',  'Red'),
    'C03': ('0.865 µm', 'Vegetation'),
    'C04': ('1.378 µm', 'Cirrus'),
    'C05': ('1.61 µm',  'Snow/Ice'),
    'C06': ('2.25 µm',  'Cloud Particle Size'),
    'C07': ('3.9 µm',   'Shortwave Window'),
    'C08': ('6.185 µm', 'Upper-level Water Vapor'),
    'C09': ('6.95 µm',  'Mid-level Water Vapor'),
    'C10': ('7.34 µm',  'Lower-level Water Vapor'),
    'C11': ('8.5 µm',   'Cloud Top Phase'),
    'C12': ('9.61 µm',  'Ozone'),
    'C13': ('10.35 µm', 'Clean Longwave Window'),
    'C14': ('11.2 µm',  'Longwave Window'),
    'C15': ('12.3 µm',  'Dirty Longwave Window'),
    'C16': ('13.3 µm',  'CO2 Longwave Window'),
}

DATA_DIR   = "./goes_data_1b"
MAX_RETRIES = 3
RETRY_WAIT  = 30  # seconds


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    """
    Parse command-line arguments.

    Usage examples:
      uv run image_visualization.py                  # all 16 bands
      uv run image_visualization.py --bands 2        # only band 2
      uv run image_visualization.py --bands 1 2 3    # bands 1, 2 and 3
      uv run image_visualization.py --bands 7 8 9 10 # bands 7-10
    """
    parser = argparse.ArgumentParser(
        description="Download and visualize GOES-19 Level 1b ABI bands."
    )
    parser.add_argument(
        '--bands',
        nargs='+',
        type=int,
        choices=range(1, 17),
        metavar='N',
        help=(
            'Band number(s) to download and visualize (1–16). '
            'If omitted, all 16 bands are processed. '
            'Example: --bands 2 7 13'
        )
    )
    return parser.parse_args()


def band_ids_from_args(bands_arg):
    """Convert a list of band numbers (e.g. [2]) to band IDs (e.g. ['C02'])"""
    if bands_arg:
        return [f'C{n:02d}' for n in sorted(bands_arg)]
    return [f'C{n:02d}' for n in range(1, 17)]  # all 16 by default


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_bands(target_band_ids):
    G = GOES(satellite=19, product="ABI-L1b-Rad", domain='F')
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  Download attempt {attempt}/{MAX_RETRIES}...")
            # Don't store the return value — just download to disk
            G.nearesttime(
                datetime.now(),
                save_dir=DATA_DIR,
                overwrite=False
            )
            # Files are already on disk; we don't need the concatenated Dataset
            print("  Download successful!")
            return True
        except MemoryError:
            print(f"  Attempt {attempt} failed: Out of memory during concatenation")
            print("  (Files were downloaded successfully; concatenation just failed)")
            # Files are still on disk, so return True anyway
            return True
        except Exception as e:
            print(f"  Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                print(f"  Waiting {RETRY_WAIT}s before retry...")
                time.sleep(RETRY_WAIT)
            else:
                print("  All download attempts failed.")
                return False


# ─────────────────────────────────────────────────────────────────────────────
# FILE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def is_valid_netcdf(filepath):
    """Check if a NetCDF file is valid and readable"""
    size_mb = os.path.getsize(filepath) / (1024 ** 2)
    if size_mb < 10:
        return False, f"File too small ({size_mb:.1f} MB) — likely incomplete download"
    try:
        with xr.open_dataset(filepath, engine='netcdf4') as ds:
            if 'Rad' not in ds.variables:
                return False, "Missing 'Rad' variable"
            _ = ds['Rad'].shape
            return True, f"Valid ({size_mb:.1f} MB)"
    except Exception as e:
        return False, str(e)


def find_band_files(root_dir, target_band_ids):
    """
    Find and validate files for the requested bands only.
    Returns (band_files dict, missing_bands list).
    """
    band_files   = {}
    missing_bands = []

    for band_id in target_band_ids:
        pattern = f"{root_dir}/**/*{band_id}*.nc"
        files   = glob.glob(pattern, recursive=True)

        if not files:
            print(f"  ✗  Band {band_id}: no file found")
            missing_bands.append(band_id)
            continue

        latest_file = max(files, key=os.path.getmtime)
        is_valid, msg = is_valid_netcdf(latest_file)

        if not is_valid:
            print(f"  ⚠️  Band {band_id} invalid: {msg}")
            missing_bands.append(band_id)
            continue

        band_files[band_id] = latest_file
        print(f"  ✓  Band {band_id}: {os.path.basename(latest_file)} — {msg}")

    if missing_bands:
        print(f"\n  ⚠️  Missing/invalid: {', '.join(missing_bands)}")

    return band_files, missing_bands


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _generate_images(band_files, output_dir):
    """Generate and save images for all available bands from Level 1b data"""

    if not band_files:
        print("ERROR: No valid band files to process!")
        return

    successful = 0
    failed     = 0

    for band_id, band_file in band_files.items():
        wavelength, description = BAND_INFO[band_id]
        print(f"  Processing {band_id} ({wavelength}) - {description}...")

        ds = None
        try:
            ds = xr.open_dataset(band_file, engine='netcdf4')
            radiance = ds['Rad'].values

            if radiance.size == 0:
                raise ValueError("Radiance array is empty")
            if np.all(np.isnan(radiance)):
                raise ValueError("All radiance values are NaN")

            # Crop a 1000x1000 region from the center
            height, width = radiance.shape
            crop_size = min(1000, height, width)
            start_row = (height - crop_size) // 2
            start_col = (width  - crop_size) // 2
            band_crop = radiance[start_row:start_row+crop_size,
                                 start_col:start_col+crop_size]

            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(band_crop, cmap='gray', origin='upper')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Spectral Radiance (W/m²/sr/µm)')
            ax.set_title(
                f'GOES-19 Band {band_id} ({wavelength}) - {description} - Level 1b',
                fontsize=12
            )
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            plt.tight_layout()

            output_path = os.path.join(output_dir, f'Band{band_id}_L1b_crop.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"    ✓ Saved: {os.path.basename(output_path)}")

            plt.close()
            ds.close()
            successful += 1

        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            failed += 1
            if ds is not None:
                try:
                    ds.close()
                except:
                    pass
            plt.close('all')

    print(f"\n{'='*60}")
    print(f"Done: {successful} successful, {failed} failed")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args           = parse_args()
    target_band_ids = band_ids_from_args(args.bands)

    print("GOES-19 Level 1b Image Visualization")
    print("=" * 60)
    print(f"Target bands: {', '.join(target_band_ids)}")

    # ── Step 1: Validate existing files for the requested bands ──────────────
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"\nValidating files in: {DATA_DIR}")
    band_files, missing_bands = find_band_files(DATA_DIR, target_band_ids)

    # ── Step 2: Download only what is missing/corrupted ──────────────────────
    if missing_bands:
        print(f"\nDownloading missing/corrupted bands: {', '.join(missing_bands)}")
        success = download_bands(missing_bands)
        if not success:
            print("\nDownload failed. Proceeding with whatever is available.")

        # Re-validate after download
        print("\nRe-validating files after download...")
        band_files, still_missing = find_band_files(DATA_DIR, target_band_ids)

        if still_missing:
            print(f"\n⚠️  Could not obtain: {', '.join(still_missing)}")
            if not band_files:
                print("No valid files at all. Exiting.")
                sys.exit(1)
            response = input("Continue with available bands? (y/n): ").strip().lower()
            if response != 'y':
                print("Aborted.")
                sys.exit(1)
    else:
        print(f"\nAll {len(target_band_ids)} requested band(s) already present and valid.")

    # ── Step 3: Generate images ───────────────────────────────────────────────
    output_dir = os.path.dirname(list(band_files.values())[0])
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating images for {len(band_files)} band(s)...")
    print("-" * 60)
    _generate_images(band_files, output_dir)


if __name__ == "__main__":
    main()