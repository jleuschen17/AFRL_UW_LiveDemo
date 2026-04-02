#!/usr/bin/env python3
"""
Plot waterfalls for Starlink SigMF frequency-hopped captures.

Output structure
----------------
  plots/
    CF_11.325GHz/          # one folder per band
      capture_001.png      # one plot per capture in that band
      capture_002.png
      ...
    CF_11.575GHz/
      capture_001.png
      ...
    band_masters/          # only with --master flag
      waterfall_CF_11.325GHz.png
      waterfall_CF_11.575GHz.png

Usage
-----
  python plot_waterfalls.py                            # per-capture plots, all bands
  python plot_waterfalls.py --bands 1                  # only the first band
  python plot_waterfalls.py --bands 1,3,5              # bands 1, 3, and 5 only
  python plot_waterfalls.py --captures 1               # only first capture per band
  python plot_waterfalls.py --bands 2 --captures 1,2   # band 2, captures 1 & 2 only
  python plot_waterfalls.py --master                   # also build band-master waterfalls
  python plot_waterfalls.py --master --ds 8            # all plots with 8x time downsample
  python plot_waterfalls.py --ds 4 --ds-method max     # 4x downsample, keep bright features
  python plot_waterfalls.py --db 130,115               # colorbar range [-130, -115] dB
  python plot_waterfalls.py --output my_plots          # save to my_plots/ instead of plots1/
"""

import argparse
import os
import gc
import glob
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sigmf


# ================== CONFIG ==================
ROOT_DIR = "/home/mowerj/AFRL_UW_LiveDemo/testCaptures/starlink_sigmf_20260402T184346"

NFFT = 1024
NOTCH_WIDTH = 40

MIN_DB = -115
MAX_DB = -90

# Max samples per chunk for chunked spectrogram (~64 MB of cf64)
CHUNK_SAMPLES = 8 * 1024 * 1024

# Default time-axis downsample factor (1 = no downsampling)
DEFAULT_DS = 1

# Base plots directory inside ROOT_DIR
PLOTS_DIR = os.path.join(ROOT_DIR, "plots1")
# ============================================


def parse_iso8601(s):
    """Parse ISO8601 string like '2025-11-20T18:57:42.123456Z'."""
    if s is None:
        return None
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def get_num_samples(data_path):
    """Compute number of complex64 samples from file size."""
    item_bytes = np.dtype(np.complex64).itemsize
    return os.path.getsize(data_path) // item_bytes


def find_sigmf_records(root_dir):
    """Scan directory for .sigmf-meta files and return a list of record dicts."""
    meta_files = glob.glob(os.path.join(root_dir, "*.sigmf-meta"))
    records = []

    for meta_path in sorted(meta_files):
        data_path = meta_path.replace(".sigmf-meta", ".sigmf-data")
        if not os.path.exists(data_path):
            print(f"[WARN] Missing data for {meta_path}, skipping")
            continue

        with open(meta_path, "r") as f:
            meta_content = f.read()

        try:
            handle = sigmf.SigMFFile(metadata=meta_content, data_file=data_path)
        except Exception as e:
            print(f"[WARN] Failed to open SigMF for {meta_path}: {e}")
            continue

        sr = handle.get_global_field("core:sample_rate")
        captures = handle.get_captures()
        if not captures:
            print(f"[WARN] No captures in {meta_path}, skipping")
            continue

        cap0 = captures[0]
        cf = cap0.get("core:frequency", None)
        dt_str = cap0.get("core:datetime", None)
        capture_time = parse_iso8601(dt_str)

        if cf is None or capture_time is None:
            print(f"[WARN] Missing cf/datetime in {meta_path}, skipping")
            continue

        records.append(
            {
                "meta": meta_path,
                "data": data_path,
                "cf": float(cf),
                "sr": float(sr),
                "capture_time": capture_time,
            }
        )

    return records


def spectrogram_chunked(data_path, sr, nfft, chunk_samples):
    """
    Compute spectrogram of a (potentially large) file by reading bounded
    chunks through a memory-mapped view.

    Returns
    -------
    f_shifted : ndarray [nfft]
    t_all     : ndarray [time_bins]   — seconds from file start
    Sxx_db    : ndarray [nfft, time_bins] — float32 dB
    """
    n_total = get_num_samples(data_path)
    if n_total < nfft:
        return None, None, None

    mmap = np.memmap(data_path, dtype=np.complex64, mode="r", shape=(n_total,))

    noverlap = nfft // 8
    f_shifted = None
    Sxx_segments = []
    t_segments = []

    start = 0
    while start < n_total:
        end = min(start + chunk_samples, n_total)
        if (end - start) < nfft:
            break

        chunk = np.array(mmap[start:end])

        f, t, Sxx = signal.spectrogram(
            chunk, fs=sr, return_onesided=False, nperseg=nfft,
        )

        Sxx_shifted = np.fft.fftshift(Sxx, axes=0)
        if f_shifted is None:
            f_shifted = np.fft.fftshift(f)

        Sxx_db = 10.0 * np.log10(np.abs(Sxx_shifted) + 1e-15)

        t_segments.append(t + start / sr)
        Sxx_segments.append(Sxx_db.astype(np.float32))

        del chunk, Sxx, Sxx_shifted, Sxx_db
        gc.collect()

        start = end - nfft
        if end == n_total:
            break

    del mmap
    gc.collect()

    if not Sxx_segments:
        return None, None, None

    Sxx_full = np.concatenate(Sxx_segments, axis=1)
    t_full = np.concatenate(t_segments)

    del Sxx_segments, t_segments
    gc.collect()

    return f_shifted, t_full, Sxx_full


def downsample_time(Sxx, t, factor, method="mean"):
    """
    Downsample spectrogram along the time axis.
    method="mean" averages blocks (smooth, but dilutes narrow features).
    method="max"  takes the max per block (preserves bright traces).
    """
    if factor <= 1:
        return Sxx, t

    n_bins = Sxx.shape[1]
    n_keep = (n_bins // factor) * factor  # trim to multiple of factor

    Sxx_trimmed = Sxx[:, :n_keep]
    t_trimmed = t[:n_keep]

    # Reshape and reduce
    Sxx_reshaped = Sxx_trimmed.reshape(Sxx.shape[0], -1, factor)
    if method == "max":
        Sxx_ds = Sxx_reshaped.max(axis=2)
    else:
        Sxx_ds = Sxx_reshaped.mean(axis=2)

    t_ds = t_trimmed.reshape(-1, factor).mean(axis=1)

    return Sxx_ds.astype(np.float32), t_ds


def plot_single_capture(rec, t0_global, band_dir, capture_idx, db_min, db_max, ds_factor=1, ds_method="mean"):
    """
    Compute spectrogram for one capture and save its plot.
    Memory is freed before returning.
    """
    data_path = rec["data"]
    cf = rec["cf"]
    sr = rec["sr"]
    t_capture = rec["capture_time"]

    f_shifted, t_local, Sxx_db = spectrogram_chunked(
        data_path, sr, NFFT, CHUNK_SAMPLES,
    )
    if f_shifted is None:
        print(f"    [WARN] No spectrogram for {os.path.basename(rec['meta'])}")
        return

    # Optional time-axis downsample (no-op when ds_factor <= 1)
    Sxx_db, t_local = downsample_time(Sxx_db, t_local, ds_factor, ds_method)

    f_axis = f_shifted + cf
    dt_capture = (t_capture - t0_global).total_seconds()
    t_global = t_local + dt_capture

    extent = [f_axis[0], f_axis[-1], t_global[0], t_global[-1]]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        Sxx_db.T,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="turbo",
        interpolation="bilinear",
        vmin=db_min,
        vmax=db_max,
    )
    fig.colorbar(im, ax=ax, label="Power (dB)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Time (s from first capture)")

    cap_time_str = t_capture.strftime("%H:%M:%S")
    ax.set_title(
        f"CF ≈ {cf/1e9:.3f} GHz | Capture {capture_idx:03d} ({cap_time_str})"
    )
    fig.tight_layout()

    out_path = os.path.join(band_dir, f"capture_{capture_idx:03d}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"    Saved {out_path}  ({Sxx_db.shape[1]} time bins)")

    del f_shifted, t_local, Sxx_db, f_axis, t_global
    gc.collect()


def build_band_master(records_band, t0_global, masters_dir, ds_factor, db_min, db_max, ds_method="mean"):
    """
    Stitch all captures for one band into a single waterfall.
    Each capture's spectrogram is downsampled before accumulating
    to keep total memory bounded.
    """
    records_band = sorted(records_band, key=lambda r: r["capture_time"])
    cf = records_band[0]["cf"]

    f_axis = None
    Sxx_segments = []
    t_segments = []

    for rec in records_band:
        data_path = rec["data"]
        sr = rec["sr"]
        t_capture = rec["capture_time"]

        f_shifted, t_local, Sxx_db = spectrogram_chunked(
            data_path, sr, NFFT, CHUNK_SAMPLES,
        )
        if f_shifted is None:
            print(f"    [WARN] Skipping {os.path.basename(rec['meta'])}")
            continue

        # Downsample this capture's time axis to save RAM
        Sxx_ds, t_ds = downsample_time(Sxx_db, t_local, ds_factor, ds_method)
        del Sxx_db, t_local
        gc.collect()

        f_final = f_shifted + cf
        dt_capture = (t_capture - t0_global).total_seconds()
        t_global = t_ds + dt_capture

        if f_axis is None:
            f_axis = f_final
        else:
            if len(f_final) != len(f_axis) or not np.allclose(
                f_final, f_axis, rtol=0, atol=1.0
            ):
                print(
                    f"    [WARN] Freq axis mismatch in "
                    f"{os.path.basename(rec['meta'])}, skipping"
                )
                del f_shifted, Sxx_ds, t_global
                gc.collect()
                continue

        Sxx_segments.append(Sxx_ds)
        t_segments.append(t_global)

        print(
            f"    Master: used {os.path.basename(rec['meta'])} "
            f"({Sxx_ds.shape[1]} bins after {ds_factor}x downsample)"
        )

        del f_shifted, Sxx_ds, t_global, f_final
        gc.collect()

    if not Sxx_segments:
        print(f"    [WARN] No data for band master CF {cf/1e9:.3f} GHz")
        return

    Sxx_full = np.concatenate(Sxx_segments, axis=1)
    t_full = np.concatenate(t_segments)
    del Sxx_segments, t_segments
    gc.collect()

    extent = [f_axis[0], f_axis[-1], t_full[0], t_full[-1]]

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        Sxx_full.T,
        extent=extent,
        aspect="auto",
        origin="lower",
        cmap="turbo",
        interpolation="bilinear",
        vmin=db_min,
        vmax=db_max,
    )
    fig.colorbar(im, ax=ax, label="Power (dB)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Time (s from first capture)")
    ax.set_title(f"Band Master: CF ≈ {cf/1e9:.3f} GHz ({ds_factor}x downsampled)")
    fig.tight_layout()

    out_path = os.path.join(masters_dir, f"waterfall_CF_{cf/1e9:.3f}GHz.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"    Saved band master: {out_path}")

    del f_axis, t_full, Sxx_full
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Plot Starlink SigMF waterfalls (per-capture + optional band masters)"
    )
    parser.add_argument(
        "--master", action="store_true",
        help="Also generate stitched band-master waterfalls (in band_masters/)"
    )
    parser.add_argument(
        "--ds", type=int, default=DEFAULT_DS,
        help=f"Time-axis downsample factor for all plots (default {DEFAULT_DS}, i.e. no downsampling). "
             "e.g. --ds 4 averages every 4 time bins into 1."
    )
    parser.add_argument(
        "--ds-method", type=str, default="mean", choices=["mean", "max"],
        help="Downsample method: 'mean' averages bins (default), 'max' preserves bright features."
    )
    parser.add_argument(
        "--bands", type=str, default=None,
        help="Comma-separated 1-indexed band numbers to process, e.g. --bands 1,2,3. "
             "Omit to process all bands."
    )
    parser.add_argument(
        "--captures", type=str, default=None,
        help="Comma-separated 1-indexed capture numbers to plot per band, e.g. --captures 1,2. "
             "Omit to plot all captures."
    )
    parser.add_argument(
        "--db", type=str, default=None,
        help="Colorbar dB range as low,high (assumes negative), e.g. --db 130,115 → [-130, -115]. "
             f"Default: {MIN_DB},{MAX_DB}"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory name (created inside ROOT_DIR). Default: plots1"
    )
    args = parser.parse_args()

    # Output directory
    plots_dir = os.path.join(ROOT_DIR, args.output) if args.output else PLOTS_DIR

    # Override dB range if specified
    db_min = MIN_DB
    db_max = MAX_DB
    if args.db is not None:
        try:
            parts = [x.strip() for x in args.db.split(",")]
            db_min = -abs(float(parts[0]))
            db_max = -abs(float(parts[1]))
        except (ValueError, IndexError):
            print(f"[ERROR] --db must be two numbers like --db 130,115, got: {args.db}")
            return
        if db_min >= db_max:
            print(f"[ERROR] --db low must be < high, got {db_min} >= {db_max}")
            return
        print(f"Using dB range: [{db_min}, {db_max}]")

    records = find_sigmf_records(ROOT_DIR)
    if not records:
        print(f"No valid SigMF captures found in {ROOT_DIR}")
        return

    print(f"Found {len(records)} SigMF captures")

    t0_global = min(r["capture_time"] for r in records)
    print(f"Global start time: {t0_global}")

    # Group by center frequency
    bands = {}
    for r in records:
        bands.setdefault(r["cf"], []).append(r)

    cfs = sorted(bands.keys())
    print(f"Discovered {len(cfs)} distinct center frequencies:")
    for i, cf in enumerate(cfs, start=1):
        print(f"  Band {i}: {cf/1e9:.3f} GHz  ({len(bands[cf])} captures)")

    # Filter to selected bands if --bands specified
    if args.bands is not None:
        try:
            selected = [int(x.strip()) for x in args.bands.split(",")]
        except ValueError:
            print(f"[ERROR] --bands must be comma-separated integers, got: {args.bands}")
            return
        for s in selected:
            if s < 1 or s > len(cfs):
                print(f"[ERROR] Band {s} out of range (1-{len(cfs)})")
                return
        cfs = [cfs[s - 1] for s in selected]
        print(f"Processing selected bands: {[f'{cf/1e9:.3f} GHz' for cf in cfs]}")

    # Parse --captures selection
    selected_captures = None
    if args.captures is not None:
        try:
            selected_captures = [int(x.strip()) for x in args.captures.split(",")]
        except ValueError:
            print(f"[ERROR] --captures must be comma-separated integers, got: {args.captures}")
            return
        for s in selected_captures:
            if s < 1:
                print(f"[ERROR] Capture index must be >= 1, got {s}")
                return

    # ---- Per-capture plots (always) ----
    os.makedirs(plots_dir, exist_ok=True)

    for cf in cfs:
        band_label = f"CF_{cf/1e9:.3f}GHz"
        band_dir = os.path.join(plots_dir, band_label)
        os.makedirs(band_dir, exist_ok=True)

        band_records = sorted(bands[cf], key=lambda r: r["capture_time"])
        n_caps = len(band_records)
        print(f"\n=== {band_label} ({n_caps} captures) ===")

        if selected_captures is not None:
            bad = [s for s in selected_captures if s > n_caps]
            if bad:
                print(f"  [WARN] Capture(s) {bad} out of range (band has {n_caps} captures)")

        for idx, rec in enumerate(band_records, start=1):
            if selected_captures is not None and idx not in selected_captures:
                continue
            print(f"  Capture {idx}/{n_caps}: "
                  f"{os.path.basename(rec['meta'])}")
            plot_single_capture(rec, t0_global, band_dir, idx, db_min, db_max, args.ds, args.ds_method)

    # ---- Band masters (only with --master) ----
    if args.master:
        masters_dir = os.path.join(plots_dir, "band_masters")
        os.makedirs(masters_dir, exist_ok=True)

        for cf in cfs:
            print(f"\n=== Band master: CF = {cf/1e9:.3f} GHz "
                  f"(ds={args.ds}x) ===")
            build_band_master(bands[cf], t0_global, masters_dir, args.ds, db_min, db_max, args.ds_method)

    print(f"\nAll plots saved in: {plots_dir}")


if __name__ == "__main__":
    main()