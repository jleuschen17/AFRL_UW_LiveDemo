#!/usr/bin/env python3
"""
demo_2.py — Starlink USRP Capture, Plotting & Doppler Correlation GUI

Three-stage GUI:
  1. Capture window: configure and run USRP frequency-hopping captures
  2. Plotting window: interactively view/adjust waterfall spectrograms
  3. Correlator window: run Doppler satellite correlation pipeline

Usage:
    python demo_2.py                  # opens capture window
    python demo_2.py --plot DIR       # skip to plotter
    python demo_2.py --correlate DIR  # skip to correlator
"""

import argparse
import gc
import glob
import json
import math
import os
import queue
import re
import sys
import threading
import time as time_mod
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy import signal, constants

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ===================================================================
# Constants
# ===================================================================
C = constants.c
COLORMAPS = ["turbo", "jet", "jet_r", "viridis", "plasma", "inferno",
             "magma", "hot", "hot_r", "coolwarm", "gray"]
NFFT_OPTIONS = [256, 512, 1024, 2048, 4096]


# ===================================================================
# Spectrogram helpers (from plotting.py)
# ===================================================================

def parse_iso8601(s):
    if s is None:
        return None
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def get_num_samples(data_path):
    return os.path.getsize(data_path) // np.dtype(np.complex64).itemsize


def find_sigmf_records(root_dir):
    """Scan directory for .sigmf-meta files, return list of record dicts."""
    meta_files = glob.glob(os.path.join(root_dir, "*.sigmf-meta"))
    records = []
    for meta_path in sorted(meta_files):
        data_path = meta_path.replace(".sigmf-meta", ".sigmf-data")
        if not os.path.exists(data_path):
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        sr = meta.get("global", {}).get("core:sample_rate")
        captures = meta.get("captures", [])
        if not captures or sr is None:
            continue
        cap0 = captures[0]
        cf = cap0.get("core:frequency")
        dt_str = cap0.get("core:datetime")
        capture_time = parse_iso8601(dt_str)
        if cf is None or capture_time is None:
            continue
        records.append({
            "meta": meta_path,
            "data": data_path,
            "cf": float(cf),
            "sr": float(sr),
            "capture_time": capture_time,
        })
    return records


def spectrogram_chunked(data_path, sr, nfft, chunk_samples=8 * 1024 * 1024):
    """Compute spectrogram of a large file via memory-mapped chunks."""
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
        Sxx_db = 10.0 * np.log10(np.abs(Sxx_shifted) + 1e-15).astype(np.float32)
        t_segments.append(t + start / sr)
        Sxx_segments.append(Sxx_db)
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
    if factor <= 1:
        return Sxx, t
    n_keep = (Sxx.shape[1] // factor) * factor
    Sxx_trimmed = Sxx[:, :n_keep]
    t_trimmed = t[:n_keep]
    Sxx_reshaped = Sxx_trimmed.reshape(Sxx.shape[0], -1, factor)
    if method == "max":
        Sxx_ds = Sxx_reshaped.max(axis=2)
    else:
        Sxx_ds = Sxx_reshaped.mean(axis=2)
    t_ds = t_trimmed.reshape(-1, factor).mean(axis=1)
    return Sxx_ds.astype(np.float32), t_ds


# ===================================================================
# Ground station defaults
# ===================================================================
DEFAULT_LAT = 47.655
DEFAULT_LON = -122.3035
DEFAULT_ALT = 60.0


# ===================================================================
# Correlator functions (from correlator_combined.py)
# ===================================================================

def corr_run_preprocessing(capture_dir, freq_ghz, sample_rate=500000.0,
                           k=10, nfft=1024, noverlap=512, log_fn=print):
    """Rebuild velocity waterfall from .txt compressed files."""
    txt_pattern = f"r*_f{freq_ghz:.3f}GHz_*.txt"
    txt_files = sorted(glob.glob(os.path.join(capture_dir, txt_pattern)))

    if not txt_files:
        log_fn(f"[preprocess] No .txt for {freq_ghz:.3f} GHz, "
               f"compressing from .sigmf...")
        sigmf_pattern = f"r*_f{freq_ghz:.3f}GHz_*.sigmf-data"
        sigmf_files = sorted(glob.glob(
            os.path.join(capture_dir, sigmf_pattern)))
        if not sigmf_files:
            raise FileNotFoundError(
                f"No .sigmf-data or .txt for {freq_ghz:.3f} GHz "
                f"in {capture_dir}")
        _corr_compress_sigmf(sigmf_files, k, nfft, noverlap, log_fn)
        txt_files = sorted(glob.glob(
            os.path.join(capture_dir, txt_pattern)))
        if not txt_files:
            raise FileNotFoundError("Compression produced no .txt files")

    log_fn(f"[preprocess] Building waterfall from "
           f"{len(txt_files)} .txt files...")
    tag = _corr_build_waterfall(
        txt_files, capture_dir, freq_ghz, sample_rate, k, nfft, noverlap,
        log_fn)
    return tag


def _corr_compress_sigmf(sigmf_files, k, nfft, noverlap, log_fn=print):
    """Compress raw .sigmf-data into sparse .txt."""
    for data_path in sigmf_files:
        base = data_path.replace(".sigmf-data", "")
        meta_path = base + ".sigmf-meta"
        if not os.path.exists(meta_path):
            continue
        log_fn(f"  Compressing {os.path.basename(base)}...")
        with open(meta_path) as f:
            meta = json.load(f)
        sr = meta.get("global", {}).get("core:sample_rate", 500000.0)
        cf = meta["captures"][0]["core:frequency"]
        dtype_str = meta.get("global", {}).get("core:datatype", "cf32_le")
        data_dtype = np.complex64 if dtype_str == "cf32_le" else np.int16

        raw = np.fromfile(data_path, dtype=data_dtype)
        f_ax, t_ax, stft = signal.stft(
            raw, fs=sr, nperseg=nfft, noverlap=noverlap,
            return_onesided=False)
        stft = np.fft.fftshift(stft, axes=0)
        f_ax = np.fft.fftshift(f_ax)

        dc_kill = np.where(np.abs(f_ax) <= 5000)
        nf = stft.shape[1]
        out_vals = np.zeros((nf, k), dtype=complex)
        out_freqs = np.zeros((nf, k))
        for i in range(nf):
            row = np.abs(stft[:, i])
            row[dc_kill] = 0
            top = np.argpartition(row, -k)[-k:]
            top = top[np.argsort(f_ax[top])]
            out_vals[i, :] = stft[top, i]
            out_freqs[i, :] = C * f_ax[top] / cf

        np.savetxt(base + ".txt",
                   np.column_stack((out_freqs, out_vals)),
                   fmt="%.10g", delimiter=" ")
        log_fn(f"  -> {os.path.basename(base)}.txt")


def _corr_build_waterfall(txt_files, capture_dir, freq_ghz, sample_rate,
                           k, nfft, noverlap, log_fn=print):
    """Reconstruct velocity waterfall from compressed .txt files."""
    v_max = 5000
    n_vel = 500
    v_axis = np.linspace(-v_max, v_max, n_vel)

    for j, txt_path in enumerate(txt_files):
        fname = os.path.basename(txt_path)
        time_match = re.search(r"_(\d{8}T\d{6})", fname)
        if not time_match:
            log_fn(f"  [WARN] Cannot parse time from {fname}")
            continue
        capture_time = datetime.strptime(
            time_match.group(1), "%Y%m%dT%H%M%S")

        data = np.loadtxt(txt_path, dtype=complex)
        num_rows = data.shape[0]
        step = nfft - noverlap
        t = (np.arange(num_rows) * step + nfft / 2) / sample_rate

        recon_freqs = np.real(data[:, 0:k])
        recon_vals = data[:, k:]

        max_freq = np.max(np.abs(recon_freqs))
        if max_freq > v_max * 2:
            cf_hz = freq_ghz * 1e9
            recon_freqs = C * recon_freqs / cf_hz
            log_fn(f"  [auto] Converted Hz->m/s "
                   f"(max {max_freq:.0f} Hz -> "
                   f"{np.max(np.abs(recon_freqs)):.0f} m/s)")

        sparse = np.zeros((len(v_axis), num_rows), dtype=complex)
        for i in range(num_rows):
            idx = np.searchsorted(v_axis, recon_freqs[i, :])
            idx = np.clip(idx, 0, len(v_axis) - 1)
            sparse[idx, i] = recon_vals[i, :]

        tag = f"r{j}_waterfall_CF_{freq_ghz:.3f}GHz"
        np.save(os.path.join(capture_dir, f"{tag}.npy"), sparse.T)
        np.save(os.path.join(capture_dir, f"rel_vel_{tag}.npy"), v_axis)
        np.save(os.path.join(capture_dir, f"time_{tag}.npy"), [t])

        stop_np = np.datetime64(capture_time)
        offset_ns = (np.asarray(t) * 1e9).astype(np.int64)
        start_np = stop_np - offset_ns[-1].astype("timedelta64[ns]")
        datetimes = start_np + offset_ns.astype("timedelta64[ns]")
        np.save(os.path.join(
            capture_dir, f"datetime_updated_{tag}.npy"), datetimes)

        log_fn(f"  -> {tag} ({num_rows} frames, "
               f"{datetimes[0]} to {datetimes[-1]})")

    return f"r0_waterfall_CF_{freq_ghz:.3f}GHz"


def corr_load_waterfall(capture_dir, tag, n_vel_out=300, log_fn=print):
    """Load waterfall, downsample to 1 Hz, convert to dB."""
    raw = np.load(os.path.join(capture_dir, f"{tag}.npy"))
    time_axis = np.load(os.path.join(
        capture_dir, f"datetime_updated_{tag}.npy"))
    vel_axis = np.load(os.path.join(
        capture_dir, f"rel_vel_{tag}.npy")) / 1000.0

    is_complex = np.iscomplexobj(raw)
    if is_complex:
        power_lin = np.abs(raw) ** 2
    else:
        power_lin = raw.astype(np.float64)

    t_utc = pd.to_datetime(
        np.asarray(time_axis), utc=True, errors="raise")
    assert len(t_utc) == power_lin.shape[0]

    df = pd.DataFrame(power_lin)
    df["t_sec"] = t_utc.floor("s")
    agg = df.groupby("t_sec").mean(numeric_only=True)
    wf = agg.to_numpy(dtype=np.float64)
    t1hz = agg.index

    if is_complex:
        wf = 10.0 * np.log10(wf + 1e-30)
    wf = wf.astype(np.float32)

    vel_in = np.asarray(vel_axis, dtype=np.float64).ravel()
    V_in = wf.shape[1]
    if len(vel_in) == V_in + 1:
        vel_in = 0.5 * (vel_in[:-1] + vel_in[1:])
    vel_out = np.linspace(vel_in.min(), vel_in.max(), n_vel_out)
    wf_out = np.empty((wf.shape[0], n_vel_out), dtype=np.float32)
    for i in range(wf.shape[0]):
        wf_out[i, :] = np.interp(
            vel_out, vel_in, wf[i, :]).astype(np.float32)

    log_fn(f"[load] Shape: {wf_out.shape}, {t1hz[0]} -> {t1hz[-1]}, "
           f"vel: [{vel_out.min():.1f}, {vel_out.max():.1f}] km/s")
    return wf_out, t1hz, vel_out


def corr_download_tle(log_fn=print):
    """Download fresh Starlink TLEs from Celestrak."""
    import requests
    url = ("https://celestrak.org/NORAD/elements/"
           "gp.php?GROUP=starlink&FORMAT=tle")
    log_fn("[tle] Downloading fresh TLE from Celestrak...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "DopplerPredictor", "starlink_fresh.txt")
    with open(out, "w") as f:
        f.write(resp.text)
    n_sats = resp.text.count("\n") // 3
    log_fn(f"[tle] Downloaded {n_sats} satellites -> {out}")
    return resp.text


def corr_predict_doppler(tle_data, start_utc, duration_sec, tx_freq_hz,
                          elevation_mask=10.0, lat=DEFAULT_LAT,
                          lon=DEFAULT_LON, alt=DEFAULT_ALT, log_fn=print):
    """Predict Doppler for all visible satellites at 1 Hz."""
    from skyfield.api import EarthSatellite, load, wgs84, utc

    ts = load.timescale()
    observer = wgs84.latlon(lat, lon, elevation_m=alt)

    lines = tle_data.strip().split("\n")
    sats = []
    i = 0
    while i < len(lines) - 2:
        name = lines[i].strip()
        l1 = lines[i + 1].strip()
        l2 = lines[i + 2].strip()
        if l1.startswith("1 ") and l2.startswith("2 "):
            try:
                sat = EarthSatellite(l1, l2, name, ts)
                cat_num = l1[2:7].strip()
                sats.append((f"SAT-{cat_num}", sat))
            except Exception:
                pass
        i += 3

    log_fn(f"[predict] {len(sats)} satellites loaded, "
           f"predicting {int(duration_sec)}s window...")

    times_dt = [start_utc + timedelta(seconds=s)
                for s in range(int(duration_sec))]
    times_sky = ts.from_datetimes(
        [t.replace(tzinfo=utc) if t.tzinfo is None else t
         for t in times_dt])

    results = {}
    n_visible = 0
    for sat_name, sat in sats:
        try:
            topo = (sat - observer).at(times_sky)
            alt_deg, _, dist = topo.altaz()
            visible = alt_deg.degrees >= elevation_mask
            if not np.any(visible):
                continue
            n_visible += 1
            dist_km = dist.km
            range_rate = np.gradient(dist_km, 1.0)
            idx = np.where(visible)[0]
            rows = [{
                "timestamp": times_dt[j],
                "relative_velocity_kms": float(range_rate[j]),
                "distance_km": float(dist_km[j]),
                "elevation_deg": float(alt_deg.degrees[j]),
            } for j in idx]
            if rows:
                results[sat_name] = pd.DataFrame(rows)
        except Exception:
            continue

    log_fn(f"[predict] {n_visible} satellites visible "
           f"above {elevation_mask} deg")
    return results


def corr_fspl_db(dist_km, tx_freq_hz):
    dist_m = np.asarray(dist_km, dtype=np.float64) * 1000.0
    return (20.0 * np.log10(dist_m + 1e-12)
            + 20.0 * np.log10(tx_freq_hz + 1e-12)
            + 20.0 * np.log10(4.0 * np.pi / C))


def corr_build_sim_waterfall(pred_df, t1hz, vel_kms, tx_freq_hz,
                              sigma_kms=0.05, background_db=250.0):
    pred_t = pd.to_datetime(
        pred_df["timestamp"], utc=True).dt.floor("s")
    row_idx = t1hz.get_indexer(pred_t)
    ok = row_idx >= 0
    if ok.sum() == 0:
        return None, None, None

    row_idx = row_idx[ok]
    df_ok = pred_df.iloc[np.where(ok)[0]]

    T = len(t1hz)
    V = len(vel_kms)
    sim = np.full((T, V), background_db, dtype=np.float32)

    vel_sim = -df_ok["relative_velocity_kms"].to_numpy()
    dist_km = df_ok["distance_km"].to_numpy()
    pl_db = corr_fspl_db(dist_km, tx_freq_hz)

    vc = np.asarray(vel_kms, dtype=np.float64)
    for ti, v_k, pl in zip(row_idx, vel_sim, pl_db):
        gaussian = np.exp(-0.5 * ((vc - v_k) / sigma_kms) ** 2)
        row = (pl * (1.0 - 0.9 * gaussian)
               + background_db * (1.0 - gaussian))
        sim[ti, :] = np.minimum(sim[ti, :], row.astype(np.float32))

    return sim, row_idx, vel_sim


def corr_correlate(waterfall, vel_kms, t1hz, predictions, tx_freq_hz,
                    log_fn=print):
    """Run NCC ridge correlation for all predicted satellites."""
    T, V = waterfall.shape
    results = []

    for sat_name, pred_df in predictions.items():
        sim, row_idx, vel_sim = corr_build_sim_waterfall(
            pred_df, t1hz, vel_kms, tx_freq_hz)
        if sim is None:
            continue

        t_lo, t_hi = int(row_idx.min()), int(row_idx.max())
        if t_hi - t_lo < 10:
            continue

        v_pad = 0.5
        col_lo = max(0, int(np.searchsorted(
            vel_kms, vel_sim.min() - v_pad)))
        col_hi = min(V, int(np.searchsorted(
            vel_kms, vel_sim.max() + v_pad)))
        if col_hi - col_lo < 10:
            continue

        real_ridge = np.argmax(
            waterfall[t_lo:t_hi + 1, col_lo:col_hi], axis=1)
        sim_ridge = np.argmin(
            sim[t_lo:t_hi + 1, col_lo:col_hi], axis=1)

        x = real_ridge.astype(float)
        y = sim_ridge.astype(float)
        x -= x.mean()
        y -= y.mean()
        xcorr = np.correlate(x, y, mode="full")
        denom = np.linalg.norm(x) * np.linalg.norm(y) + 1e-12
        ncc_full = xcorr / denom
        best_ncc = float(np.max(ncc_full))

        if np.isnan(best_ncc):
            continue

        results.append({
            "name": sat_name,
            "ncc": best_ncc,
            "sim": sim,
            "t_lo": t_lo,
            "t_hi": t_hi,
            "col_lo": col_lo,
            "col_hi": col_hi,
        })

    results.sort(key=lambda x: x["ncc"], reverse=True)
    log_fn(f"[correlate] {len(results)} satellites correlated")
    return results


# ===================================================================
# Capture class (from capture.py, with log callback)
# ===================================================================

class StarlinkSigMFCapture:
    def __init__(self, usrp_args="type=b200", sample_rate=0.5e6, gain=7,
                 lnb_lo=9.75e9, output_root="./testCaptures", log_fn=None):
        self.sample_rate = float(sample_rate)
        self.gain = float(gain)
        self.lnb_lo = float(lnb_lo)
        self.log = log_fn or print

        import uhd
        self.uhd = uhd

        self.log(f"Connecting to USRP: {usrp_args or 'default'}")
        self.usrp = uhd.usrp.MultiUSRP(usrp_args)
        self.usrp.set_rx_rate(self.sample_rate)
        self.usrp.set_rx_gain(self.gain)
        self.usrp.set_rx_antenna("RX2")

        self.log(f"Sample rate: {self.usrp.get_rx_rate()/1e6:.3f} MS/s")
        self.log(f"Gain: {self.usrp.get_rx_gain():.1f} dB")
        self.log(f"LNB LO: {self.lnb_lo/1e9:.6f} GHz")

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.output_dir = os.path.join(output_root, f"starlink_sigmf_{ts}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.log(f"Output: {self.output_dir}")

        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [0]
        self.rx_streamer = self.usrp.get_rx_stream(st_args)

        self.log_path = os.path.join(self.output_dir, "capture_log.txt")
        with open(self.log_path, "w") as f:
            f.write(f"Starlink SigMF Capture\n")
            f.write(f"Start: {datetime.now()}\n")
            f.write(f"SR: {self.sample_rate} Hz, Gain: {self.gain} dB, "
                    f"LO: {self.lnb_lo} Hz\n\n")

    def set_frequency(self, target_freq_hz):
        target_freq_hz = float(target_freq_hz)
        if_freq = target_freq_hz - self.lnb_lo
        if if_freq <= 0:
            self.log(f"WARNING: RF {target_freq_hz/1e9:.3f} GHz < "
                     f"LO {self.lnb_lo/1e9:.3f} GHz")
            return False
        self.usrp.set_rx_freq(
            self.uhd.libpyuhd.types.tune_request(if_freq))
        actual_if = self.usrp.get_rx_freq()
        self.log(f"  RF: {target_freq_hz/1e9:.3f} GHz | "
                 f"IF: {if_freq/1e6:.2f} MHz (actual {actual_if/1e6:.2f} MHz)")
        return True

    @staticmethod
    def _iso8601_utc_now():
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def capture_and_write_streaming(self, num_samples, center_freq_hz,
                                     round_idx, freq_idx,
                                     chunk_size=1_000_000,
                                     nfft=1024, noverlap=512, k=10,
                                     DC_width=5000):
        num_samples = int(num_samples)
        center_freq_hz = float(center_freq_hz)
        uhd = self.uhd

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        base_name = f"r{round_idx:03d}_f{center_freq_hz/1e9:.3f}GHz_{ts}"
        data_path = os.path.join(self.output_dir, base_name + ".sigmf-data")
        meta_path = os.path.join(self.output_dir, base_name + ".sigmf-meta")
        comp_path = os.path.join(self.output_dir, base_name + ".txt")

        recv_buffer = np.zeros(chunk_size, dtype=np.complex64)
        metadata = uhd.types.RXMetadata()
        overflow_count = 0

        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True
        self.rx_streamer.issue_stream_cmd(stream_cmd)

        total_received = 0
        power_accum = 0.0

        self.log("    Phase 1: Streaming to disk...")
        with open(data_path, "wb") as data_file:
            while total_received < num_samples:
                remaining = num_samples - total_received
                request_len = min(chunk_size, remaining)
                samps = self.rx_streamer.recv(
                    recv_buffer[:request_len], metadata)
                if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                    if metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
                        overflow_count += 1
                        continue
                    else:
                        self.log(f"    RX error: {metadata.strerror()}")
                        break
                if samps == 0:
                    continue
                chunk = recv_buffer[:samps]
                chunk.tofile(data_file)
                power_accum += np.sum(np.abs(chunk) ** 2)
                total_received += samps
                if ((total_received // 10_000_000)
                        != ((total_received - samps) // 10_000_000)):
                    pct = 100.0 * total_received / num_samples
                    self.log(f"    ... {total_received:,}/{num_samples:,} "
                             f"({pct:.0f}%)")

        stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        self.rx_streamer.issue_stream_cmd(stop_cmd)

        flush_buf = np.zeros(chunk_size, dtype=np.complex64)
        flush_md = uhd.types.RXMetadata()
        while True:
            n = self.rx_streamer.recv(flush_buf, flush_md, timeout=0.1)
            if n == 0:
                break

        if overflow_count > 0:
            self.log(f"    WARNING: {overflow_count} overflows")
        self.log(f"    Phase 1 done: {total_received:,} samples")

        # Phase 2: compress
        self.log("    Phase 2: Compressing...")
        all_compressed = []
        with open(data_path, "rb") as data_file:
            while True:
                raw = np.fromfile(data_file, dtype=np.complex64,
                                  count=chunk_size)
                if len(raw) == 0:
                    break
                if len(raw) >= nfft:
                    comp = self._compress_samples(
                        raw, self.sample_rate, center_freq_hz,
                        k, nfft, noverlap, DC_width)
                    all_compressed.append(comp)
                del raw

        if all_compressed:
            np.savetxt(comp_path, np.vstack(all_compressed),
                       fmt="%.10g", delimiter=" ")
            del all_compressed
        gc.collect()
        self.log("    Phase 2 done")

        # SigMF metadata
        meta = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": float(self.sample_rate),
                "core:version": "1.0.0",
                "core:description": "Starlink Ku-band capture",
                "core:author": "demo.py",
                "core:recorder": "USRP + LNB",
            },
            "captures": [{
                "core:sample_start": 0,
                "core:frequency": center_freq_hz,
                "core:datetime": self._iso8601_utc_now(),
            }],
            "annotations": [],
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        power_dbm = (10 * np.log10(
            power_accum / max(total_received, 1) + 1e-12) + 30)
        return meta_path, data_path, total_received, power_dbm

    def _compress_samples(self, samples, sample_rate, center_freq,
                          k=10, nfft=1024, noverlap=512, DC_width=5000):
        samples = np.array(samples)
        f, t, STFT = signal.stft(samples, fs=sample_rate, nperseg=nfft,
                                  noverlap=noverlap, return_onesided=False)
        STFT = np.fft.fftshift(STFT, axes=0)
        f = np.fft.fftshift(f)
        numFrames = STFT.shape[1]
        output_vals = np.zeros((numFrames, k), dtype=complex)
        output_freqs = np.zeros((numFrames, k))
        kill_inds = np.where(np.abs(f) <= DC_width)
        for i in range(numFrames):
            working_row = np.abs(STFT[:, i])
            working_row[kill_inds] = 0
            large_inds = np.argpartition(working_row, -k)[-k:]
            large_inds = large_inds[np.argsort(f[large_inds])]
            output_vals[i, :] = STFT[large_inds, i]
            output_freqs[i, :] = f[large_inds]
        return np.column_stack((output_freqs, output_vals))

    def frequency_hopping_capture(self, start_freq=10.7e9, end_freq=12.7e9,
                                   freq_step=125e6, capture_duration=0.5,
                                   total_duration=None, frequencies=None):
        if frequencies is not None:
            freqs = [float(f) for f in frequencies]
        else:
            num_steps = int(math.floor(
                (float(end_freq) - float(start_freq)) / float(freq_step)))
            freqs = [float(start_freq) + i * float(freq_step)
                     for i in range(num_steps + 1)]

        num_freqs = len(freqs)
        num_samples = int(self.sample_rate * float(capture_duration))
        time_per_round = num_freqs * float(capture_duration)

        if total_duration is None:
            num_rounds = 1
        else:
            num_rounds = int(math.ceil(float(total_duration) / time_per_round))

        self.log(f"\n=== Frequency-Hopping Capture ===")
        self.log(f"Frequencies: "
                 f"{[f'{fr/1e9:.3f}' for fr in freqs]} GHz")
        self.log(f"SR: {self.sample_rate/1e6:.3f} MS/s | "
                 f"Duration/freq: {capture_duration:.1f}s")
        self.log(f"Samples/freq: {num_samples:,} | Rounds: {num_rounds}")
        self.log(f"Est total time: "
                 f"{num_rounds * time_per_round:.0f}s\n")

        total_captures = 0
        with open(self.log_path, "a") as log:
            for r in range(1, num_rounds + 1):
                self.log(f"\n===== ROUND {r}/{num_rounds} =====")
                log.write(f"--- Round {r} ---\n")
                for i, rf in enumerate(freqs, 1):
                    total_captures += 1
                    self.log(f"[R{r}/{num_rounds}] [F{i}/{num_freqs}] "
                             f"RF={rf/1e9:.3f} GHz")
                    if not self.set_frequency(rf):
                        log.write(f"{datetime.now()} | R{r} | "
                                  f"{rf/1e9:.3f} GHz | FAIL\n")
                        log.flush()
                        continue
                    time_mod.sleep(0.1)
                    start_t = time_mod.time()
                    meta_path, data_path, num_recv, power_dbm = \
                        self.capture_and_write_streaming(
                            num_samples, rf, r, i)
                    elapsed = time_mod.time() - start_t
                    self.log(f"  {num_recv:,} samples ({elapsed:.1f}s), "
                             f"{power_dbm:.1f} dBm")
                    self.log(f"  -> {os.path.basename(meta_path)}")
                    log.write(f"{datetime.now()} | R{r} | "
                              f"{rf/1e9:.3f} GHz | {num_recv} samps | "
                              f"{power_dbm:.1f} dBm | {elapsed:.1f}s\n")
                    log.flush()

        self.log(f"\n=== Capture complete! {total_captures} files "
                 f"-> {self.output_dir} ===")
        return self.output_dir


# ===================================================================
# Capture GUI
# ===================================================================

class CaptureWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Starlink Demo - Capture")
        self.root.geometry("750x720")
        self.root.minsize(650, 550)

        self.log_queue = queue.Queue()
        self.capture_thread = None
        self.output_dir = None

        self._build_ui()
        self._poll_log()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # ---- Hardware ----
        hw = ttk.LabelFrame(main, text="Hardware", padding=5)
        hw.pack(fill=tk.X, pady=(0, 5))
        hw.columnconfigure(1, weight=1)
        hw.columnconfigure(3, weight=1)

        ttk.Label(hw, text="USRP Device:").grid(
            row=0, column=0, sticky=tk.W, padx=2)
        self.usrp_var = tk.StringVar(value="type=b200")
        ttk.Entry(hw, textvariable=self.usrp_var, width=20).grid(
            row=0, column=1, sticky=tk.W)

        ttk.Label(hw, text="Sample Rate (MS/s):").grid(
            row=0, column=2, sticky=tk.W, padx=(15, 2))
        self.sr_var = tk.StringVar(value="0.5")
        ttk.Entry(hw, textvariable=self.sr_var, width=8).grid(
            row=0, column=3, sticky=tk.W)

        ttk.Label(hw, text="Gain (dB):").grid(
            row=1, column=0, sticky=tk.W, padx=2)
        self.gain_var = tk.StringVar(value="40")
        ttk.Entry(hw, textvariable=self.gain_var, width=8).grid(
            row=1, column=1, sticky=tk.W)

        ttk.Label(hw, text="LNB LO (GHz):").grid(
            row=1, column=2, sticky=tk.W, padx=(15, 2))
        self.lo_var = tk.StringVar(value="9.7500002")
        ttk.Entry(hw, textvariable=self.lo_var, width=14).grid(
            row=1, column=3, sticky=tk.W)

        # ---- Frequencies ----
        freq_frame = ttk.LabelFrame(main, text="Frequencies", padding=5)
        freq_frame.pack(fill=tk.X, pady=(0, 5))

        self.freq_mode = tk.StringVar(value="specific")

        row_specific = ttk.Frame(freq_frame)
        row_specific.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(row_specific, text="Specific CFs (GHz):",
                         variable=self.freq_mode, value="specific",
                         command=self._toggle_freq_mode).pack(side=tk.LEFT)
        self.specific_var = tk.StringVar(value="11.575, 12.325")
        self.specific_entry = ttk.Entry(
            row_specific, textvariable=self.specific_var, width=35)
        self.specific_entry.pack(side=tk.LEFT, padx=5)

        row_sweep = ttk.Frame(freq_frame)
        row_sweep.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(row_sweep, text="Sweep:",
                         variable=self.freq_mode, value="sweep",
                         command=self._toggle_freq_mode).pack(side=tk.LEFT)

        ttk.Label(row_sweep, text="Start:").pack(side=tk.LEFT, padx=(10, 2))
        self.start_freq_var = tk.StringVar(value="10.7")
        self.start_entry = ttk.Entry(
            row_sweep, textvariable=self.start_freq_var, width=7,
            state="disabled")
        self.start_entry.pack(side=tk.LEFT)

        ttk.Label(row_sweep, text="End:").pack(side=tk.LEFT, padx=(10, 2))
        self.end_freq_var = tk.StringVar(value="12.7")
        self.end_entry = ttk.Entry(
            row_sweep, textvariable=self.end_freq_var, width=7,
            state="disabled")
        self.end_entry.pack(side=tk.LEFT)

        ttk.Label(row_sweep, text="Step (MHz):").pack(
            side=tk.LEFT, padx=(10, 2))
        self.step_var = tk.StringVar(value="125")
        self.step_entry = ttk.Entry(
            row_sweep, textvariable=self.step_var, width=6, state="disabled")
        self.step_entry.pack(side=tk.LEFT)

        # ---- Timing ----
        timing = ttk.LabelFrame(main, text="Timing", padding=5)
        timing.pack(fill=tk.X, pady=(0, 5))

        timing_row = ttk.Frame(timing)
        timing_row.pack(fill=tk.X)

        ttk.Label(timing_row, text="Duration per freq (sec):").pack(
            side=tk.LEFT, padx=2)
        self.dur_var = tk.StringVar(value="180")
        ttk.Entry(timing_row, textvariable=self.dur_var, width=8).pack(
            side=tk.LEFT)

        ttk.Label(timing_row, text="Num Rounds:").pack(
            side=tk.LEFT, padx=(20, 2))
        self.rounds_var = tk.StringVar(value="2")
        ttk.Entry(timing_row, textvariable=self.rounds_var, width=5).pack(
            side=tk.LEFT)

        # ---- Output ----
        out_frame = ttk.LabelFrame(main, text="Output Directory", padding=5)
        out_frame.pack(fill=tk.X, pady=(0, 5))

        out_row = ttk.Frame(out_frame)
        out_row.pack(fill=tk.X)

        default_out = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "testCaptures")
        self.outdir_var = tk.StringVar(value=default_out)
        ttk.Entry(out_row, textvariable=self.outdir_var, width=55).pack(
            side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(out_row, text="Browse",
                    command=self._browse_output).pack(side=tk.LEFT, padx=2)

        # ---- Buttons ----
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=5)

        self.start_btn = ttk.Button(
            btn_frame, text="Start Capture", command=self.start_capture)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.plot_btn = ttk.Button(
            btn_frame, text="Open Plotter", command=self.open_plotter)
        self.plot_btn.pack(side=tk.LEFT, padx=5)

        self.corr_btn = ttk.Button(
            btn_frame, text="Open Correlator", command=self.open_correlator)
        self.corr_btn.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(btn_frame, textvariable=self.status_var,
                  foreground="gray").pack(side=tk.RIGHT, padx=5)

        # ---- Log ----
        log_frame = ttk.LabelFrame(main, text="Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            log_frame, height=14, wrap=tk.WORD, state=tk.DISABLED,
            font=("Courier", 9))
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ---- Helpers ----

    def _toggle_freq_mode(self):
        if self.freq_mode.get() == "specific":
            self.specific_entry.config(state="normal")
            for e in (self.start_entry, self.end_entry, self.step_entry):
                e.config(state="disabled")
        else:
            self.specific_entry.config(state="disabled")
            for e in (self.start_entry, self.end_entry, self.step_entry):
                e.config(state="normal")

    def _browse_output(self):
        d = filedialog.askdirectory(initialdir=self.outdir_var.get())
        if d:
            self.outdir_var.set(d)

    def _log_msg(self, msg):
        """Thread-safe log callback for the capture class."""
        self.log_queue.put(msg)

    def _poll_log(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                if msg.startswith("__DONE__:"):
                    self.output_dir = msg.split(":", 1)[1]
                    self.status_var.set("Capture complete!")
                    self.start_btn.config(state="normal")
                    self._append_log(
                        "\nCapture finished. Click 'Open Plotter' to view.")
                elif msg.startswith("__ERROR__:"):
                    err = msg.split(":", 1)[1]
                    self.status_var.set("Error!")
                    self.start_btn.config(state="normal")
                    self._append_log(f"ERROR: {err}")
                else:
                    self._append_log(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log)

    def _append_log(self, msg):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    # ---- Actions ----

    def start_capture(self):
        # Parse all parameters first
        try:
            sample_rate = float(self.sr_var.get()) * 1e6
            gain_val = float(self.gain_var.get())
            lnb_lo = float(self.lo_var.get()) * 1e9
            duration = float(self.dur_var.get())
            num_rounds = int(self.rounds_var.get())
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid parameter: {e}")
            return

        params = {
            "usrp_args": self.usrp_var.get(),
            "sample_rate": sample_rate,
            "gain": gain_val,
            "lnb_lo": lnb_lo,
            "output_root": self.outdir_var.get(),
            "duration": duration,
        }

        if self.freq_mode.get() == "specific":
            try:
                freqs = [float(x.strip()) * 1e9
                         for x in self.specific_var.get().split(",")]
            except ValueError:
                messagebox.showerror(
                    "Input Error", "Invalid frequency list")
                return
            params["frequencies"] = freqs
            params["total_duration"] = len(freqs) * duration * num_rounds
        else:
            try:
                params["start_freq"] = float(
                    self.start_freq_var.get()) * 1e9
                params["end_freq"] = float(self.end_freq_var.get()) * 1e9
                params["freq_step"] = float(self.step_var.get()) * 1e6
            except ValueError:
                messagebox.showerror(
                    "Input Error", "Invalid sweep parameters")
                return
            n_steps = int(math.floor(
                (params["end_freq"] - params["start_freq"])
                / params["freq_step"]))
            n_freqs = n_steps + 1
            params["total_duration"] = n_freqs * duration * num_rounds

        self.start_btn.config(state="disabled")
        self.status_var.set("Capturing...")

        def worker(p=params):
            try:
                cap = StarlinkSigMFCapture(
                    usrp_args=p["usrp_args"],
                    sample_rate=p["sample_rate"],
                    gain=p["gain"],
                    lnb_lo=p["lnb_lo"],
                    output_root=p["output_root"],
                    log_fn=self._log_msg,
                )
                if "frequencies" in p:
                    cap.frequency_hopping_capture(
                        capture_duration=p["duration"],
                        total_duration=p["total_duration"],
                        frequencies=p["frequencies"],
                    )
                else:
                    cap.frequency_hopping_capture(
                        start_freq=p["start_freq"],
                        end_freq=p["end_freq"],
                        freq_step=p["freq_step"],
                        capture_duration=p["duration"],
                        total_duration=p["total_duration"],
                    )
                self.log_queue.put(f"__DONE__:{cap.output_dir}")
            except Exception as e:
                import traceback
                self.log_queue.put(
                    f"__ERROR__:{e}\n{traceback.format_exc()}")

        self.capture_thread = threading.Thread(target=worker, daemon=True)
        self.capture_thread.start()

    def open_plotter(self):
        d = self.output_dir
        if not d:
            d = filedialog.askdirectory(
                title="Select Capture Directory",
                initialdir=self.outdir_var.get(),
            )
        if d:
            PlottingWindow(self.root, d)

    def open_correlator(self):
        d = self.output_dir
        if not d:
            d = filedialog.askdirectory(
                title="Select Capture Directory",
                initialdir=self.outdir_var.get(),
            )
        if d:
            CorrelatorWindow(self.root, d)


# ===================================================================
# Plotting GUI
# ===================================================================

class PlottingWindow(tk.Toplevel):
    def __init__(self, parent, capture_dir=None):
        super().__init__(parent)
        self.title("Starlink Demo - Plotter")
        self.geometry("1250x820")
        self.minsize(950, 600)

        self.records = []
        self.bands = {}
        self._band_cfs = []
        self.t0_global = None
        self.cached_data = None
        self._cached_key = None
        self.im = None
        self._colorbar = None
        self._computing = False

        self._build_ui()

        if capture_dir:
            self.dir_var.set(capture_dir)
            self.after(100, self.scan_directory)

    def _build_ui(self):
        # Left panel (fixed width controls)
        left = ttk.Frame(self, width=290)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 4), pady=8)
        left.pack_propagate(False)

        # Right panel (canvas, expandable)
        right = ttk.Frame(self)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 8),
                    pady=8)

        # ---- Left: Directory ----
        dir_frame = ttk.LabelFrame(left, text="Capture Directory", padding=4)
        dir_frame.pack(fill=tk.X, pady=(0, 6))

        self.dir_var = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.dir_var).pack(fill=tk.X)
        dir_btns = ttk.Frame(dir_frame)
        dir_btns.pack(fill=tk.X, pady=3)
        ttk.Button(dir_btns, text="Browse",
                    command=self._browse_dir).pack(side=tk.LEFT, padx=2)
        ttk.Button(dir_btns, text="Scan",
                    command=self.scan_directory).pack(side=tk.LEFT, padx=2)

        # ---- Left: Band ----
        band_frame = ttk.LabelFrame(left, text="Band", padding=4)
        band_frame.pack(fill=tk.X, pady=(0, 6))

        self.band_combo = ttk.Combobox(band_frame, state="readonly")
        self.band_combo.pack(fill=tk.X)
        self.band_combo.bind("<<ComboboxSelected>>",
                              self._on_band_selected)

        # ---- Left: Captures ----
        cap_frame = ttk.LabelFrame(left, text="Captures", padding=4)
        cap_frame.pack(fill=tk.X, pady=(0, 6))

        self.cap_listbox = tk.Listbox(
            cap_frame, height=4, selectmode=tk.EXTENDED,
            exportselection=False)
        self.cap_listbox.pack(fill=tk.X)
        cap_btns = ttk.Frame(cap_frame)
        cap_btns.pack(fill=tk.X, pady=2)
        ttk.Button(cap_btns, text="All",
                    command=self._select_all_captures).pack(
                        side=tk.LEFT, padx=2)
        ttk.Button(cap_btns, text="Clear",
                    command=self._clear_captures).pack(side=tk.LEFT, padx=2)

        # ---- Left: Display settings ----
        disp = ttk.LabelFrame(left, text="Display", padding=4)
        disp.pack(fill=tk.X, pady=(0, 6))

        # dB Min
        db_min_row = ttk.Frame(disp)
        db_min_row.pack(fill=tk.X)
        ttk.Label(db_min_row, text="dB Min:").pack(side=tk.LEFT)
        self.db_min_val_label = ttk.Label(db_min_row, text="-115")
        self.db_min_val_label.pack(side=tk.RIGHT)
        self.db_min_var = tk.DoubleVar(value=-115)
        self.db_min_scale = ttk.Scale(
            disp, from_=-200, to=0, variable=self.db_min_var,
            orient=tk.HORIZONTAL, command=self._update_db_labels)
        self.db_min_scale.pack(fill=tk.X)

        # dB Max
        db_max_row = ttk.Frame(disp)
        db_max_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(db_max_row, text="dB Max:").pack(side=tk.LEFT)
        self.db_max_val_label = ttk.Label(db_max_row, text="-90")
        self.db_max_val_label.pack(side=tk.RIGHT)
        self.db_max_var = tk.DoubleVar(value=-90)
        self.db_max_scale = ttk.Scale(
            disp, from_=-200, to=0, variable=self.db_max_var,
            orient=tk.HORIZONTAL, command=self._update_db_labels)
        self.db_max_scale.pack(fill=tk.X)

        # Colormap
        cmap_row = ttk.Frame(disp)
        cmap_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(cmap_row, text="Colormap:").pack(side=tk.LEFT)
        self.cmap_var = tk.StringVar(value="turbo")
        cmap_combo = ttk.Combobox(
            cmap_row, textvariable=self.cmap_var, values=COLORMAPS,
            state="readonly", width=12)
        cmap_combo.pack(side=tk.RIGHT)
        # colormap applied on Plot click

        # NFFT
        nfft_row = ttk.Frame(disp)
        nfft_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(nfft_row, text="NFFT:").pack(side=tk.LEFT)
        self.nfft_var = tk.StringVar(value="1024")
        ttk.Combobox(nfft_row, textvariable=self.nfft_var,
                      values=[str(n) for n in NFFT_OPTIONS],
                      state="readonly", width=7).pack(side=tk.RIGHT)

        # Downsample
        ds_row = ttk.Frame(disp)
        ds_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(ds_row, text="DS Factor:").pack(side=tk.LEFT)
        self.ds_var = tk.StringVar(value="1")
        ttk.Spinbox(ds_row, from_=1, to=64, textvariable=self.ds_var,
                     width=4).pack(side=tk.LEFT, padx=5)

        self.ds_method_var = tk.StringVar(value="mean")
        ttk.Radiobutton(ds_row, text="Mean",
                         variable=self.ds_method_var,
                         value="mean").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(ds_row, text="Max",
                         variable=self.ds_method_var,
                         value="max").pack(side=tk.LEFT, padx=2)

        # ---- Left: Action buttons ----
        action = ttk.Frame(left)
        action.pack(fill=tk.X, pady=(4, 0))

        self.plot_btn = ttk.Button(
            action, text="Plot", command=self.compute_and_plot)
        self.plot_btn.pack(fill=tk.X, pady=2)

        self.save_btn = ttk.Button(
            action, text="Save Figure", command=self.save_figure)
        self.save_btn.pack(fill=tk.X, pady=2)

        self.status_var = tk.StringVar(value="Select a capture directory")
        ttk.Label(left, textvariable=self.status_var, foreground="gray",
                  wraplength=270).pack(anchor=tk.W, pady=(8, 0))

        # ---- Right: Canvas ----
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Select band and click Plot")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Time (s)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, right)
        toolbar.update()
        toolbar.pack(fill=tk.X)

    # ---- Control callbacks ----

    def _browse_dir(self):
        d = filedialog.askdirectory(initialdir=self.dir_var.get() or ".")
        if d:
            self.dir_var.set(d)
            self.scan_directory()

    def scan_directory(self):
        capture_dir = self.dir_var.get()
        if not capture_dir or not os.path.isdir(capture_dir):
            messagebox.showwarning("Warning", "Invalid directory")
            return

        self.status_var.set("Scanning...")
        self.update_idletasks()

        self.records = find_sigmf_records(capture_dir)
        if not self.records:
            self.status_var.set("No SigMF captures found")
            return

        self.t0_global = min(r["capture_time"] for r in self.records)

        self.bands = {}
        for r in self.records:
            self.bands.setdefault(r["cf"], []).append(r)

        cfs = sorted(self.bands.keys())
        self._band_cfs = cfs
        band_labels = [f"{cf/1e9:.3f} GHz  ({len(self.bands[cf])} captures)"
                       for cf in cfs]
        self.band_combo["values"] = band_labels

        if band_labels:
            self.band_combo.current(0)
            self._on_band_selected(None)

        self.status_var.set(
            f"Found {len(self.records)} captures in {len(cfs)} bands")

    def _on_band_selected(self, event):
        idx = self.band_combo.current()
        if idx < 0:
            return
        cf = self._band_cfs[idx]
        band_records = sorted(
            self.bands[cf], key=lambda r: r["capture_time"])

        self.cap_listbox.delete(0, tk.END)
        for i, rec in enumerate(band_records, 1):
            t_str = rec["capture_time"].strftime("%H:%M:%S")
            self.cap_listbox.insert(tk.END, f"Capture {i}  ({t_str})")

        # Select all by default
        self.cap_listbox.select_set(0, tk.END)

    def _select_all_captures(self):
        self.cap_listbox.select_set(0, tk.END)

    def _clear_captures(self):
        self.cap_listbox.selection_clear(0, tk.END)

    def _update_db_labels(self, *args):
        self.db_min_val_label.config(text=f"{self.db_min_var.get():.0f}")
        self.db_max_val_label.config(text=f"{self.db_max_var.get():.0f}")

    # ---- Plot (threaded) ----

    def compute_and_plot(self):
        if self._computing:
            return

        # If we already have cached data, just re-render with new
        # dB / cmap / display params (no recompute needed)
        if self.cached_data is not None:
            band_idx = self.band_combo.current()
            if band_idx >= 0:
                cf = self._band_cfs[band_idx]
                selected = list(self.cap_listbox.curselection())
                nfft = int(self.nfft_var.get())
                ds_factor = int(self.ds_var.get())
                ds_method = self.ds_method_var.get()
                # Check if data params match cached — if so, just re-render
                cached_key = self._cached_key
                new_key = (cf, tuple(selected), nfft, ds_factor, ds_method)
                if cached_key == new_key:
                    self._render_cached()
                    return

        band_idx = self.band_combo.current()
        if band_idx < 0:
            messagebox.showwarning("Warning", "Select a band first")
            return

        selected = list(self.cap_listbox.curselection())
        if not selected:
            messagebox.showwarning("Warning",
                                    "Select at least one capture")
            return

        cf = self._band_cfs[band_idx]
        band_records = sorted(
            self.bands[cf], key=lambda r: r["capture_time"])
        chosen = [band_records[i] for i in selected]

        nfft = int(self.nfft_var.get())
        ds_factor = int(self.ds_var.get())
        ds_method = self.ds_method_var.get()

        self._computing = True
        self.status_var.set("Computing spectrogram...")
        self.plot_btn.config(state="disabled")
        self.update_idletasks()

        params = dict(chosen=chosen, nfft=nfft, ds_factor=ds_factor,
                      ds_method=ds_method, cf=cf)

        def worker():
            try:
                result = self._compute_spectrogram(params)
                self.after(0, lambda: self._display_result(result, params))
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.after(0, lambda: self._plot_error(f"{e}\n{tb}"))

        threading.Thread(target=worker, daemon=True).start()

    def _compute_spectrogram(self, params):
        """Runs in background thread — pure computation, no GUI calls."""
        chosen = params["chosen"]
        nfft = params["nfft"]
        ds_factor = params["ds_factor"]
        ds_method = params["ds_method"]

        f_axis = None
        all_Sxx = []
        all_t = []

        for rec in chosen:
            f_shifted, t_local, Sxx_db = spectrogram_chunked(
                rec["data"], rec["sr"], nfft)
            if f_shifted is None:
                continue

            Sxx_db, t_local = downsample_time(
                Sxx_db, t_local, ds_factor, ds_method)

            f_final = f_shifted + rec["cf"]
            dt_offset = (
                rec["capture_time"] - self.t0_global).total_seconds()
            t_global = t_local + dt_offset

            if f_axis is None:
                f_axis = f_final
            elif len(f_final) != len(f_axis):
                continue

            all_Sxx.append(Sxx_db)
            all_t.append(t_global)

            del f_shifted, t_local, Sxx_db
            gc.collect()

        if not all_Sxx:
            return None

        Sxx_full = np.concatenate(all_Sxx, axis=1)
        t_full = np.concatenate(all_t)
        del all_Sxx, all_t
        gc.collect()

        extent = [f_axis[0], f_axis[-1], t_full[0], t_full[-1]]
        return (f_axis, t_full, Sxx_full, extent)

    def _render_cached(self):
        """Re-render cached spectrogram with current dB/cmap settings."""
        _, _, Sxx_full, extent = self.cached_data
        cf = self._band_cfs[self.band_combo.current()]
        ds_factor = int(self.ds_var.get())
        n_selected = len(list(self.cap_listbox.curselection()))

        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.im = self.ax.imshow(
            Sxx_full.T, extent=extent, aspect="auto", origin="lower",
            cmap=self.cmap_var.get(), interpolation="bilinear",
            vmin=self.db_min_var.get(), vmax=self.db_max_var.get(),
        )
        self._colorbar = self.fig.colorbar(
            self.im, ax=self.ax, label="Power (dB)")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Time (s from first capture)")
        title = f"CF = {cf/1e9:.3f} GHz"
        if n_selected > 1:
            title += f"  ({n_selected} captures stitched)"
        if ds_factor > 1:
            title += f"  [{ds_factor}x DS]"
        self.ax.set_title(title)
        self.fig.tight_layout()
        self.canvas.draw()
        self.status_var.set("Re-rendered with updated display settings")

    def _display_result(self, result, params):
        """Runs on main thread — updates the GUI."""
        self._computing = False
        self.plot_btn.config(state="normal")

        if result is None:
            self.status_var.set("No data to plot")
            return

        _, _, Sxx_full, extent = result
        self.cached_data = result
        # Store key so we can detect when only display params changed
        selected = list(self.cap_listbox.curselection())
        self._cached_key = (params["cf"], tuple(selected),
                            params["nfft"], params["ds_factor"],
                            params["ds_method"])
        cf = params["cf"]
        ds_factor = params["ds_factor"]
        n_selected = len(params["chosen"])

        # Clear figure and rebuild
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        self.im = self.ax.imshow(
            Sxx_full.T,
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap=self.cmap_var.get(),
            interpolation="bilinear",
            vmin=self.db_min_var.get(),
            vmax=self.db_max_var.get(),
        )

        self._colorbar = self.fig.colorbar(
            self.im, ax=self.ax, label="Power (dB)")

        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Time (s from first capture)")

        title = f"CF = {cf/1e9:.3f} GHz"
        if n_selected > 1:
            title += f"  ({n_selected} captures stitched)"
        if ds_factor > 1:
            title += f"  [{ds_factor}x DS]"
        self.ax.set_title(title)

        self.fig.tight_layout()
        self.canvas.draw()

        self.status_var.set(
            f"Plotted {Sxx_full.shape[1]} time bins, "
            f"{Sxx_full.shape[0]} freq bins")

    def _plot_error(self, msg):
        self._computing = False
        self.plot_btn.config(state="normal")
        self.status_var.set(f"Error: {msg[:120]}")

    # ---- Save ----

    def save_figure(self):
        if self.cached_data is None:
            messagebox.showwarning("Warning", "Nothing to save -- plot first")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"),
                       ("SVG", "*.svg"), ("All", "*.*")],
            initialdir=self.dir_var.get(),
            initialfile="waterfall.png",
        )
        if path:
            self.fig.savefig(path, dpi=200, bbox_inches="tight")
            self.status_var.set(f"Saved: {os.path.basename(path)}")


# ===================================================================
# Correlator GUI
# ===================================================================

class CorrelatorWindow(tk.Toplevel):
    def __init__(self, parent, capture_dir=None):
        super().__init__(parent)
        self.title("Starlink Demo - Correlator")
        self.geometry("1300x900")
        self.minsize(1000, 650)

        self._computing = False
        self._ranked = None
        self._waterfall = None
        self._vel_kms = None
        self._t1hz = None

        self._build_ui()

        if capture_dir:
            self.dir_var.set(capture_dir)

    def _build_ui(self):
        # Left panel (controls)
        left = ttk.Frame(self, width=300)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 4), pady=8)
        left.pack_propagate(False)

        # Right panel (canvas)
        right = ttk.Frame(self)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                    padx=(4, 8), pady=8)

        # ---- Directory ----
        dir_frame = ttk.LabelFrame(
            left, text="Capture Directory", padding=4)
        dir_frame.pack(fill=tk.X, pady=(0, 6))
        self.dir_var = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.dir_var).pack(fill=tk.X)
        dir_btns = ttk.Frame(dir_frame)
        dir_btns.pack(fill=tk.X, pady=3)
        ttk.Button(dir_btns, text="Browse",
                    command=self._browse_dir).pack(side=tk.LEFT, padx=2)

        # ---- Frequency ----
        freq_frame = ttk.LabelFrame(left, text="Frequency", padding=4)
        freq_frame.pack(fill=tk.X, pady=(0, 6))
        freq_row = ttk.Frame(freq_frame)
        freq_row.pack(fill=tk.X)
        ttk.Label(freq_row, text="CF (GHz):").pack(side=tk.LEFT)
        self.freq_var = tk.StringVar(value="12.325")
        ttk.Entry(freq_row, textvariable=self.freq_var, width=10).pack(
            side=tk.LEFT, padx=5)

        # ---- TLE ----
        tle_frame = ttk.LabelFrame(left, text="TLE Data", padding=4)
        tle_frame.pack(fill=tk.X, pady=(0, 6))

        self.tle_mode = tk.StringVar(value="download")
        ttk.Radiobutton(tle_frame, text="Download fresh",
                         variable=self.tle_mode,
                         value="download").pack(anchor=tk.W)
        file_row = ttk.Frame(tle_frame)
        file_row.pack(fill=tk.X)
        ttk.Radiobutton(file_row, text="File:",
                         variable=self.tle_mode,
                         value="file").pack(side=tk.LEFT)
        tle_default = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "DopplerPredictor", "starlink_downloaded.txt")
        self.tle_path_var = tk.StringVar(value=tle_default)
        ttk.Entry(file_row, textvariable=self.tle_path_var,
                   width=20).pack(side=tk.LEFT, padx=2, fill=tk.X,
                                   expand=True)
        ttk.Button(file_row, text="...",
                    command=self._browse_tle).pack(side=tk.LEFT)

        # ---- Ground Station ----
        gs_frame = ttk.LabelFrame(
            left, text="Ground Station", padding=4)
        gs_frame.pack(fill=tk.X, pady=(0, 6))

        gs_r1 = ttk.Frame(gs_frame)
        gs_r1.pack(fill=tk.X)
        ttk.Label(gs_r1, text="Lat:").pack(side=tk.LEFT)
        self.lat_var = tk.StringVar(value=str(DEFAULT_LAT))
        ttk.Entry(gs_r1, textvariable=self.lat_var, width=9).pack(
            side=tk.LEFT, padx=2)
        ttk.Label(gs_r1, text="Lon:").pack(side=tk.LEFT, padx=(8, 0))
        self.lon_var = tk.StringVar(value=str(DEFAULT_LON))
        ttk.Entry(gs_r1, textvariable=self.lon_var, width=9).pack(
            side=tk.LEFT, padx=2)

        gs_r2 = ttk.Frame(gs_frame)
        gs_r2.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(gs_r2, text="Alt (m):").pack(side=tk.LEFT)
        self.alt_var = tk.StringVar(value=str(DEFAULT_ALT))
        ttk.Entry(gs_r2, textvariable=self.alt_var, width=8).pack(
            side=tk.LEFT, padx=2)
        ttk.Label(gs_r2, text="Elev mask:").pack(
            side=tk.LEFT, padx=(8, 0))
        self.elev_var = tk.StringVar(value="10.0")
        ttk.Entry(gs_r2, textvariable=self.elev_var, width=6).pack(
            side=tk.LEFT, padx=2)

        # ---- Display ----
        disp = ttk.LabelFrame(left, text="Display", padding=4)
        disp.pack(fill=tk.X, pady=(0, 6))

        ntop_row = ttk.Frame(disp)
        ntop_row.pack(fill=tk.X)
        ttk.Label(ntop_row, text="Top N matches:").pack(side=tk.LEFT)
        self.ntop_var = tk.StringVar(value="3")
        ttk.Spinbox(ntop_row, from_=1, to=20,
                     textvariable=self.ntop_var, width=4).pack(
                         side=tk.LEFT, padx=5)

        # ---- Actions ----
        action = ttk.Frame(left)
        action.pack(fill=tk.X, pady=(4, 0))

        self.run_btn = ttk.Button(
            action, text="Run Correlator", command=self.run_correlator)
        self.run_btn.pack(fill=tk.X, pady=2)

        self.save_btn = ttk.Button(
            action, text="Save Figure", command=self.save_figure)
        self.save_btn.pack(fill=tk.X, pady=2)

        # ---- Status ----
        self.status_var = tk.StringVar(
            value="Set parameters and click Run")
        ttk.Label(left, textvariable=self.status_var, foreground="gray",
                  wraplength=280).pack(anchor=tk.W, pady=(6, 0))

        # ---- Ranking table ----
        rank_frame = ttk.LabelFrame(left, text="Ranking", padding=4)
        rank_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        cols = ("rank", "satellite", "ncc")
        self.rank_tree = ttk.Treeview(
            rank_frame, columns=cols, show="headings", height=10)
        self.rank_tree.heading("rank", text="#")
        self.rank_tree.heading("satellite", text="Satellite")
        self.rank_tree.heading("ncc", text="NCC")
        self.rank_tree.column("rank", width=30, anchor=tk.CENTER)
        self.rank_tree.column("satellite", width=110)
        self.rank_tree.column("ncc", width=70, anchor=tk.E)
        rank_scroll = ttk.Scrollbar(
            rank_frame, orient=tk.VERTICAL,
            command=self.rank_tree.yview)
        self.rank_tree.configure(yscrollcommand=rank_scroll.set)
        self.rank_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        rank_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # ---- Right: Scrollable Canvas ----
        toolbar_frame = ttk.Frame(right)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)

        scroll_frame = ttk.Frame(right)
        scroll_frame.pack(fill=tk.BOTH, expand=True)

        self._scroll_canvas = tk.Canvas(scroll_frame)
        scroll_y = ttk.Scrollbar(
            scroll_frame, orient=tk.VERTICAL,
            command=self._scroll_canvas.yview)
        self._scroll_canvas.configure(yscrollcommand=scroll_y.set)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self._scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._inner_frame = ttk.Frame(self._scroll_canvas)
        self._scroll_canvas.create_window(
            (0, 0), window=self._inner_frame, anchor="nw")
        self._inner_frame.bind(
            "<Configure>",
            lambda e: self._scroll_canvas.configure(
                scrollregion=self._scroll_canvas.bbox("all")))
        # mousewheel scrolling
        def _on_mousewheel(event):
            self._scroll_canvas.yview_scroll(
                int(-1 * (event.delta / 120)), "units")
        self._scroll_canvas.bind_all(
            "<MouseWheel>", _on_mousewheel)
        # Linux scroll
        self._scroll_canvas.bind_all(
            "<Button-4>",
            lambda e: self._scroll_canvas.yview_scroll(-3, "units"))
        self._scroll_canvas.bind_all(
            "<Button-5>",
            lambda e: self._scroll_canvas.yview_scroll(3, "units"))

        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self._inner_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        toolbar.pack(fill=tk.X)

    # ---- Helpers ----

    def _browse_dir(self):
        d = filedialog.askdirectory(
            initialdir=self.dir_var.get() or ".")
        if d:
            self.dir_var.set(d)

    def _browse_tle(self):
        f = filedialog.askopenfilename(
            filetypes=[("TLE/Text", "*.txt"), ("All", "*.*")])
        if f:
            self.tle_path_var.set(f)
            self.tle_mode.set("file")

    def _log(self, msg):
        """Update status from any thread."""
        self.after(0, lambda: self.status_var.set(msg))

    # ---- Run ----

    def run_correlator(self):
        if self._computing:
            return

        capture_dir = self.dir_var.get()
        if not capture_dir or not os.path.isdir(capture_dir):
            messagebox.showwarning("Warning", "Invalid capture directory")
            return

        try:
            freq_ghz = float(self.freq_var.get())
            lat = float(self.lat_var.get())
            lon = float(self.lon_var.get())
            alt = float(self.alt_var.get())
            elev_mask = float(self.elev_var.get())
            n_top = int(self.ntop_var.get())
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid parameter: {e}")
            return

        params = dict(
            capture_dir=capture_dir,
            freq_ghz=freq_ghz,
            tx_freq_hz=freq_ghz * 1e9,
            lat=lat, lon=lon, alt=alt,
            elev_mask=elev_mask,
            n_top=n_top,
            tle_mode=self.tle_mode.get(),
            tle_path=self.tle_path_var.get(),
        )

        self._computing = True
        self.run_btn.config(state="disabled")
        self.status_var.set("Running correlator pipeline...")
        self.update_idletasks()

        def worker():
            try:
                result = self._run_pipeline(params)
                self.after(0, lambda: self._display_corr_result(
                    result, params))
            except Exception as e:
                import traceback
                err_msg = f"{e}\n{traceback.format_exc()}"
                self.after(0, lambda m=err_msg: self._corr_error(m))

        threading.Thread(target=worker, daemon=True).start()

    def _run_pipeline(self, p):
        """Full correlator pipeline — runs in background thread."""
        log = self._log

        # Step 1: Preprocess
        log("Step 1/5: Preprocessing...")
        tag = corr_run_preprocessing(
            p["capture_dir"], p["freq_ghz"], log_fn=log)

        # Step 2: Load waterfall
        log("Step 2/5: Loading waterfall...")
        waterfall, t1hz, vel_kms = corr_load_waterfall(
            p["capture_dir"], tag, log_fn=log)

        # Step 3: TLE
        log("Step 3/5: Loading TLE...")
        if p["tle_mode"] == "file":
            with open(p["tle_path"]) as f:
                tle_data = f.read()
            log(f"[tle] Loaded from: {p['tle_path']}")
        else:
            tle_data = corr_download_tle(log_fn=log)

        # Step 4: Predict Doppler
        log("Step 4/5: Predicting Doppler...")
        start_utc = t1hz[0].to_pydatetime()
        duration_sec = (t1hz[-1] - t1hz[0]).total_seconds() + 1
        predictions = corr_predict_doppler(
            tle_data, start_utc, duration_sec,
            tx_freq_hz=p["tx_freq_hz"],
            elevation_mask=p["elev_mask"],
            lat=p["lat"], lon=p["lon"], alt=p["alt"],
            log_fn=log,
        )
        if not predictions:
            raise RuntimeError(
                "No visible satellites. Check TLE freshness.")

        # Step 5: Correlate
        log("Step 5/6: Correlating...")
        ranked = corr_correlate(
            waterfall, vel_kms, t1hz, predictions,
            p["tx_freq_hz"], log_fn=log)

        # Step 6: Compute raw spectrogram for display
        log("Step 6/6: Computing raw spectrogram...")
        raw_spec = None
        records = find_sigmf_records(p["capture_dir"])
        freq_records = sorted(
            [r for r in records
             if abs(r["cf"] - p["tx_freq_hz"]) < 1e6],
            key=lambda r: r["capture_time"])

        if freq_records:
            t0_cap = freq_records[0]["capture_time"]
            f_axis = None
            all_Sxx = []
            all_t = []
            for rec in freq_records:
                f_sh, t_loc, Sxx = spectrogram_chunked(
                    rec["data"], rec["sr"], 1024)
                if f_sh is None:
                    continue
                Sxx, t_loc = downsample_time(Sxx, t_loc, 4)
                f_final = f_sh + rec["cf"]
                dt = (rec["capture_time"] - t0_cap).total_seconds()
                t_g = t_loc + dt
                if f_axis is None:
                    f_axis = f_final
                elif len(f_final) != len(f_axis):
                    continue
                all_Sxx.append(Sxx)
                all_t.append(t_g)
                del f_sh, t_loc, Sxx
                gc.collect()

            if all_Sxx:
                Sxx_full = np.concatenate(all_Sxx, axis=1)
                t_full = np.concatenate(all_t)
                raw_spec = {
                    "f_axis": f_axis,
                    "t": t_full,
                    "Sxx": Sxx_full,
                }
                del all_Sxx, all_t
                gc.collect()
                log(f"[spec] Raw spectrogram: "
                    f"{Sxx_full.shape[1]} time bins")

        return {
            "waterfall": waterfall,
            "vel_kms": vel_kms,
            "t1hz": t1hz,
            "ranked": ranked,
            "raw_spec": raw_spec,
        }

    def _display_corr_result(self, result, params):
        """Render correlation results — main thread."""
        self._computing = False
        self.run_btn.config(state="normal")

        waterfall = result["waterfall"]
        vel_kms = result["vel_kms"]
        t1hz = result["t1hz"]
        ranked = result["ranked"]
        raw_spec = result.get("raw_spec")
        n_top = params["n_top"]
        tx_freq_hz = params["tx_freq_hz"]

        self._ranked = ranked
        self._waterfall = waterfall
        self._vel_kms = vel_kms
        self._t1hz = t1hz

        # Populate ranking table
        for item in self.rank_tree.get_children():
            self.rank_tree.delete(item)
        for i, r in enumerate(ranked[:20], 1):
            self.rank_tree.insert("", tk.END, values=(
                i, r["name"], f"{r['ncc']:.4f}"))

        n_show = min(n_top, len(ranked))
        n_panels = 1 + n_show

        # Velocity-space extents for sim panels (time bottom-to-top)
        dur_min = (t1hz[-1] - t1hz[0]).total_seconds() / 60.0
        vel_extent = [vel_kms.min(), vel_kms.max(), 0, dur_min]

        # Resize figure: 5 inches per panel
        self.fig.clear()
        self.fig.set_size_inches(12, 5 * n_panels)
        axes = self.fig.subplots(n_panels, 1)
        if n_panels == 1:
            axes = [axes]

        # --- Panel 0: Raw capture spectrogram with match boxes ---
        ax = axes[0]

        if raw_spec is not None:
            f_axis = raw_spec["f_axis"]
            t_spec = raw_spec["t"]
            Sxx = raw_spec["Sxx"]
            spec_extent = [f_axis[0], f_axis[-1],
                           t_spec[0], t_spec[-1]]
            vmin_s, vmax_s = np.percentile(Sxx, [5, 95])

            im = ax.imshow(Sxx.T, extent=spec_extent, aspect="auto",
                            origin="lower", cmap="turbo",
                            interpolation="bilinear",
                            vmin=vmin_s, vmax=vmax_s)
            self.fig.colorbar(im, ax=ax, label="Power (dB)")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Time (s from first capture)")
            ax.set_title(
                f"CF = {params['freq_ghz']:.3f} GHz — "
                f"Raw Capture Spectrogram")

            # Draw match boxes converted from velocity to frequency
            colors = ["white", "cyan", "yellow", "lime",
                      "magenta", "orange"]
            for i in range(n_show):
                r = ranked[i]
                # Convert velocity (km/s) to frequency offset (Hz)
                v_lo = vel_kms[r["col_lo"]]
                v_hi = vel_kms[min(r["col_hi"], len(vel_kms) - 1)]
                f_off_lo = v_lo * 1000.0 * tx_freq_hz / C
                f_off_hi = v_hi * 1000.0 * tx_freq_hz / C
                f_lo = tx_freq_hz + f_off_lo
                f_hi = tx_freq_hz + f_off_hi

                # Time: t_lo/t_hi are 1Hz indices (seconds)
                t_lo_s = float(r["t_lo"])
                t_hi_s = float(r["t_hi"])

                col = colors[i % len(colors)]
                ax.add_patch(Rectangle(
                    (f_lo, t_lo_s), f_hi - f_lo, t_hi_s - t_lo_s,
                    lw=2, edgecolor=col, facecolor="none",
                    ls="--" if i == 0 else ":"))
                ax.text(f_hi, t_hi_s,
                        f" {r['name']} NCC={r['ncc']:.3f}",
                        color=col, fontsize=9, fontweight="bold",
                        va="top",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="black", alpha=0.6))
        else:
            # Fallback: velocity waterfall if no .sigmf-data available
            vmin_r, vmax_r = np.percentile(waterfall, [5, 95])
            im = ax.imshow(waterfall, aspect="auto", extent=vel_extent,
                            cmap="jet_r", vmin=vmin_r, vmax=vmax_r,
                            interpolation="bilinear", origin="lower")
            self.fig.colorbar(im, ax=ax, label="Power (dB)")
            ax.set_ylabel("Time (min)")
            ax.set_title("Real Velocity Waterfall (no .sigmf-data found)")

            for i in range(n_show):
                r = ranked[i]
                v_lo = vel_kms[r["col_lo"]]
                v_hi = vel_kms[min(r["col_hi"], len(vel_kms) - 1)]
                y_top = r["t_lo"] / 60.0
                h = (r["t_hi"] - r["t_lo"]) / 60.0
                ax.add_patch(Rectangle(
                    (v_lo, y_top), v_hi - v_lo, h,
                    lw=2, edgecolor="white", facecolor="none",
                    ls="--" if i == 0 else ":"))

        # --- Sim panels ---
        for i in range(n_show):
            r = ranked[i]
            ax = axes[1 + i]
            sim = r["sim"]
            vmin_s = max(0, float(sim.min()) - 5)
            vmax_s = float(np.median(sim))

            im = ax.imshow(sim, aspect="auto", extent=vel_extent,
                            cmap="jet_r", vmin=vmin_s, vmax=vmax_s,
                            interpolation="bilinear", origin="lower")
            ax.axvline(0, color="white", ls="--", lw=1.5, alpha=0.8,
                        label="Zero velocity")
            ax.legend(loc="upper right", fontsize=8)
            self.fig.colorbar(im, ax=ax, label="Path Loss (dB)")
            ax.set_ylabel("Time (min)")
            ax.set_title("SIM")

            ax.text(0.5, 0.95,
                    f"{r['name']}  NCC={r['ncc']:.3f}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=12, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="black", alpha=0.5))

            v_lo = vel_kms[r["col_lo"]]
            v_hi = vel_kms[min(r["col_hi"], len(vel_kms) - 1)]
            y_top = r["t_lo"] / 60.0
            h = (r["t_hi"] - r["t_lo"]) / 60.0
            ax.add_patch(Rectangle(
                (v_lo, y_top), v_hi - v_lo, h,
                lw=2, edgecolor="white", facecolor="none", ls="--"))

        axes[-1].set_xlabel("Relative Velocity (km/s)")
        self.fig.tight_layout()
        # Resize the canvas widget to match the figure
        w, h = self.fig.get_size_inches()
        dpi = self.fig.get_dpi()
        self.canvas.get_tk_widget().configure(
            width=int(w * dpi), height=int(h * dpi))
        self.canvas.draw()

        if ranked:
            self.status_var.set(
                f"Best match: {ranked[0]['name']} "
                f"(NCC={ranked[0]['ncc']:.4f})")
        else:
            self.status_var.set("No correlations found")

    def _corr_error(self, msg):
        self._computing = False
        self.run_btn.config(state="normal")
        self.status_var.set(f"Error: {msg[:150]}")

    def save_figure(self):
        if self._ranked is None:
            messagebox.showwarning(
                "Warning", "Nothing to save -- run correlator first")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"),
                       ("SVG", "*.svg"), ("All", "*.*")],
            initialdir=self.dir_var.get(),
            initialfile="correlation_result.png",
        )
        if path:
            self.fig.savefig(path, dpi=200, bbox_inches="tight")
            self.status_var.set(f"Saved: {os.path.basename(path)}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Starlink Capture, Plot & Correlate")
    parser.add_argument("--plot", type=str, default=None,
                        help="Skip capture, open plotter on this directory")
    parser.add_argument("--correlate", type=str, default=None,
                        help="Skip capture, open correlator on this directory")
    args = parser.parse_args()

    root = tk.Tk()

    if args.plot or args.correlate:
        root.withdraw()
        if args.plot:
            PlottingWindow(root, args.plot)
        if args.correlate:
            CorrelatorWindow(root, args.correlate)

        def on_close():
            root.destroy()
        for w in root.winfo_children():
            if isinstance(w, tk.Toplevel):
                w.protocol("WM_DELETE_WINDOW", on_close)
    else:
        CaptureWindow(root)

    root.mainloop()


if __name__ == "__main__":
    main()
