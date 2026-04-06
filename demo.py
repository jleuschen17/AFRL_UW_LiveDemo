#!/usr/bin/env python3
"""
demo.py — Starlink USRP Capture & Interactive Plotting GUI

Two-stage GUI:
  1. Capture window: configure and run USRP frequency-hopping captures
  2. Plotting window: interactively view/adjust waterfall spectrograms

Usage:
    python demo.py              # opens capture window
    python demo.py --plot DIR   # skip capture, open plotter on DIR
"""

import argparse
import gc
import glob
import json
import math
import os
import queue
import re
import threading
import time as time_mod
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
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

        self.vel_btn = ttk.Button(
            btn_frame, text="Open Velocity Plotter",
            command=self.open_velocity_plotter)
        self.vel_btn.pack(side=tk.LEFT, padx=5)

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

    def open_velocity_plotter(self):
        d = self.output_dir
        if not d:
            d = filedialog.askdirectory(
                title="Select Capture Directory",
                initialdir=self.outdir_var.get(),
            )
        if d:
            VelocityPlottingWindow(self.root, d)


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
# Velocity Plotting GUI (from compressed .txt files)
# ===================================================================

def find_txt_records(root_dir):
    """Scan directory for compressed .txt files, return grouped by CF."""
    txt_files = sorted(glob.glob(os.path.join(root_dir, "r*_f*GHz_*.txt")))
    records = []
    for path in txt_files:
        fname = os.path.basename(path)
        freq_match = re.search(r"_f([\d.]+)GHz_", fname)
        time_match = re.search(r"_(\d{8}T\d{6})", fname)
        if not freq_match:
            continue
        cf_ghz = float(freq_match.group(1))
        cap_time = None
        if time_match:
            cap_time = datetime.strptime(
                time_match.group(1), "%Y%m%dT%H%M%S")
        records.append({
            "path": path,
            "cf_ghz": cf_ghz,
            "capture_time": cap_time,
        })
    return records


def build_velocity_waterfall(txt_paths, cf_ghz, sample_rate=500000.0,
                              k=10, nfft=1024, noverlap=512):
    """Reconstruct velocity waterfall from compressed .txt files.

    Returns (waterfall_db, vel_axis_kms, t_full) or (None, None, None).
    """
    v_max = 5000  # m/s
    n_vel = 500
    v_axis = np.linspace(-v_max, v_max, n_vel)
    cf_hz = cf_ghz * 1e9

    all_sparse = []
    all_t = []
    t0 = None

    for txt_path in sorted(txt_paths,
                           key=lambda p: os.path.basename(p)):
        fname = os.path.basename(txt_path)
        time_match = re.search(r"_(\d{8}T\d{6})", fname)
        if not time_match:
            continue
        capture_time = datetime.strptime(
            time_match.group(1), "%Y%m%dT%H%M%S")
        if t0 is None:
            t0 = capture_time

        data = np.loadtxt(txt_path, dtype=complex)
        num_rows = data.shape[0]
        step = nfft - noverlap
        t_local = (np.arange(num_rows) * step + nfft / 2) / sample_rate

        recon_freqs = np.real(data[:, 0:k])
        recon_vals = data[:, k:]

        # Auto-detect if values are Hz rather than m/s
        max_freq = np.max(np.abs(recon_freqs))
        if max_freq > v_max * 2:
            recon_freqs = C * recon_freqs / cf_hz

        sparse = np.zeros((len(v_axis), num_rows), dtype=complex)
        for i in range(num_rows):
            idx = np.searchsorted(v_axis, recon_freqs[i, :])
            idx = np.clip(idx, 0, len(v_axis) - 1)
            sparse[idx, i] = recon_vals[i, :]

        # Time offset from first capture
        dt = (capture_time - t0).total_seconds()
        all_sparse.append(sparse)
        all_t.append(t_local + dt)
        del data, sparse
        gc.collect()

    if not all_sparse:
        return None, None, None

    # Combine: sparse shape is (n_vel, n_time), waterfall is (n_time, n_vel)
    full_sparse = np.concatenate(all_sparse, axis=1)
    t_full = np.concatenate(all_t)
    del all_sparse, all_t
    gc.collect()

    waterfall = full_sparse.T  # (n_time, n_vel)
    wf_db = 10.0 * np.log10(np.abs(waterfall) + 1e-15).astype(np.float32)
    vel_kms = v_axis / 1000.0

    return wf_db, vel_kms, t_full


class VelocityPlottingWindow(tk.Toplevel):
    """Plot velocity waterfalls from compressed .txt files."""

    def __init__(self, parent, capture_dir=None):
        super().__init__(parent)
        self.title("Starlink Demo - Velocity Plotter")
        self.geometry("1250x820")
        self.minsize(950, 600)

        self.txt_records = []
        self.bands = {}
        self._band_cfs = []
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
        left = ttk.Frame(self, width=290)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 4), pady=8)
        left.pack_propagate(False)

        right = ttk.Frame(self)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                   padx=(4, 8), pady=8)

        # ---- Directory ----
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

        # ---- Band ----
        band_frame = ttk.LabelFrame(left, text="Band", padding=4)
        band_frame.pack(fill=tk.X, pady=(0, 6))

        self.band_combo = ttk.Combobox(band_frame, state="readonly")
        self.band_combo.pack(fill=tk.X)
        self.band_combo.bind("<<ComboboxSelected>>",
                             self._on_band_selected)

        # ---- Captures ----
        cap_frame = ttk.LabelFrame(left, text="Captures (.txt)", padding=4)
        cap_frame.pack(fill=tk.X, pady=(0, 6))

        self.cap_listbox = tk.Listbox(
            cap_frame, height=4, selectmode=tk.EXTENDED,
            exportselection=False)
        self.cap_listbox.pack(fill=tk.X)
        cap_btns = ttk.Frame(cap_frame)
        cap_btns.pack(fill=tk.X, pady=2)
        ttk.Button(cap_btns, text="All",
                   command=self._select_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(cap_btns, text="Clear",
                   command=self._clear_sel).pack(side=tk.LEFT, padx=2)

        # ---- Display settings ----
        disp = ttk.LabelFrame(left, text="Display", padding=4)
        disp.pack(fill=tk.X, pady=(0, 6))

        db_min_row = ttk.Frame(disp)
        db_min_row.pack(fill=tk.X)
        ttk.Label(db_min_row, text="dB Min:").pack(side=tk.LEFT)
        self.db_min_label = ttk.Label(db_min_row, text="-150")
        self.db_min_label.pack(side=tk.RIGHT)
        self.db_min_var = tk.DoubleVar(value=-150)
        ttk.Scale(disp, from_=-200, to=0, variable=self.db_min_var,
                  orient=tk.HORIZONTAL,
                  command=self._update_db_labels).pack(fill=tk.X)

        db_max_row = ttk.Frame(disp)
        db_max_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(db_max_row, text="dB Max:").pack(side=tk.LEFT)
        self.db_max_label = ttk.Label(db_max_row, text="-90")
        self.db_max_label.pack(side=tk.RIGHT)
        self.db_max_var = tk.DoubleVar(value=-90)
        ttk.Scale(disp, from_=-200, to=0, variable=self.db_max_var,
                  orient=tk.HORIZONTAL,
                  command=self._update_db_labels).pack(fill=tk.X)

        cmap_row = ttk.Frame(disp)
        cmap_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(cmap_row, text="Colormap:").pack(side=tk.LEFT)
        self.cmap_var = tk.StringVar(value="jet_r")
        ttk.Combobox(cmap_row, textvariable=self.cmap_var,
                     values=COLORMAPS, state="readonly",
                     width=12).pack(side=tk.RIGHT)

        # ---- Buttons ----
        action = ttk.Frame(left)
        action.pack(fill=tk.X, pady=(4, 0))

        self.plot_btn = ttk.Button(
            action, text="Plot", command=self.compute_and_plot)
        self.plot_btn.pack(fill=tk.X, pady=2)

        self.save_btn = ttk.Button(
            action, text="Save Figure", command=self.save_figure)
        self.save_btn.pack(fill=tk.X, pady=2)

        self.status_var = tk.StringVar(
            value="Select a capture directory")
        ttk.Label(left, textvariable=self.status_var, foreground="gray",
                  wraplength=270).pack(anchor=tk.W, pady=(8, 0))

        # ---- Canvas ----
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Select band and click Plot")
        self.ax.set_xlabel("Velocity (km/s)")
        self.ax.set_ylabel("Time (s)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, right)
        toolbar.update()
        toolbar.pack(fill=tk.X)

    # ---- Callbacks ----

    def _browse_dir(self):
        d = filedialog.askdirectory(
            initialdir=self.dir_var.get() or ".")
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

        self.txt_records = find_txt_records(capture_dir)
        if not self.txt_records:
            self.status_var.set("No compressed .txt files found")
            return

        self.bands = {}
        for r in self.txt_records:
            self.bands.setdefault(r["cf_ghz"], []).append(r)

        cfs = sorted(self.bands.keys())
        self._band_cfs = cfs
        labels = [f"{cf:.3f} GHz  ({len(self.bands[cf])} files)"
                  for cf in cfs]
        self.band_combo["values"] = labels

        if labels:
            self.band_combo.current(0)
            self._on_band_selected(None)

        self.status_var.set(
            f"Found {len(self.txt_records)} .txt files "
            f"in {len(cfs)} bands")

    def _on_band_selected(self, event):
        idx = self.band_combo.current()
        if idx < 0:
            return
        cf = self._band_cfs[idx]
        recs = sorted(self.bands[cf],
                      key=lambda r: r["path"])

        self.cap_listbox.delete(0, tk.END)
        for i, rec in enumerate(recs, 1):
            fname = os.path.basename(rec["path"])
            t_str = ""
            if rec["capture_time"]:
                t_str = f"  ({rec['capture_time'].strftime('%H:%M:%S')})"
            self.cap_listbox.insert(tk.END, f"{fname}{t_str}")
        self.cap_listbox.select_set(0, tk.END)

    def _select_all(self):
        self.cap_listbox.select_set(0, tk.END)

    def _clear_sel(self):
        self.cap_listbox.selection_clear(0, tk.END)

    def _update_db_labels(self, *args):
        self.db_min_label.config(
            text=f"{self.db_min_var.get():.0f}")
        self.db_max_label.config(
            text=f"{self.db_max_var.get():.0f}")

    # ---- Plot ----

    def compute_and_plot(self):
        if self._computing:
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

        # Check for cached re-render
        new_key = (cf, tuple(selected))
        if self.cached_data is not None and self._cached_key == new_key:
            self._render_cached()
            return

        recs = sorted(self.bands[cf], key=lambda r: r["path"])
        chosen_paths = [recs[i]["path"] for i in selected]

        self._computing = True
        self.status_var.set("Building velocity waterfall...")
        self.plot_btn.config(state="disabled")
        self.update_idletasks()

        params = dict(cf_ghz=cf, paths=chosen_paths,
                      selected=selected)

        def worker():
            try:
                wf_db, vel_kms, t_full = build_velocity_waterfall(
                    params["paths"], params["cf_ghz"])
                if wf_db is None:
                    self.after(0, lambda: self._plot_error(
                        "No data in selected .txt files"))
                    return
                result = (wf_db, vel_kms, t_full)
                self.after(0, lambda r=result: self._display_result(
                    r, params))
            except Exception as e:
                import traceback
                err_msg = f"{e}\n{traceback.format_exc()}"
                self.after(0, lambda m=err_msg: self._plot_error(m))

        threading.Thread(target=worker, daemon=True).start()

    def _render_cached(self):
        wf_db, vel_kms, t_full = self.cached_data
        cf = self._band_cfs[self.band_combo.current()]
        n_sel = len(list(self.cap_listbox.curselection()))

        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        extent = [vel_kms[0], vel_kms[-1], t_full[0], t_full[-1]]
        self.im = self.ax.imshow(
            wf_db, extent=extent, aspect="auto", origin="lower",
            cmap=self.cmap_var.get(), interpolation="bilinear",
            vmin=self.db_min_var.get(), vmax=self.db_max_var.get())
        self._colorbar = self.fig.colorbar(
            self.im, ax=self.ax, label="Power (dB)")
        self.ax.set_xlabel("Velocity (km/s)")
        self.ax.set_ylabel("Time (s from first capture)")
        title = f"CF = {cf:.3f} GHz — Velocity Waterfall"
        if n_sel > 1:
            title += f"  ({n_sel} files stitched)"
        self.ax.set_title(title)
        self.ax.axvline(0, color="white", ls="--", lw=1, alpha=0.5)
        self.fig.tight_layout()
        self.canvas.draw()
        self.status_var.set("Re-rendered with updated display settings")

    def _display_result(self, result, params):
        self._computing = False
        self.plot_btn.config(state="normal")

        wf_db, vel_kms, t_full = result
        self.cached_data = result
        self._cached_key = (params["cf_ghz"], tuple(params["selected"]))
        cf = params["cf_ghz"]
        n_sel = len(params["paths"])

        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        extent = [vel_kms[0], vel_kms[-1], t_full[0], t_full[-1]]
        self.im = self.ax.imshow(
            wf_db, extent=extent, aspect="auto", origin="lower",
            cmap=self.cmap_var.get(), interpolation="bilinear",
            vmin=self.db_min_var.get(), vmax=self.db_max_var.get())
        self._colorbar = self.fig.colorbar(
            self.im, ax=self.ax, label="Power (dB)")
        self.ax.set_xlabel("Velocity (km/s)")
        self.ax.set_ylabel("Time (s from first capture)")
        title = f"CF = {cf:.3f} GHz — Velocity Waterfall"
        if n_sel > 1:
            title += f"  ({n_sel} files stitched)"
        self.ax.set_title(title)
        self.ax.axvline(0, color="white", ls="--", lw=1, alpha=0.5)
        self.fig.tight_layout()
        self.canvas.draw()

        self.status_var.set(
            f"Plotted {wf_db.shape[0]} time bins, "
            f"{wf_db.shape[1]} velocity bins")

    def _plot_error(self, msg):
        self._computing = False
        self.plot_btn.config(state="normal")
        self.status_var.set(f"Error: {str(msg)[:120]}")

    def save_figure(self):
        if self.cached_data is None:
            messagebox.showwarning(
                "Warning", "Nothing to save -- plot first")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"),
                       ("SVG", "*.svg"), ("All", "*.*")],
            initialdir=self.dir_var.get(),
            initialfile="velocity_waterfall.png")
        if path:
            self.fig.savefig(path, dpi=200, bbox_inches="tight")
            self.status_var.set(f"Saved: {os.path.basename(path)}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Starlink Capture & Plot")
    parser.add_argument("--plot", type=str, default=None,
                        help="Skip capture, open plotter on this directory")
    parser.add_argument("--velocity", type=str, default=None,
                        help="Skip capture, open velocity plotter on this directory")
    args = parser.parse_args()

    root = tk.Tk()

    if args.plot or args.velocity:
        root.withdraw()
        if args.velocity:
            VelocityPlottingWindow(root, args.velocity)
        else:
            PlottingWindow(root, args.plot)
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
