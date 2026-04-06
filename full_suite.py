#!/usr/bin/env python3
"""
Full Suite: Capture -> Preprocess -> Predict -> Correlate -> Plot

Runs the complete Starlink signal analysis pipeline:
  1. USRP + LNB frequency-hopping capture (SigMF + compressed .txt)
  2. Reconstruct velocity-domain waterfalls from compressed data
  3. Generate simulated satellite Doppler predictions for the capture window
  4. NCC-correlate the real waterfall against every visible satellite
  5. Pop up a combined figure:
       - Real RX velocity waterfall (auto-scaled colorbar)
       - Best-match simulated waterfall
       - Ranked bar chart of correlation scores
"""

import os
import sys
import glob
import re
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
from scipy import constants

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOPPLER_DIR = os.path.join(SCRIPT_DIR, "DopplerPredictor")
if DOPPLER_DIR not in sys.path:
    sys.path.insert(0, DOPPLER_DIR)

# System Python that has UHD installed (capture must run outside the venv)
SYSTEM_PYTHON = "/usr/bin/python3"


# ============================================================================
# 1. CAPTURE  (runs as subprocess under system Python w/ UHD)
# ============================================================================

def run_capture(
    usrp_args="type=b200",
    sample_rate=1.0e6,
    gain=40,
    lnb_lo=9.75e9 + 0.2e6,
    output_root=None,
    frequencies=None,
    capture_duration=5,
    total_duration=None,
):
    """Spawn capture.py under the system Python (which has UHD).

    Returns (capture_output_dir, sample_rate).
    """
    if output_root is None:
        output_root = os.path.join(SCRIPT_DIR, "testCaptures")
    if frequencies is None:
        frequencies = [11.575e9, 12.325e9]
    if total_duration is None:
        total_duration = capture_duration * len(frequencies) * 2

    # We pass parameters to a small inline script that imports capture.py,
    # runs the capture, and prints the output directory on the last line
    # so we can parse it back.
    capture_script = f"""
import sys, json
sys.path.insert(0, {SCRIPT_DIR!r})
from capture import StarlinkSigMFCapture

cap = StarlinkSigMFCapture(
    usrp_args={usrp_args!r},
    sample_rate={sample_rate!r},
    gain={gain!r},
    lnb_lo={lnb_lo!r},
    output_root={output_root!r},
)
cap.frequency_hopping_capture(
    capture_duration={capture_duration!r},
    total_duration={total_duration!r},
    frequencies={frequencies!r},
)
# Sentinel line for the parent process to parse
print("CAPTURE_OUTPUT_DIR=" + cap.output_dir)
"""

    print(f"Launching capture subprocess: {SYSTEM_PYTHON}")
    proc = subprocess.run(
        [SYSTEM_PYTHON, "-c", capture_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Stream capture output to console
    print(proc.stdout)

    if proc.returncode != 0:
        raise RuntimeError(f"Capture subprocess failed (exit {proc.returncode})")

    # Parse the output directory from the sentinel line
    capture_dir = None
    for line in proc.stdout.splitlines():
        if line.startswith("CAPTURE_OUTPUT_DIR="):
            capture_dir = line.split("=", 1)[1].strip()
            break

    if capture_dir is None or not os.path.isdir(capture_dir):
        raise RuntimeError(
            f"Could not determine capture output directory from subprocess output"
        )

    return capture_dir, sample_rate


# ============================================================================
# 2. PREPROCESSING  (velocity-domain waterfall from compressed .txt)
# ============================================================================

def parse_filename_metadata(filename):
    """Extract center frequency (Hz) and capture time from filename like
    r001_f11.575GHz_20260401T230951.txt"""
    freq_match = re.search(r'_f([\d.]+)GHz_', filename)
    time_match = re.search(r'_(\d{8}T\d{6})(?:\D|$)', filename)
    if not freq_match or not time_match:
        return None, None
    cf = float(freq_match.group(1)) * 1e9
    capture_time = datetime.strptime(time_match.group(1), "%Y%m%dT%H%M%S")
    return cf, capture_time


def find_txt_records(capture_dir):
    """Scan capture directory for compressed .txt files."""
    txt_files = sorted(glob.glob(os.path.join(capture_dir, "*.txt")))
    records = []
    for txt_path in txt_files:
        fname = os.path.basename(txt_path)
        if fname == "capture_log.txt":
            continue
        cf, capture_time = parse_filename_metadata(fname)
        if cf is None:
            continue
        records.append({"data": txt_path, "cf": cf, "capture_time": capture_time})
    return records


def build_velocity_waterfall(records_band, sample_rate, k=10, nfft=1024, noverlap=512):
    """Reconstruct a velocity-domain waterfall from compressed .txt records
    for one frequency band.

    Returns (vel_axis, t_full, Sxx_db_full, datetimestamps_full).
    """
    records_band = sorted(records_band, key=lambda r: r["capture_time"])

    v_max = 5000  # m/s
    n_vel_bins = 500
    v = np.linspace(-v_max, v_max, n_vel_bins)

    Sxx_segments = []
    t_segments = []
    dt_segments = []
    step = nfft - noverlap

    for rec_idx, rec in enumerate(records_band):
        fname = os.path.basename(rec["data"])
        print(f"  [{rec_idx+1}/{len(records_band)}] Loading {fname}  "
              f"(CF={rec['cf']/1e9:.3f} GHz, time={rec['capture_time']})")

        data = np.loadtxt(rec["data"], dtype=complex)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        num_rows = data.shape[0]
        if num_rows == 0:
            print(f"         -> 0 rows, skipping")
            continue

        t = (np.arange(num_rows) * step + nfft / 2) / sample_rate
        print(f"         -> {num_rows} STFT frames, {t[-1]:.2f}s duration")

        recon_freqs_hz = data[:, :k].real
        # capture.py stores raw STFT frequencies (Hz); convert to velocity (m/s)
        recon_vels = C * recon_freqs_hz / rec["cf"]
        recon_vals = data[:, k:]

        vel_range = (recon_vels.min(), recon_vels.max())
        print(f"         -> Velocity range: {vel_range[0]:.1f} to {vel_range[1]:.1f} m/s")

        sparse = np.zeros((n_vel_bins, num_rows), dtype=complex)

        for i in range(num_rows):
            indices = np.searchsorted(v, recon_vels[i, :])
            indices = np.clip(indices, 0, n_vel_bins - 1)
            sparse[indices, i] = recon_vals[i, :]

        Sxx_db = 10.0 * np.log10(np.abs(sparse) + 1e-15)

        # Report dB stats for non-background bins
        valid = Sxx_db[Sxx_db > -140]
        if valid.size > 0:
            print(f"         -> dB stats (signal): min={valid.min():.1f}  "
                  f"median={np.median(valid):.1f}  max={valid.max():.1f}  "
                  f"p5={np.percentile(valid,5):.1f}  p95={np.percentile(valid,95):.1f}")
        else:
            print(f"         -> WARNING: no signal bins above -140 dB")

        # Build absolute timestamps
        stop_np = np.datetime64(rec["capture_time"])
        offset_ns = (t * 1e9).astype(np.int64)
        start_time = stop_np - np.timedelta64(int(offset_ns[-1]), 'ns')
        timestamps = start_time + offset_ns.astype('timedelta64[ns]')

        Sxx_segments.append(Sxx_db)
        t_segments.append(t)
        dt_segments.append(timestamps)

    if not Sxx_segments:
        return None, None, None, None

    Sxx_full = np.concatenate(Sxx_segments, axis=1)
    t_full = np.concatenate(t_segments)
    dt_full = np.concatenate(dt_segments)

    return v, t_full, Sxx_full, dt_full


# ============================================================================
# 3. SATELLITE PREDICTION  (generate simulated Doppler curves)
# ============================================================================

C = constants.c

def load_tle_data(tle_path=None):
    """Load TLE text from file."""
    if tle_path is None:
        tle_path = os.path.join(DOPPLER_DIR, "starlink.txt")
    with open(tle_path, "r") as f:
        return f.read()


def generate_satellite_predictions(
    tle_data,
    start_time_utc,
    duration_sec,
    ue_lat=47.655,
    ue_lon=-122.3035,
    ue_alt_m=60.0,
    tx_freq_hz=10.5e9,
    elevation_mask=10.0,
    max_sats=1000,
    time_step_sec=1.0,
):
    """Generate per-satellite DataFrames of Doppler/velocity predictions
    for all satellites visible during [start_time_utc, start_time_utc + duration_sec].

    Returns dict: sat_name -> DataFrame with columns
        [timestamp, relative_velocity_kms, distance_km, tx_freq_ghz, elevation_deg, ...]
    """
    from skyfield.api import EarthSatellite, load, wgs84, utc

    tle_lines = tle_data.strip().split('\n')
    ts = load.timescale()
    observer = wgs84.latlon(ue_lat, ue_lon, elevation_m=ue_alt_m)

    # Parse TLEs
    satellites = []
    i = 0
    while i < len(tle_lines) - 2 and len(satellites) < max_sats:
        line1 = tle_lines[i + 1].strip() if i + 1 < len(tle_lines) else ""
        line2 = tle_lines[i + 2].strip() if i + 2 < len(tle_lines) else ""
        if line1.startswith('1 ') and line2.startswith('2 '):
            cat_num = line1[2:7].strip()
            try:
                sat = EarthSatellite(line1, line2)
                satellites.append((f"SAT-{cat_num}", sat))
            except Exception:
                pass
        i += 3

    num_steps = int(duration_sec / time_step_sec)
    print(f"  Loaded {len(satellites)} TLEs, propagating orbits...")
    print(f"  Time window: {num_steps} steps @ {time_step_sec}s each")
    times_dt = [start_time_utc + timedelta(seconds=j * time_step_sec)
                for j in range(num_steps)]
    times_sf = ts.from_datetimes([t.replace(tzinfo=utc) if t.tzinfo is None else t
                                  for t in times_dt])

    sat_results = {}

    for sat_name, sat_obj in satellites:
        try:
            diff = sat_obj - observer
            topo = diff.at(times_sf)
            alt, az, dist = topo.altaz()

            el = alt.degrees
            visible_mask = el >= elevation_mask
            if not np.any(visible_mask):
                continue

            dist_km = dist.km

            # Range rate via finite difference on distance
            range_rate_kms = np.gradient(dist_km, time_step_sec)
            doppler_hz = -tx_freq_hz * range_rate_kms * 1000.0 / C
            vel_kms = -doppler_hz * C / (tx_freq_hz * 1000.0)

            rows = []
            for idx in np.where(visible_mask)[0]:
                rows.append({
                    "timestamp": times_dt[idx],
                    "satellite": sat_name,
                    "elevation_deg": el[idx],
                    "azimuth_deg": az.degrees[idx],
                    "distance_km": dist_km[idx],
                    "relative_velocity_kms": vel_kms[idx],
                    "doppler_shift_hz": doppler_hz[idx],
                    "tx_freq_ghz": tx_freq_hz / 1e9,
                    "rx_freq_hz": tx_freq_hz + doppler_hz[idx],
                })
            if rows:
                df_sat = pd.DataFrame(rows)
                # Reject satellites with unrealistic velocities (bad TLE)
                max_vel = df_sat["relative_velocity_kms"].abs().max()
                if max_vel > 20.0:
                    continue
                sat_results[sat_name] = df_sat
        except Exception:
            continue

    print(f"  Found {len(sat_results)} satellites visible during capture window")
    if sat_results:
        # Show a few example satellites with their velocity ranges
        shown = 0
        for sname, sdf in list(sat_results.items())[:5]:
            vr = sdf["relative_velocity_kms"]
            el = sdf["elevation_deg"]
            print(f"    {sname}: vel=[{vr.min():.2f}, {vr.max():.2f}] km/s, "
                  f"elev=[{el.min():.1f}, {el.max():.1f}] deg, {len(sdf)} pts")
            shown += 1
        if len(sat_results) > shown:
            print(f"    ... and {len(sat_results) - shown} more")
    return sat_results


# ============================================================================
# 4. CORRELATION  (NCC between real and simulated waterfalls)
# ============================================================================

def fspl_db(dist_km, tx_freq_hz):
    dist_m = np.asarray(dist_km, dtype=np.float64) * 1000.0
    return (20.0 * np.log10(dist_m + 1e-12)
            + 20.0 * np.log10(tx_freq_hz + 1e-12)
            + 20.0 * np.log10(4.0 * np.pi / C))


def build_sim_waterfall(df_sat, t1hz_utc, vel_centers_kms, sigma_kms=0.05,
                        background_db=250.0):
    """Build a simulated velocity waterfall aligned to the real data axes."""
    t_sim = pd.to_datetime(df_sat["timestamp"], utc=True).dt.floor("s")
    row_idx = t1hz_utc.get_indexer(t_sim)
    ok = row_idx >= 0
    if ok.sum() == 0:
        return None

    row_idx = row_idx[ok]
    df_ok = df_sat.loc[ok.values].copy() if isinstance(ok, pd.Series) else df_sat.iloc[np.where(ok)[0]].copy()

    T = len(t1hz_utc)
    V = len(vel_centers_kms)
    sim = np.full((T, V), background_db, dtype=np.float32)

    tx_freq_hz = float(df_ok["tx_freq_ghz"].iloc[0]) * 1e9
    pl = fspl_db(df_ok["distance_km"].to_numpy(), tx_freq_hz)
    vel = df_ok["relative_velocity_kms"].to_numpy(dtype=np.float64)

    for ti, v_k, p in zip(row_idx, vel, pl):
        gaussian = np.exp(-0.5 * ((vel_centers_kms - v_k) / sigma_kms) ** 2)
        row = p * (1.0 - 0.9 * gaussian) + background_db * (1.0 - gaussian)
        sim[ti, :] = np.minimum(sim[ti, :], row.astype(np.float32))

    return sim


def downsample_to_1hz(Sxx_db, timestamps, vel_axis, n_vel_out=300):
    """Downsample the real waterfall to 1 Hz and resample velocity bins.

    Uses MAX (not mean) for time aggregation since data is sparse in dB domain —
    averaging -30 dB signal with -150 dB background produces garbage.
    """
    # Sxx_db shape: [vel_bins, time_bins] -> transpose to [time, vel]
    W = Sxx_db.T.copy()
    t_pd = pd.to_datetime(timestamps, utc=True)

    df = pd.DataFrame(W)
    df["t_sec"] = t_pd.floor("s")
    # Use MAX to preserve signal peaks in this sparse representation
    agg = df.groupby("t_sec").max(numeric_only=True)

    real_1hz = agg.to_numpy(dtype=np.float32)
    t1hz = agg.index

    print(f"    1Hz downsampled: {real_1hz.shape}, "
          f"dB range [{real_1hz.min():.1f}, {real_1hz.max():.1f}]")

    # Resample velocity axis
    vel_in = np.asarray(vel_axis, dtype=np.float64)
    vel_out = np.linspace(vel_in.min(), vel_in.max(), n_vel_out)
    T = real_1hz.shape[0]
    real_out = np.empty((T, n_vel_out), dtype=np.float32)
    for i in range(T):
        real_out[i, :] = np.interp(vel_out, vel_in, real_1hz[i, :])

    return real_out, t1hz, vel_out


def ridge_extract(H, vel_window=(180, 290), mode="max"):
    """Extract ridge index from a velocity window."""
    lo, hi = vel_window
    hi = min(hi, H.shape[1])
    sub = H[:, lo:hi]
    if mode == "max":
        return np.argmax(sub, axis=1)
    return np.argmin(sub, axis=1)


def ncc_1d(x, y):
    """Normalised cross-correlation, returns best NCC and lag."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xcorr = np.correlate(x - x.mean(), y - y.mean(), mode="full")
    denom = np.linalg.norm(x - x.mean()) * np.linalg.norm(y - y.mean()) + 1e-12
    ncc = xcorr / denom
    lags = np.arange(-len(y) + 1, len(x))
    best_idx = np.argmax(ncc)
    return float(ncc[best_idx]), int(lags[best_idx])


def correlate_all(real_1hz, vel_centers_kms, t1hz, sat_predictions):
    """Score every predicted satellite against the real waterfall via NCC.

    Returns sorted list of (sat_name, ncc_score, sim_waterfall).
    """
    # Flip real waterfall to match corelator.py convention
    real_flipped = np.flip(real_1hz, axis=1)
    real_ridge = ridge_extract(real_flipped, mode="max")

    total = len(sat_predictions)
    scores = []
    for idx, (sat_name, df_sat) in enumerate(sat_predictions.items(), 1):
        if idx % 50 == 0 or idx == 1 or idx == total:
            print(f"    Correlating satellite {idx}/{total}: {sat_name}...")
        sim = build_sim_waterfall(df_sat, t1hz, vel_centers_kms)
        if sim is None:
            continue
        sim_ridge = ridge_extract(sim, mode="min")

        # Use the middle portion of the pass for correlation
        n = len(real_ridge)
        start = n // 5
        end = 4 * n // 5
        if end - start < 10:
            start, end = 0, n

        best_ncc, _best_lag = ncc_1d(real_ridge[start:end], sim_ridge[start:end])
        if not np.isnan(best_ncc):
            scores.append((sat_name, best_ncc, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ============================================================================
# 5. PLOTTING
# ============================================================================

def auto_db_range(W, plow=2, phigh=98):
    """Compute colorbar limits from data percentiles, ignoring background."""
    valid = W[W > -100]  # ignore zero-padded bins (background ~ -150 dB)
    if valid.size > 0:
        lo = float(np.percentile(valid, plow))
        hi = float(np.percentile(valid, phigh))
        # Ensure at least 3 dB spread
        if hi - lo < 3.0:
            lo = hi - 10.0
        return lo, hi
    return -50.0, -20.0


def show_results(vel_axis, t_axis, Sxx_db, scores, vel_centers_kms, t1hz, cf_hz,
                  real_1hz=None):
    """Pop up the combined results figure."""
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1])

    # --- Panel 1: Real RX waterfall (use 1Hz downsampled for visibility) ---
    ax_real = fig.add_subplot(gs[0, 0])
    if real_1hz is not None:
        # Plot the 1 Hz downsampled version — much more visible than raw sparse
        W_plot = real_1hz
        vmin, vmax = auto_db_range(W_plot)
        duration_min = (t1hz[-1] - t1hz[0]).total_seconds() / 60.0
        extent_real = [vel_centers_kms[0], vel_centers_kms[-1], duration_min, 0]
        ylabel = "Time (min)"
    else:
        # Fallback to raw sparse waterfall
        W_plot = Sxx_db.T
        vmin, vmax = auto_db_range(Sxx_db)
        extent_real = [vel_axis[0] / 1000, vel_axis[-1] / 1000, t_axis[-1], t_axis[0]]
        ylabel = "Time (s)"
    print(f"  Real waterfall auto dB range: [{vmin:.1f}, {vmax:.1f}]")
    im1 = ax_real.imshow(
        W_plot, extent=extent_real, aspect="auto", origin="upper",
        cmap="turbo", interpolation="bilinear", vmin=vmin, vmax=vmax,
    )
    ax_real.set_xlabel("Relative Velocity (km/s)")
    ax_real.set_ylabel(ylabel)
    ax_real.set_title(f"Real RX Waterfall  |  CF = {cf_hz/1e9:.3f} GHz")
    fig.colorbar(im1, ax=ax_real, label="Power (dB)")

    # --- Panel 2: Best-match simulated waterfall ---
    ax_sim = fig.add_subplot(gs[0, 1])
    if scores:
        best_name, best_ncc, best_sim = scores[0]
        valid_sim = best_sim[best_sim < 249]
        if valid_sim.size > 0:
            sv_min, sv_max = np.percentile(valid_sim, [5, 95])
        else:
            sv_min, sv_max = 170, 185
        duration_min = (t1hz[-1] - t1hz[0]).total_seconds() / 60.0
        extent_sim = [vel_centers_kms.min(), vel_centers_kms.max(), duration_min, 0]
        im2 = ax_sim.imshow(
            best_sim, extent=extent_sim, aspect="auto", origin="upper",
            cmap="jet_r", interpolation="bilinear", vmin=sv_min, vmax=sv_max,
        )
        ax_sim.axvline(0, color="white", ls="--", lw=1.5, alpha=0.8)
        ax_sim.set_xlabel("Relative Velocity (km/s)")
        ax_sim.set_ylabel("Time into pass (min)")
        ax_sim.set_title(f"Best Match: {best_name}  (NCC = {best_ncc:.3f})")
        fig.colorbar(im2, ax=ax_sim, label="Path Loss (dB)")
    else:
        ax_sim.text(0.5, 0.5, "No satellite matches found",
                    ha="center", va="center", transform=ax_sim.transAxes, fontsize=14)
        ax_sim.set_title("Simulated Waterfall")

    # --- Panel 3: Correlation score bar chart ---
    ax_bar = fig.add_subplot(gs[1, :])
    if scores:
        top_n = min(20, len(scores))
        names = [s[0] for s in scores[:top_n]]
        nccs = [s[1] for s in scores[:top_n]]
        colors = plt.cm.RdYlGn(np.linspace(0.3, 1.0, top_n))[::-1]
        bars = ax_bar.barh(range(top_n), nccs, color=colors)
        ax_bar.set_yticks(range(top_n))
        ax_bar.set_yticklabels(names, fontsize=9)
        ax_bar.set_xlabel("NCC Score")
        ax_bar.set_title(f"Satellite Correlation Ranking  ({len(scores)} candidates)")
        ax_bar.invert_yaxis()
        ax_bar.axvline(0.5, color="gray", ls="--", alpha=0.5, label="0.5 threshold")
        ax_bar.legend(loc="lower right")
        # Annotate score values
        for bar, val in zip(bars, nccs):
            ax_bar.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=8)
    else:
        ax_bar.text(0.5, 0.5, "No correlation results",
                    ha="center", va="center", transform=ax_bar.transAxes, fontsize=14)

    fig.suptitle("Starlink Full Suite — Capture + Correlate", fontsize=16, fontweight="bold")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    # ---- User-configurable parameters ----
    USRP_ARGS      = "type=b200"
    SAMPLE_RATE     = 0.5e6         # Hz
    GAIN            = 40            # dB
    LNB_LO          = 9.75e9 + 0.2e6
    OUTPUT_ROOT     = os.path.join(SCRIPT_DIR, "testCaptures")
    FREQUENCIES     = [11.575e9, 12.325e9]
    CAPTURE_DURATION = 120          # seconds per frequency
    TOTAL_DURATION   = CAPTURE_DURATION * len(FREQUENCIES) * 2

    # Ground station
    UE_LAT  = 47.655
    UE_LON  = -122.3035
    UE_ALT  = 60.0          # meters

    # Satellite prediction
    TX_FREQ = 10.5e9         # Starlink downlink
    TLE_PATH = os.path.join(DOPPLER_DIR, "starlink.txt")
    ELEVATION_MASK = 10.0

    # Preprocessing
    K       = 10
    NFFT    = 1024
    NOVERLAP = 512

    # ==================================================================
    # Step 1: Capture
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 1/5: USRP FREQUENCY-HOPPING CAPTURE")
    print("=" * 60)
    print(f"  Sample rate:      {SAMPLE_RATE/1e3:.0f} kS/s")
    print(f"  Gain:             {GAIN} dB")
    print(f"  LNB LO:           {LNB_LO/1e9:.4f} GHz")
    print(f"  Frequencies:      {[f'{f/1e9:.3f} GHz' for f in FREQUENCIES]}")
    print(f"  Capture duration: {CAPTURE_DURATION}s per freq")
    print(f"  Total duration:   {TOTAL_DURATION}s")
    print(f"  Output root:      {OUTPUT_ROOT}")

    capture_dir, sr = run_capture(
        usrp_args=USRP_ARGS,
        sample_rate=SAMPLE_RATE,
        gain=GAIN,
        lnb_lo=LNB_LO,
        output_root=OUTPUT_ROOT,
        frequencies=FREQUENCIES,
        capture_duration=CAPTURE_DURATION,
        total_duration=TOTAL_DURATION,
    )
    print(f"  Capture saved to: {capture_dir}")

    # ==================================================================
    # Step 2: Preprocess — build velocity waterfalls
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 2/5: PREPROCESSING — Building velocity waterfalls")
    print("=" * 60)

    records = find_txt_records(capture_dir)
    if not records:
        print("[ERROR] No compressed .txt files found — check capture output")
        return

    # Group by center frequency
    bands = {}
    for r in records:
        bands.setdefault(r["cf"], []).append(r)

    print(f"  Found {len(records)} compressed .txt files across {len(bands)} band(s)")
    for cf_key in sorted(bands.keys()):
        print(f"    Band {cf_key/1e9:.3f} GHz: {len(bands[cf_key])} capture(s)")

    # Process first band (primary)
    primary_cf = sorted(bands.keys())[0]
    band_records = bands[primary_cf]
    print(f"\n  >>> Sampling band {primary_cf/1e9:.3f} GHz now <<<")
    print(f"  Loading {len(band_records)} .txt record(s) for this band...")

    vel_axis, t_axis, Sxx_db, timestamps = build_velocity_waterfall(
        band_records, sr, k=K, nfft=NFFT, noverlap=NOVERLAP,
    )
    if vel_axis is None:
        print("[ERROR] Could not build waterfall")
        return

    print(f"\n  Waterfall shape: {Sxx_db.shape}  (vel_bins x time_bins)")
    print(f"  Velocity axis: {vel_axis[0]:.0f} to {vel_axis[-1]:.0f} m/s  ({len(vel_axis)} bins)")
    print(f"  Time axis: {t_axis[0]:.3f} to {t_axis[-1]:.3f} s  ({len(t_axis)} frames)")

    # Report overall dB stats
    signal_mask = Sxx_db > -100
    n_signal = signal_mask.sum()
    n_total = Sxx_db.size
    print(f"  Signal bins (> -100 dB): {n_signal}/{n_total} ({100*n_signal/n_total:.2f}%)")
    if n_signal > 0:
        sig_vals = Sxx_db[signal_mask]
        print(f"  Signal dB range: min={sig_vals.min():.1f}  median={np.median(sig_vals):.1f}  "
              f"max={sig_vals.max():.1f}")

    # Downsample to 1 Hz for correlation
    print(f"\n  >>> Downsampling waterfall to 1 Hz for correlation <<<")
    real_1hz, t1hz, vel_centers_kms = downsample_to_1hz(
        Sxx_db, timestamps, vel_axis / 1000.0, n_vel_out=300,
    )
    print(f"  Downsampled shape: {real_1hz.shape}  (time_sec x vel_bins)")
    print(f"  Time range: {t1hz[0]} -> {t1hz[-1]}")
    print(f"  Velocity range: {vel_centers_kms[0]:.2f} to {vel_centers_kms[-1]:.2f} km/s")

    # ==================================================================
    # Step 3: Generate satellite predictions
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 3/5: SATELLITE DOPPLER PREDICTION")
    print("=" * 60)

    tle_data = load_tle_data(TLE_PATH)
    tle_line_count = len(tle_data.strip().split('\n'))
    print(f"  TLE file: {TLE_PATH}")
    print(f"  TLE entries: ~{tle_line_count // 3} satellites in catalog")

    # Determine capture time window from timestamps
    t_start = pd.Timestamp(t1hz[0]).to_pydatetime()
    t_end = pd.Timestamp(t1hz[-1]).to_pydatetime()
    # Add padding so we catch satellites entering/leaving the window
    pad = timedelta(minutes=5)
    duration_sec = (t_end - t_start + 2 * pad).total_seconds()

    print(f"  Capture window:   {t_start} -> {t_end}")
    print(f"  Padded duration:  {duration_sec:.0f}s ({duration_sec/60:.1f} min)")
    print(f"  Observer:         ({UE_LAT}, {UE_LON}), alt {UE_ALT}m")
    print(f"  TX freq:          {TX_FREQ/1e9:.3f} GHz")
    print(f"  Elevation mask:   {ELEVATION_MASK} deg")
    print(f"\n  >>> Propagating TLEs and scanning for visible passes... <<<")

    sat_predictions = generate_satellite_predictions(
        tle_data,
        start_time_utc=t_start - pad,
        duration_sec=duration_sec,
        ue_lat=UE_LAT,
        ue_lon=UE_LON,
        ue_alt_m=UE_ALT,
        tx_freq_hz=TX_FREQ,
        elevation_mask=ELEVATION_MASK,
    )

    # ==================================================================
    # Step 4: Correlate
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 4/5: NCC CORRELATION — Real vs Simulated")
    print("=" * 60)
    print(f"  Visible satellites to correlate: {len(sat_predictions)}")
    print(f"  Real waterfall shape:  {real_1hz.shape}")
    print(f"  Velocity bins:         {len(vel_centers_kms)}  "
          f"({vel_centers_kms[0]:.2f} to {vel_centers_kms[-1]:.2f} km/s)")
    print(f"\n  >>> Correlating each satellite now... <<<")

    scores = correlate_all(real_1hz, vel_centers_kms, t1hz, sat_predictions)
    if scores:
        print(f"\n  Scored {len(scores)} satellites. Top 10 matches:")
        print(f"  {'Rank':>4s}  {'Satellite':>12s}  {'NCC':>8s}")
        print(f"  {'-'*4}  {'-'*12}  {'-'*8}")
        for rank, (name, ncc, _) in enumerate(scores[:10], 1):
            marker = " <-- BEST" if rank == 1 else ""
            print(f"  {rank:4d}  {name:>12s}  {ncc:8.4f}{marker}")
    else:
        print("  [WARNING] No matching satellites found")

    # ==================================================================
    # Step 5: Show results
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 5/5: PLOTTING RESULTS")
    print("=" * 60)
    print(f"  Plotting real waterfall:   {Sxx_db.shape}")
    if scores:
        print(f"  Best match:               {scores[0][0]} (NCC={scores[0][1]:.4f})")
        print(f"  Total ranked satellites:   {len(scores)}")
    else:
        print(f"  No satellite matches to plot")

    show_results(vel_axis, t_axis, Sxx_db, scores, vel_centers_kms, t1hz, primary_cf,
                 real_1hz=real_1hz)
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
