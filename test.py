"""
correlator_combined.py — End-to-end satellite Doppler correlation pipeline.

Always runs fresh: downloads TLE, preprocesses from .txt, predicts Doppler, correlates.

Usage:
    python correlator_combined.py <capture_dir> [--freq FREQ_GHZ]

Example:
    python correlator_combined.py ./testCaptures/same_direction_extra_rx1 --freq 12.325
"""

import argparse
import glob
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy import signal as sp_signal

C = 299792458.0

# Ground station defaults (UW campus)
DEFAULT_LAT = 47.655
DEFAULT_LON = -122.3035
DEFAULT_ALT = 60.0


# =====================================================================
# STAGE 1 — Preprocessing: always rebuild from .txt compressed data
# =====================================================================

def run_preprocessing(capture_dir, freq_ghz, sample_rate=500000.0,
                      k=10, nfft=1024, noverlap=512):
    """
    Always rebuild waterfall from .txt compressed files.
    If no .txt exist, compress from .sigmf-data first.
    """
    txt_pattern = f"r*_f{freq_ghz:.3f}GHz_*.txt"
    txt_files = sorted(glob.glob(os.path.join(capture_dir, txt_pattern)))

    if not txt_files:
        print(f"[preprocess] No .txt files for {freq_ghz:.3f} GHz, compressing from .sigmf...")
        sigmf_pattern = f"r*_f{freq_ghz:.3f}GHz_*.sigmf-data"
        sigmf_files = sorted(glob.glob(os.path.join(capture_dir, sigmf_pattern)))
        if not sigmf_files:
            raise FileNotFoundError(
                f"No .sigmf-data or .txt files for {freq_ghz:.3f} GHz in {capture_dir}"
            )
        _compress_sigmf_files(sigmf_files, k, nfft, noverlap)
        txt_files = sorted(glob.glob(os.path.join(capture_dir, txt_pattern)))
        if not txt_files:
            raise FileNotFoundError("Compression produced no .txt files")

    print(f"[preprocess] Building waterfall from {len(txt_files)} .txt files...")
    tag = _build_waterfall(txt_files, capture_dir, freq_ghz, sample_rate, k, nfft, noverlap)
    return tag


def _compress_sigmf_files(sigmf_files, k, nfft, noverlap):
    """Compress raw .sigmf-data into sparse .txt."""
    import sigmf as sigmf_lib

    for data_path in sigmf_files:
        base = data_path.replace(".sigmf-data", "")
        meta_path = base + ".sigmf-meta"
        if not os.path.exists(meta_path):
            continue

        print(f"  Compressing {os.path.basename(base)}...")
        datafile = sigmf_lib.fromfile(base)
        global_meta = datafile.get_global_info()
        captures = datafile.get_captures()
        sr = global_meta.get("core:sample_rate", 500000.0)
        cf = captures[0]["core:frequency"]

        dtype_str = global_meta.get("core:datatype", "cf32_le")
        data_dtype = np.complex64 if dtype_str == "cf32_le" else np.int16

        raw = np.fromfile(data_path, dtype=data_dtype)
        f, t, stft = sp_signal.stft(raw, fs=sr, nperseg=nfft,
                                     noverlap=noverlap, return_onesided=False)
        stft = np.fft.fftshift(stft, axes=0)
        f = np.fft.fftshift(f)

        dc_kill = np.where(np.abs(f) <= 5000)
        nf = stft.shape[1]
        out_vals = np.zeros((nf, k), dtype=complex)
        out_freqs = np.zeros((nf, k))

        for i in range(nf):
            row = np.abs(stft[:, i])
            row[dc_kill] = 0
            top = np.argpartition(row, -k)[-k:]
            top = top[np.argsort(f[top])]
            out_vals[i, :] = stft[top, i]
            out_freqs[i, :] = C * f[top] / cf

        np.savetxt(base + ".txt",
                   np.column_stack((out_freqs, out_vals)),
                   fmt="%.10g", delimiter=" ")
        print(f"  -> {os.path.basename(base)}.txt")


def _build_waterfall(txt_files, capture_dir, freq_ghz, sample_rate, k, nfft, noverlap):
    """Reconstruct velocity waterfall from compressed .txt files."""
    v_max = 5000
    n_vel = 500
    v_axis = np.linspace(-v_max, v_max, n_vel)

    for j, txt_path in enumerate(txt_files):
        fname = os.path.basename(txt_path)
        time_match = re.search(r"_(\d{8}T\d{6})", fname)
        if not time_match:
            print(f"  [WARN] Cannot parse time from {fname}")
            continue
        capture_time = datetime.strptime(time_match.group(1), "%Y%m%dT%H%M%S")

        data = np.loadtxt(txt_path, dtype=complex)
        num_rows = data.shape[0]
        step = nfft - noverlap
        t = (np.arange(num_rows) * step + nfft / 2) / sample_rate

        recon_freqs = data[:, 0:k]
        recon_vals = data[:, k:]
        sparse = np.zeros((len(v_axis), num_rows), dtype=complex)

        for i in range(num_rows):
            idx = np.searchsorted(v_axis, np.real(recon_freqs[i, :]))
            idx = np.clip(idx, 0, len(v_axis) - 1)
            sparse[idx, i] = recon_vals[i, :]

        tag = f"r{j}_waterfall_CF_{freq_ghz:.3f}GHz"

        np.save(os.path.join(capture_dir, f"{tag}.npy"), sparse.T)
        np.save(os.path.join(capture_dir, f"rel_vel_{tag}.npy"), v_axis)
        np.save(os.path.join(capture_dir, f"time_{tag}.npy"), [t])

        # Absolute timestamps
        stop_np = np.datetime64(capture_time)
        offset_ns = (np.asarray(t) * 1e9).astype(np.int64)
        start_np = stop_np - offset_ns[-1].astype("timedelta64[ns]")
        datetimes = start_np + offset_ns.astype("timedelta64[ns]")
        np.save(os.path.join(capture_dir, f"datetime_updated_{tag}.npy"), datetimes)

        print(f"  -> {tag} ({num_rows} frames, "
              f"{datetimes[0]} to {datetimes[-1]})")

    return f"r0_waterfall_CF_{freq_ghz:.3f}GHz"


# =====================================================================
# STAGE 2 — Load real waterfall
# =====================================================================

def load_real_waterfall(capture_dir, tag, n_vel_out=300):
    """Load waterfall, downsample to 1 Hz, convert to dB.

    Aggregates in LINEAR power domain (mean), then converts to dB.
    This is critical for sparse data (k=10 out of 500 bins):
    - Mean of dB values is wrong (zeros→-300 dB dominate the average)
    - Max of linear elevates all noise bins equally (no contrast)
    - Mean of linear correctly preserves signal-to-noise ratio
    """
    raw = np.load(os.path.join(capture_dir, f"{tag}.npy"))
    time_axis = np.load(os.path.join(capture_dir, f"datetime_updated_{tag}.npy"))
    vel_axis = np.load(os.path.join(capture_dir, f"rel_vel_{tag}.npy")) / 1000.0

    # Work in linear power for aggregation (NOT dB)
    is_complex = np.iscomplexobj(raw)
    if is_complex:
        power_lin = np.abs(raw) ** 2
    else:
        power_lin = raw.astype(np.float64)

    t_utc = pd.to_datetime(np.asarray(time_axis), utc=True, errors="raise")
    assert len(t_utc) == power_lin.shape[0]

    # 1 Hz downsample: MEAN in linear domain preserves SNR correctly
    df = pd.DataFrame(power_lin)
    df["t_sec"] = t_utc.floor("s")
    agg = df.groupby("t_sec").mean(numeric_only=True)
    wf = agg.to_numpy(dtype=np.float64)
    t1hz = agg.index

    # Convert to dB AFTER aggregation
    if is_complex:
        wf = 10.0 * np.log10(wf + 1e-30)
    wf = wf.astype(np.float32)

    # Resample velocity
    vel_in = np.asarray(vel_axis, dtype=np.float64).ravel()
    V_in = wf.shape[1]
    if len(vel_in) == V_in + 1:
        vel_in = 0.5 * (vel_in[:-1] + vel_in[1:])
    vel_out = np.linspace(vel_in.min(), vel_in.max(), n_vel_out)
    wf_out = np.empty((wf.shape[0], n_vel_out), dtype=np.float32)
    for i in range(wf.shape[0]):
        wf_out[i, :] = np.interp(vel_out, vel_in, wf[i, :]).astype(np.float32)

    print(f"[load] Shape: {wf_out.shape}, {t1hz[0]} → {t1hz[-1]}, "
          f"vel: [{vel_out.min():.1f}, {vel_out.max():.1f}] km/s")
    return wf_out, t1hz, vel_out


# =====================================================================
# STAGE 3 — Download TLE and predict Doppler
# =====================================================================

def download_tle():
    """Download fresh Starlink TLEs from Celestrak."""
    import requests
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
    print("[tle] Downloading fresh TLE from Celestrak...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "DopplerPredictor", "starlink_fresh.txt")
    with open(out, "w") as f:
        f.write(resp.text)
    n_sats = resp.text.count("\n") // 3
    print(f"[tle] Downloaded {n_sats} satellites → {out}")
    return resp.text


def predict_doppler(tle_data, start_utc, duration_sec, tx_freq_hz,
                    elevation_mask=10.0,
                    lat=DEFAULT_LAT, lon=DEFAULT_LON, alt=DEFAULT_ALT):
    """Predict Doppler for all visible satellites in a time window at 1 Hz."""
    from skyfield.api import EarthSatellite, load, wgs84, utc

    ts = load.timescale()
    observer = wgs84.latlon(lat, lon, elevation_m=alt)

    # Parse TLE
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

    print(f"[predict] {len(sats)} satellites loaded, "
          f"predicting {int(duration_sec)}s window...")

    # Time grid at 1 Hz
    times_dt = [start_utc + timedelta(seconds=s) for s in range(int(duration_sec))]
    times_sky = ts.from_datetimes(
        [t.replace(tzinfo=utc) if t.tzinfo is None else t for t in times_dt]
    )

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
            range_rate = np.gradient(dist_km, 1.0)  # km/s

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

    print(f"[predict] {n_visible} satellites visible above {elevation_mask}°")
    return results


# =====================================================================
# STAGE 4 — Build simulated FSPL waterfalls & Correlate via ridge NCC
# =====================================================================

def fspl_db(dist_km, tx_freq_hz):
    """Free-space path loss in dB."""
    dist_m = np.asarray(dist_km, dtype=np.float64) * 1000.0
    return (20.0 * np.log10(dist_m + 1e-12)
            + 20.0 * np.log10(tx_freq_hz + 1e-12)
            + 20.0 * np.log10(4.0 * np.pi / C))


def build_sim_waterfall(pred_df, t1hz, vel_kms, tx_freq_hz,
                        sigma_kms=0.05, background_db=250.0):
    """
    Build a free-space path loss simulated waterfall aligned to real axes.
    Returns (sim_waterfall, matched_row_indices, negated_velocity_array)
    or (None, None, None) if no overlap.
    """
    pred_t = pd.to_datetime(pred_df["timestamp"], utc=True).dt.floor("s")
    row_idx = t1hz.get_indexer(pred_t)
    ok = row_idx >= 0
    if ok.sum() == 0:
        return None, None, None

    row_idx = row_idx[ok]
    df_ok = pred_df.iloc[np.where(ok)[0]]

    T = len(t1hz)
    V = len(vel_kms)
    sim = np.full((T, V), background_db, dtype=np.float32)

    # Negate velocity: preprocessing convention (positive = approaching)
    # vs prediction convention (negative = approaching)
    vel_sim = -df_ok["relative_velocity_kms"].to_numpy()
    dist_km = df_ok["distance_km"].to_numpy()
    pl_db = fspl_db(dist_km, tx_freq_hz)

    vc = np.asarray(vel_kms, dtype=np.float64)
    for ti, v_k, pl in zip(row_idx, vel_sim, pl_db):
        gaussian = np.exp(-0.5 * ((vc - v_k) / sigma_kms) ** 2)
        row = pl * (1.0 - 0.9 * gaussian) + background_db * (1.0 - gaussian)
        sim[ti, :] = np.minimum(sim[ti, :], row.astype(np.float32))

    return sim, row_idx, vel_sim


def correlate(waterfall, vel_kms, t1hz, predictions, tx_freq_hz):
    """
    For each predicted satellite:
      1. Build FSPL sim waterfall
      2. Extract ridges from real (argmax) and sim (argmin) in a velocity window
         around the predicted Doppler curve
      3. Compute NCC between ridge shapes (full cross-correlation, best lag)
    """
    T, V = waterfall.shape
    results = []

    for sat_name, pred_df in predictions.items():
        sim, row_idx, vel_sim = build_sim_waterfall(
            pred_df, t1hz, vel_kms, tx_freq_hz
        )
        if sim is None:
            continue

        # Time window where satellite is visible
        t_lo, t_hi = int(row_idx.min()), int(row_idx.max())
        if t_hi - t_lo < 10:
            continue

        # Velocity window from predicted curve (with padding)
        v_pad = 0.5  # km/s
        col_lo = max(0, int(np.searchsorted(vel_kms, vel_sim.min() - v_pad)))
        col_hi = min(V, int(np.searchsorted(vel_kms, vel_sim.max() + v_pad)))
        if col_hi - col_lo < 10:
            continue

        # Ridge extraction in velocity window
        real_ridge = np.argmax(waterfall[t_lo:t_hi + 1, col_lo:col_hi], axis=1)
        sim_ridge = np.argmin(sim[t_lo:t_hi + 1, col_lo:col_hi], axis=1)

        # NCC (full cross-correlation, take best lag)
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
    return results


# =====================================================================
# STAGE 5 — Plot: real waterfall + sim waterfalls with NCC scores
# =====================================================================

def plot_results(waterfall, vel_kms, t1hz, ranked, output_path, n_top=2):
    """Top panel = real waterfall, lower panels = sim waterfalls with NCC labels."""
    T = waterfall.shape[0]
    dur_min = (t1hz[-1] - t1hz[0]).total_seconds() / 60.0
    extent = [vel_kms.min(), vel_kms.max(), dur_min, 0]

    # Real waterfall color scale (percentile-based)
    vmin_r, vmax_r = np.percentile(waterfall, [5, 95])

    n_top = min(n_top, len(ranked))
    n_panels = 1 + n_top
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 5 * n_panels))
    if n_panels == 1:
        axes = [axes]

    # --- Panel 0: Real waterfall ---
    ax = axes[0]
    im = ax.imshow(waterfall, aspect="auto", extent=extent,
                   cmap="jet_r", vmin=vmin_r, vmax=vmax_r,
                   interpolation="bilinear", origin="upper")
    ax.axvline(0, color="white", ls="--", lw=1.5, alpha=0.8, label="Zero velocity")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="Power (dB)")
    ax.set_ylabel("Time into pass (min)")
    ax.set_title("Real")

    # White dashed boxes on real panel for each top match
    for i in range(n_top):
        r = ranked[i]
        v_lo = vel_kms[r["col_lo"]]
        v_hi = vel_kms[min(r["col_hi"], len(vel_kms) - 1)]
        y_top = r["t_lo"] / 60.0
        h = (r["t_hi"] - r["t_lo"]) / 60.0
        ls = "--" if i == 0 else ":"
        ax.add_patch(Rectangle(
            (v_lo, y_top), v_hi - v_lo, h,
            lw=2, edgecolor="white", facecolor="none", ls=ls
        ))

    # --- Sim panels ---
    for i in range(n_top):
        r = ranked[i]
        ax = axes[1 + i]
        sim = r["sim"]

        # Sim color scale: valley (~0.1*FSPL) to background (250)
        vmin_s = max(0, float(sim.min()) - 5)
        vmax_s = float(np.median(sim))

        im = ax.imshow(sim, aspect="auto", extent=extent,
                       cmap="jet_r", vmin=vmin_s, vmax=vmax_s,
                       interpolation="bilinear", origin="upper")
        ax.axvline(0, color="white", ls="--", lw=1.5, alpha=0.8,
                   label="Zero velocity")
        ax.legend(loc="upper right")
        fig.colorbar(im, ax=ax, label="Path Loss (dB)")
        ax.set_ylabel("Time into pass (min)")
        ax.set_title("SIM")

        # NCC score label
        ax.text(0.5, 0.95, f"{r['name']}  NCC score {r['ncc']:.3f}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=14, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="black", alpha=0.5))

        # Dashed box on sim panel
        v_lo = vel_kms[r["col_lo"]]
        v_hi = vel_kms[min(r["col_hi"], len(vel_kms) - 1)]
        y_top = r["t_lo"] / 60.0
        h = (r["t_hi"] - r["t_lo"]) / 60.0
        ax.add_patch(Rectangle(
            (v_lo, y_top), v_hi - v_lo, h,
            lw=2, edgecolor="white", facecolor="none", ls="--"
        ))

    axes[-1].set_xlabel("Relative Velocity (km/s)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[plot] Saved: {output_path}")
    plt.show()


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end satellite Doppler correlation pipeline"
    )
    parser.add_argument("capture_dir",
                        help="Capture directory with .sigmf/.txt compressed data")
    parser.add_argument("--freq", type=float, default=12.325,
                        help="Center frequency in GHz (default: 12.325)")
    parser.add_argument("--tle", type=str, default=None,
                        help="TLE file path (default: download fresh from Celestrak)")
    parser.add_argument("--n-top", type=int, default=2,
                        help="Number of top matches to show (default: 2)")
    parser.add_argument("--elev-mask", type=float, default=10.0)
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon", type=float, default=DEFAULT_LON)
    parser.add_argument("--alt", type=float, default=DEFAULT_ALT)

    args = parser.parse_args()
    capture_dir = os.path.abspath(args.capture_dir)
    tx_freq_hz = args.freq * 1e9

    print("=" * 60)
    print("SATELLITE DOPPLER CORRELATOR")
    print("=" * 60)

    # --- 1. Preprocess (always fresh from .txt) ---
    print(f"\n--- Step 1: Preprocess ({args.freq:.3f} GHz) ---")
    tag = run_preprocessing(capture_dir, args.freq)

    # --- 2. Load waterfall ---
    print("\n--- Step 2: Load waterfall ---")
    waterfall, t1hz, vel_kms = load_real_waterfall(capture_dir, tag)

    # --- 3. Get TLE (download fresh or use provided file) ---
    print("\n--- Step 3: TLE ---")
    if args.tle:
        with open(args.tle) as f:
            tle_data = f.read()
        print(f"[tle] Using: {args.tle}")
    else:
        tle_data = download_tle()

    # --- 4. Predict Doppler ---
    print("\n--- Step 4: Predict Doppler ---")
    start_utc = t1hz[0].to_pydatetime()
    duration_sec = (t1hz[-1] - t1hz[0]).total_seconds() + 1
    predictions = predict_doppler(
        tle_data, start_utc, duration_sec,
        tx_freq_hz=tx_freq_hz,
        elevation_mask=args.elev_mask,
        lat=args.lat, lon=args.lon, alt=args.alt,
    )
    if not predictions:
        print("[ERROR] No visible satellites. Check TLE freshness vs capture date.")
        sys.exit(1)

    # --- 5. Correlate ---
    print("\n--- Step 5: Correlate ---")
    ranked = correlate(waterfall, vel_kms, t1hz, predictions, tx_freq_hz)

    print(f"\n{'=' * 55}")
    print("SATELLITE RANKING")
    print(f"{'=' * 55}")
    for i, r in enumerate(ranked[:20], 1):
        print(f"  {i:3d}. {r['name']:15s}  NCC={r['ncc']:.4f}")
    if ranked:
        print(f"\nBest match: {ranked[0]['name']}")

    # --- 6. Plot ---
    print("\n--- Step 6: Plot ---")
    out_path = os.path.join(capture_dir, "correlation_result.png")
    plot_results(waterfall, vel_kms, t1hz, ranked, out_path, n_top=args.n_top)


if __name__ == "__main__":
    main()
