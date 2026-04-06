import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# ------------------------------------------------------------
# Load real waterfall data
# ------------------------------------------------------------
# real_spec = np.load("./real data/waterfall_CF_r3_11.575GHz.npy")
# time_axis = np.load("./real data/datetime_waterfall_CF_r3_11.575GHz.npy")
# vel_axis = np.load("./real data/rel_vel_waterfall_CF_r3_11.575GHz.npy") / 1000.0


real_spec = np.load("/home/mowerj/AFRL_UW_LiveDemo/testCaptures/starlink_sigmf_20260402T201229/r0_waterfall_CF_11.575GHz.npy")
time_axis = np.load("/home/mowerj/AFRL_UW_LiveDemo/testCaptures/starlink_sigmf_20260402T201229/datetime_updated_r0_waterfall_CF_11.575GHz.npy")
vel_axis = np.load("/home/mowerj/AFRL_UW_LiveDemo/testCaptures/starlink_sigmf_20260402T201229/rel_vel_r0_waterfall_CF_11.575GHz.npy") / 1000.0

# Speed of light (m/s)
C = 299792458.0


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
def create_waterfallplot(W, vel_centers_kms, t1hz, title):
    """
    Plot a waterfall heatmap.

    Parameters
    ----------
    W : ndarray
        Waterfall matrix with shape (time, velocity).
    vel_centers_kms : ndarray
        Velocity bin centers in km/s.
    t1hz : DatetimeIndex
        Time axis sampled at 1 Hz.
    title : str
        Figure title.
    """
    valid = W[W < 200]
    if valid.size > 0:
        vmin, vmax = np.percentile(valid, [5, 95])
    else:
        vmin, vmax = 170, 185

    duration_min = (t1hz[-1] - t1hz[0]).total_seconds() / 60.0
    extent = [vel_centers_kms.min(), vel_centers_kms.max(), duration_min, 0]

    plt.figure(figsize=(12, 5))
    im = plt.imshow(
        W,
        aspect="auto",
        extent=extent,
        cmap="jet_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        origin="upper"
    )
    plt.axvline(
        0,
        color="white",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="Zero velocity"
    )
    plt.xlabel("Relative Velocity (km/s)")
    plt.ylabel("Time into pass (min)")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.colorbar(im, label="Path Loss (dB)")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Path loss model
# ------------------------------------------------------------
def fspl_db_from_dist_and_freq(dist_km: np.ndarray, tx_freq_hz: float) -> np.ndarray:
    """
    Compute free-space path loss (FSPL) in dB.

    Parameters
    ----------
    dist_km : ndarray
        Distance in kilometers.
    tx_freq_hz : float
        Transmit frequency in Hz.
    """
    dist_m = np.asarray(dist_km, dtype=np.float64) * 1000.0
    return (
        20.0 * np.log10(dist_m + 1e-12)
        + 20.0 * np.log10(tx_freq_hz + 1e-12)
        + 20.0 * np.log10(4.0 * np.pi / C)
    )


# ------------------------------------------------------------
# Build simulated waterfall aligned with real waterfall axes
# ------------------------------------------------------------
def build_sim_waterfall_on_real_axes(
    csv_path: str,
    t1hz_utc: pd.DatetimeIndex,
    vel_centers_kms: np.ndarray,
    sigma_kms: float = 0.05,
    background_db: float = 250.0,
    ts_col: str = "timestamp",
    vel_col: str = "relative_velocity_kms",
    dist_col: str = "distance_km",
    freq_col: str = "tx_freq_ghz",
):
    """
    Align the simulated waterfall with the real waterfall time axis and velocity axis.
    """
    df = pd.read_csv(csv_path)

    # Convert simulated timestamps to UTC and round to second
    t_sim = pd.to_datetime(df[ts_col], utc=True, errors="raise").dt.floor("S")
    df["t_sec"] = t_sim

    if getattr(t1hz_utc, "tz", None) is None:
        raise ValueError("t1hz_utc must be tz-aware UTC DatetimeIndex.")

    # Match simulated timestamps to the 1 Hz real timestamps
    row_idx = t1hz_utc.get_indexer(df["t_sec"])
    ok = row_idx >= 0
    print(f"[sim->real] matched {ok.sum()} / {len(ok)} samples")

    if ok.sum() == 0:
        print("real UTC range:", t1hz_utc.min(), "->", t1hz_utc.max())
        print("sim  UTC range:", df["t_sec"].min(), "->", df["t_sec"].max())
        raise RuntimeError("No timestamp overlap after timezone alignment.")

    row_idx = row_idx[ok]
    df = df.loc[ok].copy()

    T = len(t1hz_utc)
    V = len(vel_centers_kms)
    sim = np.full((T, V), background_db, dtype=np.float32)

    tx_freq_hz = float(df[freq_col].iloc[0]) * 1e9
    pl_db = fspl_db_from_dist_and_freq(df[dist_col].to_numpy(), tx_freq_hz)

    vel_kms = df[vel_col].to_numpy(dtype=np.float64)
    vel_centers_kms = np.asarray(vel_centers_kms, dtype=np.float64)

    # Build a Gaussian-shaped valley around the simulated velocity
    for ti, v_k, pl in zip(row_idx, vel_kms, pl_db):
        gaussian = np.exp(-0.5 * ((vel_centers_kms - v_k) / sigma_kms) ** 2)
        row = pl * (1.0 - 0.9 * gaussian) + background_db * (1.0 - gaussian)
        sim[ti, :] = np.minimum(sim[ti, :], row.astype(np.float32))

    return sim


# ------------------------------------------------------------
# Velocity-axis utilities
# ------------------------------------------------------------
def _vel_centers_from_axis(vel_axis, V_expected=None):
    """
    Convert a velocity axis into velocity bin centers.
    """
    vel_axis = np.asarray(vel_axis, dtype=np.float64).ravel()

    if V_expected is not None:
        if len(vel_axis) == V_expected + 1:
            # Treat as bin edges
            return 0.5 * (vel_axis[:-1] + vel_axis[1:])
        if len(vel_axis) == V_expected:
            # Treat as bin centers
            return vel_axis
        raise ValueError(f"vel_axis length {len(vel_axis)} not compatible with V={V_expected}")

    return vel_axis


def _resample_velocity_interp(H, vel_centers_in, n_out):
    """
    Resample the heatmap along the velocity axis using 1D interpolation.

    Parameters
    ----------
    H : ndarray
        Input heatmap with shape (T, V_in).
    vel_centers_in : ndarray
        Input velocity centers.
    n_out : int
        Number of output velocity bins.
    """
    vel_centers_in = np.asarray(vel_centers_in, dtype=np.float64)
    vmin, vmax = float(vel_centers_in.min()), float(vel_centers_in.max())
    vel_centers_out = np.linspace(vmin, vmax, n_out)

    H = np.asarray(H, dtype=np.float64)
    T, V_in = H.shape
    H_out = np.empty((T, n_out), dtype=np.float32)

    # Interpolate each row independently along velocity
    for i in range(T):
        H_out[i, :] = np.interp(
            vel_centers_out, vel_centers_in, H[i, :]
        ).astype(np.float32)

    return H_out, vel_centers_out


# ------------------------------------------------------------
# Downsample real waterfall to 1 Hz and resample velocity bins
# ------------------------------------------------------------
def downsample_real_to_1hz_and_vel300(
    real_spec,
    time_axis,
    vel_axis,
    method="mean",
    n_vel_out=300
):
    """
    1) Downsample time axis to 1 Hz by grouping samples within each second.
    2) Resample velocity bins to n_vel_out using interpolation.

    Returns
    -------
    real_1hz_vel300 : ndarray
        Real waterfall after 1 Hz time downsampling and velocity resampling.
    t1hz : DatetimeIndex
        Time axis at 1 Hz.
    vel_centers_out : ndarray
        Output velocity centers.
    """
    real_spec = np.asarray(real_spec)
    if real_spec.ndim != 2:
        raise ValueError("real_spec must be 2D (N, V).")

    t = pd.to_datetime(np.asarray(time_axis), utc=True, errors="raise")
    if len(t) != real_spec.shape[0]:
        raise ValueError(f"time_axis length {len(t)} != real_spec rows {real_spec.shape[0]}")

    # Group by second
    df = pd.DataFrame(real_spec)
    df["t_sec"] = t.floor("S")
    group = df.groupby("t_sec")

    if method == "mean":
        agg = group.mean(numeric_only=True)
    elif method == "median":
        agg = group.median(numeric_only=True)
    else:
        raise ValueError("method must be 'mean' or 'median'")

    real_1hz = agg.to_numpy(dtype=np.float32)
    t1hz = agg.index

    V_in = real_1hz.shape[1]
    vel_centers_in = _vel_centers_from_axis(vel_axis, V_expected=V_in)

    real_1hz_vel300, vel_centers_out = _resample_velocity_interp(
        real_1hz, vel_centers_in, n_vel_out
    )

    return real_1hz_vel300, t1hz, vel_centers_out


# ------------------------------------------------------------
# Ridge extraction and NCC
# ------------------------------------------------------------
def ridge_from_heatmap(H, mode="max"):
    """
    Extract the ridge index from a selected velocity window.
    """
    H = np.asarray(H)

    if mode == "max":
        return np.argmax(H[:, 180:290], axis=1)
    elif mode == "min":
        return np.argmin(H[:, 180:290], axis=1)
    else:
        raise ValueError("mode must be 'max' or 'min'")


def ncc_1d(x, y, demean=False, eps=1e-12):
    """
    Compute normalized cross-correlation (NCC) between two 1D arrays.
    """
    x = np.asarray(x, dtype=float).astype(float, copy=True)
    y = np.asarray(y, dtype=float).astype(float, copy=True)

    if demean:
        x = x - x.mean()
        y = y - y.mean()

    xcorr = np.correlate(x, y, mode="full")
    denom = np.linalg.norm(x) * np.linalg.norm(y) + eps
    ncc = xcorr / denom

    lags = np.arange(-len(y) + 1, len(x))

    best_idx = np.argmax(ncc)
    best_ncc = float(ncc[best_idx])
    best_lag = int(lags[best_idx])

    return ncc, lags, best_ncc, best_lag


# ------------------------------------------------------------
# Preprocess real data
# ------------------------------------------------------------
real_1hz, t1hz, vel_axis_out = downsample_real_to_1hz_and_vel300(
    real_spec, time_axis, vel_axis, method="mean"
)

vel_axis_out = np.asarray(vel_axis_out, dtype=np.float64)

if len(vel_axis_out) == real_1hz.shape[1] + 1:
    vel_centers_kms = 0.5 * (vel_axis_out[:-1] + vel_axis_out[1:])
elif len(vel_axis_out) == real_1hz.shape[1]:
    vel_centers_kms = vel_axis_out
else:
    raise ValueError(f"Unexpected vel_axis length {len(vel_axis_out)} for V={real_1hz.shape[1]}")


# ------------------------------------------------------------
# Compare all simulated files
# ------------------------------------------------------------
scores = []
files = []

folders = os.listdir("./sim data/")

for ff in folders:
    if ff.find("20260211_205621") == -1:
        continue

    fnames = os.listdir("./sim data/" + ff)

    for f in fnames:
        if f.find("csv") == -1 or f.find("waterfall") != -1:
            continue

        try:
            sim_1hz = build_sim_waterfall_on_real_axes(
                csv_path="./sim data/" + ff + "/" + f,
                t1hz_utc=t1hz,
                vel_centers_kms=vel_centers_kms,
                sigma_kms=0.05,
                background_db=250.0
            )
        except Exception:
            continue

        # IMPORTANT:
        # This flip is intentionally kept inside the loop to preserve
        # the exact behavior of the original code.
        real_1hz = np.flip(real_1hz, axis=1)

        real_r = ridge_from_heatmap(real_1hz, mode="max")
        sim_r = ridge_from_heatmap(sim_1hz, mode="min")

        starts = [125, 190]
        ends = [160, 215]
        start_ = starts[0]
        end_ = ends[0]

        x = real_r[start_:end_]
        y = sim_r[start_:end_]

        ncc_vals, lags, best_ncc, best_lag = ncc_1d(x, y)

        if np.isnan(best_ncc):
            continue

        files.append(f)
        scores.append(best_ncc)

        if f == "SAT-49745.csv":
            create_waterfallplot(sim_1hz, vel_centers_kms, t1hz, "SIM")

            plt.figure()
            plt.plot(real_r[start_:end_])
            plt.plot(sim_r[start_:end_])

            plt.figure()
            plt.pcolormesh(real_1hz)
            plt.colorbar()

            create_waterfallplot(real_1hz, vel_centers_kms, t1hz, "Real")
            plt.show()

    scores = np.array(scores)
    
    print("All sorted NCC scores: ", np.sort(scores))
    print("Best matched satellite: ", files[np.argmax(scores)])