import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import glob
from datetime import datetime, timedelta, timezone
from skyfield.api import EarthSatellite, load, wgs84

# Constants
C = 299792458.0         # m/s

def read_tle_from_file(sat_id, tle_file='starlink.txt'):
    """Read TLE for a given satellite ID from TLE file."""
    with open(tle_file, 'r') as f:
        lines = f.readlines()
    
    # Search for the satellite
    for i in range(len(lines) - 2):
        line1 = lines[i + 1].strip()
        line2 = lines[i + 2].strip()
        
        # Check if this is the satellite we're looking for
        if line1.startswith('1 ') and line2.startswith('2 '):
            # Extract catalog number from TLE line 1 (positions 2-7)
            catalog_num = line1[2:7].strip()
            if catalog_num == str(sat_id):
                return line1, line2
    
    raise ValueError(f"Satellite {sat_id} not found in {tle_file}")

def find_csv_file(sat_id):
    """Find CSV file for the given satellite in any satellites_* folder."""
    # Look for satellites_* directories
    sat_dirs = glob.glob('satellites_*')
    if not sat_dirs:
        raise ValueError("No satellites_* directory found")
    
    # Use the most recent directory (sorted by name, which includes timestamp)
    sat_dir = sorted(sat_dirs)[-1]
    
    # Look for the CSV file
    csv_path = os.path.join(sat_dir, f'SAT-{sat_id}.csv')
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV file not found: {csv_path}")
    
    return csv_path

def main(sat_id=None):
    if sat_id is None:
        # Check command line arguments
        if len(sys.argv) > 1:
            sat_id = sys.argv[1]
        else:
            sat_id = '45047'  # Default
    
    # Remove 'SAT-' prefix if provided
    sat_id = str(sat_id).replace('SAT-', '').replace('sat-', '')
    
    print(f"Processing satellite: SAT-{sat_id}")
    
    # Read TLE from starlink.txt
    try:
        TLE_LINE1, TLE_LINE2 = read_tle_from_file(sat_id)
        print(f"Found TLE in starlink.txt")
    except Exception as e:
        print(f"Error reading TLE: {e}")
        return
    
    # Find CSV file
    try:
        csv_path = find_csv_file(sat_id)
        print(f"Found CSV: {csv_path}")
    except Exception as e:
        print(f"Error finding CSV: {e}")
        return
    
    print("Using Skyfield SGP4 propagator (same as doppler_predictor_gui.py)")
    
    # Create satellite object using Skyfield
    satellite = EarthSatellite(TLE_LINE1, TLE_LINE2)
    ts = load.timescale()
    
    print(f"Satellite: SAT-{sat_id}")
    print(f"TLE Epoch: {satellite.epoch.utc_datetime()}")
    
    # Load CSV to get the timestamps
    try:
        df_ref = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not load {csv_path}")
        return

    # Parse timestamps
    df_ref['dt'] = pd.to_datetime(df_ref['timestamp'])
    
    # Ground station
    ue_lat = 47.655
    ue_lon = -122.3035
    ue_alt = 60.0
    tx_freq = 10.5e9  # Hz
    
    print(f"Ground Station: {ue_lat}°N, {ue_lon}°E, {ue_alt}m")
    print(f"CSV time range: {df_ref['dt'].iloc[0]} to {df_ref['dt'].iloc[-1]}")
    
    # Extend time range by 10 minutes before and after
    start_time = df_ref['dt'].iloc[0] - timedelta(minutes=10)
    end_time = df_ref['dt'].iloc[-1] + timedelta(minutes=10)
    
    print(f"Simulating extended range: {start_time} to {end_time}")
    
    duration_sec = (end_time - start_time).total_seconds()
    time_step_sec = 1  # 1 second steps
    num_samples = int(duration_sec / time_step_sec)
    
    # Generate time array
    times = [start_time + timedelta(seconds=i * time_step_sec) for i in range(num_samples)]
    
    print(f"\nPropagating orbit using SGP4 for {num_samples} time points...")
    
    # Create observer location
    observer = wgs84.latlon(ue_lat, ue_lon, elevation_m=ue_alt)
    
    # Arrays to store results
    azimuth_deg = np.zeros(num_samples)
    elevation_deg = np.zeros(num_samples)
    distance_km = np.zeros(num_samples)
    doppler_hz = np.zeros(num_samples)
    
    # Calculate for each time point
    for i, t in enumerate(times):
        # Convert to Skyfield time
        from skyfield.api import utc
        t_utc = t.replace(tzinfo=utc) if t.tzinfo is None else t
        ts_time = ts.from_datetime(t_utc)
        
        # Get satellite position relative to observer
        difference = satellite - observer
        topocentric = difference.at(ts_time)
        
        # Use Skyfield's correct altaz() method
        alt, az, distance = topocentric.altaz()
        
        azimuth_deg[i] = az.degrees
        elevation_deg[i] = alt.degrees
        distance_km[i] = distance.km
        
        # Calculate Doppler shift (using range rate)
        # Get position at slightly later time for velocity
        dt_seconds = 1.0
        ts_time_plus = ts.from_datetime(
            datetime.utcfromtimestamp(t.timestamp() + dt_seconds).replace(tzinfo=utc)
        )
        topocentric_later = difference.at(ts_time_plus)
        
        # Range rate in km/s
        range_rate_km_s = (topocentric_later.distance().km - distance.km) / dt_seconds
        
        # Doppler shift: f' = f * (c - v_r) / c
        # For recession (positive range_rate): negative shift
        doppler_hz[i] = -tx_freq * range_rate_km_s * 1000.0 / C
    
    # Calculate relative velocity
    Rdot = -doppler_hz * C / (tx_freq * 1000.0)  # km/s
    rx_freq_hz = tx_freq + doppler_hz
    
    # Print statistics
    print(f"\nElevation statistics:")
    print(f"  Min elevation: {elevation_deg.min():.2f}°")
    print(f"  Max elevation: {elevation_deg.max():.2f}°")
    print(f"  Mean elevation: {elevation_deg.mean():.2f}°")
    visible_count = np.sum(elevation_deg > 0)
    above_threshold = np.sum(elevation_deg > 10)
    print(f"  Points above horizon (0°): {visible_count} / {num_samples}")
    print(f"  Points above threshold (10°): {above_threshold} / {num_samples}")
    
    # Create DataFrame with computed data
    df = pd.DataFrame({
        'timestamp': times,
        'dt': times,
        'satellite': f'SAT-{sat_id}',
        'azimuth_deg': azimuth_deg,
        'elevation_deg': elevation_deg,
        'distance_km': distance_km,
        'relative_velocity_kms': Rdot,
        'doppler_shift_hz': doppler_hz,
        'tx_freq_ghz': tx_freq / 1e9,
        'rx_freq_hz': rx_freq_hz,
        'ue_lat': ue_lat,
        'ue_lon': ue_lon,
        'ue_alt_m': ue_alt,
        'time_minutes': [(t - times[0]).total_seconds() / 60.0 for t in times]
    })
    
    print(f"Generated {len(df)} data points")
    print(f"Time range: {df['dt'].iloc[0]} to {df['dt'].iloc[-1]}")
    print(f"Distance range: {df['distance_km'].min():.1f} to {df['distance_km'].max():.1f} km")
    
    # Filter to only visible portion (elevation >= 0)
    df_visible = df[df['elevation_deg'] >= 0].copy()
    print(f"\nVisible pass: {len(df_visible)} points above horizon")
    
    if len(df_visible) == 0:
        print("ERROR: No visible points in the pass!")
        return
    
    # Get azimuth and elevation for polar plot (visible only)
    azimuth_deg = df_visible['azimuth_deg'].values
    elevation_deg = df_visible['elevation_deg'].values
    
    # Convert Doppler shift to relative velocity
    velocity_ms = -df_visible['doppler_shift_hz'].values * C / tx_freq
    velocity_kms = velocity_ms / 1000  # km/s
    
    # Time in minutes from start of visible pass
    time_minutes = np.array([(t - df_visible['dt'].iloc[0]).total_seconds() / 60.0 for t in df_visible['dt']])
    
    # Plotting
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(14, 10), dpi=100)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.2, 1], hspace=0.3)
    
    # 1. Doppler Waterfall Plot
    ax_waterfall = fig.add_subplot(gs[0])
    
    # Create 2D waterfall data with velocity on x-axis
    velocity_range_kms = 8.0  # ±8 km/s
    velocity_start_kms = -velocity_range_kms
    velocity_end_kms = velocity_range_kms
    
    n_velocity_bins = 300
    n_time_samples = len(time_minutes)
    
    velocity_grid_kms = np.linspace(velocity_start_kms, velocity_end_kms, n_velocity_bins)
    
    # Create waterfall matrix (time x velocity) - initialize with noise floor
    noise_floor_db = 200.0
    waterfall_data = np.full((n_time_samples, n_velocity_bins), noise_floor_db)
    
    # Fill in signal data using Gaussian spreading
    sigma_kms = 0.05  # 50 m/s = 0.05 km/s
    for i, (vel_kms, dist_km) in enumerate(zip(velocity_kms, df_visible['distance_km'].values)):
        # Calculate FSPL (Free Space Path Loss)
        distance_m = dist_km * 1000
        fspl_db = 20 * np.log10(distance_m) + 20 * np.log10(tx_freq) + 20 * np.log10(4 * np.pi / C)
        
        # Create Gaussian profile centered at current velocity
        gaussian = np.exp(-0.5 * ((velocity_grid_kms - vel_kms) / sigma_kms) ** 2)
        waterfall_data[i, :] = noise_floor_db - gaussian * (noise_floor_db - fspl_db)
    
    # Use actual data duration
    actual_duration_min = time_minutes[-1] - time_minutes[0] if len(time_minutes) > 0 else 0
    extent = [velocity_start_kms, velocity_end_kms, actual_duration_min, 0]
    
    # Set colorbar range based on actual signal values (exclude noise floor)
    signal_data = waterfall_data[waterfall_data < noise_floor_db - 1]
    if len(signal_data) > 0:
        vmin, vmax = np.min(signal_data), np.max(signal_data)
    else:
        vmin, vmax = 100, 200
    
    im = ax_waterfall.imshow(waterfall_data, aspect='auto', extent=extent,
                            cmap='jet_r', vmin=vmin, vmax=vmax, interpolation='bilinear')
    ax_waterfall.axvline(x=0, color='white', linestyle='--', linewidth=1.5, alpha=0.8, label='Zero velocity')
    ax_waterfall.set_xlabel('Relative Velocity (km/s) [- approaching, + receding]', fontsize=11)
    ax_waterfall.set_ylabel('Time into pass (min)', fontsize=11)
    ax_waterfall.set_title(f'Velocity Waterfall - SAT-{sat_id}\nStart: {df_visible["dt"].iloc[0]}', fontsize=12)
    ax_waterfall.legend(loc='upper right', fontsize=9)
    fig.colorbar(im, ax=ax_waterfall, label='Path Loss (dB)')
    
    # 2. Trajectory Polar Plot (Sky Map)
    ax_traj = fig.add_subplot(gs[1], projection='polar')
    ax_traj.set_theta_zero_location('N')
    ax_traj.set_theta_direction(-1)
    ax_traj.set_ylim(0, 90)
    ax_traj.set_yticks([0, 15, 30, 45, 60, 75, 90])
    ax_traj.set_yticklabels(['90°', '75°', '60°', '45°', '30°', '15°', '0°'], fontsize=8)
    ax_traj.set_rlabel_position(22.5)
    
    # Set azimuth ticks
    az_deg_ticks = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    az_rad_ticks = np.radians(az_deg_ticks)
    ax_traj.set_xticks(az_rad_ticks)
    ax_traj.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], 
                            fontsize=10, fontweight='bold')
    ax_traj.grid(True, alpha=0.4, linestyle='--')
    
    # Draw elevation mask circle (assume 10 degrees)
    elevation_mask = 10.0
    mask_r = 90 - elevation_mask
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax_traj.plot(theta_circle, [mask_r]*100, 'r--', linewidth=1, alpha=0.5, 
                label=f'Elev mask ({elevation_mask}°)')
    
    # Plot zenith (ground station location)
    ax_traj.plot(0, 0, 'r+', markersize=12, markeredgewidth=2, zorder=10, label='Zenith (UE)')
    
    # Convert trajectory to polar coordinates
    thetas = np.radians(azimuth_deg)
    rs = 90 - elevation_deg  # r=0 is zenith (90° elev), r=90 is horizon (0° elev)
    
    # Plot trajectory with color gradient (time progression)
    colors = np.linspace(0, 1, len(thetas))
    scatter = ax_traj.scatter(thetas, rs, c=colors, cmap='cool', s=30, alpha=0.7, zorder=3)
    ax_traj.plot(thetas, rs, 'c-', linewidth=1.5, alpha=0.5, zorder=2)
    
    # Mark start and end points
    if len(thetas) > 0:
        ax_traj.scatter([thetas[0]], [rs[0]], c='green', s=150, marker='^', 
                       edgecolors='white', linewidths=2, zorder=4, label='Start')
        ax_traj.scatter([thetas[-1]], [rs[-1]], c='red', s=150, marker='v', 
                       edgecolors='white', linewidths=2, zorder=4, label='End')
    
    ax_traj.set_title(f'Sky Trajectory - SAT-{sat_id}', fontsize=12, pad=10)
    ax_traj.legend(loc='upper left', bbox_to_anchor=(-0.25, 1.15), fontsize=8)
    
    plt.tight_layout()
    output_file = f"sat{sat_id}_simulation.png"
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"\nSimulation complete. Saved plot to {output_file}")
    print(f"  - Waterfall plot: Velocity vs Time")
    print(f"  - Polar plot: Sky trajectory (Azimuth/Elevation)")

if __name__ == "__main__":
    main()
