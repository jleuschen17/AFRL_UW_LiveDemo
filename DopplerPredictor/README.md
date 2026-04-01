# Starlink Doppler Predictor GUI

A PyQt5-based GUI application for real-time Starlink satellite tracking and Doppler shift visualization.

## Dependencies

### Required Python Packages

```bash
pip install PyQt5 numpy matplotlib skyfield requests cartopy
```


## Features

### 1. Sky Map Visualization
- Real-time polar plot showing satellite positions
- Azimuth (N, NE, E, SE, S, SW, W, NW) and elevation (0°-90°) display
- Zenith marker at center (ground station location)
- Configurable elevation mask filter
- Efficient two-tier update system:
  - Fast updates for visible satellites
  - Slower visibility checks for hidden satellites

### 2. Live Doppler Waterfall
- Real-time frequency vs. time waterfall display
- Shows Free Space Path Loss (FSPL) in dB
- Configurable duration and update interval
- Gaussian signal envelope modeling

### 3. Ground Station Inset Map
- Shows UE (User Equipment) location on a map
- Displays visibility circle (~2500 km radius)
- Uses Cartopy for proper map projection (falls back to simple plot if unavailable)

## Operation

### Starting the Application

```bash
python3 doppler_predictor_gui.py
```

### Control Panel Options

| Section | Field | Description |
|---------|-------|-------------|
| **TLE Data** | Load TLE File | Load TLE data from a local file |
| | Download Latest TLE | Download current Starlink TLEs from CelesTrak |
| **Ground Station** | Latitude (°N) | Ground station latitude (default: 47.6550) |
| | Longitude (°E) | Ground station longitude (default: -122.3035) |
| | Altitude (m) | Ground station altitude (default: 60) |
| **Settings** | Elevation Mask (°) | Minimum elevation angle for visibility (default: 10.0) |
| | Max Satellites | Maximum number of satellites to load (default: 1000) |
| | Sky Map Update (ms) | Update interval for sky map animation (default: 100) |
| | Waterfall Duration (min) | Time span shown in waterfall (default: 5) |
| | Waterfall Update (sec) | Update interval for waterfall (default: 5) |

### Visualization Controls

- **▶ Start Sky Map** - Begin real-time satellite position tracking
- **■ Stop** - Stop sky map animation
- **▶ Start Live Waterfall** - Open waterfall display window
- **■ Stop Waterfall** - Stop waterfall updates
- **Save Current View** - Export current figure as PNG or PDF
- **🔄 Update Settings** - Apply changed settings without restarting

## Technical Details

### Satellite Position Calculation

1. **TLE Propagation**: Uses SGP4 orbital propagator (via Skyfield library) to compute satellite geocentric position from Two-Line Element data
2. **Observer-Relative Position**: Transforms satellite position to topocentric coordinates relative to the ground station
3. **Elevation Angle**: Computed as `elevation = arctan2(z, √(x² + y²))` where (x, y, z) is the relative position vector
4. **Azimuth Angle**: Computed as `azimuth = arctan2(x, y)`, measured clockwise from North (0° = N, 90° = E, 180° = S, 270° = W)
5. **Sky Map Coordinates**: Polar plot uses `r = 90 - elevation` so zenith (90° elevation) is at center and horizon (0°) is at edge

### Doppler Shift Calculation

1. **Range Rate**: Computed by differencing slant distances at two time instants (1 second apart)
   ```
   range_rate = (distance_later - distance_now) / Δt
   ```
2. **Doppler Shift**: Applied using the classical Doppler formula
   ```
   Δf = -f_tx × (v_radial / c)
   ```
   - Positive range rate (satellite receding) → negative frequency shift
   - Negative range rate (satellite approaching) → positive frequency shift

### Waterfall Display

1. **Frequency Grid**: Creates frequency bins spanning TX frequency ± 500 kHz
2. **Signal Modeling**: Each visible satellite contributes a Gaussian-shaped signal centered at its Doppler-shifted frequency
   ```
   signal(f) = exp(-(f - f_received)² / (2σ²))
   ```
   where σ = 5 kHz (signal bandwidth)
3. **Path Loss Calculation**: Free Space Path Loss (FSPL) computed for each satellite
   ```
   FSPL(dB) = 20·log₁₀(d) + 20·log₁₀(f) - 147.55
   ```
4. **Waterfall Scrolling**: New spectrum row added at top, older data scrolls down
5. **Color Mapping**: Lower path loss (stronger signal) shown in warmer colors

### Coordinate System (Sky Map)
- **Center**: Zenith (90° elevation, directly overhead)
- **Edge**: Horizon (0° elevation)
- **Radial distance**: `r = 90 - elevation`
- **Azimuth**: 0° = North, 90° = East, 180° = South, 270° = West

### Starlink Parameters
- **TX Frequency**: ~10.5 GHz (X-band downlink)
- **Orbital Altitude**: ~550 km
- **Typical FSPL**: 170-185 dB

### Fallback Mode

If PyQt5 is unavailable, the application falls back to a terminal-based interface with basic menu options.

## TLE Data Sources

- **CelesTrak**: https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle
- Downloaded TLEs are saved to `starlink_downloaded.txt`

## Architecture

### High-Level Structure

```
┌─────────────────────────────────────────────────────────────┐
│                         main()                              │
│   Entry point - tries PyQt5, falls back to terminal UI     │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│     try_pyqt5()         │     │    try_terminal_ui()        │
│   (Primary GUI)         │     │   (Fallback CLI)            │
└─────────────────────────┘     └─────────────────────────────┘
```

### Core Classes

#### 1. `DopplerPredictor`
Single satellite Doppler calculation engine.

| Attribute | Description |
|-----------|-------------|
| `STARLINK_TX_FREQ` | 10.5 GHz reference frequency |
| `SPEED_OF_LIGHT` | 299,792,458 m/s |
| Uses **Skyfield** library | For SGP4 TLE propagation |

**Key Method:**
- `calculate_doppler_shift(obs_time)` → Returns Doppler shift in Hz based on range-rate between satellite and ground station

#### 2. `MultiSatellitePredictor`
Manages multiple `DopplerPredictor` instances.

- Parses bulk TLE data (3-line format: name, line1, line2)
- Creates a list of `DopplerPredictor` objects
- Limits to `num_satellites` for performance

#### 3. `DopplerPredictorGUI`
Main PyQt5 `QMainWindow` application.

```
┌─────────────────────────────────────────────────────────────────┐
│                    DopplerPredictorGUI                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────────────────────────┐ │
│  │  Control Panel   │  │       Visualization Panel            │ │
│  │  (Left side)     │  │       (Right side)                   │ │
│  │                  │  │                                      │ │
│  │  • TLE loading   │  │  ┌────────────────────────────────┐  │ │
│  │  • Location input│  │  │   Polar Plot (Sky Map)         │  │ │
│  │  • Settings      │  │  │   - Satellite positions        │  │ │
│  │  • Action buttons│  │  │   - Azimuth/Elevation grid     │  │ │
│  │  • Statistics    │  │  └────────────────────────────────┘  │ │
│  │                  │  │  ┌────────────────────────────────┐  │ │
│  │                  │  │  │   Inset Map (Location)         │  │ │
│  │                  │  │  │   - Uses Cartopy if available  │  │ │
│  │                  │  │  └────────────────────────────────┘  │ │
│  └──────────────────┘  └──────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                        Status Bar                               │
└─────────────────────────────────────────────────────────────────┘
```

### Timer-Based Update System

The app uses **three QTimers** for efficient real-time updates:

| Timer | Interval | Purpose |
|-------|----------|---------|
| `self.timer` | ~100ms (configurable) | Update **visible** satellite positions |
| `self.visibility_timer` | 10 seconds | Check if **hidden** satellites became visible |
| `self.waterfall_timer` | ~5 seconds (configurable) | Update Doppler waterfall spectrogram |

**Optimization Strategy:**
- **Visible satellites** → Updated frequently (fast timer)
- **Hidden satellites** → Checked less often (slow timer)
- Satellites swap between `visible_predictors` and `hidden_predictors` lists

### Data Flow

```
TLE File/Download
       │
       ▼
┌──────────────────┐
│ MultiSatellitePredictor │
│  (parses TLEs)   │
└────────┬─────────┘
         │ creates
         ▼
┌──────────────────┐
│ DopplerPredictor │ ×N satellites
│  (per satellite) │
└────────┬─────────┘
         │ calculates
         ▼
┌──────────────────────────────────┐
│  Satellite Position & Doppler   │
│  - Azimuth/Elevation            │
│  - Doppler shift (Hz)           │
│  - Slant distance               │
└────────┬─────────────────────────┘
         │ displays
         ▼
┌──────────────────────────────────┐
│  Matplotlib Visualizations      │
│  - Polar sky map                │
│  - Waterfall spectrogram        │
└──────────────────────────────────┘
```

### Key Features Summary

| Feature | Implementation |
|---------|----------------|
| **Sky Map** | Polar plot showing satellite positions (az/el) |
| **Live Waterfall** | Spectrogram of received frequency vs time |
| **FSPL Calculation** | Free Space Path Loss in dB |
| **Location Inset** | Mini-map with Cartopy (optional) |
| **TLE Download** | From CelesTrak API |
| **Threaded Downloads** | Non-blocking TLE fetch |
