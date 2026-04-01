# Starlink Doppler Predictor - User Tutorial

A step-by-step guide to tracking Starlink satellites and visualizing Doppler shifts in real-time.

---

## Quick Start (30 seconds)

1. **Launch the application**
   ```bash
   python3 doppler_predictor_gui.py
   ```
   Or run the standalone executable:
   ```bash
   ./dist/doppler_predictor_gui
   ```

2. **Click "▶ Start Sky Map"** - satellites will appear on the polar plot

3. **Click "▶ Start Live Waterfall"** - opens the Doppler frequency display

That's it! The app auto-loads TLE data if `starlink.txt` exists in the same folder.

---

## Step-by-Step Tutorial

### Step 1: Load Satellite Data

When you first open the app, you need TLE (Two-Line Element) data to track satellites.

**Option A: Auto-load (easiest)**
- If `starlink.txt` exists in the app folder, it loads automatically
- Check the status bar at the bottom for "Auto-loaded X TLEs"

**Option B: Download fresh data**
1. Click **"Download Latest TLE"** in the TLE Data section
2. Wait a few seconds for download from CelesTrak
3. Status bar shows "Downloaded X TLEs" when complete
4. Data is saved to `starlink_downloaded.txt` for future use

**Option C: Load from file**
1. Click **"Load TLE File"**
2. Browse to your `.txt` or `.tle` file
3. Select and open

✅ **Success indicator**: The "TLE in file" and "Loaded" counters update with satellite counts.

---

### Step 2: Set Your Location

The default location is UW, Seattle, WA (47.655°N, -122.304°E). To change it:

1. Enter your **Latitude** in degrees North (negative for South)
2. Enter your **Longitude** in degrees East (negative for West)
3. Enter your **Altitude** in meters above sea level
4. Click **"🔄 Update Settings"** to apply changes

**Example locations:**
| City | Latitude | Longitude | Altitude |
|------|----------|-----------|----------|
| Seattle, WA | 47.655 | -122.304 | 60 |
| New York, NY | 40.713 | -74.006 | 10 |
| London, UK | 51.507 | -0.128 | 20 |
| Tokyo, Japan | 35.682 | 139.759 | 40 |
| Sydney, Australia | -33.869 | 151.209 | 50 |

The inset map (bottom-left) updates to show your location with a cyan visibility circle.

---

### Step 3: Start the Sky Map

1. Click **"▶ Start Sky Map"**
2. The polar plot shows satellites as white dots with red edges

**Reading the Sky Map:**
- **Center (red cross)** = Zenith (directly overhead, 90° elevation)
- **Edge** = Horizon (0° elevation)
- **N/E/S/W** = Cardinal directions (North at top)
- **Concentric rings** = Elevation angles (labeled 0°-90°)

**What you'll see:**
- Satellites move slowly across the sky
- More satellites appear near the horizon than at zenith
- The title shows current time and visible satellite count

**Controls:**
- **"■ Stop"** - Pause the animation
- Adjust **"Sky Map Update (ms)"** for faster/slower refresh (lower = faster)
- Adjust **"Elevation Mask (°)"** to hide satellites below a certain angle

---

### Step 4: View the Doppler Waterfall

1. Click **"▶ Start Live Waterfall"**
2. A new window opens showing the frequency spectrum over time

**Reading the Waterfall:**
- **X-axis** = Frequency in GHz (centered on 10.5 GHz TX frequency)
- **Y-axis** = Time (0 = now, increasing downward = past)
- **Colors** = Signal strength (path loss in dB)
  - Warmer colors (red/orange) = Stronger signals (lower path loss)
  - Cooler colors (blue/purple) = Weaker signals (higher path loss)
- **White dashed line** = Nominal TX frequency (no Doppler)

**What to look for:**
- **Horizontal streaks** = Individual satellites with constant Doppler
- **Curved traces** = Satellites passing overhead (Doppler changes as geometry changes)
- **Frequency spread** = Satellites at different angles have different Doppler shifts
  - Approaching satellites → Higher frequency (right of center)
  - Receding satellites → Lower frequency (left of center)

**Waterfall Settings:**
- **"Waterfall Duration (min)"** - How much history to show (default: 5 min)
- **"Waterfall Update (sec)"** - How often to add new data (default: 5 sec)

---

### Step 5: Save Your Results

Click **"Save Current View"** to export the current sky map as:
- PNG image (for presentations)
- PDF file (for publications)

---

## Tips & Tricks

### Performance Optimization
- Reduce **"Max Satellites"** from 1000 to 200-500 for faster updates
- Increase **"Sky Map Update (ms)"** from 100 to 500+ if animation is choppy
- The app uses a two-tier update system:
  - Visible satellites update frequently
  - Hidden satellites are checked every 10 seconds

### Best Viewing Times
- Starlink satellites are in ~550 km polar/inclined orbits
- You'll typically see 20-80 satellites at any time depending on location
- More satellites visible from higher latitudes

### Understanding Doppler Shift
- Maximum Doppler shift: ~±300 kHz at 10.5 GHz
- Satellites directly overhead have near-zero Doppler (perpendicular velocity)
- Satellites near horizon have maximum Doppler (tangential velocity)

### Troubleshooting

| Problem | Solution |
|---------|----------|
| No satellites visible | Check elevation mask isn't too high (try 5°), verify TLE data loaded |
| App runs slowly | Reduce Max Satellites, increase update interval |
| Waterfall is empty | Make sure satellites are loaded, check elevation mask |
| Map inset shows "Install cartopy" | Install cartopy: `pip install cartopy` |
| Download fails | Check internet connection, try loading from local file |

---

## Understanding the Science

### Why Doppler Shift Matters
When a satellite moves relative to your ground station:
- **Approaching** → Radio waves are compressed → **Higher frequency**
- **Receding** → Radio waves are stretched → **Lower frequency**

The Doppler shift formula:
```
Δf = f × v_radial / c
```
Where:
- Δf = frequency shift (Hz)
- f = transmitted frequency (10.5 GHz)
- v_radial = velocity toward/away from you (m/s)
- c = speed of light (3×10⁸ m/s)

### What is Path Loss?
Free Space Path Loss (FSPL) is how much signal weakens over distance:
```
FSPL = 20·log₁₀(distance) + 20·log₁₀(frequency) - 147.55 dB
```

For Starlink at 550 km altitude, expect:
- **Overhead (90° elevation)**: ~170 dB loss (shortest path)
- **Near horizon (10° elevation)**: ~185 dB loss (longest path)

---

## Keyboard Shortcuts

The app uses standard matplotlib toolbar shortcuts when focused on the plot:
- **H** - Home (reset view)
- **P** - Pan mode
- **O** - Zoom mode
- **S** - Save figure

---

## Example Use Cases

1. **Amateur Radio Operators**: Track satellite passes for communication windows
2. **RF Engineers**: Understand Doppler compensation requirements
3. **Students**: Visualize orbital mechanics and Doppler physics
4. **Researchers**: Analyze LEO satellite constellation dynamics

---

Enjoy exploring the Starlink constellation! 🛰️
