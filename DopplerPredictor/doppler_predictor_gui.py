"""
Doppler Predictor GUI for Satellite Communications
GUI application for real-time Starlink satellite tracking and Doppler visualization.
Uses PyQt5 for better macOS compatibility.

This is a standalone version with integrated predictor classes.
"""

import sys
import os
import numpy as np
from datetime import datetime, timezone
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Doppler Predictor Classes (integrated from doppler_predictor2.py)
# ============================================================================

class DopplerPredictor:
    """
    Predicts Doppler shift for satellites using TLE data.
    X-band reference frequency: 10.5 GHz (Starlink downlink)
    """
    
    # Constants
    EARTH_RADIUS_KM = 6371.0  # km
    STARLINK_TX_FREQ = 10.5e9  # Hz (10.5 GHz - Starlink transmit frequency)
    SPEED_OF_LIGHT = 299792458  # m/s
    RX_BAND_START = 10e9  # Hz (10 GHz - UE receive band start)
    RX_BAND_END = 11e9  # Hz (11 GHz - UE receive band end)
    
    def __init__(self, tle_line1: str, tle_line2: str, ue_location: dict, tx_freq_hz: float = None):
        """
        Initialize the Doppler predictor.
        
        Args:
            tle_line1: First line of TLE data
            tle_line2: Second line of TLE data
            ue_location: Dict with keys 'latitude' (deg), 'longitude' (deg), 'altitude' (m)
            tx_freq_hz: Transmit frequency in Hz (default: 10.5 GHz)
        """
        self.tle_line1 = tle_line1
        self.tle_line2 = tle_line2
        
        # Set TX frequency (use provided value or default)
        if tx_freq_hz is not None:
            self.STARLINK_TX_FREQ = tx_freq_hz
        
        # Parse TLE to extract satellite catalog number (columns 3-7 of line 1)
        # TLE line 1 format: "1 NNNNNC ..."
        try:
            # Extract catalog number from positions 2-7 (0-indexed)
            catalog_num = tle_line1[2:7].strip()
            self.sat_name = f"SAT-{catalog_num}"
        except:
            self.sat_name = tle_line1.strip()
        
        # Store UE location
        self.ue_lat = ue_location.get('latitude', 0)
        self.ue_lon = ue_location.get('longitude', 0)
        self.ue_alt = ue_location.get('altitude', 0) / 1000  # Convert to km
        
        # Import SGP4 for TLE propagation
        try:
            from skyfield.api import EarthSatellite, load, wgs84
            self.skyfield = True
            self.wgs84 = wgs84
            
            # Create satellite object
            self.satellite = EarthSatellite(self.tle_line1, self.tle_line2)
            self.ts = load.timescale()
            
        except ImportError:
            self.skyfield = False
            print("Warning: Skyfield not installed. Install with: pip install skyfield")
    
    def calculate_doppler_shift(self, obs_time: datetime = None) -> float:
        """
        Calculate Doppler shift in Hz for given observation time.
        
        Args:
            obs_time: Observation time (datetime). If None, uses current time.
            
        Returns:
            Doppler shift in Hz
        """
        if not self.skyfield:
            return 0.0
        
        if obs_time is None:
            obs_time = datetime.utcnow()
        
        try:
            from skyfield.api import utc
            
            # Convert to Skyfield time
            ts_time = self.ts.from_datetime(obs_time.replace(tzinfo=utc) if obs_time.tzinfo is None else obs_time)
            
            # Create observer location using wgs84.latlon
            observer_location = self.wgs84.latlon(
                self.ue_lat,
                self.ue_lon,
                elevation_m=self.ue_alt * 1000
            )
            
            # Get satellite position relative to observer at two times
            relative_now = (self.satellite - observer_location).at(ts_time)
            
            # Calculate at slightly later time for velocity
            dt_seconds = 1.0
            ts_time_plus = self.ts.from_datetime(
                datetime.utcfromtimestamp(obs_time.timestamp() + dt_seconds).replace(tzinfo=utc)
            )
            relative_later = (self.satellite - observer_location).at(ts_time_plus)
            
            # Get distance at both times
            dist_now_au = (relative_now.position.au[0]**2 + 
                          relative_now.position.au[1]**2 + 
                          relative_now.position.au[2]**2)**0.5
            
            dist_later_au = (relative_later.position.au[0]**2 + 
                            relative_later.position.au[1]**2 + 
                            relative_later.position.au[2]**2)**0.5
            
            # Range rate in AU/second
            range_rate_au_per_s = (dist_later_au - dist_now_au) / dt_seconds
            
            # Convert to m/s (1 AU = 150e9 m)
            range_rate_m_per_s = range_rate_au_per_s * 150e9
            
            # Calculate Doppler shift: f' = f * (c - v_r) / c
            # For recession (positive range_rate): frequency decreases (negative shift)
            doppler_shift = -self.STARLINK_TX_FREQ * range_rate_m_per_s / self.SPEED_OF_LIGHT
            
            return doppler_shift
            
        except Exception as e:
            return 0.0


class MultiSatellitePredictor:
    """
    Predicts combined Doppler spectrum from multiple Starlink satellites.
    """
    
    def __init__(self, tle_data: str, ue_location: dict, num_satellites: int = 20, tx_freq_hz: float = None):
        """
        Initialize with multiple satellite TLEs.
        
        Args:
            tle_data: TLE data string from CelesTrak (multiple 3-line TLE sets)
            ue_location: Dict with keys 'latitude', 'longitude', 'altitude'
            num_satellites: Number of satellites to use from TLE data
            tx_freq_hz: Transmit frequency in Hz (default: 10.5 GHz)
        """
        self.ue_location = ue_location
        self.tx_freq_hz = tx_freq_hz
        self.predictors = []
        
        # Parse TLE data into groups of 3 lines
        tle_lines = tle_data.strip().split('\n')
        
        # Process TLE data (groups of 3 lines: name, line1, line2)
        sat_count = 0
        i = 0
        while i < len(tle_lines) - 2 and sat_count < num_satellites:
            tle_name = tle_lines[i].strip()
            tle_line1 = tle_lines[i + 1].strip()
            tle_line2 = tle_lines[i + 2].strip()
            
            # Validate TLE format
            if tle_line1.startswith('1 ') and tle_line2.startswith('2 '):
                try:
                    predictor = DopplerPredictor(tle_line1, tle_line2, ue_location, tx_freq_hz=self.tx_freq_hz)
                    if predictor.skyfield:
                        self.predictors.append(predictor)
                        sat_count += 1
                except:
                    pass
            
            i += 3
        
        print(f"Loaded {len(self.predictors)} Starlink satellites")


# ============================================================================
# GUI Code
# ============================================================================

# Try different GUI backends
def try_pyqt5():
    """Try PyQt5-based GUI."""
    import matplotlib
    matplotlib.use('Qt5Agg')
    
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                  QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                                  QGroupBox, QFormLayout, QFileDialog, QMessageBox,
                                  QStatusBar, QSplitter, QFrame)
    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtGui import QFont
    
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    import threading
    
    class DopplerPredictorGUI(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Starlink Doppler Predictor")
            self.setGeometry(100, 100, 1200, 800)
            
            # Initialize variables
            self.multi_predictor = None
            self.timer = None
            self.visibility_timer = None  # Slower timer for checking visibility of out-of-view satellites
            self.waterfall_timer = None   # Timer for waterfall animation
            self.single_sat_waterfall_timer = None  # Timer for single satellite waterfall
            self.tle_data = None
            self.is_running = False
            self.waterfall_running = False
            self.single_sat_waterfall_running = False
            self.datagen_running = False  # Flag for data generation
            self.current_mode = 'realtime'  # 'realtime' or 'datagen'
            
            # Track which satellites are currently visible (for efficient updates)
            self.visible_predictors = []  # Satellites currently in view (updated frequently)
            self.hidden_predictors = []   # Satellites not in view (checked less often)
            self.visible_sat_data = []    # Store satellite data for click detection
            
            # Selected satellite for individual waterfall
            self.selected_predictor = None
            self.selected_sat_name = None
            
            # Waterfall data storage
            self.waterfall_data = None
            self.waterfall_times = None
            
            # Single satellite waterfall data
            self.single_sat_waterfall_data = None
            
            # Velocity arrows storage
            self.velocity_arrows = []
            
            # Flag to prevent duplicate loading during initialization
            self.is_initializing = True
            
            self.setup_ui()
            
            # Initialization complete
            self.is_initializing = False
            
            # Now safe to create matplotlib canvas and auto-load TLE
            self.initialize_matplotlib_canvas()
            self.auto_load_tle()
            
        def setup_ui(self):
            """Set up the GUI layout."""
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            main_layout = QHBoxLayout(central_widget)
            
            # Create tab widget for mode selection
            from PyQt5.QtWidgets import QTabWidget
            self.mode_tabs = QTabWidget()
            self.mode_tabs.currentChanged.connect(self.on_mode_changed)
            
            # Real-time Mode Tab
            realtime_widget = QWidget()
            realtime_layout = QHBoxLayout(realtime_widget)
            
            # Left panel for controls
            control_panel = self.create_control_panel()
            realtime_layout.addWidget(control_panel)
            
            # Right panel for visualization
            viz_panel = self.create_visualization_panel()
            realtime_layout.addWidget(viz_panel, stretch=1)
            
            self.mode_tabs.addTab(realtime_widget, "📡 Real-time Mode")
            
            # Data Generation Mode Tab
            datagen_widget = self.create_datagen_panel()
            self.mode_tabs.addTab(datagen_widget, "🔬 Data Generation Mode")
            
            main_layout.addWidget(self.mode_tabs)
            
            # Status bar
            self.status_bar = QStatusBar()
            self.setStatusBar(self.status_bar)
            self.status_bar.showMessage("Ready. Load TLE data to begin.")
            
        def create_control_panel(self):
            """Create the control panel."""
            panel = QFrame()
            panel.setFrameStyle(QFrame.StyledPanel)
            panel.setMaximumWidth(300)
            layout = QVBoxLayout(panel)
            
            # TLE Data Section
            tle_group = QGroupBox("TLE Data")
            tle_layout = QVBoxLayout(tle_group)
            
            self.load_btn = QPushButton("Load TLE File")
            self.load_btn.clicked.connect(self.load_tle_file)
            tle_layout.addWidget(self.load_btn)
            
            self.download_btn = QPushButton("Download Latest TLE")
            self.download_btn.clicked.connect(self.download_tle)
            tle_layout.addWidget(self.download_btn)
            
            layout.addWidget(tle_group)
            
            # Location Section
            loc_group = QGroupBox("Ground Station")
            loc_layout = QFormLayout(loc_group)
            
            self.lat_input = QLineEdit("47.6550")
            self.lon_input = QLineEdit("-122.3035")
            self.alt_input = QLineEdit("60")
            
            loc_layout.addRow("Latitude (°N):", self.lat_input)
            loc_layout.addRow("Longitude (°E):", self.lon_input)
            loc_layout.addRow("Altitude (m):", self.alt_input)
            
            layout.addWidget(loc_group)
            
            # Settings Section
            settings_group = QGroupBox("Settings")
            settings_layout = QFormLayout(settings_group)
            
            self.carrier_freq_input = QLineEdit("10.5")
            self.carrier_freq_input.setToolTip("Carrier/TX frequency in GHz (e.g., 10.5 for Starlink Ku-band, 12.0 for Ku-band DL)")
            self.elev_input = QLineEdit("10.0")
            self.num_sats_input = QLineEdit("1000")
            self.interval_input = QLineEdit("100")
            self.duration_input = QLineEdit("5")
            self.wf_update_input = QLineEdit("5")
            
            settings_layout.addRow("Carrier Freq (GHz):", self.carrier_freq_input)
            settings_layout.addRow("Elevation Mask (°):", self.elev_input)
            settings_layout.addRow("Max Satellites:", self.num_sats_input)
            settings_layout.addRow("Sky Map Update (ms):", self.interval_input)
            settings_layout.addRow("Waterfall Duration (min):", self.duration_input)
            settings_layout.addRow("Waterfall Update (sec):", self.wf_update_input)
            
            # Update button for settings
            self.update_settings_btn = QPushButton("🔄 Update Settings")
            self.update_settings_btn.clicked.connect(self.apply_settings_update)
            self.update_settings_btn.setToolTip("Apply changes to location and settings")
            settings_layout.addRow(self.update_settings_btn)
            
            layout.addWidget(settings_group)
            
            # Action Buttons
            action_group = QGroupBox("Visualization")
            action_layout = QVBoxLayout(action_group)
            
            self.start_btn = QPushButton("▶ Start Sky Map")
            self.start_btn.clicked.connect(self.start_sky_map)
            action_layout.addWidget(self.start_btn)
            
            self.stop_btn = QPushButton("■ Stop")
            self.stop_btn.clicked.connect(self.stop_animation)
            self.stop_btn.setEnabled(False)
            action_layout.addWidget(self.stop_btn)
            
            self.waterfall_btn = QPushButton("▶ Start Live Waterfall")
            self.waterfall_btn.clicked.connect(self.start_live_waterfall)
            action_layout.addWidget(self.waterfall_btn)
            
            self.stop_waterfall_btn = QPushButton("■ Stop Waterfall")
            self.stop_waterfall_btn.clicked.connect(self.stop_waterfall)
            self.stop_waterfall_btn.setEnabled(False)
            action_layout.addWidget(self.stop_waterfall_btn)
            
            self.save_btn = QPushButton("Save Current View")
            self.save_btn.clicked.connect(self.save_figure)
            action_layout.addWidget(self.save_btn)
            
            layout.addWidget(action_group)
            
            # Statistics
            stats_group = QGroupBox("Statistics")
            stats_layout = QVBoxLayout(stats_group)
            
            self.tle_count_label = QLabel("TLE in file: 0")
            self.sats_label = QLabel("Loaded: 0")
            self.visible_label = QLabel("Visible: 0")
            self.tle_update_label = QLabel("TLE Updated: N/A")
            self.tle_update_label.setToolTip("Date and time when TLE data was last loaded or downloaded")
            
            stats_layout.addWidget(self.tle_count_label)
            stats_layout.addWidget(self.sats_label)
            stats_layout.addWidget(self.visible_label)
            stats_layout.addWidget(self.tle_update_label)
            
            layout.addWidget(stats_group)
            
            layout.addStretch()
            
            return panel
        
        def create_datagen_panel(self):
            """Create the data generation mode panel."""
            from PyQt5.QtWidgets import QDateTimeEdit, QTextEdit, QComboBox
            from PyQt5.QtCore import QDateTime
            
            main_widget = QWidget()
            main_layout = QHBoxLayout(main_widget)
            
            # Left panel - Data Generation Controls
            control_panel = QFrame()
            control_panel.setFrameStyle(QFrame.StyledPanel)
            control_panel.setMaximumWidth(350)
            layout = QVBoxLayout(control_panel)
            
            # Title
            title = QLabel("Data Generation Mode")
            title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
            layout.addWidget(title)
            
            # TLE Data Section
            tle_group = QGroupBox("TLE Data")
            tle_layout = QVBoxLayout(tle_group)
            
            self.datagen_load_btn = QPushButton("Load TLE File")
            self.datagen_load_btn.clicked.connect(self.load_tle_file)
            tle_layout.addWidget(self.datagen_load_btn)
            
            self.datagen_download_btn = QPushButton("Download Latest TLE")
            self.datagen_download_btn.clicked.connect(self.download_tle)
            tle_layout.addWidget(self.datagen_download_btn)
            
            layout.addWidget(tle_group)
            
            # Location Section
            loc_group = QGroupBox("Ground Station")
            loc_layout = QFormLayout(loc_group)
            
            self.datagen_lat_input = QLineEdit("47.6550")
            self.datagen_lon_input = QLineEdit("-122.3035")
            self.datagen_alt_input = QLineEdit("60")
            
            loc_layout.addRow("Latitude (°N):", self.datagen_lat_input)
            loc_layout.addRow("Longitude (°E):", self.datagen_lon_input)
            loc_layout.addRow("Altitude (m):", self.datagen_alt_input)
            
            layout.addWidget(loc_group)
            
            # Simulation Settings
            sim_group = QGroupBox("Simulation Settings")
            sim_layout = QFormLayout(sim_group)
            
            # Start time picker
            self.sim_start_time = QDateTimeEdit()
            self.sim_start_time.setCalendarPopup(True)
            self.sim_start_time.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
            self.sim_start_time.setDateTime(QDateTime.currentDateTime())
            self.sim_start_time.setToolTip("Select simulation start time (UTC)")
            sim_layout.addRow("Start Time (UTC):", self.sim_start_time)
            
            # Duration input
            self.sim_duration_input = QLineEdit("1")
            self.sim_duration_input.setToolTip("Simulation duration in minutes")
            sim_layout.addRow("Duration (min):", self.sim_duration_input)
            
            # Time step input
            self.sim_timestep_input = QLineEdit("10")
            self.sim_timestep_input.setToolTip("Time step between samples in seconds")
            sim_layout.addRow("Time Step (sec):", self.sim_timestep_input)
            
            # Additional settings
            self.datagen_elev_input = QLineEdit("10.0")
            sim_layout.addRow("Elevation Mask (°):", self.datagen_elev_input)
            
            self.datagen_num_sats_input = QLineEdit("100")
            sim_layout.addRow("Max Satellites:", self.datagen_num_sats_input)
            
            layout.addWidget(sim_group)
            
            # Satellite Selection
            sat_group = QGroupBox("Satellite Selection")
            sat_layout = QVBoxLayout(sat_group)
            
            self.satellite_combo = QComboBox()
            self.satellite_combo.addItem("-- Generate data first --")
            self.satellite_combo.currentIndexChanged.connect(self.on_satellite_selected)
            sat_layout.addWidget(self.satellite_combo)
            
            layout.addWidget(sat_group)
            
            # Action Buttons
            action_group = QGroupBox("Actions")
            action_layout = QVBoxLayout(action_group)
            
            self.generate_btn = QPushButton("▶ Generate Data")
            self.generate_btn.clicked.connect(self.generate_simulation_data)
            self.generate_btn.setStyleSheet("font-weight: bold; padding: 8px;")
            action_layout.addWidget(self.generate_btn)
            
            self.stop_datagen_btn = QPushButton("■ Stop Generation")
            self.stop_datagen_btn.clicked.connect(self.stop_data_generation)
            self.stop_datagen_btn.setEnabled(False)
            action_layout.addWidget(self.stop_datagen_btn)
            
            self.export_csv_btn = QPushButton("💾 Export Data & Image")
            self.export_csv_btn.clicked.connect(self.export_data_to_csv)
            self.export_csv_btn.setEnabled(False)
            self.export_csv_btn.setToolTip("Export CSV data, settings, and plot images")
            action_layout.addWidget(self.export_csv_btn)
            
            layout.addWidget(action_group)
            
            # Progress/Status
            progress_group = QGroupBox("Progress")
            progress_layout = QVBoxLayout(progress_group)
            
            from PyQt5.QtWidgets import QProgressBar
            self.datagen_progress = QProgressBar()
            self.datagen_progress.setValue(0)
            progress_layout.addWidget(self.datagen_progress)
            
            self.datagen_status_label = QLabel("Ready to generate data")
            self.datagen_status_label.setWordWrap(True)
            progress_layout.addWidget(self.datagen_status_label)
            
            layout.addWidget(progress_group)
            
            layout.addStretch()
            
            main_layout.addWidget(control_panel)
            
            # Right panel - Visualization (waterfall + trajectory)
            viz_panel = QFrame()
            viz_panel.setFrameStyle(QFrame.StyledPanel)
            viz_layout = QVBoxLayout(viz_panel)
            
            viz_title = QLabel("Satellite Visualization")
            viz_title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
            viz_layout.addWidget(viz_title)
            
            # Create matplotlib figure for data generation mode
            self.datagen_fig = Figure(figsize=(12, 10), dpi=100)
            
            # Create subplots: waterfall on top, trajectory on bottom
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(2, 1, figure=self.datagen_fig, height_ratios=[1.2, 1], hspace=0.5)
            
            self.datagen_waterfall_ax = self.datagen_fig.add_subplot(gs[0])
            self.datagen_trajectory_ax = self.datagen_fig.add_subplot(gs[1], projection='polar')
            
            # Create canvas first
            self.datagen_canvas = FigureCanvas(self.datagen_fig)
            viz_layout.addWidget(self.datagen_canvas)
            
            datagen_toolbar = NavigationToolbar(self.datagen_canvas, viz_panel)
            viz_layout.addWidget(datagen_toolbar)
            
            # Initialize plots after canvas is created
            self.setup_datagen_plots()
            
            main_layout.addWidget(viz_panel, stretch=1)
            
            return main_widget
        
        def setup_datagen_plots(self):
            """Initialize the data generation mode plots."""
            # Waterfall plot
            self.datagen_waterfall_ax.clear()
            self.datagen_waterfall_ax.set_xlabel('Frequency (GHz)', fontsize=11)
            self.datagen_waterfall_ax.set_ylabel('Time into pass (min)', fontsize=11)
            self.datagen_waterfall_ax.set_title('Doppler Waterfall - Select satellite to view', fontsize=12)
            self.datagen_waterfall_ax.text(0.5, 0.5, 'Generate data and select a satellite', 
                                          ha='center', va='center', transform=self.datagen_waterfall_ax.transAxes,
                                          fontsize=14, color='gray')
            
            # Trajectory plot
            self.datagen_trajectory_ax.clear()
            self.datagen_trajectory_ax.set_theta_zero_location('N')
            self.datagen_trajectory_ax.set_theta_direction(-1)
            self.datagen_trajectory_ax.set_ylim(0, 90)
            self.datagen_trajectory_ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
            self.datagen_trajectory_ax.set_yticklabels(['90°', '75°', '60°', '45°', '30°', '15°', '0°'], fontsize=8)
            self.datagen_trajectory_ax.set_rlabel_position(22.5)
            
            az_deg = np.array([0, 45, 90, 135, 180, 225, 270, 315])
            az_rad = np.radians(az_deg)
            self.datagen_trajectory_ax.set_xticks(az_rad)
            self.datagen_trajectory_ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], 
                                                       fontsize=10, fontweight='bold')
            self.datagen_trajectory_ax.grid(True, alpha=0.4, linestyle='--')
            self.datagen_trajectory_ax.set_title('Sky Trajectory - Select satellite to view', fontsize=12, pad=10)
            
            self.datagen_fig.tight_layout()
            self.datagen_canvas.draw()
            
        def create_visualization_panel(self):
            """Create the visualization panel."""
            panel = QFrame()
            panel.setFrameStyle(QFrame.StyledPanel)
            self.viz_layout = QVBoxLayout(panel)
            
            # Placeholder label until figure is created
            self.viz_placeholder = QLabel("Initializing visualization...")
            self.viz_placeholder.setAlignment(Qt.AlignCenter)
            self.viz_placeholder.setStyleSheet("font-size: 14px; color: gray;")
            self.viz_layout.addWidget(self.viz_placeholder)
            
            # Will create figure after initialization
            self.canvas = None
            self.toolbar = None
            self.fig = None
            self.ax = None
            self.inset_ax = None
            
            return panel
        
        def initialize_matplotlib_canvas(self):
            """Create matplotlib canvas after window initialization."""
            if self.canvas is not None:
                return  # Already created
            
            # Create matplotlib figure with space for inset map
            self.fig = Figure(figsize=(8, 7), dpi=100)
            # Position polar plot to the right to make room for inset map
            # [left, bottom, width, height] in figure coordinates (0-1)
            self.ax = self.fig.add_axes([0.35, 0.1, 0.6, 0.8], projection='polar')
            self.setup_polar_plot()
            
            # Canvas
            self.canvas = FigureCanvas(self.fig)
            self.viz_layout.addWidget(self.canvas)
            
            # Connect click event for satellite selection
            self.canvas.mpl_connect('pick_event', self.on_satellite_click)
            
            # Toolbar
            panel = self.canvas.parent()
            self.toolbar = NavigationToolbar(self.canvas, panel)
            self.viz_layout.addWidget(self.toolbar)
            
            # Remove placeholder
            if hasattr(self, 'viz_placeholder') and self.viz_placeholder:
                self.viz_placeholder.deleteLater()
                self.viz_placeholder = None
            
            # Create inset map
            self.create_location_inset()
        
        def create_location_inset(self):
            """Create a small inset map showing the ground station location."""
            # Only create after initialization
            if self.is_initializing:
                return
            # Create inset axes in the left side (larger map now that polar plot is shifted right)
            self.inset_ax = self.fig.add_axes([0.07, 0.05, 0.30, 0.35])  # [left, bottom, width, height]
            self.update_location_inset()
        
        def update_location_inset(self):
            """Update the inset map with current UE location."""
            if not hasattr(self, 'inset_ax') or self.inset_ax is None:
                return
                
            self.inset_ax.clear()
            
            try:
                lat = float(self.lat_input.text())
                lon = float(self.lon_input.text())
            except:
                lat, lon = 47.655, -122.3035  # Default
            
            import numpy as np
            
            # Try to use cartopy for proper map, fallback to simple version
            try:
                import cartopy.crs as ccrs
                import cartopy.feature as cfeature
                
                # Remove old inset and create new one with cartopy projection
                self.inset_ax.remove()
                self.inset_ax = self.fig.add_axes([0.05, 0.05, 0.25, 0.35], 
                                                   projection=ccrs.PlateCarree())
                
                # Set regional extent centered on UE (±10° lat/lon for closer zoom)
                lon_range = 3
                lat_range = 3
                self.inset_ax.set_extent([lon - lon_range, lon + lon_range, 
                                          max(-90, lat - lat_range), min(90, lat + lat_range)],
                                         crs=ccrs.PlateCarree())
                
                self.inset_ax.add_feature(cfeature.OCEAN, color='#1a1a2e')
                self.inset_ax.add_feature(cfeature.LAND, color='#2d4a3e')
                self.inset_ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='white')
                self.inset_ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray', alpha=0.5)
                self.inset_ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor='gray', alpha=0.3)
                
                # Plot UE location (red cross to match sky map)
                self.inset_ax.plot(lon, lat, 'r+', markersize=12, markeredgewidth=2,
                                  transform=ccrs.PlateCarree(), zorder=10)
                
                # Visibility circle (~2500 km radius ≈ 22°)
                circle_angles = np.linspace(0, 2*np.pi, 100)
                vis_radius = 22  # degrees, approximate visibility range
                circle_lats = lat + vis_radius * np.cos(circle_angles)
                circle_lons = lon + vis_radius / np.cos(np.radians(lat)) * np.sin(circle_angles)
                self.inset_ax.plot(circle_lons, circle_lats, 'c-', linewidth=1.5, alpha=0.7,
                                  transform=ccrs.PlateCarree(), label='Visibility')
                self.inset_ax.fill(circle_lons, circle_lats, color='cyan', alpha=0.1,
                                  transform=ccrs.PlateCarree())
                
                # Add gridlines
                gl = self.inset_ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, 
                                             color='gray', linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': 6}
                gl.ylabel_style = {'size': 6}
                
                self.inset_ax.set_title(f'UE: {lat:.2f}°, {lon:.2f}°', fontsize=8, pad=2)
                
            except ImportError:
                # Fallback: simple plot without cartopy
                lon_range = 10
                lat_range = 8
                self.inset_ax.set_xlim(lon - lon_range, lon + lon_range)
                self.inset_ax.set_ylim(max(-90, lat - lat_range), min(90, lat + lat_range))
                self.inset_ax.set_facecolor('#1a1a2e')
                
                # Plot UE location (red cross to match sky map)
                self.inset_ax.plot(lon, lat, 'r+', markersize=12, markeredgewidth=2, zorder=10)
                
                # Visibility circle
                circle_angles = np.linspace(0, 2*np.pi, 100)
                vis_radius = 22
                circle_lats = lat + vis_radius * np.cos(circle_angles)
                circle_lons = lon + vis_radius / np.cos(np.radians(lat)) * np.sin(circle_angles)
                self.inset_ax.plot(circle_lons, circle_lats, 'c-', linewidth=1.5, alpha=0.7)
                self.inset_ax.fill(circle_lons, circle_lats, color='cyan', alpha=0.1)
                
                self.inset_ax.tick_params(labelsize=6)
                self.inset_ax.set_title(f'UE: {lat:.2f}°, {lon:.2f}°', fontsize=8, pad=2)
                self.inset_ax.grid(True, alpha=0.3, linewidth=0.3)
                self.inset_ax.text(lon, lat - lat_range + 3, 'Install cartopy for map', fontsize=5, 
                                  ha='center', color='yellow', alpha=0.7)
            
        def setup_polar_plot(self):
            """Initialize the polar plot."""
            if self.ax is None:
                return
            
            self.ax.clear()
            self.ax.set_theta_zero_location('N')
            self.ax.set_theta_direction(-1)
            
            # Plot zenith marker at center (directly overhead the ground station) - red cross
            self.ax.plot(0, 0, 'r+', markersize=15, markeredgewidth=3, label='Zenith (UE location)', zorder=10)
            self.ax.scatter([0], [0], c='red', s=100, marker='o', alpha=0.3, zorder=9)
            
            # Initialize empty scatter plots (picker=True enables click detection)
            self.satellite_glow = self.ax.scatter([], [], c='yellow', s=150, marker='o', alpha=0.3, zorder=4)
            self.satellite_scatter = self.ax.scatter([], [], c='white', s=50, marker='o', zorder=5,
                                                      edgecolors='red', linewidths=0.5, label='Starlink Satellites',
                                                      picker=True, pickradius=10)
            
            # Selected satellite marker (initially hidden)
            self.selected_marker = self.ax.scatter([], [], c='lime', s=200, marker='*', zorder=6,
                                                   edgecolors='white', linewidths=1.5, label='Selected')
            
            # Set up axes - center is 90° elevation (zenith), edge is 0° elevation (horizon)
            # r = 90 - elevation, so r=0 means 90° elev (center), r=90 means 0° elev (edge)
            self.ax.set_ylim(0, 90)
            self.ax.set_rscale('linear')  # Must be set BEFORE yticklabels!
            self.ax.grid(True, alpha=0.4, linestyle='--')
            
            # Now set ticks and labels (after rscale and grid)
            self.ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
            self.ax.set_yticklabels(['90°', '80°', '70°', '60°', '50°', '40°', '30°', '20°', '10°', '0°'], fontsize=9)
            self.ax.set_rlabel_position(22.5)  # Position the radial labels
            
            # Set azimuth ticks: cardinal (N, E, S, W) and intercardinal (NE, SE, SW, NW)
            az_deg = np.array([0, 45, 90, 135, 180, 225, 270, 315])  # 0° = North
            az_rad = np.radians(az_deg)
            self.ax.set_xticks(az_rad)
            self.ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=12, fontweight='bold')
            
            self.ax.set_title('Sky Map - Load TLE data to begin', fontsize=12, pad=10)
            self.ax.legend(loc='upper right', bbox_to_anchor=(-0.1, 1.05), fontsize=8)


            
        def auto_load_tle(self):
            """Auto-load starlink.txt if it exists."""
            default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'starlink.txt')
            if os.path.exists(default_path):
                try:
                    with open(default_path, 'r') as f:
                        self.tle_data = f.read()
                    tle_lines = self.tle_data.strip().split('\n')
                    num_sats = len([line for line in tle_lines if line.startswith('1 ')])
                    
                    # Get file modification time
                    import time
                    mod_time = os.path.getmtime(default_path)
                    update_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
                    
                    self.status_bar.showMessage(f"Auto-loaded {num_sats} TLEs from starlink.txt")
                    self.tle_count_label.setText(f"TLE in file: {num_sats}")
                    self.tle_update_label.setText(f"TLE Updated: {update_time}")
                    self.initialize_predictor()
                except:
                    pass
                    
        def load_tle_file(self):
            """Load TLE data from file."""
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Select TLE File", 
                os.path.dirname(os.path.abspath(__file__)),
                "Text files (*.txt);;TLE files (*.tle);;All files (*.*)"
            )
            
            if filepath:
                try:
                    with open(filepath, 'r') as f:
                        self.tle_data = f.read()
                    
                    tle_lines = self.tle_data.strip().split('\n')
                    num_sats = len([line for line in tle_lines if line.startswith('1 ')])
                    
                    # Get file modification time
                    import time
                    mod_time = os.path.getmtime(filepath)
                    update_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
                    
                    self.status_bar.showMessage(f"Loaded {num_sats} TLEs from {os.path.basename(filepath)}")
                    self.tle_count_label.setText(f"TLE in file: {num_sats}")
                    self.tle_update_label.setText(f"TLE Updated: {update_time}")
                    self.initialize_predictor()
                    
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load TLE file: {e}")
                    
        def download_tle(self):
            """Download latest TLE data."""
            self.status_bar.showMessage("Downloading TLE data...")
            self.tle_update_label.setText("TLE Updated: Downloading...")
            
            def download_thread():
                try:
                    import requests
                    
                    QTimer.singleShot(0, lambda: self.status_bar.showMessage("Connecting to CelesTrak..."))
                    
                    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    self.tle_data = response.text
                    
                    tle_lines = self.tle_data.strip().split('\n')
                    num_sats = len([line for line in tle_lines if line.startswith('1 ')])
                    
                    if num_sats == 0:
                        QTimer.singleShot(0, lambda: QMessageBox.warning(self, "Warning", "No valid TLE data found in download"))
                        QTimer.singleShot(0, lambda: self.status_bar.showMessage("Download failed - no valid TLE data"))
                        QTimer.singleShot(0, lambda: self.tle_update_label.setText("TLE Updated: N/A"))
                        return
                    
                    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'starlink_downloaded.txt')
                    with open(save_path, 'w') as f:
                        f.write(self.tle_data)
                    
                    # Get current time for update timestamp
                    from datetime import datetime
                    update_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Update UI from main thread
                    QTimer.singleShot(0, lambda n=num_sats: self.status_bar.showMessage(f"Downloaded {n} TLEs successfully"))
                    QTimer.singleShot(0, lambda n=num_sats: self.tle_count_label.setText(f"TLE in file: {n}"))
                    QTimer.singleShot(0, lambda t=update_time: self.tle_update_label.setText(f"TLE Updated: {t}"))
                    QTimer.singleShot(0, self.initialize_predictor)
                    QTimer.singleShot(0, lambda: QMessageBox.information(self, "Success", 
                                      f"Downloaded {num_sats} TLE entries from CelesTrak\nSaved to: starlink_downloaded.txt"))
                    
                except requests.exceptions.RequestException as e:
                    QTimer.singleShot(0, lambda: QMessageBox.critical(self, "Download Error", 
                                      f"Failed to download TLE data:\n{str(e)}\n\nPlease check your internet connection."))
                    QTimer.singleShot(0, lambda: self.status_bar.showMessage("Download failed - connection error"))
                    QTimer.singleShot(0, lambda: self.tle_update_label.setText("TLE Updated: Failed"))
                except Exception as e:
                    QTimer.singleShot(0, lambda: QMessageBox.critical(self, "Error", f"Download failed: {e}"))
                    QTimer.singleShot(0, lambda: self.status_bar.showMessage("Download failed"))
                    QTimer.singleShot(0, lambda: self.tle_update_label.setText("TLE Updated: Failed"))
                    
            threading.Thread(target=download_thread, daemon=True).start()
            
        def initialize_predictor(self):
            """Initialize the predictor."""
            if not self.tle_data:
                return
            
            # Don't initialize during initial UI setup to avoid conflicts
            if self.is_initializing:
                return
                
            try:
                ue_location = {
                    'latitude': float(self.lat_input.text()),
                    'longitude': float(self.lon_input.text()),
                    'altitude': float(self.alt_input.text())
                }
                
                num_sats = int(self.num_sats_input.text())
                
                # Get carrier frequency in Hz (input is in GHz)
                carrier_freq_ghz = float(self.carrier_freq_input.text())
                carrier_freq_hz = carrier_freq_ghz * 1e9
                
                if not self.is_initializing:
                    self.status_bar.showMessage("Initializing predictor...")
                
                self.multi_predictor = MultiSatellitePredictor(
                    self.tle_data, ue_location, 
                    num_satellites=num_sats, 
                    tx_freq_hz=carrier_freq_hz
                )
                
                self.status_bar.showMessage(f"Ready. {len(self.multi_predictor.predictors)} satellites loaded @ {carrier_freq_ghz} GHz.")
                self.sats_label.setText(f"Loaded: {len(self.multi_predictor.predictors)}")
                
                # Update the location inset map
                self.update_location_inset()
                if not self.is_initializing and self.canvas:
                    self.canvas.draw_idle()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to initialize: {e}")
        
        def apply_settings_update(self):
            """Apply updated settings from the input fields."""
            if not self.tle_data:
                QMessageBox.warning(self, "Warning", "Please load TLE data first.")
                return
            
            # Remember if animation was running
            was_running = self.is_running
            was_waterfall_running = self.waterfall_running
            
            # Stop any running animations
            if self.is_running:
                self.stop_animation()
            if self.waterfall_running:
                self.stop_waterfall()
            
            # Reinitialize predictor with new settings
            self.initialize_predictor()
            
            # Update timer interval if sky map was running
            if was_running and self.multi_predictor:
                self.start_sky_map()
            
            # Restart waterfall if it was running
            if was_waterfall_running and self.multi_predictor:
                self.start_live_waterfall()
            
            self.status_bar.showMessage("Settings updated successfully.")
        
        def on_mode_changed(self, index):
            """Handle mode tab change."""
            if index == 0:
                self.current_mode = 'realtime'
                if hasattr(self, 'status_bar'):
                    self.status_bar.showMessage("Switched to Real-time Mode")
            else:
                self.current_mode = 'datagen'
                if hasattr(self, 'status_bar'):
                    self.status_bar.showMessage("Switched to Data Generation Mode")
        
        def stop_data_generation(self):
            """Stop the data generation process."""
            self.datagen_running = False
            self.stop_datagen_btn.setEnabled(False)
            self.generate_btn.setEnabled(True)
            self.datagen_status_label.setText("Stopping generation...")
            self.status_bar.showMessage("Data generation stopped by user.")
        
        def generate_simulation_data(self):
            """Generate simulation data based on user settings."""
            if not self.tle_data:
                QMessageBox.warning(self, "Warning", "Please load TLE data first.")
                return
            
            from skyfield.api import utc
            from datetime import timedelta
            import threading
            
            def generate_thread():
                try:
                    self.datagen_running = True
                    self.generate_btn.setEnabled(False)
                    self.stop_datagen_btn.setEnabled(True)
                    self.datagen_progress.setValue(0)
                    self.datagen_status_label.setText("Initializing...")
                    
                    # Get parameters
                    ue_location = {
                        'latitude': float(self.datagen_lat_input.text()),
                        'longitude': float(self.datagen_lon_input.text()),
                        'altitude': float(self.datagen_alt_input.text())
                    }
                    
                    # Use default carrier frequency (10.5 GHz for Starlink)
                    carrier_freq_ghz = 10.5
                    carrier_freq_hz = carrier_freq_ghz * 1e9
                    num_sats = int(self.datagen_num_sats_input.text())
                    elevation_mask = float(self.datagen_elev_input.text())
                    
                    # Get simulation parameters
                    start_dt = self.sim_start_time.dateTime().toPyDateTime()
                    start_time_utc = start_dt.replace(tzinfo=utc)
                    duration_min = float(self.sim_duration_input.text())
                    time_step_sec = float(self.sim_timestep_input.text())
                    
                    # Create predictor
                    predictor = MultiSatellitePredictor(
                        self.tle_data, ue_location,
                        num_satellites=num_sats,
                        tx_freq_hz=carrier_freq_hz
                    )
                    
                    self.datagen_status_label.setText(f"Loaded {len(predictor.predictors)} satellites")
                    
                    # Generate time points
                    num_samples = int((duration_min * 60) / time_step_sec)
                    time_points = [start_time_utc + timedelta(seconds=i*time_step_sec) for i in range(num_samples)]
                    
                    # Store generated data - both aggregated and per-satellite
                    self.generated_data = []
                    self.satellite_data_dict = {}  # Dict: satellite_name -> list of data points
                    
                    for idx, current_time in enumerate(time_points):
                        # Check if stop was requested
                        if not self.datagen_running:
                            self.datagen_status_label.setText("Generation stopped by user")
                            self.datagen_progress.setValue(0)
                            return
                        
                        progress = int((idx / num_samples) * 100)
                        self.datagen_progress.setValue(progress)
                        self.datagen_status_label.setText(f"Generating... {idx+1}/{num_samples} time points")
                        
                        # Calculate positions for all satellites at this time
                        for pred in predictor.predictors:
                            try:
                                ts_time = pred.ts.from_datetime(current_time)
                                observer_location = pred.wgs84.latlon(
                                    pred.ue_lat, pred.ue_lon, elevation_m=pred.ue_alt * 1000
                                )
                                
                                relative = (pred.satellite - observer_location).at(ts_time)
                                
                                # Use Skyfield's proper altaz() method for correct horizon coordinates
                                # FIXED: Previously used inertial frame coordinates incorrectly
                                # Old buggy method calculated azimuth/elevation from J2000 frame xyz directly
                                # which gave wrong values (azimuth ~257° off, elevation errors increasing with angle)
                                alt, az, distance = relative.altaz()
                                elevation_deg = alt.degrees
                                azimuth_deg = az.degrees
                                distance_km = distance.km
                                
                                if elevation_deg >= elevation_mask:
                                    # Calculate Doppler shift
                                    doppler_hz = pred.calculate_doppler_shift(current_time)
                                    rx_freq_hz = pred.STARLINK_TX_FREQ + doppler_hz
                                    
                                    # Calculate relative velocity from Doppler
                                    c = 299792458  # m/s
                                    velocity_ms = -doppler_hz * c / carrier_freq_hz
                                    velocity_kms = velocity_ms / 1000  # km/s
                                    
                                    # Validate realistic values for LEO satellites
                                    # Typical Starlink orbit: 500-600 km altitude, so distance 500-5000 km
                                    # (max slant range at 10° elevation for 600km altitude satellite)
                                    # Typical relative velocity: -8 to +8 km/s
                                    if distance_km > 5000 or distance_km < 0:
                                        # Skip satellites with unrealistic distance (likely bad TLE data)
                                        continue
                                    if abs(velocity_kms) > 20:
                                        # Skip satellites with unrealistic velocity
                                        continue
                                    
                                    # Store data point
                                    data_point = {
                                        'timestamp': current_time.isoformat(),
                                        'satellite': pred.sat_name,
                                        'azimuth_deg': azimuth_deg,
                                        'elevation_deg': elevation_deg,
                                        'distance_km': distance_km,
                                        'relative_velocity_kms': velocity_kms,
                                        'doppler_shift_hz': doppler_hz,
                                        'tx_freq_ghz': carrier_freq_ghz,
                                        'rx_freq_hz': rx_freq_hz,
                                        'ue_lat': ue_location['latitude'],
                                        'ue_lon': ue_location['longitude'],
                                        'ue_alt_m': ue_location['altitude'],
                                        'time_minutes': (current_time - start_time_utc).total_seconds() / 60.0
                                    }
                                    
                                    self.generated_data.append(data_point)
                                    
                                    # Store per-satellite
                                    if pred.sat_name not in self.satellite_data_dict:
                                        self.satellite_data_dict[pred.sat_name] = []
                                    self.satellite_data_dict[pred.sat_name].append(data_point)
                            except:
                                continue
                    
                    # Update progress
                    self.datagen_progress.setValue(100)
                    num_visible_sats = len(self.satellite_data_dict)
                    self.datagen_status_label.setText(f"✓ Generated {len(self.generated_data)} data points from {num_visible_sats} satellites")
                    
                    # Populate satellite dropdown
                    self.satellite_combo.clear()
                    self.satellite_combo.addItem("-- Select a satellite --")
                    for sat_name in sorted(self.satellite_data_dict.keys()):
                        num_points = len(self.satellite_data_dict[sat_name])
                        self.satellite_combo.addItem(f"{sat_name} ({num_points} points)")
                    
                    # Enable export button
                    self.export_csv_btn.setEnabled(True)
                    self.generate_btn.setEnabled(True)
                    self.stop_datagen_btn.setEnabled(False)
                    self.datagen_running = False
                    
                    # Store simulation parameters for later use
                    self.sim_params = {
                        'start_time': start_time_utc,
                        'duration_min': duration_min,
                        'carrier_freq_ghz': carrier_freq_ghz,
                        'elevation_mask': elevation_mask
                    }
                    
                except Exception as e:
                    self.datagen_status_label.setText(f"Error: {str(e)}")
                    self.datagen_progress.setValue(0)
                    self.generate_btn.setEnabled(True)
                    self.stop_datagen_btn.setEnabled(False)
                    self.datagen_running = False
                    QMessageBox.critical(self, "Error", f"Failed to generate data: {e}")
            
            threading.Thread(target=generate_thread, daemon=True).start()
        
        def on_satellite_selected(self, index):
            """Handle satellite selection in data generation mode."""
            if index <= 0 or not hasattr(self, 'satellite_data_dict'):
                self.setup_datagen_plots()
                return
            
            # Get selected satellite name (remove point count suffix)
            sat_display = self.satellite_combo.currentText()
            sat_name = sat_display.split(' (')[0]
            
            if sat_name not in self.satellite_data_dict:
                return
            
            # Get data for this satellite
            sat_data = self.satellite_data_dict[sat_name]
            
            # Extract arrays for plotting
            time_minutes = np.array([d['time_minutes'] for d in sat_data])
            doppler_hz = np.array([d['doppler_shift_hz'] for d in sat_data])
            rx_freq_hz = np.array([d['rx_freq_hz'] for d in sat_data])
            elevation_deg = np.array([d['elevation_deg'] for d in sat_data])
            azimuth_deg = np.array([d['azimuth_deg'] for d in sat_data])
            distance_km = np.array([d['distance_km'] for d in sat_data])
            
            # Get parameters
            tx_freq_ghz = self.sim_params['carrier_freq_ghz']
            tx_freq_hz = tx_freq_ghz * 1e9
            duration_min = self.sim_params['duration_min']
            elevation_mask = self.sim_params['elevation_mask']
            
            # Create waterfall display
            self.plot_datagen_waterfall(sat_name, time_minutes, doppler_hz, rx_freq_hz, 
                                       tx_freq_ghz, tx_freq_hz, duration_min, distance_km)
            
            # Create trajectory display
            self.plot_datagen_trajectory(sat_name, azimuth_deg, elevation_deg, elevation_mask)
            
            self.datagen_fig.tight_layout()
            self.datagen_canvas.draw()
        
        def plot_datagen_waterfall(self, sat_name, time_minutes, doppler_hz, rx_freq_hz,
                                   tx_freq_ghz, tx_freq_hz, duration_min, distance_km):
            """Plot doppler waterfall for selected satellite."""
            self.plot_satellite_waterfall_to_ax(self.datagen_waterfall_ax, sat_name, time_minutes, 
                                               doppler_hz, rx_freq_hz, tx_freq_ghz, tx_freq_hz, 
                                               duration_min, distance_km)
        
        def plot_satellite_waterfall_to_ax(self, ax, sat_name, time_minutes, doppler_hz, rx_freq_hz,
                                           tx_freq_ghz, tx_freq_hz, duration_min, distance_km):
            """Plot doppler waterfall to a specific axes object."""
            ax.clear()
            
            # Convert Doppler shift to relative velocity
            # doppler_shift = -tx_freq * range_rate / c
            # range_rate (m/s) = -doppler_shift * c / tx_freq
            c = 299792458  # m/s
            velocity_ms = -doppler_hz * c / tx_freq_hz  # m/s (negative = approaching, positive = receding)
            velocity_kms = velocity_ms / 1000  # km/s
            
            # Create 2D waterfall data with velocity on x-axis
            velocity_range_kms = 5.0  # ±5 km/s
            velocity_start_kms = -velocity_range_kms
            velocity_end_kms = velocity_range_kms
            
            n_velocity_bins = 300
            n_time_samples = len(time_minutes)
            
            velocity_grid_kms = np.linspace(velocity_start_kms, velocity_end_kms, n_velocity_bins)
            
            # Create waterfall matrix (time x velocity)
            waterfall_data = np.full((n_time_samples, n_velocity_bins), 250.0)
            
            # Fill in signal data using Gaussian spreading
            sigma_kms = 0.05  # 50 m/s = 0.05 km/s
            for i, (vel_kms, dist_km) in enumerate(zip(velocity_kms, distance_km)):
                # Calculate path loss
                if dist_km > 0:
                    dist_m = dist_km * 1000
                    fspl_db = 20 * np.log10(dist_m) + 20 * np.log10(tx_freq_hz) + 20 * np.log10(4 * np.pi / 299792458)
                    
                    # Spread signal across velocity bins
                    gaussian = np.exp(-0.5 * ((velocity_grid_kms - vel_kms) / sigma_kms) ** 2)
                    waterfall_data[i, :] = fspl_db * (1 - 0.9 * gaussian) + 250.0 * (1 - gaussian)
            
            # Plot waterfall - use actual data duration, not configured duration
            actual_duration_min = (time_minutes[-1] - time_minutes[0]) if len(time_minutes) > 0 else duration_min
            extent = [velocity_start_kms, velocity_end_kms, actual_duration_min, 0]
            
            valid_data = waterfall_data[waterfall_data < 200]
            if len(valid_data) > 0:
                vmin, vmax = np.percentile(valid_data, [5, 95])
            else:
                vmin, vmax = 170, 185
            
            im = ax.imshow(waterfall_data, aspect='auto', extent=extent,
                          cmap='jet_r', vmin=vmin, vmax=vmax,
                          interpolation='bilinear')
            
            # Mark zero velocity (stationary)
            ax.axvline(x=0, color='white', linestyle='--', 
                      linewidth=1.5, alpha=0.8, label='Zero velocity')
            
            ax.set_xlabel('Relative Velocity (km/s) [- approaching, + receding]', fontsize=11)
            ax.set_ylabel('Time into pass (min)', fontsize=11)
            # Split long satellite name into two lines if needed
            if len(sat_name) > 40:
                title_text = f'Velocity Waterfall\n{sat_name}'
            else:
                title_text = f'Velocity Waterfall - {sat_name}'
            ax.set_title(title_text, fontsize=12)
            ax.legend(loc='upper right', fontsize=9)
            
            # Format x-axis to show velocity with 2 decimal places
            from matplotlib.ticker import FuncFormatter
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
        
        def plot_datagen_trajectory(self, sat_name, azimuth_deg, elevation_deg, elevation_mask):
            """Plot sky trajectory for selected satellite."""
            self.plot_satellite_trajectory_to_ax(self.datagen_trajectory_ax, sat_name, 
                                                azimuth_deg, elevation_deg, elevation_mask)
        
        def plot_satellite_trajectory_to_ax(self, ax, sat_name, azimuth_deg, elevation_deg, elevation_mask):
            """Plot sky trajectory to a specific axes object."""
            ax.clear()
            
            # Configure polar plot
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_ylim(0, 90)
            ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
            ax.set_yticklabels(['90°', '75°', '60°', '45°', '30°', '15°', '0°'], fontsize=8)
            ax.set_rlabel_position(22.5)
            
            az_deg_ticks = np.array([0, 45, 90, 135, 180, 225, 270, 315])
            az_rad_ticks = np.radians(az_deg_ticks)
            ax.set_xticks(az_rad_ticks)
            ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], 
                              fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.4, linestyle='--')
            
            # Draw elevation mask circle
            mask_r = 90 - elevation_mask
            theta_circle = np.linspace(0, 2*np.pi, 100)
            ax.plot(theta_circle, [mask_r]*100, 'r--', linewidth=1, alpha=0.5,
                   label=f'Elev mask ({elevation_mask}°)')
            
            # Plot zenith
            ax.plot(0, 0, 'r+', markersize=12, markeredgewidth=2, zorder=10)
            
            # Convert trajectory to polar coordinates
            thetas = np.radians(azimuth_deg)
            rs = 90 - elevation_deg
            
            # Plot trajectory with color gradient
            colors = np.linspace(0, 1, len(thetas))
            scatter = ax.scatter(thetas, rs, c=colors, cmap='cool', 
                                s=20, alpha=0.7, zorder=3)
            
            # Plot trajectory line
            ax.plot(thetas, rs, 'c-', linewidth=1.5, alpha=0.5, zorder=2)
            
            # Mark start and end
            if len(thetas) > 0:
                ax.scatter([thetas[0]], [rs[0]], c='green', s=150, 
                          marker='^', edgecolors='white', linewidths=2, 
                          zorder=6, label='Rise')
                ax.scatter([thetas[-1]], [rs[-1]], c='red', s=150, 
                          marker='v', edgecolors='white', linewidths=2, 
                          zorder=6, label='Set')
            
            # Split long satellite name into two lines if needed
            if len(sat_name) > 40:
                title_text = f'Sky Trajectory\n{sat_name}'
            else:
                title_text = f'Sky Trajectory - {sat_name}'
            ax.set_title(title_text, fontsize=12, pad=10)
            ax.legend(loc='upper left', bbox_to_anchor=(-0.55, 1.15), fontsize=8)
        
        def export_data_to_csv(self):
            """Export generated data to CSV file, along with settings and plot images."""
            if not hasattr(self, 'generated_data') or not self.generated_data:
                QMessageBox.warning(self, "Warning", "No data to export. Generate data first.")
                return
            
            # Ask for directory instead of file
            from PyQt5.QtWidgets import QFileDialog
            directory = QFileDialog.getExistingDirectory(
                self, "Select Export Directory", 
                os.path.expanduser("~"),
                QFileDialog.ShowDirsOnly
            )
            
            if directory:
                try:
                    import csv
                    from datetime import datetime
                    
                    # Create timestamp for file naming
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_name = f"satellite_export_{timestamp}"
                    
                    # 1. Export all data to CSV
                    csv_path = os.path.join(directory, f"{base_name}_data.csv")
                    with open(csv_path, 'w', newline='') as f:
                        # Write settings as comments at the top
                        f.write("# Satellite Data Generation Settings\n")
                        f.write(f"# Ground Station: Lat={self.datagen_lat_input.text()}°N, Lon={self.datagen_lon_input.text()}°E, Alt={self.datagen_alt_input.text()}m\n")
                        f.write(f"# Start Time (UTC): {self.sim_start_time.dateTime().toString('yyyy-MM-dd HH:mm:ss')}\n")
                        f.write(f"# Duration: {self.sim_duration_input.text()} min, Time Step: {self.sim_timestep_input.text()} sec\n")
                        f.write(f"# Carrier Frequency: 10.5 GHz (Starlink default), Elevation Mask: {self.datagen_elev_input.text()}°\n")
                        f.write(f"# Total Data Points: {len(self.generated_data)}, Satellites Tracked: {len(self.satellite_data_dict)}\n")
                        f.write("#\n")
                        
                        if self.generated_data:
                            writer = csv.DictWriter(f, fieldnames=self.generated_data[0].keys())
                            writer.writeheader()
                            writer.writerows(self.generated_data)
                    
                    # 2. Export settings to text file
                    settings_path = os.path.join(directory, f"{base_name}_settings.txt")
                    with open(settings_path, 'w') as f:
                        f.write("Satellite Data Generation Settings\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(f"Ground Station Location:\n")
                        f.write(f"  Latitude: {self.datagen_lat_input.text()}°N\n")
                        f.write(f"  Longitude: {self.datagen_lon_input.text()}°E\n")
                        f.write(f"  Altitude: {self.datagen_alt_input.text()} m\n\n")
                        f.write(f"Simulation Parameters:\n")
                        f.write(f"  Start Time (UTC): {self.sim_start_time.dateTime().toString('yyyy-MM-dd HH:mm:ss')}\n")
                        f.write(f"  Duration: {self.sim_duration_input.text()} minutes\n")
                        f.write(f"  Time Step: {self.sim_timestep_input.text()} seconds\n")
                        f.write(f"  Carrier Frequency: 10.5 GHz (Starlink default)\n")
                        f.write(f"  Elevation Mask: {self.datagen_elev_input.text()}°\n")
                        f.write(f"  Max Satellites: {self.datagen_num_sats_input.text()}\n\n")
                        f.write(f"Results:\n")
                        f.write(f"  Total Data Points: {len(self.generated_data)}\n")
                        f.write(f"  Satellites Tracked: {len(self.satellite_data_dict)}\n")
                    
                    # 3. Export images for ALL satellites
                    self.status_bar.showMessage("Generating plots for all satellites...")
                    exported_sats = []
                    
                    for idx, sat_name in enumerate(sorted(self.satellite_data_dict.keys())):
                        self.status_bar.showMessage(f"Generating plots... {idx+1}/{len(self.satellite_data_dict)}")
                        
                        sat_data = self.satellite_data_dict[sat_name]
                        
                        # Extract arrays for this satellite
                        time_minutes = np.array([d['time_minutes'] for d in sat_data])
                        doppler_hz = np.array([d['doppler_shift_hz'] for d in sat_data])
                        rx_freq_hz = np.array([d['rx_freq_hz'] for d in sat_data])
                        elevation_deg = np.array([d['elevation_deg'] for d in sat_data])
                        azimuth_deg = np.array([d['azimuth_deg'] for d in sat_data])
                        distance_km = np.array([d['distance_km'] for d in sat_data])
                        
                        # Get parameters
                        tx_freq_ghz = self.sim_params['carrier_freq_ghz']
                        tx_freq_hz = tx_freq_ghz * 1e9
                        duration_min = self.sim_params['duration_min']
                        elevation_mask = self.sim_params['elevation_mask']
                        
                        # Create a temporary figure for this satellite
                        from matplotlib.gridspec import GridSpec
                        temp_fig = Figure(figsize=(12, 10), dpi=100)
                        gs = GridSpec(2, 1, figure=temp_fig, height_ratios=[1.2, 1], hspace=0.5)
                        
                        temp_wf_ax = temp_fig.add_subplot(gs[0])
                        temp_traj_ax = temp_fig.add_subplot(gs[1], projection='polar')
                        
                        # Plot waterfall
                        self.plot_satellite_waterfall_to_ax(temp_wf_ax, sat_name, time_minutes, doppler_hz, 
                                                           rx_freq_hz, tx_freq_ghz, tx_freq_hz, duration_min, distance_km)
                        
                        # Plot trajectory
                        self.plot_satellite_trajectory_to_ax(temp_traj_ax, sat_name, azimuth_deg, elevation_deg, elevation_mask)
                        
                        temp_fig.tight_layout()
                        
                        # Save image
                        image_path = os.path.join(directory, f"{base_name}_{sat_name}.png")
                        temp_fig.savefig(image_path, dpi=150, bbox_inches='tight')
                        
                        # Save CSV for this satellite
                        sat_csv_path = os.path.join(directory, f"{base_name}_{sat_name}.csv")
                        with open(sat_csv_path, 'w', newline='') as f:
                            # Write settings as comments at the top
                            f.write(f"# Satellite: {sat_name}\n")
                            f.write(f"# Ground Station: Lat={self.datagen_lat_input.text()}°N, Lon={self.datagen_lon_input.text()}°E, Alt={self.datagen_alt_input.text()}m\n")
                            f.write(f"# Start Time (UTC): {self.sim_start_time.dateTime().toString('yyyy-MM-dd HH:mm:ss')}\n")
                            f.write(f"# Duration: {self.sim_duration_input.text()} min, Time Step: {self.sim_timestep_input.text()} sec\n")
                            f.write(f"# Carrier Frequency: 10.5 GHz (Starlink default), Elevation Mask: {self.datagen_elev_input.text()}°\n")
                            f.write(f"# Data Points for this satellite: {len(sat_data)}\n")
                            f.write("#\n")
                            
                            writer = csv.DictWriter(f, fieldnames=sat_data[0].keys())
                            writer.writeheader()
                            writer.writerows(sat_data)
                        
                        exported_sats.append(sat_name)
                        
                        # Clean up
                        import matplotlib.pyplot as plt
                        plt.close(temp_fig)
                    
                    msg = f"Export successful!\n\n"
                    msg += f"Files saved to: {directory}\n\n"
                    msg += f"- {base_name}_data.csv (all data)\n"
                    msg += f"- {base_name}_settings.txt (settings)\n"
                    msg += f"- {len(exported_sats)} satellite plots and CSV files\n"
                    msg += f"  (format: {base_name}_SAT-XXXXX.png/csv)"
                    
                    QMessageBox.information(self, "Success", msg)
                    self.status_bar.showMessage(f"Exported {len(exported_sats)} satellites to {directory}")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    QMessageBox.critical(self, "Error", f"Failed to export: {e}")
                
        def start_sky_map(self):
            """Start the real-time animation."""
            if not self.multi_predictor:
                QMessageBox.warning(self, "Warning", "Please load TLE data first.")
                return
                
            self.is_running = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            
            # Initialize all satellites as hidden (will be checked on first visibility update)
            self.visible_predictors = []
            self.hidden_predictors = list(self.multi_predictor.predictors)
                
            self.setup_polar_plot()
            
            # Create fast timer for updating visible satellites
            interval = int(self.interval_input.text())
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_visible_satellites)
            self.timer.start(interval)
            
            # Create slow timer for checking visibility of hidden satellites
            self.visibility_timer = QTimer()
            self.visibility_timer.timeout.connect(self.check_hidden_satellites)
            self.visibility_timer.start(10000)  # Check every 10 seconds
            
            # Do initial full scan
            self.check_hidden_satellites()
            
            self.status_bar.showMessage("Sky map running...")
            
        def update_sky_map(self):
            """Update the sky map - DEPRECATED, kept for compatibility."""
            self.update_visible_satellites()
        
        def calculate_satellite_position(self, predictor, current_time_utc, elevation_mask, calc_velocity=False):
            """Calculate position for a single satellite.
            
            Returns:
                If calc_velocity=False: (is_visible, theta, r) or (False, None, None)
                If calc_velocity=True: (is_visible, theta, r, d_theta, d_r) - includes velocity components
            """
            try:
                ts_time = predictor.ts.from_datetime(current_time_utc)
                
                observer_location = predictor.wgs84.latlon(
                    predictor.ue_lat,
                    predictor.ue_lon,
                    elevation_m=predictor.ue_alt * 1000
                )
                
                relative = (predictor.satellite - observer_location).at(ts_time)
                
                # Use Skyfield's proper altaz() method for correct horizon coordinates
                alt, az, distance = relative.altaz()
                elevation_deg = alt.degrees
                azimuth_deg = az.degrees
                
                if elevation_deg >= elevation_mask:
                    theta_rad = np.radians(azimuth_deg)
                    r = 90 - elevation_deg
                    
                    if calc_velocity:
                        # Calculate future position (1 second ahead) for velocity arrow
                        from skyfield.api import utc
                        future_time = datetime.utcfromtimestamp(
                            current_time_utc.timestamp() + 1.0
                        ).replace(tzinfo=utc)
                        ts_time_future = predictor.ts.from_datetime(future_time)
                        
                        relative_future = (predictor.satellite - observer_location).at(ts_time_future)
                        alt_future, az_future, distance_future = relative_future.altaz()
                        elevation_deg_future = alt_future.degrees
                        azimuth_deg_future = az_future.degrees
                        
                        theta_rad_future = np.radians(azimuth_deg_future)
                        r_future = 90 - elevation_deg_future
                        
                        # Calculate delta (direction of movement)
                        d_theta = theta_rad_future - theta_rad
                        d_r = r_future - r
                        
                        # Handle wrap-around for azimuth
                        if d_theta > np.pi:
                            d_theta -= 2 * np.pi
                        elif d_theta < -np.pi:
                            d_theta += 2 * np.pi
                        
                        return (True, theta_rad, r, d_theta, d_r)
                    
                    return (True, theta_rad, r)
                else:
                    if calc_velocity:
                        return (False, None, None, None, None)
                    return (False, None, None)
            except:
                if calc_velocity:
                    return (False, None, None, None, None)
                return (False, None, None)
        
        def check_hidden_satellites(self):
            """Check hidden satellites for visibility changes (runs every 10 seconds)."""
            if not self.is_running:
                return
            
            from skyfield.api import utc
            
            elevation_mask = float(self.elev_input.text())
            current_time_utc = datetime.utcnow().replace(tzinfo=utc)
            
            # Check all hidden satellites
            newly_visible = []
            still_hidden = []
            
            for predictor in self.hidden_predictors:
                is_visible, theta, r = self.calculate_satellite_position(predictor, current_time_utc, elevation_mask)
                if is_visible:
                    newly_visible.append(predictor)
                else:
                    still_hidden.append(predictor)
            
            # Also check currently visible satellites to see if any went out of view
            still_visible = []
            newly_hidden = []
            
            for predictor in self.visible_predictors:
                is_visible, theta, r = self.calculate_satellite_position(predictor, current_time_utc, elevation_mask)
                if is_visible:
                    still_visible.append(predictor)
                else:
                    newly_hidden.append(predictor)
            
            # Update the lists
            self.visible_predictors = still_visible + newly_visible
            self.hidden_predictors = still_hidden + newly_hidden
            
            # Update display immediately after visibility check
            self.update_visible_satellites()
        
        def update_visible_satellites(self):
            """Update positions of only visible satellites (runs frequently)."""
            if not self.is_running:
                return
                
            from skyfield.api import utc
            import numpy as np
            
            elevation_mask = float(self.elev_input.text())
            current_time_utc = datetime.utcnow().replace(tzinfo=utc)
            
            satellites_visible = []
            
            # Only update satellites that are already known to be visible
            self.visible_sat_data = []  # Reset visible satellite data for click detection
            for predictor in self.visible_predictors:
                result = self.calculate_satellite_position(predictor, current_time_utc, elevation_mask, calc_velocity=True)
                if result[0]:  # is_visible
                    is_visible, theta, r, d_theta, d_r = result
                    satellites_visible.append({
                        'theta': theta, 'r': r, 
                        'd_theta': d_theta, 'd_r': d_r,
                        'predictor': predictor
                    })
                    self.visible_sat_data.append({'theta': theta, 'r': r, 'predictor': predictor})
            
            # Update scatter plots
            if satellites_visible:
                thetas = [s['theta'] for s in satellites_visible]
                rs = [s['r'] for s in satellites_visible]
                offsets = np.c_[thetas, rs]
                self.satellite_scatter.set_offsets(offsets)
                self.satellite_glow.set_offsets(offsets)
            else:
                self.satellite_scatter.set_offsets(np.c_[[], []])
                self.satellite_glow.set_offsets(np.c_[[], []])
            
            # Update velocity arrows
            self.update_velocity_arrows(satellites_visible)
            
            # Update selected satellite marker if one is selected
            if self.selected_predictor is not None:
                is_visible, theta, r = self.calculate_satellite_position(self.selected_predictor, current_time_utc, elevation_mask)
                if is_visible:
                    self.selected_marker.set_offsets(np.c_[[theta], [r]])
                else:
                    self.selected_marker.set_offsets(np.c_[[], []])
            
            # Update title
            current_time_aware = current_time_utc.astimezone()
            current_time_str = current_time_aware.strftime('%Y-%m-%d %H:%M:%S %Z')
            num_visible = len(self.visible_predictors)
            num_hidden = len(self.hidden_predictors)
            self.ax.set_title(f'Sky Map - Starlink Satellites (LIVE)\n{current_time_str}\n({num_visible} visible, {num_hidden} hidden, Elevation > {elevation_mask}°)',
                             fontsize=11, pad=-20)
            
            self.visible_label.setText(f"Visible: {num_visible}")
            self.canvas.draw_idle()
        
        def update_velocity_arrows(self, satellites_visible):
            """Update velocity arrows showing satellite direction of movement."""
            # Remove existing arrows
            for arrow in self.velocity_arrows:
                arrow.remove()
            self.velocity_arrows = []
            
            # Arrow scale factor for visibility
            arrow_scale = 8.0  # Scale the arrow length for visibility
            
            for sat in satellites_visible:
                theta = sat['theta']
                r = sat['r']
                d_theta = sat['d_theta']
                d_r = sat['d_r']
                
                if d_theta is None or d_r is None:
                    continue
                
                # Convert current position to Cartesian (in plot space)
                x = r * np.sin(theta)
                y = r * np.cos(theta)
                
                # Calculate future position in polar
                theta_future = theta + d_theta
                r_future = r + d_r
                
                # Convert future position to Cartesian
                x_future = r_future * np.sin(theta_future)
                y_future = r_future * np.cos(theta_future)
                
                # Calculate velocity vector in Cartesian
                dx = x_future - x
                dy = y_future - y
                
                # Calculate arrow length and skip if too small
                arrow_length = np.sqrt(dx**2 + dy**2)
                if arrow_length < 0.001:
                    continue
                
                # Normalize and scale the arrow
                dx_scaled = (dx / arrow_length) * arrow_scale
                dy_scaled = (dy / arrow_length) * arrow_scale
                
                # Calculate arrow endpoint in Cartesian
                x_end = x + dx_scaled
                y_end = y + dy_scaled
                
                # Convert arrow endpoint back to polar
                r_end = np.sqrt(x_end**2 + y_end**2)
                theta_end = np.arctan2(x_end, y_end)  # Note: sin/cos swap for polar plot orientation
                
                # Create arrow using annotate
                arrow = self.ax.annotate(
                    '',  # No text
                    xy=(theta_end, r_end),  # Arrow head
                    xytext=(theta, r),  # Arrow tail (satellite position)
                    arrowprops=dict(
                        arrowstyle='->,head_width=0.3,head_length=0.2',
                        color='cyan',
                        lw=1.2,
                        alpha=0.8
                    ),
                    zorder=3
                )
                self.velocity_arrows.append(arrow)
            
        def stop_animation(self):
            """Stop the animation."""
            self.is_running = False
            if self.timer:
                self.timer.stop()
                self.timer = None
            if self.visibility_timer:
                self.visibility_timer.stop()
                self.visibility_timer = None
                
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_bar.showMessage("Animation stopped.")
        
        def start_live_waterfall(self):
            """Start live updating waterfall display."""
            if not self.multi_predictor:
                QMessageBox.warning(self, "Warning", "Please load TLE data first.")
                return
            
            from skyfield.api import utc
            
            self.waterfall_running = True
            self.waterfall_btn.setEnabled(False)
            self.stop_waterfall_btn.setEnabled(True)
            
            # Get settings
            self.wf_elevation_mask = float(self.elev_input.text())
            self.wf_duration_minutes = float(self.duration_input.text())
            
            # Get frequency info
            predictor = self.multi_predictor.predictors[0]
            self.wf_tx_hz = predictor.STARLINK_TX_FREQ
            self.wf_tx_ghz = self.wf_tx_hz / 1e9
            self.wf_doppler_range_hz = 500e3  # ±500 kHz
            self.wf_freq_start_hz = self.wf_tx_hz - self.wf_doppler_range_hz
            self.wf_freq_end_hz = self.wf_tx_hz + self.wf_doppler_range_hz
            
            # Waterfall dimensions
            self.wf_n_freq_bins = 300
            self.wf_n_time_samples = max(int(self.wf_duration_minutes * 6), 30)  # ~10 sec per row
            self.wf_sigma = 5e3  # 5 kHz
            
            # Create frequency grid
            self.wf_freq_grid_hz = np.linspace(self.wf_freq_start_hz, self.wf_freq_end_hz, self.wf_n_freq_bins)
            
            # Initialize waterfall with high path loss values (no signal = high loss)
            self.waterfall_data = np.full((self.wf_n_time_samples, self.wf_n_freq_bins), 250.0)
            self.waterfall_times = []
            
            # Create waterfall window
            self.waterfall_window = QMainWindow(self)
            self.waterfall_window.setWindowTitle("Live Doppler Waterfall")
            self.waterfall_window.setGeometry(150, 150, 1000, 700)
            
            central = QWidget()
            self.waterfall_window.setCentralWidget(central)
            layout = QVBoxLayout(central)
            
            # Create figure
            self.wf_fig = Figure(figsize=(12, 8), dpi=100)
            self.wf_ax = self.wf_fig.add_subplot(111)
            
            # Initial plot - use absolute frequency in GHz
            freq_start_ghz = self.wf_freq_start_hz / 1e9
            freq_end_ghz = self.wf_freq_end_hz / 1e9
            self.wf_extent = [freq_start_ghz, freq_end_ghz, self.wf_duration_minutes, 0]
            
            self.wf_im = self.wf_ax.imshow(self.waterfall_data, aspect='auto', extent=self.wf_extent,
                                           cmap='jet_r', vmin=170, vmax=185,
                                           interpolation='bilinear')
            
            self.wf_cbar = self.wf_fig.colorbar(self.wf_im, ax=self.wf_ax, pad=0.02)
            self.wf_cbar.set_label('Free Space Path Loss (dB)', fontsize=11)
            
            # Mark TX frequency (no Doppler)
            self.wf_ax.axvline(x=self.wf_tx_ghz, color='white', linestyle='--', linewidth=1.5, alpha=0.8,
                              label=f'TX: {self.wf_tx_ghz:.3f} GHz')
            self.wf_ax.set_xlabel('Frequency (GHz)', fontsize=12)
            self.wf_ax.set_ylabel('Time ago (minutes)', fontsize=12)
            self.wf_title = self.wf_ax.set_title(f'Live Doppler Waterfall - Path Loss\nInitializing...', fontsize=12)
            self.wf_ax.legend(loc='upper right', fontsize=9)
            
            # Format x-axis to show full GHz values without offset notation
            from matplotlib.ticker import FuncFormatter
            self.wf_ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.4f}'))
            
            self.wf_fig.tight_layout()
            
            self.wf_canvas = FigureCanvas(self.wf_fig)
            layout.addWidget(self.wf_canvas)
            
            toolbar = NavigationToolbar(self.wf_canvas, central)
            layout.addWidget(toolbar)
            
            self.waterfall_window.show()
            
            # Start update timer based on user setting
            wf_update_sec = float(self.wf_update_input.text())
            self.waterfall_timer = QTimer()
            self.waterfall_timer.timeout.connect(self.update_waterfall)
            self.waterfall_timer.start(int(wf_update_sec * 1000))
            
            # Do initial update
            self.update_waterfall()
            
            self.status_bar.showMessage("Live waterfall running...")
        
        def calculate_fspl_db(self, distance_m, frequency_hz):
            """Calculate Free Space Path Loss in dB.
            
            FSPL(dB) = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/c)
                     = 20*log10(d) + 20*log10(f) - 147.55
            
            Args:
                distance_m: Slant distance in meters
                frequency_hz: Frequency in Hz
                
            Returns:
                FSPL in dB (positive value representing loss)
            """
            c = 299792458  # Speed of light in m/s
            if distance_m <= 0:
                return 200  # Return very high loss for invalid distance
            
            fspl_db = 20 * np.log10(distance_m) + 20 * np.log10(frequency_hz) + 20 * np.log10(4 * np.pi / c)
            return fspl_db
        
        def update_waterfall(self):
            """Update the live waterfall display with FSPL-based signal power."""
            if not self.waterfall_running:
                return
            
            from skyfield.api import utc
            from datetime import timezone
            
            current_time_utc = datetime.utcnow().replace(tzinfo=utc)
            
            # Calculate current Doppler spectrum (single time slice)
            # Store FSPL in dB directly for each frequency bin (lower = stronger signal)
            current_spectrum_db = np.full(self.wf_n_freq_bins, 250.0)  # Initialize with very high path loss
            visible_count = 0
            
            for pred in self.multi_predictor.predictors:
                try:
                    ts_time = pred.ts.from_datetime(current_time_utc)
                    observer_location = pred.wgs84.latlon(
                        pred.ue_lat,
                        pred.ue_lon,
                        elevation_m=pred.ue_alt * 1000
                    )
                    
                    relative = (pred.satellite - observer_location).at(ts_time)
                    xyz = relative.position.au
                    
                    # Calculate slant distance in meters (1 AU = 149597870700 m)
                    AU_TO_METERS = 149597870700
                    slant_distance_m = np.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2) * AU_TO_METERS
                    
                    horizontal_dist = np.sqrt(xyz[0]**2 + xyz[1]**2)
                    magnitude = (xyz[0]**2 + xyz[1]**2 + xyz[2]**2)**0.5
                    
                    if magnitude > 0:
                        elevation_rad = np.arctan2(xyz[2], horizontal_dist)
                        elevation_deg = np.degrees(elevation_rad)
                    else:
                        elevation_deg = -90
                    
                    if elevation_deg >= self.wf_elevation_mask:
                        shift = pred.calculate_doppler_shift(current_time_utc)
                        received_freq = pred.STARLINK_TX_FREQ + shift
                        
                        if self.wf_freq_start_hz <= received_freq <= self.wf_freq_end_hz:
                            # Calculate absolute FSPL in dB
                            fspl_db = self.calculate_fspl_db(slant_distance_m, received_freq)
                            
                            # Add signal with Gaussian shape
                            # Create Gaussian envelope centered at received frequency
                            gaussian = np.exp(-((self.wf_freq_grid_hz - received_freq) ** 2) / (2 * self.wf_sigma ** 2))
                            
                            # Convert gaussian to dB attenuation (peak = 0 dB, tails = positive addition to loss)
                            gaussian_db = -10 * np.log10(gaussian + 1e-10)
                            
                            # Total path loss: FSPL + gaussian shape (minimum = best signal)
                            signal_loss_db = fspl_db + gaussian_db
                            
                            # Take minimum path loss (strongest signal)
                            current_spectrum_db = np.minimum(current_spectrum_db, signal_loss_db)
                            visible_count += 1
                except:
                    pass
            
            # Scroll waterfall down and add new data at top (data is FSPL in dB)
            self.waterfall_data = np.roll(self.waterfall_data, 1, axis=0)
            self.waterfall_data[0, :] = current_spectrum_db
            
            # Data is path loss in dB
            waterfall_db = self.waterfall_data.copy()
            
            # Update image data - find appropriate display range
            # For Starlink at ~550-2000 km, FSPL at 12 GHz is roughly 170-185 dB
            valid_data = waterfall_db[waterfall_db < 200]
            if len(valid_data) > 0:
                vmin = np.min(valid_data)  # Minimum path loss (strongest)
                vmax = vmin + 15  # 15 dB dynamic range
            else:
                vmin, vmax = 170, 185
            
            self.wf_im.set_data(waterfall_db)
            self.wf_im.set_clim(vmin=vmin, vmax=vmax)
            
            # Update title with current time
            current_time_aware = current_time_utc.astimezone()
            current_time_str = current_time_aware.strftime('%H:%M:%S %Z')
            self.wf_title.set_text(f'Live Doppler Waterfall - Path Loss (TX: {self.wf_tx_ghz:.3f} GHz)\n{current_time_str} ({visible_count} satellites visible)')
            
            self.wf_canvas.draw_idle()
        
        def stop_waterfall(self):
            """Stop the live waterfall."""
            self.waterfall_running = False
            if self.waterfall_timer:
                self.waterfall_timer.stop()
                self.waterfall_timer = None
            
            self.waterfall_btn.setEnabled(True)
            self.stop_waterfall_btn.setEnabled(False)
            self.status_bar.showMessage("Live waterfall stopped.")
        
        def on_satellite_click(self, event):
            """Handle click on a satellite in the sky map."""
            if event.artist != self.satellite_scatter:
                return
            
            # Get the index of the clicked point
            ind = event.ind[0]
            
            if ind < len(self.visible_sat_data):
                sat_data = self.visible_sat_data[ind]
                self.selected_predictor = sat_data['predictor']
                
                # Extract satellite name from TLE
                try:
                    # Parse NORAD ID from TLE line 1
                    tle_line1 = self.selected_predictor.tle_line1
                    norad_id = tle_line1.split()[1] if len(tle_line1.split()) > 1 else "Unknown"
                    self.selected_sat_name = f"STARLINK (NORAD {norad_id})"
                except:
                    self.selected_sat_name = "Selected Satellite"
                
                # Update the selected marker position
                self.selected_marker.set_offsets(np.c_[[sat_data['theta']], [sat_data['r']]])
                self.canvas.draw_idle()
                
                self.status_bar.showMessage(f"Selected: {self.selected_sat_name} - Computing full pass waterfall...")
                
                # Start single satellite waterfall for full pass
                self.start_single_satellite_waterfall()
        
        def find_pass_window(self, predictor, elevation_mask=10.0, search_hours=2.0):
            """
            Find the current pass window (rise to set) for a satellite.
            
            Returns:
                (rise_time, set_time, max_el_time) as datetime objects, or None if not found
            """
            from skyfield.api import utc
            from datetime import timedelta
            
            current_time = datetime.utcnow().replace(tzinfo=utc)
            
            observer_location = predictor.wgs84.latlon(
                predictor.ue_lat,
                predictor.ue_lon,
                elevation_m=predictor.ue_alt * 1000
            )
            
            # Sample times to find the pass
            dt_step = 10  # seconds
            num_samples = int(search_hours * 3600 / dt_step)
            
            elevations = []
            times = []
            
            for i in range(-num_samples // 4, num_samples):  # Start a bit in the past
                t = current_time + timedelta(seconds=i * dt_step)
                ts_time = predictor.ts.from_datetime(t)
                
                try:
                    relative = (predictor.satellite - observer_location).at(ts_time)
                    xyz = relative.position.au
                    horizontal_dist = np.sqrt(xyz[0]**2 + xyz[1]**2)
                    magnitude = np.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2)
                    
                    if magnitude > 0:
                        elevation_rad = np.arctan2(xyz[2], horizontal_dist)
                        elevation_deg = np.degrees(elevation_rad)
                    else:
                        elevation_deg = -90
                    
                    elevations.append(elevation_deg)
                    times.append(t)
                except:
                    elevations.append(-90)
                    times.append(t)
            
            elevations = np.array(elevations)
            
            # Find the current pass that includes "now"
            # Look for where elevation crosses the mask
            above_mask = elevations >= elevation_mask
            
            # Find current index (where i=0, which is at num_samples//4)
            current_idx = num_samples // 4
            
            # If currently not visible, find next pass
            if not above_mask[current_idx]:
                # Find next rise
                for i in range(current_idx, len(above_mask)):
                    if above_mask[i]:
                        current_idx = i
                        break
                else:
                    return None  # No pass found
            
            # Find rise time (go backwards from current visible position)
            rise_idx = current_idx
            for i in range(current_idx, -1, -1):
                if above_mask[i]:
                    rise_idx = i
                else:
                    break
            
            # Find set time (go forwards from current visible position)
            set_idx = current_idx
            for i in range(current_idx, len(above_mask)):
                if above_mask[i]:
                    set_idx = i
                else:
                    break
            
            # Find max elevation time
            pass_elevations = elevations[rise_idx:set_idx+1]
            if len(pass_elevations) > 0:
                max_el_idx = rise_idx + np.argmax(pass_elevations)
            else:
                max_el_idx = current_idx
            
            return (times[rise_idx], times[set_idx], times[max_el_idx])
        
        def start_single_satellite_waterfall(self):
            """Start waterfall display for the selected satellite's complete pass."""
            if self.selected_predictor is None:
                return
            
            # Stop existing single satellite waterfall if running
            if self.single_sat_waterfall_running:
                self.stop_single_satellite_waterfall()
            
            from skyfield.api import utc
            from datetime import timedelta
            
            self.single_sat_waterfall_running = True
            
            # Get settings
            self.ss_wf_elevation_mask = float(self.elev_input.text())
            
            # Get frequency info
            self.ss_wf_tx_hz = self.selected_predictor.STARLINK_TX_FREQ
            self.ss_wf_tx_ghz = self.ss_wf_tx_hz / 1e9
            self.ss_wf_doppler_range_hz = 500e3  # ±500 kHz
            self.ss_wf_freq_start_hz = self.ss_wf_tx_hz - self.ss_wf_doppler_range_hz
            self.ss_wf_freq_end_hz = self.ss_wf_tx_hz + self.ss_wf_doppler_range_hz
            
            # Find the pass window
            pass_info = self.find_pass_window(self.selected_predictor, self.ss_wf_elevation_mask)
            
            if pass_info is None:
                QMessageBox.warning(self, "Warning", "Could not find a visible pass for this satellite.")
                self.single_sat_waterfall_running = False
                return
            
            self.ss_rise_time, self.ss_set_time, self.ss_max_el_time = pass_info
            
            # Calculate pass duration
            pass_duration_sec = (self.ss_set_time - self.ss_rise_time).total_seconds()
            pass_duration_min = pass_duration_sec / 60.0
            
            # Waterfall dimensions - 1 row per 2 seconds for good resolution
            self.ss_wf_n_freq_bins = 300
            self.ss_wf_n_time_samples = max(int(pass_duration_sec / 2), 30)
            self.ss_wf_sigma = 5e3  # 5 kHz
            
            # Create frequency grid
            self.ss_wf_freq_grid_hz = np.linspace(self.ss_wf_freq_start_hz, self.ss_wf_freq_end_hz, self.ss_wf_n_freq_bins)
            
            # Pre-compute the entire pass waterfall
            self.status_bar.showMessage(f"Computing waterfall for {pass_duration_min:.1f} min pass...")
            
            self.single_sat_waterfall_data = np.full((self.ss_wf_n_time_samples, self.ss_wf_n_freq_bins), 250.0)
            self.ss_time_points = []  # Store time for each row
            self.ss_elevation_data = []  # Store elevation for each row
            self.ss_azimuth_data = []  # Store azimuth for each row (for trajectory plot)
            self.ss_doppler_data = []  # Store Doppler for each row
            
            pred = self.selected_predictor
            observer_location = pred.wgs84.latlon(
                pred.ue_lat, pred.ue_lon, elevation_m=pred.ue_alt * 1000
            )
            
            for i in range(self.ss_wf_n_time_samples):
                # Time for this row (from rise to set)
                t_offset = (i / (self.ss_wf_n_time_samples - 1)) * pass_duration_sec
                t = self.ss_rise_time + timedelta(seconds=t_offset)
                self.ss_time_points.append(t)
                
                try:
                    ts_time = pred.ts.from_datetime(t)
                    relative = (pred.satellite - observer_location).at(ts_time)
                    xyz = relative.position.au
                    
                    AU_TO_METERS = 149597870700
                    slant_distance_m = np.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2) * AU_TO_METERS
                    
                    # Use Skyfield's proper altaz() method
                    alt, az, distance = relative.altaz()
                    elevation_deg = alt.degrees
                    azimuth_deg = az.degrees
                    
                    self.ss_elevation_data.append(elevation_deg)
                    self.ss_azimuth_data.append(azimuth_deg)
                    
                    # Calculate Doppler
                    shift = pred.calculate_doppler_shift(t)
                    self.ss_doppler_data.append(shift / 1000)  # kHz
                    received_freq = pred.STARLINK_TX_FREQ + shift
                    
                    if elevation_deg >= self.ss_wf_elevation_mask:
                        fspl_db = self.calculate_fspl_db(slant_distance_m, received_freq)
                        gaussian = np.exp(-((self.ss_wf_freq_grid_hz - received_freq) ** 2) / (2 * self.ss_wf_sigma ** 2))
                        gaussian_db = -10 * np.log10(gaussian + 1e-10)
                        signal_loss_db = fspl_db + gaussian_db
                        self.single_sat_waterfall_data[i, :] = signal_loss_db
                except:
                    self.ss_elevation_data.append(-90)
                    self.ss_azimuth_data.append(0)
                    self.ss_doppler_data.append(0)
            
            # Create waterfall window
            self.single_sat_waterfall_window = QMainWindow(self)
            self.single_sat_waterfall_window.setWindowTitle(f"Full Pass Waterfall - {self.selected_sat_name}")
            self.single_sat_waterfall_window.setGeometry(200, 200, 1400, 850)
            
            central = QWidget()
            self.single_sat_waterfall_window.setCentralWidget(central)
            layout = QVBoxLayout(central)
            
            # Info label
            rise_str = self.ss_rise_time.astimezone().strftime('%H:%M:%S')
            set_str = self.ss_set_time.astimezone().strftime('%H:%M:%S')
            max_el = max(self.ss_elevation_data) if self.ss_elevation_data else 0
            info_text = f"Satellite: {self.selected_sat_name} | Pass: {rise_str} → {set_str} ({pass_duration_min:.1f} min) | Max Elevation: {max_el:.1f}°"
            info_label = QLabel(info_text)
            info_label.setStyleSheet("font-weight: bold; font-size: 20px; padding: 5px; background-color: #2a2a2a; color: white;")
            layout.addWidget(info_label)
            
            # Create figure with GridSpec for flexible layout
            self.ss_wf_fig = Figure(figsize=(14, 9), dpi=100)
            
            # Use GridSpec: 2 rows, 2 columns
            # Left column (col 0): waterfall (top) and elevation/doppler (bottom)
            # Right column (col 1): polar trajectory plot (spans both rows)
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(2, 2, figure=self.ss_wf_fig, width_ratios=[2, 1], height_ratios=[1, 1],
                         hspace=0.3, wspace=0.25)
            
            # Main waterfall plot (top-left)
            self.ss_wf_ax = self.ss_wf_fig.add_subplot(gs[0, 0])
            
            # Elevation/Doppler plot (bottom-left)
            self.ss_info_ax = self.ss_wf_fig.add_subplot(gs[1, 0])
            
            # Polar trajectory plot (right side, spans both rows)
            self.ss_polar_ax = self.ss_wf_fig.add_subplot(gs[:, 1], projection='polar')
            
            # Plot waterfall
            freq_start_ghz = self.ss_wf_freq_start_hz / 1e9
            freq_end_ghz = self.ss_wf_freq_end_hz / 1e9
            self.ss_wf_extent = [freq_start_ghz, freq_end_ghz, pass_duration_min, 0]
            
            # Find valid range for color scaling
            valid_data = self.single_sat_waterfall_data[self.single_sat_waterfall_data < 200]
            if len(valid_data) > 0:
                vmin = np.min(valid_data)
                vmax = vmin + 15
            else:
                vmin, vmax = 170, 185
            
            self.ss_wf_im = self.ss_wf_ax.imshow(self.single_sat_waterfall_data, aspect='auto', 
                                                  extent=self.ss_wf_extent, cmap='jet_r', 
                                                  vmin=vmin, vmax=vmax, interpolation='bilinear')
            
            self.ss_wf_cbar = self.ss_wf_fig.colorbar(self.ss_wf_im, ax=self.ss_wf_ax, pad=0.02)
            self.ss_wf_cbar.set_label('Path Loss (dB)', fontsize=10)
            
            # Mark TX frequency
            self.ss_wf_ax.axvline(x=self.ss_wf_tx_ghz, color='white', linestyle='--', linewidth=1, alpha=0.7)
            
            # Current time marker (horizontal line)
            self.ss_current_time_line = self.ss_wf_ax.axhline(y=0, color='lime', linestyle='-', linewidth=2, alpha=0.9)
            
            self.ss_wf_ax.set_xlabel('Frequency (GHz)', fontsize=11)
            self.ss_wf_ax.set_ylabel('Time into pass (min)', fontsize=11)
            self.ss_wf_ax.set_title(f'Doppler Waterfall - Complete Pass\nTX: {self.ss_wf_tx_ghz:.3f} GHz', fontsize=11)
            
            # Format x-axis
            from matplotlib.ticker import FuncFormatter
            self.ss_wf_ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.4f}'))
            
            # Plot elevation and Doppler
            time_axis = np.linspace(0, pass_duration_min, self.ss_wf_n_time_samples)
            
            ax2 = self.ss_info_ax
            color1 = 'tab:blue'
            ax2.set_xlabel('Time into pass (min)', fontsize=11)
            ax2.set_ylabel('Elevation (°)', color=color1, fontsize=11)
            ax2.plot(time_axis, self.ss_elevation_data, color=color1, linewidth=2, label='Elevation')
            ax2.axhline(y=self.ss_wf_elevation_mask, color=color1, linestyle=':', alpha=0.5, label=f'Mask ({self.ss_wf_elevation_mask}°)')
            ax2.tick_params(axis='y', labelcolor=color1)
            ax2.set_ylim(0, max(self.ss_elevation_data) * 1.1 + 5)
            
            # Doppler on secondary y-axis
            ax3 = ax2.twinx()
            color2 = 'tab:red'
            ax3.set_ylabel('Doppler Shift (kHz)', color=color2, fontsize=11)
            ax3.plot(time_axis, self.ss_doppler_data, color=color2, linewidth=2, label='Doppler')
            ax3.axhline(y=0, color=color2, linestyle=':', alpha=0.5)
            ax3.tick_params(axis='y', labelcolor=color2)
            
            # Current time marker for info plot
            self.ss_info_time_line = ax2.axvline(x=0, color='lime', linestyle='-', linewidth=2, alpha=0.9)
            
            # Current position markers
            self.ss_el_marker, = ax2.plot([0], [self.ss_elevation_data[0] if self.ss_elevation_data else 0], 
                                          'o', color=color1, markersize=10, zorder=5)
            self.ss_doppler_marker, = ax3.plot([0], [self.ss_doppler_data[0] if self.ss_doppler_data else 0], 
                                               'o', color=color2, markersize=10, zorder=5)
            
            ax2.set_title('Elevation & Doppler Profile', fontsize=11)
            ax2.grid(True, alpha=0.3)
            
            # === Polar trajectory plot ===
            self.setup_trajectory_polar_plot()
            
            self.ss_wf_fig.tight_layout()
            
            self.ss_wf_canvas = FigureCanvas(self.ss_wf_fig)
            layout.addWidget(self.ss_wf_canvas)
            
            toolbar = NavigationToolbar(self.ss_wf_canvas, central)
            layout.addWidget(toolbar)
            
            # Status label for current time
            self.ss_status_label = QLabel("Initializing...")
            self.ss_status_label.setStyleSheet("font-size: 11px; padding: 3px;")
            layout.addWidget(self.ss_status_label)
            
            # Close button
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(self.stop_single_satellite_waterfall)
            layout.addWidget(close_btn)
            
            self.single_sat_waterfall_window.show()
            
            # Connect close event
            self.single_sat_waterfall_window.closeEvent = lambda e: self.stop_single_satellite_waterfall()
            
            # Store pass duration for updates
            self.ss_pass_duration_min = pass_duration_min
            self.ss_pass_duration_sec = pass_duration_sec
            
            # Start update timer to move the current time marker
            self.single_sat_waterfall_timer = QTimer()
            self.single_sat_waterfall_timer.timeout.connect(self.update_single_satellite_waterfall)
            self.single_sat_waterfall_timer.start(1000)  # Update every second
            
            # Initial update
            self.update_single_satellite_waterfall()
            
            self.status_bar.showMessage(f"Showing full pass waterfall for {self.selected_sat_name}")
        
        def setup_trajectory_polar_plot(self):
            """Set up the polar plot showing the satellite's full trajectory across the sky."""
            ax = self.ss_polar_ax
            
            # Configure polar plot (same style as main sky map)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_ylim(0, 90)
            ax.set_rscale('linear')
            ax.grid(True, alpha=0.4, linestyle='--')
            
            # Set ticks
            ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
            ax.set_yticklabels(['90°', '75°', '60°', '45°', '30°', '15°', '0°'], fontsize=8)
            ax.set_rlabel_position(22.5)
            
            # Azimuth labels
            az_deg = np.array([0, 45, 90, 135, 180, 225, 270, 315])
            az_rad = np.radians(az_deg)
            ax.set_xticks(az_rad)
            ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=10, fontweight='bold')
            
            # Draw elevation mask circle
            mask_r = 90 - self.ss_wf_elevation_mask
            theta_circle = np.linspace(0, 2*np.pi, 100)
            ax.plot(theta_circle, [mask_r]*100, 'r--', linewidth=1, alpha=0.5, label=f'Elev mask ({self.ss_wf_elevation_mask}°)')
            
            # Plot zenith marker (center)
            ax.plot(0, 0, 'r+', markersize=12, markeredgewidth=2, zorder=10)
            
            # Convert trajectory data to polar coordinates
            thetas = np.radians(self.ss_azimuth_data)
            rs = 90 - np.array(self.ss_elevation_data)
            
            # Plot the full trajectory with color gradient (time)
            # Use scatter for color-coded trajectory
            colors = np.linspace(0, 1, len(thetas))
            scatter = ax.scatter(thetas, rs, c=colors, cmap='cool', s=15, alpha=0.7, zorder=3)
            
            # Plot trajectory line
            ax.plot(thetas, rs, 'c-', linewidth=1.5, alpha=0.5, zorder=2)
            
            # Mark start (rise) and end (set) points
            if len(thetas) > 0:
                ax.scatter([thetas[0]], [rs[0]], c='lime', s=100, marker='^', 
                          edgecolors='white', linewidths=1, zorder=5, label='Rise')
                ax.scatter([thetas[-1]], [rs[-1]], c='red', s=100, marker='v', 
                          edgecolors='white', linewidths=1, zorder=5, label='Set')
                
                # Mark max elevation point
                max_el_idx = np.argmax(self.ss_elevation_data)
                ax.scatter([thetas[max_el_idx]], [rs[max_el_idx]], c='yellow', s=120, marker='*', 
                          edgecolors='black', linewidths=0.5, zorder=6, label=f'Max El ({max(self.ss_elevation_data):.1f}°)')
            
            # Current position marker (will be updated)
            self.ss_trajectory_marker = ax.scatter([thetas[0] if len(thetas) > 0 else 0], 
                                                    [rs[0] if len(rs) > 0 else 90], 
                                                    c='white', s=150, marker='o', 
                                                    edgecolors='lime', linewidths=3, zorder=7)
            
            ax.set_title('Sky Trajectory', fontsize=11, pad=10)
            ax.legend(loc='upper left', bbox_to_anchor=(-0.15, 1.15), fontsize=8)
        
        def update_single_satellite_waterfall(self):
            """Update the current time marker on the pre-computed waterfall."""
            if not self.single_sat_waterfall_running or self.selected_predictor is None:
                return
            
            from skyfield.api import utc
            
            current_time_utc = datetime.utcnow().replace(tzinfo=utc)
            
            # Calculate position in the pass
            time_since_rise = (current_time_utc - self.ss_rise_time).total_seconds()
            time_into_pass_min = time_since_rise / 60.0
            
            # Clamp to pass duration
            time_into_pass_min = max(0, min(time_into_pass_min, self.ss_pass_duration_min))
            
            # Update current time line on waterfall
            self.ss_current_time_line.set_ydata([time_into_pass_min, time_into_pass_min])
            
            # Update current time line on info plot
            self.ss_info_time_line.set_xdata([time_into_pass_min, time_into_pass_min])
            
            # Find nearest data index
            idx = int((time_into_pass_min / self.ss_pass_duration_min) * (self.ss_wf_n_time_samples - 1))
            idx = max(0, min(idx, len(self.ss_elevation_data) - 1))
            
            # Update markers
            self.ss_el_marker.set_data([time_into_pass_min], [self.ss_elevation_data[idx]])
            self.ss_doppler_marker.set_data([time_into_pass_min], [self.ss_doppler_data[idx]])
            
            # Update trajectory marker on polar plot
            if hasattr(self, 'ss_trajectory_marker') and idx < len(self.ss_azimuth_data):
                theta = np.radians(self.ss_azimuth_data[idx])
                r = 90 - self.ss_elevation_data[idx]
                self.ss_trajectory_marker.set_offsets([[theta, r]])
            
            # Update status
            current_time_str = current_time_utc.astimezone().strftime('%H:%M:%S %Z')
            el = self.ss_elevation_data[idx]
            az = self.ss_azimuth_data[idx] if idx < len(self.ss_azimuth_data) else 0
            doppler = self.ss_doppler_data[idx]
            
            if time_since_rise < 0:
                status = f"Pass starts in {-time_since_rise:.0f} sec"
            elif time_since_rise > self.ss_pass_duration_sec:
                status = f"Pass ended {time_since_rise - self.ss_pass_duration_sec:.0f} sec ago"
            else:
                status = f"LIVE | El: {el:.1f}° Az: {az:.1f}° | Doppler: {doppler:+.1f} kHz"
            
            self.ss_status_label.setText(f"{current_time_str} | {status}")
            
            self.ss_wf_canvas.draw_idle()
        
        def stop_single_satellite_waterfall(self):
            """Stop the single satellite waterfall."""
            self.single_sat_waterfall_running = False
            if self.single_sat_waterfall_timer:
                self.single_sat_waterfall_timer.stop()
                self.single_sat_waterfall_timer = None
            
            # Clear selected satellite marker
            if hasattr(self, 'selected_marker'):
                self.selected_marker.set_offsets(np.c_[[], []])
                self.canvas.draw_idle()
            
            self.selected_predictor = None
            self.selected_sat_name = None
            
            self.status_bar.showMessage("Single satellite waterfall stopped.")
                
        def save_figure(self):
            """Save figure to file."""
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Save Figure", "", "PNG files (*.png);;PDF files (*.pdf)"
            )
            
            if filepath:
                try:
                    self.fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                    self.status_bar.showMessage(f"Saved to {os.path.basename(filepath)}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save: {e}")
                    
        def closeEvent(self, event):
            """Handle window close."""
            self.stop_animation()
            self.stop_waterfall()
            self.stop_single_satellite_waterfall()
            event.accept()
    
    # Run the application
    app = QApplication(sys.argv)
    window = DopplerPredictorGUI()
    window.show()
    sys.exit(app.exec_())


def main():
    """Main entry point - try different GUI backends."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Starlink Doppler Predictor')
    parser.add_argument('--terminal', action='store_true', 
                       help='Force terminal mode (data generation only)')
    parser.add_argument('--no-gui', action='store_true',
                       help='Same as --terminal')
    args = parser.parse_args()
    
    # Force terminal mode if requested
    if args.terminal or args.no_gui:
        print("Starting in terminal mode...")
        terminal_data_generation()
        return
    
    # Check for display availability
    import os
    if not os.environ.get('DISPLAY') and sys.platform != 'darwin' and sys.platform != 'win32':
        print("No display detected. Starting in terminal mode...")
        terminal_data_generation()
        return
    
    print("Starting Doppler Predictor GUI...")
    
    # Try PyQt5 first
    try:
        try_pyqt5()
        return
    except ImportError as e:
        print(f"PyQt5 not available: {e}")
    except Exception as e:
        print(f"PyQt5 failed: {e}")
    
    # Fallback to terminal UI
    print("\nFalling back to terminal interface...")
    terminal_data_generation()


def terminal_data_generation():
    """Terminal-based data generation interface for headless environments."""
    print("\n" + "="*60)
    print("DOPPLER PREDICTOR - DATA GENERATION MODE (Terminal)")
    print("="*60)
    
    # Load TLE data
    print("\n[1/8] TLE Data")
    default_tle = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'starlink.txt')
    if os.path.exists(default_tle):
        tle_path = input(f"TLE file path (press Enter for '{default_tle}'): ").strip()
        if not tle_path:
            tle_path = default_tle
    else:
        tle_path = input("TLE file path: ").strip()
    
    if not os.path.exists(tle_path):
        print(f"Error: TLE file not found: {tle_path}")
        return
    
    with open(tle_path, 'r') as f:
        tle_data = f.read()
    print(f"✓ Loaded TLE data from {tle_path}")
    
    # Ground station location
    print("\n[2/8] Ground Station Location")
    lat = float(input("Latitude (°N) [default: 47.6550]: ").strip() or "47.6550")
    lon = float(input("Longitude (°E) [default: -122.3035]: ").strip() or "-122.3035")
    alt = float(input("Altitude (m) [default: 60]: ").strip() or "60")
    
    ue_location = {
        'latitude': lat,
        'longitude': lon,
        'altitude': alt
    }
    print(f"✓ Ground station: {lat}°N, {lon}°E, {alt}m")
    
    # Simulation settings
    print("\n[3/8] Simulation Time Window")
    start_time_str = input("Start time (UTC, YYYY-MM-DD HH:MM:SS) [press Enter for now]: ").strip()
    if start_time_str:
        from datetime import datetime
        from skyfield.api import utc
        start_time_utc = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=utc)
    else:
        from datetime import datetime
        from skyfield.api import utc
        start_time_utc = datetime.utcnow().replace(tzinfo=utc)
    print(f"✓ Start time: {start_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    print("\n[4/8] Simulation Duration")
    duration_min = float(input("Duration (minutes) [default: 1]: ").strip() or "1")
    time_step_sec = float(input("Time step (seconds) [default: 10]: ").strip() or "10")
    print(f"✓ Duration: {duration_min} min, Time step: {time_step_sec} sec")
    
    print("\n[5/8] Satellite Parameters")
    elevation_mask = float(input("Elevation mask (degrees) [default: 10.0]: ").strip() or "10.0")
    num_sats = int(input("Max satellites to process [default: 100]: ").strip() or "100")
    print(f"✓ Elevation mask: {elevation_mask}°, Max satellites: {num_sats}")
    
    # Use default carrier frequency
    carrier_freq_ghz = 10.5
    carrier_freq_hz = carrier_freq_ghz * 1e9
    print(f"✓ Using carrier frequency: {carrier_freq_ghz} GHz (Starlink default)")
    
    # Output directory
    print("\n[6/8] Output Settings")
    output_dir = input("Output directory [default: current directory]: ").strip() or "."
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"✓ Output directory: {os.path.abspath(output_dir)}")
    
    # Create predictor
    print("\n[7/8] Loading Satellites...")
    predictor = MultiSatellitePredictor(
        tle_data, ue_location,
        num_satellites=num_sats,
        tx_freq_hz=carrier_freq_hz
    )
    print(f"✓ Loaded {len(predictor.predictors)} satellites")
    
    # Generate data
    print("\n[8/8] Generating Data...")
    from datetime import timedelta
    
    num_samples = int((duration_min * 60) / time_step_sec)
    time_points = [start_time_utc + timedelta(seconds=i*time_step_sec) for i in range(num_samples)]
    
    generated_data = []
    satellite_data_dict = {}
    
    for idx, current_time in enumerate(time_points):
        progress = int((idx / num_samples) * 100)
        print(f"\rProgress: [{progress:3d}%] {idx+1}/{num_samples} time points", end='', flush=True)
        
        for pred in predictor.predictors:
            try:
                ts_time = pred.ts.from_datetime(current_time)
                observer_location = pred.wgs84.latlon(
                    pred.ue_lat, pred.ue_lon, elevation_m=pred.ue_alt * 1000
                )
                
                relative = (pred.satellite - observer_location).at(ts_time)
                
                # Use Skyfield's proper altaz() method
                alt, az, distance = relative.altaz()
                elevation_deg = alt.degrees
                azimuth_deg = az.degrees
                distance_km = distance.km
                
                if elevation_deg >= elevation_mask:
                    doppler_hz = pred.calculate_doppler_shift(current_time)
                    rx_freq_hz = pred.STARLINK_TX_FREQ + doppler_hz
                    
                    # Calculate relative velocity from Doppler
                    c = 299792458  # m/s
                    velocity_ms = -doppler_hz * c / carrier_freq_hz
                    velocity_kms = velocity_ms / 1000  # km/s
                    
                    # Validate realistic values for LEO satellites
                    # Typical Starlink orbit: 500-600 km altitude, so distance 500-5000 km
                    # (max slant range at 10° elevation for 600km altitude satellite)
                    # Typical relative velocity: -8 to +8 km/s
                    if distance_km > 5000 or distance_km < 0:
                        # Skip satellites with unrealistic distance (likely bad TLE data)
                        continue
                    if abs(velocity_kms) > 20:
                        # Skip satellites with unrealistic velocity
                        continue
                    
                    data_point = {
                        'timestamp': current_time.isoformat(),
                        'satellite': pred.sat_name,
                        'azimuth_deg': azimuth_deg,
                        'elevation_deg': elevation_deg,
                        'distance_km': distance_km,
                        'relative_velocity_kms': velocity_kms,
                        'doppler_shift_hz': doppler_hz,
                        'tx_freq_ghz': carrier_freq_ghz,
                        'rx_freq_hz': rx_freq_hz,
                        'ue_lat': ue_location['latitude'],
                        'ue_lon': ue_location['longitude'],
                        'ue_alt_m': ue_location['altitude'],
                        'time_minutes': (current_time - start_time_utc).total_seconds() / 60.0
                    }
                    
                    generated_data.append(data_point)
                    
                    if pred.sat_name not in satellite_data_dict:
                        satellite_data_dict[pred.sat_name] = []
                    satellite_data_dict[pred.sat_name].append(data_point)
            except:
                continue
    
    print(f"\n✓ Generated {len(generated_data)} data points from {len(satellite_data_dict)} visible satellites")
    
    # Save data
    print("\nSaving data and generating plots...")
    import csv
    from datetime import datetime
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create satellite directory
    sat_dir = os.path.join(output_dir, f"satellites_{timestamp}")
    os.makedirs(sat_dir, exist_ok=True)
    
    # Save settings in the satellite directory
    settings_file = os.path.join(sat_dir, "settings.txt")
    with open(settings_file, 'w') as f:
        f.write("Doppler Predictor - Simulation Settings\n")
        f.write("="*60 + "\n\n")
        f.write(f"TLE File: {tle_path}\n")
        f.write(f"Start Time (UTC): {start_time_utc.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration (min): {duration_min}\n")
        f.write(f"Time Step (sec): {time_step_sec}\n")
        f.write(f"Carrier Frequency (GHz): {carrier_freq_ghz}\n")
        f.write(f"Elevation Mask (deg): {elevation_mask}\n")
        f.write(f"Max Satellites: {num_sats}\n")
        f.write(f"Ground Station Latitude (deg): {lat}\n")
        f.write(f"Ground Station Longitude (deg): {lon}\n")
        f.write(f"Ground Station Altitude (m): {alt}\n")
        f.write(f"\nResults:\n")
        f.write(f"Total Data Points: {len(generated_data)}\n")
        f.write(f"Visible Satellites: {len(satellite_data_dict)}\n")
    print(f"✓ Saved settings: {settings_file}")
    
    # Process each satellite
    total_sats = len(satellite_data_dict)
    for idx, (sat_name, sat_data) in enumerate(satellite_data_dict.items(), 1):
        print(f"\r[{idx}/{total_sats}] Processing {sat_name}...", end='', flush=True)
        
        safe_name = sat_name.replace('/', '_').replace(' ', '_')
        
        # Save CSV
        sat_csv = os.path.join(sat_dir, f"{safe_name}.csv")
        with open(sat_csv, 'w', newline='') as csvfile:
            if sat_data:
                fieldnames = sat_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(sat_data)
        
        # Extract data for plotting
        time_minutes = np.array([d['time_minutes'] for d in sat_data])
        doppler_hz = np.array([d['doppler_shift_hz'] for d in sat_data])
        elevation_deg = np.array([d['elevation_deg'] for d in sat_data])
        azimuth_deg = np.array([d['azimuth_deg'] for d in sat_data])
        distance_km = np.array([d['distance_km'] for d in sat_data])
        rx_freq_hz = np.array([d['rx_freq_hz'] for d in sat_data])
        
        # Convert Doppler to velocity
        c = 299792458  # m/s
        velocity_ms = -doppler_hz * c / carrier_freq_hz
        velocity_kms = velocity_ms / 1000  # km/s
        
        # Create figure with 2 subplots
        fig = plt.figure(figsize=(14, 10), dpi=100)
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 1, figure=fig, height_ratios=[1.2, 1], hspace=0.3)
        
        # Waterfall plot
        ax_waterfall = fig.add_subplot(gs[0])
        
        velocity_range_kms = 8.0
        velocity_start_kms = -velocity_range_kms
        velocity_end_kms = velocity_range_kms
        n_velocity_bins = 300
        n_time_samples = len(time_minutes)
        
        velocity_grid_kms = np.linspace(velocity_start_kms, velocity_end_kms, n_velocity_bins)
        
        # Initialize with constant noise floor (no signal)
        noise_floor_db = 200.0
        waterfall_data = np.full((n_time_samples, n_velocity_bins), noise_floor_db)
        
        sigma_kms = 0.05
        for i, (vel_kms, dist_km) in enumerate(zip(velocity_kms, distance_km)):
            if dist_km > 0:
                dist_m = dist_km * 1000
                # Calculate actual Free Space Path Loss
                fspl_db = 20 * np.log10(dist_m) + 20 * np.log10(carrier_freq_hz) + 20 * np.log10(4 * np.pi / 299792458)
                
                # Only show path loss at satellite's actual velocity (Gaussian peak)
                gaussian = np.exp(-0.5 * ((velocity_grid_kms - vel_kms) / sigma_kms) ** 2)
                # Background = noise_floor, Signal at satellite velocity = actual path loss
                waterfall_data[i, :] = noise_floor_db - (noise_floor_db - fspl_db) * gaussian
        
        # Use actual data duration instead of configured duration
        actual_duration_min = (time_minutes[-1] - time_minutes[0]) if len(time_minutes) > 0 else duration_min
        extent = [velocity_start_kms, velocity_end_kms, actual_duration_min, 0]
        # Set colorbar range based on actual signal values (exclude noise floor)
        signal_data = waterfall_data[waterfall_data < noise_floor_db - 1]
        if len(signal_data) > 0:
            vmin, vmax = np.percentile(signal_data, [5, 95])
        else:
            vmin, vmax = 170, 185
        
        im = ax_waterfall.imshow(waterfall_data, aspect='auto', extent=extent,
                                cmap='jet_r', vmin=vmin, vmax=vmax, interpolation='bilinear')
        ax_waterfall.axvline(x=0, color='white', linestyle='--', linewidth=1.5, alpha=0.8, label='Zero velocity')
        ax_waterfall.set_xlabel('Relative Velocity (km/s) [- approaching, + receding]', fontsize=11)
        ax_waterfall.set_ylabel('Time into pass (min)', fontsize=11)
        ax_waterfall.set_title(f'Velocity Waterfall - {sat_name}', fontsize=12)
        ax_waterfall.legend(loc='upper right', fontsize=9)
        fig.colorbar(im, ax=ax_waterfall, label='Path Loss (dB)')
        
        # Trajectory plot
        ax_traj = fig.add_subplot(gs[1], projection='polar')
        ax_traj.set_theta_zero_location('N')
        ax_traj.set_theta_direction(-1)
        ax_traj.set_ylim(0, 90)
        ax_traj.set_yticks([0, 15, 30, 45, 60, 75, 90])
        ax_traj.set_yticklabels(['90°', '75°', '60°', '45°', '30°', '15°', '0°'], fontsize=8)
        ax_traj.set_rlabel_position(22.5)
        
        az_deg_ticks = np.array([0, 45, 90, 135, 180, 225, 270, 315])
        az_rad_ticks = np.radians(az_deg_ticks)
        ax_traj.set_xticks(az_rad_ticks)
        ax_traj.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=10, fontweight='bold')
        ax_traj.grid(True, alpha=0.4, linestyle='--')
        
        mask_r = 90 - elevation_mask
        theta_circle = np.linspace(0, 2*np.pi, 100)
        ax_traj.plot(theta_circle, [mask_r]*100, 'r--', linewidth=1, alpha=0.5, label=f'Elev mask ({elevation_mask}°)')
        ax_traj.plot(0, 0, 'r+', markersize=12, markeredgewidth=2, zorder=10)
        
        thetas = np.radians(azimuth_deg)
        rs = 90 - elevation_deg
        colors = np.linspace(0, 1, len(thetas))
        ax_traj.scatter(thetas, rs, c=colors, cmap='cool', s=20, alpha=0.7, zorder=3)
        ax_traj.plot(thetas, rs, 'c-', linewidth=1.5, alpha=0.5, zorder=2)
        
        if len(thetas) > 0:
            ax_traj.scatter([thetas[0]], [rs[0]], c='green', s=150, marker='^', 
                          edgecolors='white', linewidths=2, zorder=6, label='Rise')
            ax_traj.scatter([thetas[-1]], [rs[-1]], c='red', s=150, marker='v',
                          edgecolors='white', linewidths=2, zorder=6, label='Set')
        
        ax_traj.set_title(f'Sky Trajectory - {sat_name}', fontsize=12, pad=10)
        ax_traj.legend(loc='upper left', bbox_to_anchor=(-0.25, 1.15), fontsize=8)
        
        # Save figure
        fig.tight_layout()
        plot_file = os.path.join(sat_dir, f"{safe_name}_plot.png")
        fig.savefig(plot_file, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    print(f"\n✓ Saved {len(satellite_data_dict)} satellites to: {sat_dir}/")
    print(f"  - CSV data files")
    print(f"  - Waterfall + trajectory plots")
    print(f"  - Settings file")
    
    print("\n" + "="*60)
    print("DATA GENERATION COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {os.path.abspath(sat_dir)}")


if __name__ == "__main__":
    main()
