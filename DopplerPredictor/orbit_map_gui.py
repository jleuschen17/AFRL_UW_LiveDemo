"""
Orbit Map GUI - 2D and 3D Visualization
Displays satellite orbits with 6 Keplerian parameters and user location input.
Uses PyQt5 for GUI, Matplotlib for 2D map, and PyVista for 3D visualization.
"""

import sys
import numpy as np
import warnings
import os
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QSlider, QPushButton, QTabWidget, QGroupBox,
    QFormLayout, QCheckBox, QSpinBox, QDoubleSpinBox, QDateTimeEdit,
    QGridLayout, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, QDateTime
from PyQt5.QtGui import QFont, QValidator, QDoubleValidator

# Optional Numba acceleration (best-effort; falls back cleanly if unavailable)
try:
    import numba
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    # Fallback decorator and prange alias so code can call the same symbols when numba is absent
    def njit(func=None, **kwargs):
        def decorator(f):
            return f
        if func is None:
            return decorator
        else:
            return func
    prange = range

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import pyvista as pv
from pyvistaqt import QtInteractor
try:
    from PIL import Image
    import urllib.request
    from io import BytesIO
except ImportError:
    print("PIL not available - using simplified Earth rendering")

warnings.filterwarnings('ignore')


class OrbitalMechanics:
    """Calculate orbital positions using Keplerian elements"""
    
    # Constants
    MU = 3.986004418e14  # m^3/s^2 - Earth's gravitational parameter
    EARTH_RADIUS = 6.371e6  # meters
    DEG_TO_RAD = np.pi / 180.0
    RAD_TO_DEG = 180.0 / np.pi
    
    @staticmethod
    def kepler_to_cartesian(a, e, i, Omega, omega, nu):
        """
        Convert Keplerian elements to Cartesian coordinates (ECI frame).
        
        Args:
            a: Semi-major axis (m)
            e: Eccentricity
            i: Inclination (radians)
            Omega: Right Ascension of Ascending Node (radians)
            omega: Argument of Perigee (radians)
            nu: True Anomaly (radians)
        
        Returns:
            position [x, y, z] in meters (ECI frame)
        """
        # Distance from focus
        r = a * (1 - e**2) / (1 + e * np.cos(nu))
        
        # Perifocal coordinates
        x_peri = r * np.cos(nu)
        y_peri = r * np.sin(nu)
        z_peri = 0
        
        # Rotation matrices
        cos_Omega = np.cos(Omega)
        sin_Omega = np.sin(Omega)
        cos_omega = np.cos(omega)
        sin_omega = np.sin(omega)
        cos_i = np.cos(i)
        sin_i = np.sin(i)
        
        # Transform from perifocal to ECI
        x = (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i) * x_peri + \
            (-cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i) * y_peri
        y = (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i) * x_peri + \
            (-sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i) * y_peri
        z = sin_i * sin_omega * x_peri + sin_i * cos_omega * y_peri
        
        return np.array([x, y, z])
    
    @staticmethod
    def generate_orbit(a, e, i, Omega, omega, M0, num_points=200):
        """
        Generate orbit points around Earth (vectorized).
        Uses a Numba-accelerated implementation when available for heavy workloads.
        """
        if NUMBA_AVAILABLE:
            return _generate_orbit_numba(a, e, i, Omega, omega, num_points)

        nu = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        
        # Distance from focus (vectorized)
        r = a * (1 - e**2) / (1 + e * np.cos(nu))
        
        # Perifocal coordinates
        x_peri = r * np.cos(nu)
        y_peri = r * np.sin(nu)
        
        # Rotation constants
        cos_O = np.cos(Omega)
        sin_O = np.sin(Omega)
        cos_w = np.cos(omega)
        sin_w = np.sin(omega)
        cos_i = np.cos(i)
        sin_i = np.sin(i)
        
        # Transform from perifocal to ECI (vectorized)
        x = (cos_O * cos_w - sin_O * sin_w * cos_i) * x_peri + \
            (-cos_O * sin_w - sin_O * cos_w * cos_i) * y_peri
        y = (sin_O * cos_w + cos_O * sin_w * cos_i) * x_peri + \
            (-sin_O * sin_w + cos_O * cos_w * cos_i) * y_peri
        z = sin_i * sin_w * x_peri + sin_i * cos_w * y_peri
        
        return np.column_stack([x, y, z])


# -- Numba-accelerated helpers (optional) ---------------------------------
if NUMBA_AVAILABLE:
    @njit
    def _kepler_to_cartesian_numba(a, e, i, Omega, omega, nu):
        r = a * (1.0 - e * e) / (1.0 + e * np.cos(nu))
        x_peri = r * np.cos(nu)
        y_peri = r * np.sin(nu)
        cos_O = np.cos(Omega)
        sin_O = np.sin(Omega)
        cos_w = np.cos(omega)
        sin_w = np.sin(omega)
        cos_i = np.cos(i)
        sin_i = np.sin(i)
        x = (cos_O * cos_w - sin_O * sin_w * cos_i) * x_peri + (-cos_O * sin_w - sin_O * cos_w * cos_i) * y_peri
        y = (sin_O * cos_w + cos_O * sin_w * cos_i) * x_peri + (-sin_O * sin_w + cos_O * cos_w * cos_i) * y_peri
        z = sin_i * sin_w * x_peri + sin_i * cos_w * y_peri
        out = np.empty(3)
        out[0] = x
        out[1] = y
        out[2] = z
        return out

    @njit(parallel=True)
    def _generate_orbit_numba(a, e, i, Omega, omega, num_points):
        arr = np.empty((num_points, 3))
        two_pi = 2.0 * np.pi
        for j in prange(num_points):
            nu = j * two_pi / num_points
            arr[j, :] = _kepler_to_cartesian_numba(a, e, i, Omega, omega, nu)
        return arr

    @njit(parallel=True)
    def _compute_elevations_numba(a, e, i, Omega, omega, M0, user_pos, time_points_search, T):
        n = time_points_search.shape[0]
        out = np.empty(n)
        user_norm = np.sqrt(user_pos[0]*user_pos[0] + user_pos[1]*user_pos[1] + user_pos[2]*user_pos[2])
        lu0 = user_pos[0]/user_norm; lu1 = user_pos[1]/user_norm; lu2 = user_pos[2]/user_norm
        two_pi = 2.0 * np.pi
        for idx in prange(n):
            t = time_points_search[idx]
            M = M0 + (t / T) * two_pi
            pos = _kepler_to_cartesian_numba(a, e, i, Omega, omega, M)
            to0 = pos[0] - user_pos[0]; to1 = pos[1] - user_pos[1]; to2 = pos[2] - user_pos[2]
            dist = np.sqrt(to0*to0 + to1*to1 + to2*to2)
            to0 /= dist; to1 /= dist; to2 /= dist
            dot = to0*lu0 + to1*lu1 + to2*lu2
            if dot > 1.0:
                dot = 1.0
            elif dot < -1.0:
                dot = -1.0
            out[idx] = np.degrees(np.arcsin(dot))
        return out
else:
    # Fallback Python implementations (invoked when Numba is unavailable)
    def _generate_orbit_numba(a, e, i, Omega, omega, num_points):
        return OrbitalMechanics.generate_orbit(a, e, i, Omega, omega, 0, num_points)

    def _compute_elevations_numba(a, e, i, Omega, omega, M0, user_pos, time_points_search, T):
        out = np.empty(time_points_search.shape[0])
        for idx, t in enumerate(time_points_search):
            M = M0 + (t / T) * 2.0 * np.pi
            pos = OrbitalMechanics.kepler_to_cartesian(a, e, i, Omega, omega, M)
            to = pos - user_pos
            dist = np.linalg.norm(to)
            if dist == 0:
                out[idx] = 0.0
                continue
            to_norm = to / dist
            lu = user_pos / np.linalg.norm(user_pos)
            elevation_angle = np.arcsin(np.dot(to_norm, lu))
            out[idx] = np.degrees(elevation_angle)


class OrbitMapGUI(QMainWindow):
    """Main GUI window for orbit visualization"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Satellite Orbit Map Viewer")
        self.setGeometry(100, 100, 1600, 900)
        
        # Initialize parameters with default values
        self.params = {
            'a': 6.96e6,  # Semi-major axis (m) - ~325 km altitude
            'e': 0.0,     # Eccentricity
            'i': 55.0,    # Inclination (degrees)
            'Omega': 0.0, # RAAN (degrees)
            'omega': 0.0, # Argument of Perigee (degrees)
            'M0': 0.0     # Mean Anomaly (degrees)
        }
        
        self.user_location = {
            'lat': 47.6550,
            'lon': -122.3035,
            'alt': 0
        }
        
        # Load Earth texture
        self.earth_texture = self.load_earth_texture()
        self.earth_sphere_cache = None  # Cache for 3D sphere geometry and colors
        self.load_sphere_cache()
        
        # Link budget parameters
        self.carrier_freq = 2.4  # GHz
        
        # Simulation state
        self.simulation_time = datetime.now()
        self.is_playing = False
        self.time_step = 50  # seconds per update
        
        # Store ground track data for click interaction
        self.ground_track_data = []  # List of (lon, lat, time_offset) tuples
        
        # Cache for link plot data (so it doesn't recalculate during animation)
        self.link_plot_cache = None
        self.link_plot_artists = None
        
        # 3D scene: track dynamic label actors for manual removal
        self._3d_label_actors = []

        # Debounce timer for 3D updates (slider dragging causes many rapid calls)
        self._3d_update_timer = QTimer()
        self._3d_update_timer.setSingleShot(True)
        self._3d_update_timer.setInterval(100)  # 100ms debounce for responsiveness on Windows
        self._3d_update_timer.timeout.connect(self._deferred_3d_update)
        # Flag to indicate a slider is actively being dragged
        self._slider_dragging = False
        # Orbit detail levels (points) for interactive vs full rendering
        self._orbit_detail_full = 200
        self._orbit_detail_low = 36
        # Throttle for fast updates (seconds)
        self._fast_update_throttle = 0.05
        self._last_fast_update_time = 0.0

        # Create UI
        self.init_ui()
        
        # Setup simulation timer
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.update_simulation)
        self.sim_timer.setInterval(10)  # Fast interval for responsive pause
        
    def load_earth_texture(self):
        """Load Earth texture image from NASA Blue Marble"""
        texture_file = 'earth_texture.jpg'
        
        try:
            # Check if texture file already exists
            if os.path.exists(texture_file):
                print("Loading Earth texture from cache...")
                img = Image.open(texture_file)
                return np.array(img)
            
            # Download if not cached
            url = "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73751/world.topo.bathy.200407.3x5400x2700.jpg"
            print("Downloading Earth texture (first run only)...")
            
            with urllib.request.urlopen(url, timeout=10) as response:
                image_data = response.read()
            
            # Load and resize to 4K resolution for high quality
            img = Image.open(BytesIO(image_data))
            img = img.resize((3840, 2160), Image.LANCZOS)  # 4K resolution with high-quality resampling
            
            # Save to cache for future runs
            img.save(texture_file, 'JPEG', quality=95)
            print(f"Earth texture saved to {texture_file}")
            
            return np.array(img)
            
        except Exception as e:
            print(f"Could not load Earth texture: {e}")
            print("Using fallback Earth rendering")
            return None
    
    def load_sphere_cache(self):
        """Load precomputed sphere from disk cache"""
        cache_file = 'earth_sphere_cache.npz'
        try:
            if os.path.exists(cache_file):
                print("Loading precomputed sphere from cache...")
                data = np.load(cache_file)
                # Ensure colors are in valid range [0, 1]
                colors = np.clip(data['colors'], 0, 1)
                self.earth_sphere_cache = {
                    'x': data['x'],
                    'y': data['y'],
                    'z': data['z'],
                    'colors': colors
                }
                print("Sphere cache loaded successfully.")
        except Exception as e:
            print(f"Could not load sphere cache: {e}")
            self.earth_sphere_cache = None
    
    def save_sphere_cache(self, x, y, z, colors):
        """Save precomputed sphere to disk cache"""
        cache_file = 'earth_sphere_cache.npz'
        try:
            print("Saving sphere to cache...")
            np.savez_compressed(cache_file, x=x, y=y, z=z, colors=colors)
            print("Sphere cache saved successfully.")
        except Exception as e:
            print(f"Could not save sphere cache: {e}")
        
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Overall layout: top row (controls + 3D) and bottom row (2D map + link plots)
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        # === Top row: Controls on left, 3D view on right ===
        top_layout = QHBoxLayout()
        
        # Left panel - Controls only
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.create_simulation_group())
        controls_layout.addWidget(self.create_parameter_group())
        controls_layout.addWidget(self.create_location_group())
        controls_layout.addStretch()
        
        # Right panel - 3D Visualization with PyVista
        right_layout = QVBoxLayout()
        
        # 3D camera controls
        camera_controls = QHBoxLayout()
        self.home_button = QPushButton("🏠 Home")
        self.home_button.clicked.connect(self.reset_3d_camera)
        self.zoom_in_button = QPushButton("🔍+ Zoom In")
        self.zoom_in_button.clicked.connect(self.zoom_in_3d)
        self.zoom_out_button = QPushButton("🔍- Zoom Out")
        self.zoom_out_button.clicked.connect(self.zoom_out_3d)
        
        camera_controls.addWidget(self.home_button)
        camera_controls.addWidget(self.zoom_in_button)
        camera_controls.addWidget(self.zoom_out_button)
        camera_controls.addStretch()
        
        right_layout.addLayout(camera_controls)
        
        self.plotter = QtInteractor(right_layout.parentWidget())
        self.plotter.set_background('black')
        right_layout.addWidget(self.plotter.interactor, 1)
        
        top_layout.addLayout(controls_layout, 1)
        top_layout.addLayout(right_layout, 2)
        
        # === Bottom row: 2D map + polar/elevation plots side by side ===
        bottom_layout = QHBoxLayout()
        
        # 2D map (left side of bottom row)
        self.fig_2d = Figure(figsize=(6, 3), dpi=100)
        self.canvas_2d = FigureCanvas(self.fig_2d)
        self.canvas_2d.mpl_connect('button_press_event', self.on_2d_map_press)
        self.canvas_2d.mpl_connect('motion_notify_event', self.on_2d_map_drag)
        self.canvas_2d.mpl_connect('button_release_event', self.on_2d_map_release)
        self._dragging_satellite = False
        bottom_layout.addWidget(self.canvas_2d, 1)
        
        # Link plots - polar + elevation/Doppler (right side of bottom row)
        self.fig_link = Figure(figsize=(8, 3), dpi=100)
        self.canvas_link = FigureCanvas(self.fig_link)
        bottom_layout.addWidget(self.canvas_link, 1)
        
        # Initialize static 3D elements (Earth sphere + axis arrows) once
        self.init_3d_scene()
        
        # Draw initial plots
        self.update_plots()

        # Precompile Numba-accelerated functions in background to avoid first-call lag
        if NUMBA_AVAILABLE:
            QTimer.singleShot(100, self._precompile_numba)
        
        main_layout.addLayout(top_layout, 2)
        main_layout.addLayout(bottom_layout, 1)
        
        central_widget.setLayout(main_layout)
    
    def create_simulation_group(self):
        """Create simulation control group"""
        group = QGroupBox("Simulation Control")
        layout = QVBoxLayout()
        
        # Date/Time input
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Simulation Time:"))
        self.time_input = QDateTimeEdit()
        self.time_input.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.time_input.setDateTime(QDateTime.currentDateTime())
        self.time_input.setCalendarPopup(True)
        self.time_input.dateTimeChanged.connect(self.on_time_changed)
        time_layout.addWidget(self.time_input)
        layout.addLayout(time_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.play_button = QPushButton("▶ Play")
        self.play_button.clicked.connect(self.toggle_play)
        self.reset_button = QPushButton("⟲ Reset")
        self.reset_button.clicked.connect(self.reset_time)
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.reset_button)
        layout.addLayout(button_layout)
        
        # Time navigation buttons
        nav_layout = QHBoxLayout()
        self.prev_min_button = QPushButton("◀ -1 min")
        self.prev_min_button.clicked.connect(self.previous_minute)
        self.next_min_button = QPushButton("+1 min ▶")
        self.next_min_button.clicked.connect(self.next_minute)
        nav_layout.addWidget(self.prev_min_button)
        nav_layout.addWidget(self.next_min_button)
        layout.addLayout(nav_layout)
        
        # Time step control
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Time Step (sec):"))
        self.time_step_input = QDoubleSpinBox()
        self.time_step_input.setRange(0.1, 1000)
        self.time_step_input.setValue(self.time_step)
        self.time_step_input.valueChanged.connect(lambda v: setattr(self, 'time_step', v))
        step_layout.addWidget(self.time_step_input)
        layout.addLayout(step_layout)
        
        group.setLayout(layout)
        return group
    
    def create_parameter_group(self):
        """Create orbital parameters input group"""
        group = QGroupBox("Orbital Parameters")
        layout = QGridLayout()
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setColumnStretch(0, 0)  # Label - fixed width
        layout.setColumnStretch(1, 0)  # Input box - fixed width
        layout.setColumnStretch(2, 1)  # Slider - takes all remaining space
        
        # Define parameter specifications
        param_specs = {
            'a': ('Satellite Height (km)', 200, 3600, 50),  # altitude above Earth surface
            'e': ('Eccentricity', 0.0, 1.0, 0.01),
            'i': ('Inclination (°)', 0, 180, 1),
            'Omega': ('RAAN (°)', 0, 360, 1),
            'omega': ('Arg of Perigee (°)', 0, 360, 1),
            'M0': ('Mean Anomaly (°)', 0, 360, 1)
        }
        
        self.param_inputs = {}
        self.param_sliders = {}
        
        for row, (param, (label, min_val, max_val, step)) in enumerate(param_specs.items()):
            # Input box
            if param == 'a':
                # Convert semi-major axis (stored) to height above surface (displayed)
                default = (self.params[param] - OrbitalMechanics.EARTH_RADIUS) / 1000  # height in km
            elif param in ['i', 'Omega', 'omega', 'M0']:
                default = self.params[param]
            else:
                default = self.params[param]
            
            lbl = QLabel(label)
            lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            
            input_box = QDoubleSpinBox()
            input_box.setMinimum(min_val)
            input_box.setMaximum(max_val)
            input_box.setSingleStep(step)
            input_box.setValue(default)
            input_box.setDecimals(2 if param == 'e' else 0)
            # Use minimum width + flexible size policy so the widget adapts to platform font/DPI
            input_box.setMinimumWidth(90)
            input_box.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
            input_box.valueChanged.connect(self.on_param_changed)
            
            self.param_inputs[param] = input_box
            
            # Slider
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(min_val * 100))
            slider.setMaximum(int(max_val * 100))
            slider.setValue(int(default * 100))
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(int((max_val - min_val) * 100 / 10))
            slider.valueChanged.connect(self.on_param_changed)
            # Track dragging start/end to avoid heavy per-update work
            slider.sliderPressed.connect(self._on_slider_pressed)
            slider.sliderReleased.connect(self._on_slider_released)
            
            self.param_sliders[param] = slider
            
            # Add to grid: label | input | slider
            layout.addWidget(lbl, row, 0)
            layout.addWidget(input_box, row, 1)
            layout.addWidget(slider, row, 2)
        
        group.setLayout(layout)
        return group
    
    def create_location_group(self):
        """Create user location input group"""
        group = QGroupBox("User Location")
        layout = QGridLayout()
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)
        
        # Row 0: Latitude | Longitude
        self.lat_input = QDoubleSpinBox()
        self.lat_input.setMinimum(-90)
        self.lat_input.setMaximum(90)
        self.lat_input.setValue(self.user_location['lat'])
        self.lat_input.setDecimals(4)
        self.lat_input.setSingleStep(0.1)
        self.lat_input.valueChanged.connect(self.on_location_changed)
        self.lat_input.setMinimumWidth(100)
        layout.addWidget(QLabel("Lat (°)"), 0, 0)
        layout.addWidget(self.lat_input, 0, 1)
        
        self.lon_input = QDoubleSpinBox()
        self.lon_input.setMinimum(-180)
        self.lon_input.setMaximum(180)
        self.lon_input.setValue(self.user_location['lon'])
        self.lon_input.setDecimals(4)
        self.lon_input.setSingleStep(0.1)
        self.lon_input.valueChanged.connect(self.on_location_changed)
        self.lon_input.setMinimumWidth(100)
        layout.addWidget(QLabel("Lon (°)"), 0, 2)
        layout.addWidget(self.lon_input, 0, 3)
        
        # Row 1: Altitude | Carrier Freq
        self.alt_input = QDoubleSpinBox()
        self.alt_input.setMinimum(0)
        self.alt_input.setMaximum(5000)
        self.alt_input.setValue(self.user_location['alt'])
        self.alt_input.setDecimals(1)
        self.alt_input.setSingleStep(10)
        layout.addWidget(QLabel("Alt (m)"), 1, 0)
        layout.addWidget(self.alt_input, 1, 1)
        
        self.freq_input = QDoubleSpinBox()
        self.freq_input.setMinimum(0.01)
        self.freq_input.setMaximum(100.0)
        self.freq_input.setValue(self.carrier_freq)
        self.freq_input.setDecimals(3)
        self.freq_input.setSingleStep(0.1)
        self.freq_input.setSuffix(" GHz")
        self.freq_input.valueChanged.connect(self.on_freq_changed)
        layout.addWidget(QLabel("Freq"), 1, 2)
        layout.addWidget(self.freq_input, 1, 3)
        
        group.setLayout(layout)
        return group
    
    def set_parameters_enabled(self, enabled):
        """Enable or disable all parameter inputs"""
        for input_box in self.param_inputs.values():
            input_box.setEnabled(enabled)
        for slider in self.param_sliders.values():
            slider.setEnabled(enabled)
    
    def on_param_changed(self):
        """Handle parameter value changes"""
        sender = self.sender()
        
        # Find which parameter was changed and sync its counterpart
        for param in self.param_inputs:
            input_box = self.param_inputs[param]
            slider = self.param_sliders[param]
            
            if sender == input_box:
                # Input box changed → sync slider to match
                slider.blockSignals(True)
                slider.setValue(int(input_box.value() * 100))
                slider.blockSignals(False)
            elif sender == slider:
                # Slider changed → sync input box to match
                input_box.blockSignals(True)
                input_box.setValue(slider.value() / 100.0)
                input_box.blockSignals(False)
        
        # Now read all parameters from input boxes (which are all up-to-date)
        for param, input_box in self.param_inputs.items():
            value = input_box.value()
            if param == 'a':
                # Convert height above surface (input in km) to semi-major axis (stored in m)
                self.params[param] = (value * 1000) + OrbitalMechanics.EARTH_RADIUS
            else:
                self.params[param] = value
        
        # If a slider is being dragged, avoid heavy recalculations on every event.
        if getattr(self, '_slider_dragging', False):
            # Quick visual feedback for slider movement
            self.update_2d_plot()
            # Do a low-detail 3D update immediately for responsiveness
            self.update_3d_plot(num_points=self._orbit_detail_low, full_render=False)
            # Defer heavier updates (full 3D + link plots) to debounce timer
            self._3d_update_timer.start()
            return

        # Full update when not actively dragging
        self.update_2d_plot()
        self.update_link_plots(update_only_current_position=False)
        # Update 3D directly (full detail)
        self.update_3d_plot(num_points=self._orbit_detail_full, full_render=True)
    
    def on_location_changed(self):
        """Handle location changes"""
        self.user_location['lat'] = self.lat_input.value()
        self.user_location['lon'] = self.lon_input.value()
        self.user_location['alt'] = self.alt_input.value()
        self.update_plots()
    
    def on_freq_changed(self):
        """Handle carrier frequency changes"""
        self.carrier_freq = self.freq_input.value()
        self.link_plot_cache = None  # Force recalculation with new frequency
        self.link_plot_artists = None
        self.update_link_plots(update_only_current_position=False)
    
    def reset_3d_camera(self):
        """Reset 3D camera to center on user location"""
        # Calculate user position
        user_lat = np.radians(self.user_location['lat'])
        user_lon = np.radians(self.user_location['lon'])
        user_alt = self.user_location['alt']
        
        user_r = (OrbitalMechanics.EARTH_RADIUS + user_alt) / 1e6
        user_x = user_r * np.cos(user_lat) * np.cos(user_lon)
        user_y = user_r * np.cos(user_lat) * np.sin(user_lon)
        user_z = user_r * np.sin(user_lat)
        
        # Set camera to look at user location from a good viewing distance
        self.plotter.camera_position = [
            (user_x * 3, user_y * 3, user_z * 3),  # Camera position (3x distance from center)
            (user_x, user_y, user_z),               # Focal point (user location)
            (0, 0, 1)                               # View up direction
        ]
        self.plotter.render()
    
    def zoom_in_3d(self):
        """Zoom in on 3D view"""
        self.plotter.camera.zoom(1.2)
        self.plotter.render()
    
    def zoom_out_3d(self):
        """Zoom out on 3D view"""
        self.plotter.camera.zoom(0.8)
        self.plotter.render()
    
    def toggle_play(self):
        """Toggle simulation play/pause"""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_button.setText("⏸ Pause")
            self.sim_timer.start()
            self.set_parameters_enabled(False)
        else:
            self.play_button.setText("▶ Play")
            self.sim_timer.stop()
            self.set_parameters_enabled(True)
    
    def reset_time(self):
        """Reset simulation time to current time"""
        self.simulation_time = datetime.now()
        self.time_input.setDateTime(QDateTime.currentDateTime())
        self.update_plots()
    
    def previous_minute(self):
        """Go back 1 minute"""
        from datetime import timedelta
        self.simulation_time -= timedelta(minutes=1)
        self.time_input.setDateTime(QDateTime(self.simulation_time))
        self.update_plots()
    
    def next_minute(self):
        """Go forward 1 minute"""
        from datetime import timedelta
        self.simulation_time += timedelta(minutes=1)
        self.time_input.setDateTime(QDateTime(self.simulation_time))
        self.update_plots()
    
    def on_time_changed(self):
        """Handle manual time input changes"""
        qt_datetime = self.time_input.dateTime()
        self.simulation_time = qt_datetime.toPyDateTime()
        self.update_plots()
    
    def update_simulation(self):
        """Update simulation time and mean anomaly"""
        from datetime import timedelta
        
        # Advance simulation time
        self.simulation_time += timedelta(seconds=self.time_step)
        self.time_input.setDateTime(QDateTime(self.simulation_time))
        
        # Calculate orbital period (simplified)
        a = self.params['a']
        mu = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
        T = 2 * np.pi * np.sqrt(a**3 / mu)  # Orbital period in seconds
        
        # Update mean anomaly based on time step
        delta_M = (self.time_step / T) * 360  # degrees
        self.params['M0'] = (self.params['M0'] + delta_M) % 360
        
        # Update the input box and slider (block signals to avoid triggering on_param_changed)
        self.param_inputs['M0'].blockSignals(True)
        self.param_inputs['M0'].setValue(self.params['M0'])
        self.param_inputs['M0'].blockSignals(False)
        self.param_sliders['M0'].blockSignals(True)
        self.param_sliders['M0'].setValue(int(self.params['M0'] * 100))
        self.param_sliders['M0'].blockSignals(False)
        
        # Calculate current elevation to check if satellite is visible
        i = np.radians(self.params['i'])
        Omega = np.radians(self.params['Omega'])
        omega = np.radians(self.params['omega'])
        M0 = np.radians(self.params['M0'])
        pos_sat_current = OrbitalMechanics.kepler_to_cartesian(a, self.params['e'], i, Omega, omega, M0)
        
        user_lat = np.radians(self.user_location['lat'])
        user_lon = np.radians(self.user_location['lon'])
        user_alt = self.user_location['alt']
        user_r = OrbitalMechanics.EARTH_RADIUS + user_alt
        user_x = user_r * np.cos(user_lat) * np.cos(user_lon)
        user_y = user_r * np.cos(user_lat) * np.sin(user_lon)
        user_z = user_r * np.sin(user_lat)
        user_pos = np.array([user_x, user_y, user_z])
        
        to_sat_current = pos_sat_current - user_pos
        local_up = user_pos / np.linalg.norm(user_pos)
        to_sat_norm_current = to_sat_current / np.linalg.norm(to_sat_current)
        current_elevation = np.degrees(np.arcsin(np.dot(to_sat_norm_current, local_up)))
        
        # Update 2D plot first (faster)
        self.update_2d_plot()
        
        # Process events to allow pause button to respond
        QApplication.processEvents()
        
        # Update link plots based on satellite visibility
        if current_elevation > 0:
            # Satellite is visible
            if self.link_plot_cache is not None:
                # Try to use cached data for fast update
                self.update_link_plots(update_only_current_position=True)
                # If cache was cleared (outside window), do full recalculation
                if self.link_plot_cache is None:
                    self.update_link_plots(update_only_current_position=False)
            else:
                # No cache, do full calculation
                self.update_link_plots(update_only_current_position=False)
        else:
            # Satellite not visible - clear cache and plots
            if self.link_plot_cache is not None:
                self.link_plot_cache = None
                self.link_plot_artists = None
                # Clear the plots
                self.fig_link.clear()
                gs = GridSpec(1, 2, figure=self.fig_link, width_ratios=[3, 2])
                ax1 = self.fig_link.add_subplot(gs[0], projection='polar')
                ax2 = self.fig_link.add_subplot(gs[1])
                
                ax1.set_theta_zero_location('N')
                ax1.set_theta_direction(-1)
                ax1.set_ylim(0, 90)
                ax1.set_yticks([0, 30, 60, 90])
                ax1.set_yticklabels(['90°', '60°', '30°', '0°'])
                ax1.set_title('Polar Sky View', fontsize=8, fontweight='bold', pad=12)
                ax1.grid(True, alpha=0.3)
                
                ax2.set_xlabel('Time (minutes)', fontsize=8)
                ax2.set_ylabel('Elevation Angle (°)', fontsize=8, color='b')
                ax2.tick_params(axis='y', labelcolor='b')
                ax2.set_title('Elevation & Doppler vs Time', fontsize=8, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                self.fig_link.tight_layout()
                self.canvas_link.draw()
        
        # Only update 3D if still playing (allows pause to be responsive)
        if self.is_playing:
            self.update_3d_plot()
    
    def _find_closest_track_point(self, click_lon, click_lat):
        """Find closest ground track point to the given coordinates.
        Returns (min_dist, relative_time) or (inf, 0) if no data."""
        min_dist = float('inf')
        closest_time = 0
        for lon, lat, time_offset in self.ground_track_data:
            dist = np.sqrt((lon - click_lon)**2 + (lat - click_lat)**2)
            if dist < min_dist:
                min_dist = dist
                closest_time = time_offset
        return min_dist, closest_time

    def _jump_satellite_to_time(self, relative_time):
        """Move the satellite by the given relative time offset."""
        a = self.params['a']
        mu = 3.986004418e14
        T = 2 * np.pi * np.sqrt(a**3 / mu)

        delta_M = (relative_time / T) * 360  # degrees
        self.params['M0'] = (self.params['M0'] + delta_M) % 360

        # Sync input box and slider
        self.param_inputs['M0'].blockSignals(True)
        self.param_inputs['M0'].setValue(self.params['M0'])
        self.param_inputs['M0'].blockSignals(False)
        self.param_sliders['M0'].blockSignals(True)
        self.param_sliders['M0'].setValue(int(self.params['M0'] * 100))
        self.param_sliders['M0'].blockSignals(False)

        # Clear link plot cache
        self.link_plot_cache = None
        self.link_plot_artists = None

    def on_2d_map_press(self, event):
        """Start dragging if click is near the satellite or on the track"""
        if event.inaxes is None or len(self.ground_track_data) == 0:
            return
        min_dist, closest_time = self._find_closest_track_point(event.xdata, event.ydata)
        if min_dist < 20:
            self._dragging_satellite = True
            self._jump_satellite_to_time(closest_time)
            self.update_plots()

    def on_2d_map_drag(self, event):
        """While dragging, continuously move satellite to closest track point"""
        if not self._dragging_satellite or event.inaxes is None or len(self.ground_track_data) == 0:
            return
        min_dist, closest_time = self._find_closest_track_point(event.xdata, event.ydata)
        if min_dist < 30:
            self._jump_satellite_to_time(closest_time)
            # Only update 2D for responsiveness during drag
            self.update_2d_plot()
            self.update_link_plots(update_only_current_position=False)

    def on_2d_map_release(self, event):
        """Stop dragging and do a full update"""
        if self._dragging_satellite:
            self._dragging_satellite = False
            self.update_plots()
    
    def update_plots(self):
        """Update both 2D and 3D orbit visualizations"""
        self.update_2d_plot()
        self.update_3d_plot()
        self.update_link_plots(update_only_current_position=False)  # Full recalculation
    
    def update_2d_plot(self):
        """Update 2d world map projection with satellite ground tracks"""
        self.fig_2d.clear()
        ax = self.fig_2d.add_subplot(111)
        
        # Draw Earth map
        self.draw_earth_2d_map(ax)
        
        # Get orbital parameters in radians
        a = self.params['a']
        e = self.params['e']
        i = np.radians(self.params['i'])
        Omega = np.radians(self.params['Omega'])
        omega = np.radians(self.params['omega'])
        M0 = np.radians(self.params['M0'])
        
        # Generate ground track (satellite positions projected onto Earth surface)
        self.draw_ground_track(ax, a, e, i, Omega, omega, M0)
        
        # Add user location
        user_lat = self.user_location['lat']
        user_lon = self.user_location['lon']
        ax.plot(user_lon, user_lat, 'go', markersize=8, label='User Location', zorder=7)
        
        # Calculate satellite sub-point and draw link line if visible
        pos_sat = OrbitalMechanics.kepler_to_cartesian(a, self.params['e'], i, Omega, omega, M0)
        sx, sy, sz = pos_sat
        sat_r_xy = np.sqrt(sx**2 + sy**2)
        
        # Apply Earth rotation to match ground track
        mu = 3.986004418e14
        T = 2 * np.pi * np.sqrt(a**3 / mu)
        earth_rotation_rate = 2 * np.pi / 86400.0
        time_from_track_start = (M0 / (2 * np.pi)) * T
        earth_rot = earth_rotation_rate * time_from_track_start
        
        sat_lon_eci = np.arctan2(sy, sx)
        sat_lon = np.degrees(sat_lon_eci - earth_rot) % 360
        if sat_lon > 180:
            sat_lon -= 360
        sat_lat = np.degrees(np.arctan2(sz, sat_r_xy))
        
        # Check elevation to decide link line color
        user_r = OrbitalMechanics.EARTH_RADIUS + self.user_location['alt']
        user_rad_lat = np.radians(user_lat)
        user_rad_lon = np.radians(user_lon)
        user_pos = np.array([
            user_r * np.cos(user_rad_lat) * np.cos(user_rad_lon),
            user_r * np.cos(user_rad_lat) * np.sin(user_rad_lon),
            user_r * np.sin(user_rad_lat)
        ])
        to_sat = pos_sat - user_pos
        local_up = user_pos / np.linalg.norm(user_pos)
        to_sat_norm = to_sat / np.linalg.norm(to_sat)
        elevation_deg = np.degrees(np.arcsin(np.dot(to_sat_norm, local_up)))
        
        if elevation_deg > 0:
            ax.plot([user_lon, sat_lon], [user_lat, sat_lat], 'c-', linewidth=2,
                    alpha=0.8, label=f'Link (El: {elevation_deg:.1f}°)', zorder=6)
        else:
            ax.plot([user_lon, sat_lon], [user_lat, sat_lat], 'r--', linewidth=1,
                    alpha=0.4, label=f'No Link (El: {elevation_deg:.1f}°)', zorder=6)
        
        # Formatting
        ax.set_xlabel('Longitude (°)', fontsize=10)
        ax.set_ylabel('Latitude (°)', fontsize=10)
        ax.set_title('2d Satellite Ground Track Map', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9, framealpha=0.8, borderaxespad=0,
                 prop={'weight': 'bold', 'size': 9})
        self.fig_2d.subplots_adjust(right=0.75)
        
        # Set limits to world map
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect('equal')
        
        self.canvas_2d.draw()
    
    def draw_earth_2d_map(self, ax):
        """Draw world map with real Earth imagery"""
        if self.earth_texture is not None:
            # Display the Earth texture as background
            ax.imshow(self.earth_texture, extent=[-180, 180, -90, 90], 
                     origin='upper', zorder=1, aspect='auto')
        else:
            # Fallback: simple blue background for oceans
            ax.set_facecolor('#1f77b4')
    
    def draw_ground_track(self, ax, a, e, i, Omega, omega, M0):
        """Draw satellite ground track (orbit projected onto Earth surface)"""
        # Clear previous ground track data
        self.ground_track_data = []
        
        # Calculate orbital period
        mu = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
        T = 2 * np.pi * np.sqrt(a**3 / mu)  # Orbital period in seconds
        
        # Earth rotation rate (radians per second)
        earth_rotation_rate = 2 * np.pi / 86400.0  # radians/second
        
        # J2 perturbation - nodal precession rate (RAAN change per orbit)
        # This causes the orbital plane to rotate, making subsequent passes different
        J2 = 1.08263e-3
        Re = 6378137.0  # Earth radius in meters
        n = np.sqrt(mu / a**3)  # Mean motion (rad/s)
        
        # RAAN precession rate (radians per second) due to J2
        Omega_dot = -1.5 * n * J2 * (Re / a)**2 * np.cos(i)
        
        # Current mean anomaly as fraction of orbit
        M0_frac = M0 / (2 * np.pi)  # Fraction of orbit elapsed
        
        # Generate multiple passes to show ground track pattern
        colors = ['red', 'orange', 'yellow']
        num_points = 180  # Points per orbit
        
        for orbit_num in range(3):  # Show 3 consecutive orbits
            track_lats = []
            track_lons = []
            
            # Time offset for this orbit (start of orbit_num-th orbit)
            time_offset = T * orbit_num
            
            # RAAN for this specific pass (accounts for precession)
            Omega_pass = Omega + Omega_dot * time_offset
            
            for j in range(num_points + 1):
                # True anomaly for this point
                nu = 2 * np.pi * j / num_points
                
                # Time elapsed since epoch for this point in this orbit
                # Approximate: assume circular orbit for time calculation
                time_in_orbit = T * j / num_points
                total_time = time_offset + time_in_orbit
                
                # Time relative to current satellite position (for click-to-jump)
                # Current position is at M0, this point is at nu in orbit orbit_num
                # The anomaly difference from current position:
                nu_diff = nu - M0  # radians difference within this orbit
                relative_time = time_offset + (nu_diff / (2 * np.pi)) * T
                
                # Get satellite position in ECI (inertial frame) with updated RAAN
                pos_eci = OrbitalMechanics.kepler_to_cartesian(a, e, i, Omega_pass, omega, nu)
                x, y, z = pos_eci
                
                # Earth has rotated by this amount since epoch
                earth_rotation = earth_rotation_rate * total_time
                
                # Convert ECI to geodetic coordinates
                # In ECI frame, account for Earth rotation
                lon_eci = np.arctan2(y, x)
                
                # Subtract Earth rotation to get ground track longitude
                lon_ground = lon_eci - earth_rotation
                
                # Convert to degrees and wrap to [-180, 180]
                lon = (lon_ground * 180 / np.pi) % 360
                if lon > 180:
                    lon -= 360
                
                # Latitude calculation
                r_xy = np.sqrt(x**2 + y**2)
                lat = np.arctan2(z, r_xy) * 180 / np.pi
                
                track_lats.append(lat)
                track_lons.append(lon)
                
                # Store relative time for click interaction
                self.ground_track_data.append((lon, lat, relative_time))
            
            # Split track at discontinuities (when crossing ±180° boundary)
            color = colors[orbit_num % len(colors)]
            segments_lons = []
            segments_lats = []
            current_lons = [track_lons[0]]
            current_lats = [track_lats[0]]
            
            for pt_idx in range(1, len(track_lons)):
                # Check for large jump in longitude (wrap around)
                if abs(track_lons[pt_idx] - track_lons[pt_idx-1]) > 180:
                    # Save current segment
                    segments_lons.append(current_lons)
                    segments_lats.append(current_lats)
                    # Start new segment
                    current_lons = [track_lons[pt_idx]]
                    current_lats = [track_lats[pt_idx]]
                else:
                    current_lons.append(track_lons[pt_idx])
                    current_lats.append(track_lats[pt_idx])
            
            # Add last segment
            segments_lons.append(current_lons)
            segments_lats.append(current_lats)
            
            # Plot each segment separately
            for seg_idx, (seg_lons, seg_lats) in enumerate(zip(segments_lons, segments_lats)):
                label = f'Pass k={orbit_num}' if seg_idx == 0 else None  # Only label first segment
                ax.plot(seg_lons, seg_lats, color=color, linewidth=1.5, alpha=0.7, 
                       label=label, zorder=5)
        
        # Plot current satellite position on map (using current mean anomaly M0)
        nu = M0  # Use current mean anomaly as position (valid for circular orbits)
        pos_eci = OrbitalMechanics.kepler_to_cartesian(a, e, i, Omega, omega, nu)
        x, y, z = pos_eci
        
        # Convert to ground position with Earth rotation matching the ground track
        # The ground track starts at nu=0 (t=0), so at nu=M0 the elapsed time is (M0/2pi)*T
        time_from_track_start = (M0 / (2 * np.pi)) * T
        earth_rotation_sat = earth_rotation_rate * time_from_track_start
        
        r_xy = np.sqrt(x**2 + y**2)
        lon_eci = np.arctan2(y, x)
        lon_ground = lon_eci - earth_rotation_sat
        lon = (lon_ground * 180 / np.pi) % 360
        if lon > 180:
            lon -= 360
        lat = np.arctan2(z, r_xy) * 180 / np.pi
        
        ax.plot(lon, lat, 'ro', markersize=6, label='Satellite', zorder=6)
    
    def init_3d_scene(self):
        """Initialize static 3D elements (Earth sphere + axis arrows). Called once."""
        self.draw_earth_pyvista()
        
        # Draw XYZ axis arrows on the sphere (static, never change)
        R = OrbitalMechanics.EARTH_RADIUS / 1e6
        arrow_len = R * 0.6
        tip_r = 0.15
        shaft_r = 0.03
        tip_len = 0.3
        
        # X axis (Greenwich meridian on equator) - Red
        x_arrow = pv.Arrow(start=(R, 0, 0), direction=(1, 0, 0),
                          scale=arrow_len, tip_radius=tip_r, shaft_radius=shaft_r, tip_length=tip_len)
        self.plotter.add_mesh(x_arrow, color='red', opacity=0.9, name='axis_x')
        self.plotter.add_point_labels(
            np.array([[R + arrow_len * 1.1, 0, 0]]), ['X (0°lon)'],
            font_size=10, text_color='red', shape=None, render_points_as_spheres=False,
            point_size=0, always_visible=True, name='axis_x_label')
        
        # Y axis (90°E on equator) - Green
        y_arrow = pv.Arrow(start=(0, R, 0), direction=(0, 1, 0),
                          scale=arrow_len, tip_radius=tip_r, shaft_radius=shaft_r, tip_length=tip_len)
        self.plotter.add_mesh(y_arrow, color='lime', opacity=0.9, name='axis_y')
        self.plotter.add_point_labels(
            np.array([[0, R + arrow_len * 1.1, 0]]), ['Y (90°E)'],
            font_size=10, text_color='lime', shape=None, render_points_as_spheres=False,
            point_size=0, always_visible=True, name='axis_y_label')
        
        # Z axis (North Pole) - Blue
        z_arrow = pv.Arrow(start=(0, 0, R), direction=(0, 0, 1),
                          scale=arrow_len, tip_radius=tip_r, shaft_radius=shaft_r, tip_length=tip_len)
        self.plotter.add_mesh(z_arrow, color='dodgerblue', opacity=0.9, name='axis_z')
        self.plotter.add_point_labels(
            np.array([[0, 0, R + arrow_len * 1.1]]), ['Z (North)'],
            font_size=10, text_color='dodgerblue', shape=None, render_points_as_spheres=False,
            point_size=0, always_visible=True, name='axis_z_label')
        
        self.plotter.reset_camera()
        self.plotter.render()
    
    def _deferred_3d_update(self):
        """Called by debounce timer to update 3D and link plots after slider activity."""
        # Run the heavier updates after a short debounce to avoid UI lag during slider dragging
        self.update_3d_plot()
        self.update_link_plots(update_only_current_position=False)

    def _on_slider_pressed(self):
        """Called when a slider drag starts."""
        self._slider_dragging = True

    def _on_slider_released(self):
        """Called when a slider drag ends — do a final full update."""
        self._slider_dragging = False
        # Stop any pending debounce and do a full update immediately
        if hasattr(self, '_3d_update_timer') and self._3d_update_timer.isActive():
            self._3d_update_timer.stop()
        self.update_plots()

    def _precompile_numba(self):
        """Warm up Numba-compiled functions to avoid visible lag on first use."""
        try:
            print("Precompiling Numba functions (this may take a moment)...")
            a = self.params['a']
            e = self.params['e']
            i = np.radians(self.params['i'])
            Omega = np.radians(self.params['Omega'])
            omega = np.radians(self.params['omega'])
            # small orbit to compile orbit generator
            _generate_orbit_numba(a, e, i, Omega, omega, 8)
            # small search to compile elevation kernel
            mu = 3.986004418e14
            T = 2 * np.pi * np.sqrt(a**3 / mu)
            times = np.linspace(-T, T, 64)
            user_r = OrbitalMechanics.EARTH_RADIUS + self.user_location['alt']
            user_lat = np.radians(self.user_location['lat'])
            user_lon = np.radians(self.user_location['lon'])
            user_pos = np.array([user_r * np.cos(user_lat) * np.cos(user_lon),
                                 user_r * np.cos(user_lat) * np.sin(user_lon),
                                 user_r * np.sin(user_lat)])
            _compute_elevations_numba(a, e, i, Omega, omega, np.radians(self.params['M0']), user_pos, times, T)
            print("Numba precompile finished.")
        except Exception as exc:
            print("Numba precompile failed:", exc)
    
    def update_3d_plot(self, num_points=None, full_render=True):
        """Update 3D orbit visualization — only dynamic elements (orbit, satellite, user, LOS).
        If num_points is provided, use that resolution for the orbit; this allows fast low-detail
        updates while dragging sliders.
        Static elements (Earth, axes) are drawn once in init_3d_scene()."""
        import time
        # Get orbital parameters in radians
        a = self.params['a']
        e = self.params['e']
        i = np.radians(self.params['i'])
        Omega = np.radians(self.params['Omega'])
        omega = np.radians(self.params['omega'])
        M0 = np.radians(self.params['M0'])

        # Choose number of points for this update
        if num_points is None:
            num_points = self._orbit_detail_full

        # Generate orbit (use named actor to auto-replace previous)
        orbit = OrbitalMechanics.generate_orbit(a, e, i, Omega, omega, M0, num_points)
        orbit_scaled = orbit / 1e6
        n_pts = len(orbit_scaled)
        orbit_polyline = pv.PolyData(orbit_scaled)
        orbit_polyline.lines = np.hstack([[n_pts] + list(range(n_pts))])
        self.plotter.add_mesh(orbit_polyline, color='red', line_width=3, name='orbit',
                             render_lines_as_tubes=False, reset_camera=False)

        # Plot current satellite position (named actor replaces previous)
        sat_pos = OrbitalMechanics.kepler_to_cartesian(a, e, i, Omega, omega, M0)
        sat_sphere = pv.Sphere(radius=0.3 if full_render else 0.22, center=sat_pos / 1e6,
                               theta_resolution=8, phi_resolution=8)
        self.plotter.add_mesh(sat_sphere, color='yellow', name='satellite', reset_camera=False)

        # Add user location (keep small sphere)
        user_lat = np.radians(self.user_location['lat'])
        user_lon = np.radians(self.user_location['lon'])
        user_alt = self.user_location['alt']

        user_r = (OrbitalMechanics.EARTH_RADIUS + user_alt) / 1e6
        user_x = user_r * np.cos(user_lat) * np.cos(user_lon)
        user_y = user_r * np.cos(user_lat) * np.sin(user_lon)
        user_z = user_r * np.sin(user_lat)

        user_sphere = pv.Sphere(radius=0.15, center=[user_x, user_y, user_z],
                               theta_resolution=8, phi_resolution=8)
        self.plotter.add_mesh(user_sphere, color='green', name='user', reset_camera=False)

        # Calculate line of sight and elevation
        sat_pos_3d = sat_pos / 1e6
        user_pos_3d = np.array([user_x, user_y, user_z])
        to_sat = sat_pos_3d - user_pos_3d
        local_up = user_pos_3d / np.linalg.norm(user_pos_3d)
        to_sat_norm = to_sat / np.linalg.norm(to_sat)
        elevation_angle = np.arcsin(np.dot(to_sat_norm, local_up))
        elevation_deg = np.degrees(elevation_angle)

        # If satellite is visible (elevation > 0), draw line of sight
        if elevation_deg > 0:
            los_line = pv.Line(user_pos_3d, sat_pos_3d)
            self.plotter.add_mesh(los_line, color='cyan', line_width=2, name='los_line',
                                 reset_camera=False)
        else:
            # Remove LOS line if satellite not visible
            try:
                self.plotter.remove_actor('los_line')
            except Exception:
                pass

        # For fast updates, throttle render calls to avoid saturating UI
        now = time.time()
        # If low-detail (interactive), only render at most once per throttle window
        if num_points <= self._orbit_detail_low:
            if now - self._last_fast_update_time > self._fast_update_throttle:
                self.plotter.render()
                self._last_fast_update_time = now
        else:
            self.plotter.render()
    
    def update_link_plots(self, update_only_current_position=False):
        """Update polar sky view and elevation angle plots
        
        Args:
            update_only_current_position: If True, only update the red dot position using cached data.
                                         If False, recalculate the entire pass window.
        """
        # If we're only updating current position and have cached data, use it
        if update_only_current_position and self.link_plot_cache is not None and self.link_plot_artists is not None:
            cached_data = self.link_plot_cache
            times = cached_data['times']
            elevations = cached_data['elevations']
            azimuths = cached_data['azimuths']
            zenith_angles = cached_data['zenith_angles']
            dopplers = cached_data['dopplers']
            cache_M0 = cached_data['cache_M0']
            T = cached_data['orbital_period']
            peak_time_offset = cached_data.get('peak_time_offset', 0)
            
            # Calculate current time offset relative to when cache was built
            # times[] are in minutes with t=0 at peak elevation
            M0 = np.radians(self.params['M0'])
            
            # Find how far M0 has advanced since cache was built
            M_diff = (M0 - cache_M0 + np.pi) % (2 * np.pi) - np.pi  # Normalized difference
            current_time_offset = (M_diff / (2 * np.pi)) * T / 60 - peak_time_offset  # Convert to minutes, shift to peak-centered
            
            # Check if current time is still within the cached pass window (with 5 minute margin)
            if current_time_offset < min(times) - 5 or current_time_offset > max(times) + 5:
                # Current time is outside cached window, clear cache and return
                # The calling code will handle whether to show plots or not based on elevation
                self.link_plot_cache = None
                self.link_plot_artists = None
                return
            
            # Find the index closest to current time
            current_idx = np.argmin(np.abs(np.array(times) - current_time_offset))
            
            # If we're way outside the data range, extrapolate by using boundary points
            if current_time_offset < min(times):
                current_idx = 0
            elif current_time_offset > max(times):
                current_idx = len(times) - 1
            
            # Update only the marker positions using stored artist references
            artists = self.link_plot_artists
            
            # Update polar plot marker
            if 'polar_marker' in artists:
                artists['polar_marker'].set_data([azimuths[current_idx]], [zenith_angles[current_idx]])
            
            # Update elevation plot marker
            if 'elev_marker' in artists:
                artists['elev_marker'].set_data([times[current_idx]], [elevations[current_idx]])
            
            # Update doppler plot marker
            if 'doppler_marker' in artists:
                artists['doppler_marker'].set_data([times[current_idx]], [dopplers[current_idx]])
            
            # Redraw efficiently using blit
            self.canvas_link.draw_idle()
            self.canvas_link.flush_events()
            return
        
        # Full recalculation (when update_only_current_position=False or no cache)
        self.fig_link.clear()
        
        # Get orbital parameters
        a = self.params['a']
        e = self.params['e']
        i = np.radians(self.params['i'])
        Omega = np.radians(self.params['Omega'])
        omega = np.radians(self.params['omega'])
        M0 = np.radians(self.params['M0'])
        
        # User location
        user_lat = np.radians(self.user_location['lat'])
        user_lon = np.radians(self.user_location['lon'])
        user_alt = self.user_location['alt']
        
        # Calculate current elevation first
        pos_sat_current = OrbitalMechanics.kepler_to_cartesian(a, e, i, Omega, omega, M0)
        user_r = OrbitalMechanics.EARTH_RADIUS + user_alt
        user_x = user_r * np.cos(user_lat) * np.cos(user_lon)
        user_y = user_r * np.cos(user_lat) * np.sin(user_lon)
        user_z = user_r * np.sin(user_lat)
        user_pos = np.array([user_x, user_y, user_z])
        
        to_sat_current = pos_sat_current - user_pos
        local_up = user_pos / np.linalg.norm(user_pos)
        to_sat_norm_current = to_sat_current / np.linalg.norm(to_sat_current)
        current_elevation = np.degrees(np.arcsin(np.dot(to_sat_norm_current, local_up)))
        
        # Create subplot layout with balanced space (3:2 ratio)
        gs = GridSpec(1, 2, figure=self.fig_link, width_ratios=[3, 2])
        ax1 = self.fig_link.add_subplot(gs[0], projection='polar')  # Polar plot (left)
        ax2 = self.fig_link.add_subplot(gs[1])  # Regular plot (right, wider than before)
        
        # Setup empty plots first
        ax1.set_theta_zero_location('N')  # North at top
        ax1.set_theta_direction(-1)  # Clockwise
        ax1.set_ylim(0, 90)  # 0° at center (zenith), 90° at edge (horizon)
        ax1.set_yticks([0, 30, 60, 90])
        ax1.set_yticklabels(['90°', '60°', '30°', '0°'])  # Elevation labels
        ax1.set_title('Polar Sky View', fontsize=8, fontweight='bold', pad=12)
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Time (minutes)', fontsize=8)
        ax2.set_ylabel('Elevation Angle (°)', fontsize=8, color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.set_title('Elevation & Doppler vs Time', fontsize=8, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-30, 30)
        ax2.set_ylim(0, 90)
        
        # Calculate orbital period
        mu = 3.986004418e14
        T = 2 * np.pi * np.sqrt(a**3 / mu)  # seconds
        
        # Find the complete contact window (pass) that includes current time
        # Search ±1 orbital period
        search_range = T
        time_points_search = np.linspace(-search_range, search_range, 1000)
        
        # Local East, North, Up vectors at user location
        local_east = np.array([-np.sin(user_lon), np.cos(user_lon), 0])
        local_north = np.array([-np.sin(user_lat) * np.cos(user_lon), 
                                -np.sin(user_lat) * np.sin(user_lon), 
                                np.cos(user_lat)])
        
        # Transmit frequency for Doppler calculation
        f_tx = self.carrier_freq * 1e9  # Convert GHz to Hz
        c = 299792458  # speed of light m/s
        
        # First pass: find all elevation values to identify contact windows
        # Use Numba-accelerated routine when available to speedup large searches
        if NUMBA_AVAILABLE:
            # time_points_search is already a NumPy array
            all_elevations = _compute_elevations_numba(a, e, i, Omega, omega, M0, user_pos, time_points_search, T)
        else:
            all_elevations = []
            for t in time_points_search:
                M = M0 + (t / T) * 2 * np.pi
                nu = M
                pos_sat = OrbitalMechanics.kepler_to_cartesian(a, e, i, Omega, omega, nu)
                to_sat = pos_sat - user_pos
                to_sat_norm = to_sat / np.linalg.norm(to_sat)
                elevation_angle = np.arcsin(np.dot(to_sat_norm, local_up))
                elevation_deg = np.degrees(elevation_angle)
                all_elevations.append(elevation_deg)
            all_elevations = np.array(all_elevations)
        
        # Check if satellite is currently in contact (at t=0)
        current_time_idx = len(time_points_search) // 2  # t=0 is at the middle
        in_contact = all_elevations > 0
        
        # Find the boundaries of the current pass (if any)
        # Find start (search backwards from current time)
        contact_start_idx = 0
        for idx_back in range(current_time_idx, -1, -1):
            if not in_contact[idx_back]:
                contact_start_idx = idx_back + 1
                break
        
        # Find end (search forwards from current time)
        contact_end_idx = len(in_contact) - 1
        for idx_fwd in range(current_time_idx, len(in_contact)):
            if not in_contact[idx_fwd]:
                contact_end_idx = idx_fwd - 1
                break
        
        # Only plot if we found a valid contact window around current time
        if contact_end_idx > contact_start_idx and current_time_idx >= contact_start_idx and current_time_idx <= contact_end_idx:
            # Extract the contact window
            contact_time_points = time_points_search[contact_start_idx:contact_end_idx+1]
            
            elevations = []
            azimuths = []
            dopplers = []
            times = []
            current_idx = None
            
            for idx, t in enumerate(contact_time_points):
                # Calculate mean anomaly at this time
                M = M0 + (t / T) * 2 * np.pi
                nu = M
                
                # Get satellite position
                pos_sat = OrbitalMechanics.kepler_to_cartesian(a, e, i, Omega, omega, nu)
                
                # Vector from user to satellite
                to_sat = pos_sat - user_pos
                distance = np.linalg.norm(to_sat)
                
                # Elevation angle
                to_sat_norm = to_sat / distance
                elevation_angle = np.arcsin(np.dot(to_sat_norm, local_up))
                elevation_deg = np.degrees(elevation_angle)
                
                elevations.append(elevation_deg)
                times.append(t / 60)  # Convert to minutes
                
                # Calculate azimuth angle
                to_sat_horiz = to_sat - np.dot(to_sat, local_up) * local_up
                horiz_dist = np.linalg.norm(to_sat_horiz)
                
                # Handle case when satellite is near zenith (horizontal distance ~0)
                if horiz_dist < 1e-6:
                    # Satellite is directly overhead, azimuth is undefined, use 0
                    azimuth_rad = 0.0
                else:
                    to_sat_horiz_norm = to_sat_horiz / horiz_dist
                    # Azimuth from North (0°=North, 90°=East)
                    azimuth_rad = np.arctan2(np.dot(to_sat_horiz_norm, local_east), 
                                            np.dot(to_sat_horiz_norm, local_north))
                azimuths.append(azimuth_rad)
                
                # Track which point is the current time (t=0)
                if abs(t) < 1.0:  # Within 1 second of current time
                    current_idx = idx
                
                # Doppler shift calculation
                dt = 1.0
                M_future = M + (dt / T) * 2 * np.pi
                nu_future = M_future
                pos_sat_future = OrbitalMechanics.kepler_to_cartesian(a, e, i, Omega, omega, nu_future)
                vel_sat = (pos_sat_future - pos_sat) / dt
                radial_vel = np.dot(vel_sat, to_sat_norm)
                doppler_shift = -(radial_vel / c) * f_tx
                dopplers.append(doppler_shift / 1e3)  # Convert to kHz
            
            # Plot the complete contact window
            if len(elevations) > 0:
                # Polar sky plot with line trajectory
                zenith_angles = [90 - el for el in elevations]
                
                # Shift times so t=0 is at peak elevation
                peak_idx = np.argmax(elevations)
                peak_time_offset = times[peak_idx]  # minutes
                times = [t - peak_time_offset for t in times]
                
                # Plot satellite trajectory as both line and points for better visibility
                ax1.plot(azimuths, zenith_angles, 'b-', linewidth=3, alpha=0.8, label='Trajectory')
                ax1.scatter(azimuths, zenith_angles, c='blue', s=20, alpha=0.5, zorder=3)
                
                # Mark current position with red dot (use closest time to t=0 before shift)
                if current_idx is None and len(times) > 0:
                    # Find the closest point to current satellite time
                    current_idx = np.argmin(np.abs(np.array(times) + peak_time_offset))
                
                # Store artist references for efficient animation updates
                polar_marker = None
                elev_marker = None
                doppler_marker = None
                
                if current_idx is not None:
                    current_azimuth = azimuths[current_idx]
                    current_zenith = zenith_angles[current_idx]
                    polar_marker_line, = ax1.plot(current_azimuth, current_zenith, 'ro', markersize=12, 
                            markeredgecolor='black', markeredgewidth=2, label='Current', zorder=5)
                    polar_marker = polar_marker_line
                ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=7, framealpha=0.8, markerscale=0.6, borderaxespad=0)
                
                # Elevation vs time plot with Doppler overlaid
                ax2.plot(times, elevations, 'b-', linewidth=2.5, label='Elevation')
                ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax2.fill_between(times, 0, elevations, alpha=0.2, color='blue')
                
                # Set x-axis limits based on data with some padding
                time_range = max(times) - min(times)
                time_padding = time_range * 0.1
                ax2.set_xlim(min(times) - time_padding, max(times) + time_padding)
                
                # Mark current time with red dot
                if current_idx is not None:
                    elev_marker_line, = ax2.plot(times[current_idx], elevations[current_idx], 'ro', 
                            markersize=8, markeredgecolor='black', markeredgewidth=1, 
                            label='Current Time', zorder=5)
                    elev_marker = elev_marker_line
                
                # Secondary y-axis: Doppler
                ax2_doppler = ax2.twinx()
                ax2_doppler.plot(times, dopplers, 'r-', linewidth=2, label='Doppler', alpha=0.7)
                ax2_doppler.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
                ax2_doppler.set_ylabel('Doppler Shift (kHz)', fontsize=9, color='r')
                ax2_doppler.tick_params(axis='y', labelcolor='r')
                
                # Mark current time on Doppler curve
                if current_idx is not None:
                    doppler_marker_line, = ax2_doppler.plot(times[current_idx], dopplers[current_idx], 'ro', 
                                    markersize=8, markeredgecolor='black', markeredgewidth=1, zorder=5)
                    doppler_marker = doppler_marker_line
                
                # Combine legends
                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax2_doppler.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7)
                
                # Cache the pass data for efficient updates during animation
                # Store the M0 at cache-build time as reference, since times[] are relative to t=0 at that moment
                self.link_plot_cache = {
                    'times': times,
                    'elevations': elevations,
                    'azimuths': azimuths,
                    'zenith_angles': zenith_angles,
                    'dopplers': dopplers,
                    'cache_M0': M0,
                    'orbital_period': T,
                    'peak_time_offset': peak_time_offset
                }
                
                # Store artist references for efficient animation
                self.link_plot_artists = {
                    'polar_marker': polar_marker,
                    'elev_marker': elev_marker,
                    'doppler_marker': doppler_marker,
                    'ax1': ax1,
                    'ax2': ax2,
                    'ax2_doppler': ax2_doppler
                }
        
        self.fig_link.tight_layout()
        self.canvas_link.draw()
    
    def draw_earth_pyvista(self):
        """Draw Earth sphere with texture using PyVista for high performance"""
        # High resolution mesh for smooth appearance
        sphere = pv.Sphere(radius=OrbitalMechanics.EARTH_RADIUS / 1e6, 
                         theta_resolution=360, phi_resolution=180)
        
        if self.earth_texture is not None:
            # Generate texture coordinates manually
            # Compute theta (longitude) and phi (latitude) for each point
            points = sphere.points
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            
            # Convert to spherical coordinates
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arctan2(y, x)  # longitude
            phi = np.arcsin(z / r)     # latitude
            
            # Map to texture coordinates [0, 1]
            u = (theta + np.pi) / (2 * np.pi)  # longitude to [0, 1]
            v = (phi + np.pi/2) / np.pi         # latitude to [0, 1]
            
            # Add texture coordinates to sphere using the correct PyVista method
            tex_coords = np.c_[u, v]
            sphere.active_texture_coordinates = tex_coords
            
            # Create texture from image
            texture = pv.numpy_to_texture(self.earth_texture)
            self.plotter.add_mesh(sphere, texture=texture, smooth_shading=True, name='earth')
        else:
            self.plotter.add_mesh(sphere, color='blue', smooth_shading=True, name='earth')


def main():
    """Main entry point"""
    # Enable high-DPI scaling on Windows to avoid cramped widgets and overlapping text
    if sys.platform.startswith('win'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    # Set a reasonable default font on Windows
    if sys.platform.startswith('win'):
        app.setFont(QFont('Segoe UI', 9))
    window = OrbitMapGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
