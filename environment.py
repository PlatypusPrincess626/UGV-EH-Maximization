"""
File: environment.py
Author: Mason Conkel
Creation Date: 2026-02-10
Description: This script establishes the environment
"""
# Import Dependencies
from pvlib import spectrum, solarposition, irradiance, atmosphere
import pandas as pd
from numpy.typing import NDArray
import numpy as np
import random
from sklearn.cluster import KMeans
from scipy import signal
from scipy.signal import windows
from scipy.ndimage import gaussian_filter
import math
import rasterio
from scipy.ndimage import zoom
from pathlib import Path

# Custom Packages
import ugv_simulator


def gaussian_kernel(n, std, normalised=False):
    '''
    Generates a n x n matrix with a centered gaussian
    of standard deviation std centered on it. If normalised,
    its volume equals 1.
    '''
    gaussian1D = windows.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalised:
        gaussian2D /= (2 * np.pi * (std ** 2))
    return gaussian2D


def dist(pt1: NDArray[np.int32], pt2: NDArray[np.int32]):
    assert pt1.shape == (2,)
    assert pt2.shape == (2,)
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


class sim_env:
    def __init__(self, scene, num_sensors, max_num_steps):

        self.ch_table = None
        self.sensor_table = None
        self.total_sensors = num_sensors
        self.max_num_steps = max_num_steps

        self.sensor_pts = []
        self.ch_pt = [0, 0]
        self.ch = None

        if (scene == "test"):
            # Test scene will be Yellowstone National Park
            self.lat_center = 44.424  # Latitude
            self.long_center = -110.589  # Longitude
            self.stp = 0.000009  # 1 degree lat/long is ~111km
            self.pressure = 101253  # Sea Level is 1013.25 mb, Average Pressure in Yellowstone is +4.09mb
            self.water_vapor_content = 0.35  # Roughly 0.35 cm in Yellowstone
            self.tau500 = 0.75  # Aerosol Turbidity 500nm
            self.ozone = 0.23  # Ozone in atm-cm
            self.albedo = 0.2  # Bare Ground and Grassy
            self.dim = 800  # Map dimension n x n
            self.numObst = 500  # Number of obstacles decided
            self.stepSize = 'min'  # Frequency of time steps
            self.times = pd.date_range('2021-01-01 8:00', freq=self.stepSize, periods=self.max_num_steps, tz="MST")
            random.seed('2021-01-01 8:00')

        self.r_move = self.dim
        self.env_map = self.make_map()

        self.view_dist = self.dim

        # insert interference creation loop
        self.topo_mask, self.foliage_mask = None, None
        script_dir = Path(__file__).resolve().parent
        self.topo_file_path = script_dir / 'topo_data.tif'
        self.obfuscation_array = self.init_interference()



    def step_simulation(self, current_step: int, target_x: float, target_y: float):
        """
        Advances the simulation by one environment step using trajectory coordinates.

        Args:
            target_x: Selected X coordinate target from the model
            target_y: Selected Y coordinate target from the model
            current_step: The integer index of the current simulation time step.

        Returns:
            tuple: (telemetry_dict, next_local_observation_patch)
        """
        # 1. Capture battery context prior to making the transition step
        battery_before = self.ch.get_battery()
        ugv_x, ugv_y, ugv_yaw = self.ch.get_position()

        # 2. Step the UGV forward toward the destination
        new_position, battery_after = self.ch.step(self, current_step, float(target_x), float(target_y))

        # 3. Extract the NEXT local 2D observation patch of shadows for RL state observation
        lat_offset = new_position[0] * self.stp
        long_offset = new_position[1] * self.stp
        solpos = solarposition.get_solarposition(self.times[current_step], self.lat_center + lat_offset,
                                                 self.long_center + long_offset)

        azimuth = solpos['azimuth'].iloc[0]
        zenith = solpos['apparent_zenith'].iloc[0]

        next_local_observation = self.get_obfuscation(new_position[0], new_position[1], current_step, azimuth, zenith)

        telemetry = {
            "step": current_step,
            "previous_position": (ugv_x, ugv_y, ugv_yaw),
            "new_position": new_position,
            "target_trajectory": (target_x, target_y),
            "battery_before": battery_before,
            "battery_after": battery_after,
            "net_battery_change": battery_after - battery_before
        }

        return telemetry, next_local_observation

    def set_view_dist(self, view):
        if view < self.dim:
            self.view_dist = view
        else:
            self.view_dist = self.dim
        return self.view_dist

    # Create Map and Grid
    def make_map(self):
        if self.dim % 2 == 0:
            self.dim += 1

        flg_done = self.place_devices()

        return flg_done

    def init_interference(self):
        """
        Initializes two distinct masks: Topography (loaded from CSV) and Foliage.
        """
        with open(self.topo_file_path, 'rb') as f:
            header = f.read(4)
            print(f"File header (hex): {header.hex()}")
        # 1. Topography: Load from local CSV for speed
        try:
            with rasterio.open(self.topo_file_path) as src:
                # Read the first band (elevation data)
                data = src.read(1)

                # 1. Convert to relative height (subtract the minimum elevation)
                # This centers your terrain around 0, making it compatible with your ray-casting height checks
                data = data - np.min(data)

                max_val = np.max(data)
                if max_val > 0:
                    data = (data / max_val) * 50.0

                # 2. Resizing logic
                if data.shape != (self.dim, self.dim):
                    zoom_factors = (self.dim / data.shape[0], self.dim / data.shape[1])
                    self.topo_mask = zoom(data, zoom_factors, order=1)
                else:
                    self.topo_mask = data

                print(f"Successfully loaded topography: {self.topo_mask.shape}")
        except Exception as e:
            print(f"Error loading topography file: {e}")
            self.topo_mask = np.zeros((self.dim, self.dim))

        # 2. Foliage: Clustered distribution
        # Create a base distribution of potential tree locations
        # Weights: [Empty, 5m, 10m, 15m, 20m]
        choices = [0, 5, 10, 15, 20]
        probs = [0.4, 0.3, 0.15, 0.1, 0.05]

        # Create raw random height map
        raw_foliage = np.random.choice(choices, size=(self.dim, self.dim), p=probs)

        # Apply Gaussian filter to create "patches" (Spatial Correlation)
        # Sigma controls the size of the tree clumps
        smoothed = gaussian_filter(raw_foliage.astype(float), sigma=3)

        # Re-quantize to snap back to our height classes
        # This keeps the "patches" but ensures the ray-caster gets expected height values
        bins = [0, 2.5, 7.5, 12.5, 17.5, 25]
        self.foliage_mask = np.digitize(smoothed, bins)
        # Map indices to height values: 0->0, 1->5, 2->10, 3->15, 4->20
        height_map = np.array([0, 5, 10, 15, 20])
        self.foliage_mask = height_map[np.clip(self.foliage_mask - 1, 0, 4)]
        return True

    # Place obstructions and devices in initial positions
    def place_devices(self) -> list:
        """
        self.dim -> dimension of environment
        self.total_sensors -> number of sensors in the environment
        max_dist_ambc = 800
        """
        max_dist_ambc = 800
        sensor_pts = np.array([[0, 0]] * self.total_sensors, np.int32)

        print("Placing Sensors")
        for sensor in range(self.total_sensors):
            position = random.randint(0, self.dim * self.dim - 1)
            sensor_pts[sensor] = [int(position % self.dim), int(position / self.dim)]

        k_means = KMeans(n_clusters=1, random_state=0, n_init=10).fit(sensor_pts)
        cluster = k_means.cluster_centers_[0]

        pt_max_dist = 0
        for true_pt in sensor_pts:
            pt_dist = dist(true_pt, cluster)
            if pt_dist > pt_max_dist:
                pt_max_dist = pt_dist

        self.r_move = max_dist_ambc - pt_max_dist
        self.sensor_pts = sensor_pts
        self.ch_pt = cluster.astype(int).tolist()  # FIXED: Removed undefined 'centroids' reference

        # FIXED: Pass r_move parameter directly to initialization safely
        self.ch = ugv_simulator.UGVSimulator(
            x=float(self.ch_pt[0]),
            y=float(self.ch_pt[1]),
            yaw=np.pi / 4,
            battery_level=100,
            r_move=self.r_move
        )
        return True

    def get_spectrum(self, x, y, tilt, azimuth, step):
        lat_offset = x * self.stp
        long_offset = y * self.stp

        solpos = solarposition.get_solarposition(self.times[step], self.lat_center + lat_offset,
                                                 self.long_center + long_offset)
        aoi = irradiance.aoi(solpos.apparent_zenith, solpos.azimuth, solpos.apparent_zenith, solpos.azimuth)
        relative_airmass = atmosphere.get_relative_airmass(solpos.apparent_zenith, model='kasten1966')
        spectra = spectrum.spectrl2(
            apparent_zenith=solpos.apparent_zenith,
            aoi=aoi,
            surface_tilt=solpos.apparent_zenith,
            ground_albedo=self.albedo,
            surface_pressure=self.pressure,
            relative_airmass=relative_airmass,
            precipitable_water=self.water_vapor_content,
            ozone=self.ozone,
            aerosol_turbidity_500nm=self.tau500,
        )
        return spectra, solpos

    def get_obfuscation(self, x: int, y: int, step, azimuth: float, zenith: float):

        v_dist = int(self.view_dist)
        patch_size = 2 * v_dist + 1

        obfuscation_patch = np.zeros(
            (patch_size, patch_size),
            dtype=np.float32
        )

        # Sun below horizon
        if zenith >= 90.0:
            return np.ones(
                (patch_size, patch_size),
                dtype=np.float32
            )

        # --------------------------------------------------
        # Solar geometry
        # --------------------------------------------------

        az_rad = math.radians(90.0 - azimuth)
        el_rad = math.radians(90.0 - zenith)

        tan_elevation = math.tan(el_rad)

        step_x = math.cos(az_rad)
        step_y = math.sin(az_rad)

        perp_x = -step_y
        perp_y = step_x

        center_x = int(x)
        center_y = int(y)

        MAX_FOLIAGE_ATTENUATION = 0.25

        # --------------------------------------------------
        # Iterate through visible patch
        # --------------------------------------------------

        for j in range(patch_size):

            for i in range(patch_size):

                global_x = center_x - v_dist + i
                global_y = center_y - v_dist + j

                if not (
                        0 <= global_x < self.dim
                        and
                        0 <= global_y < self.dim
                ):
                    continue

                # Remaining sunlight after attenuation
                transmittance = 1.0

                # --------------------------------------------------
                # Ray cast toward sun
                # --------------------------------------------------

                for d in range(1, self.dim):

                    h_min = d * tan_elevation

                    # Nothing in the environment exceeds this
                    # if h_min > 10.0:
                    #     break

                    ray_x = int(round(global_x + d * step_x))
                    ray_y = int(round(global_y + d * step_y))

                    if not (
                            0 <= ray_x < self.dim
                            and
                            0 <= ray_y < self.dim
                    ):
                        break

                    # ==================================================
                    # TERRAIN CHECK
                    # ==================================================

                    terrain_height = self.topo_mask[ray_y, ray_x]

                    if terrain_height >= h_min:
                        obfuscation_patch[j, i] = 1.0
                        transmittance = 0.0
                        break

                    # ==================================================
                    # FOLIAGE CHECK
                    # ==================================================

                    foliage_height = self.foliage_mask[ray_y, ray_x]

                    if foliage_height <= 0:
                        continue

                    canopy_start = foliage_height / 3.0
                    canopy_radius = foliage_height / 2.0

                    # Ray passes below foliage
                    if h_min < canopy_start:
                        continue

                    # Ray passes above tree
                    if h_min > foliage_height:
                        continue

                    # --------------------------------------------------
                    # Sample canopy cross-section
                    # --------------------------------------------------

                    half_width = int(math.ceil(canopy_radius))

                    local_attenuation = 0.0

                    for k in range(-half_width, half_width + 1):

                        check_x = int(round(ray_x + k * perp_x))
                        check_y = int(round(ray_y + k * perp_y))

                        if not (
                                0 <= check_x < self.dim
                                and
                                0 <= check_y < self.dim
                        ):
                            continue

                        if self.foliage_mask[check_y, check_x] <= 0:
                            continue

                        r = abs(k)

                        if r > canopy_radius:
                            continue

                        r_norm = r / canopy_radius

                        # Dense near trunk
                        density = math.exp(
                            -2.0 * r_norm * r_norm
                        )

                        attenuation = (
                                density *
                                MAX_FOLIAGE_ATTENUATION
                        )

                        if attenuation > local_attenuation:
                            local_attenuation = attenuation

                    # --------------------------------------------------
                    # Beer-Lambert style accumulation
                    # --------------------------------------------------

                    transmittance *= (
                            1.0 - local_attenuation
                    )

                    # Early exit if nearly opaque
                    if transmittance < 0.01:
                        transmittance = 0.0
                        break

                obfuscation_patch[j, i] = (
                        1.0 - transmittance
                )

        return obfuscation_patch