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
import math
import datetime

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
        # Base matrix initialized to 0 (ground level)
        envStaticInter = np.zeros((self.dim, self.dim), dtype=np.int32)

        obstacles_count = int(self.dim)
        print("Placing Discrete Height Points")
        for _ in range(obstacles_count):
            pt_x = random.randint(0, self.dim - 1)
            pt_y = random.randint(0, self.dim - 1)

            # Select discrete integer height profile
            obstacle_height = random.choice([3, 5, 10])

            # Direct single point assignment. If a coordinate is selected twice,
            # it resolves to the higher obstacle.
            envStaticInter[pt_y, pt_x] = max(envStaticInter[pt_y, pt_x], obstacle_height)

        return envStaticInter

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
        """
        Generates a 2D float matrix of shape (2 * view_dist + 1, 2 * view_dist + 1)
        representing localized solar block weights based on object height canopy projections.
        """
        v_dist = int(self.view_dist)
        patch_size = 2 * v_dist + 1
        obfuscation_patch = np.zeros((patch_size, patch_size), dtype=np.float32)

        if zenith >= 90.0:
            return np.ones((patch_size, patch_size), dtype=np.float32)

        azimuth_rad = math.radians(90.0 - azimuth)
        step_x = math.cos(azimuth_rad)
        step_y = math.sin(azimuth_rad)

        perp_x = -step_y
        perp_y = step_x

        elevation_rad = math.radians(90.0 - zenith)
        tan_elevation = math.tan(elevation_rad) if zenith > 0 else float('inf')

        center_x = int(x)
        center_y = int(y)

        for j in range(patch_size):
            for i in range(patch_size):
                global_x = center_x - v_dist + i
                global_y = center_y - v_dist + j

                if global_x < 0 or global_x >= self.dim or global_y < 0 or global_y >= self.dim:
                    continue

                max_block_strength = 0.0
                found_block = False

                for d in range(1, self.dim):
                    h_min = d * tan_elevation
                    if h_min > 10.0:
                        break

                    ray_x = global_x + d * step_x
                    ray_y = global_y + d * step_y

                    for k in range(-5, 6):
                        check_x = int(round(ray_x + k * perp_x))
                        check_y = int(round(ray_y + k * perp_y))

                        if check_x < 0 or check_x >= self.dim or check_y < 0 or check_y >= self.dim:
                            continue

                        height = self.obfuscation_array[check_y, check_x]
                        if height > 0:
                            half_height = int(height / 2)
                            if -half_height <= k < half_height:
                                if height >= h_min:
                                    block_strength = 1.0 - (abs(k) * (1.0 / height))
                                    if block_strength > max_block_strength:
                                        max_block_strength = block_strength
                                        found_block = True
                    if found_block:
                        obfuscation_patch[j, i] = max_block_strength
                        break

        return obfuscation_patch