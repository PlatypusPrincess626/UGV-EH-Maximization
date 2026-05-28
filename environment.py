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
        self.chkpt_div = 15
        checkpoints = int(720 / self.chkpt_div)
        shadow_array = []
        for _ in range(checkpoints):
            shadows = self.init_interference()
            shadow_array.append(shadows)

        self.obfuscation_array = np.array(shadow_array)



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

        # 3. Extract the NEXT local 2D observation patch for the next step calculation
        v_dist = int(self.view_dist)
        checkpoint_idx = min(int(current_step / self.chkpt_div), self.obfuscation_array.shape[0] - 1)
        grid_2d = self.obfuscation_array[checkpoint_idx].reshape((self.dim, self.dim))

        padded_grid = np.ones((self.dim + 2 * v_dist, self.dim + 2 * v_dist), dtype=np.float32)
        padded_grid[v_dist: v_dist + self.dim, v_dist: v_dist + self.dim] = grid_2d

        next_x, next_y, _ = new_position
        center_x_padded = int(next_x) + v_dist
        center_y_padded = int(next_y) + v_dist

        next_local_observation = padded_grid[
            center_y_padded - v_dist: center_y_padded + v_dist + 1,
            center_x_padded - v_dist: center_x_padded + v_dist + 1
        ]

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

        envStaticInter = np.zeros((self.dim, self.dim))

        shadows = int(self.dim)
        print("Making Happy Trees")
        for _ in range(shadows):
            start_x = random.randint(0, self.dim - 1)
            start_y = random.randint(0, self.dim - 1)
            shadeSize = random.randint(8, 50)
            intensity = random.randint(int(0.25 * shadeSize), int(0.75 * shadeSize))
            data2D = gaussian_kernel(shadeSize, intensity, normalised=False)

            # FIXED: Safe 2D overlapping matrix alignment block to prevent index exceptions
            end_x = min(start_x + shadeSize, self.dim)
            end_y = min(start_y + shadeSize, self.dim)
            kernel_w = end_x - start_x
            kernel_h = end_y - start_y

            envStaticInter[start_y:end_y, start_x:end_x] += data2D[0:kernel_h, 0:kernel_w]

        envStaticInter = np.clip(envStaticInter, 0.0, 1.0)
        return envStaticInter.flatten()

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

    def get_spectrum(self, lat, long, tilt, azimuth, step):
        lat_offset = x * self.stp
        long_offset = y * self.stp

        solpos = solarposition.get_solarposition(self.times[step], self.lat_center + lat_offset,
                                                 self.long_center + long_offset)
        aoi = irradiance.aoi(tilt, azimuth, solpos.apparent_zenith, solpos.azimuth)
        relative_airmass = atmosphere.get_relative_airmass(solpos.apparent_zenith, model='kasten1966')
        spectra = spectrum.spectrl2(
            apparent_zenith=solpos.apparent_zenith,
            aoi=aoi,
            surface_tilt=tilt,
            ground_albedo=self.albedo,
            surface_pressure=self.pressure,
            relative_airmass=relative_airmass,
            precipitable_water=self.water_vapor_content,
            ozone=self.ozone,
            aerosol_turbidity_500nm=self.tau500,
        )
        return spectra

    def get_obfuscation(self, x: int, y: int, step):
        safe_x = max(0, min(int(x), self.dim - 1))
        safe_y = max(0, min(int(y), self.dim - 1))
        checkpoint_idx = min(int(step / self.chkpt_div), self.obfuscation_array.shape[0] - 1)

        return self.obfuscation_array[checkpoint_idx, int(safe_y * self.dim + safe_x)]