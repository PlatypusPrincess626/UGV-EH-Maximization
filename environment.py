# Standard Imports
from pvlib import spectrum, solarposition, irradiance, atmosphere
import pandas as pd
from numpy.typing import NDArray
import numpy as np
import random
from scipy import signal
from scipy.signal import windows
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
import math

# Custom Packages
import ugv_simulator


MIN_USABLE_ELEVATION = 12

###############################################################
# Episode date
#
# The episode window is anchored to solar noon on this date (see
# sim_env.__init__), not to a hardcoded clock time.
#
# The date matters far more than it looks. At the Yellowstone
# scene's latitude (44.424 N) the sun's peak elevation swings from
# ~22.5 deg on Jan 1 to ~69 deg on Jun 21, and get_obfuscation()
# short-circuits to "fully obstructed" below MIN_USABLE_ELEVATION.
# Hours per day above that threshold, at this latitude:
#
#   Jan 1   peak 22.5 deg   5.7 h usable   09:34 - 15:17
#   Mar 20  peak 45.1 deg   9.7 h usable   07:40 - 17:21
#   May 1   peak 60.4 deg  11.7 h usable   06:29 - 18:10
#   Jun 21  peak 69.0 deg  12.8 h usable   05:59 - 18:48
#   Aug 1   peak 63.8 deg  12.1 h usable   06:25 - 18:33
#   Sep 22  peak 46.2 deg   9.8 h usable   07:20 - 17:10
#
# The previous setting ('2021-01-01 8:00', freq='min', 720 steps)
# was a full 12-hour episode as intended, but only 5h43m of it had
# any sun: measured directly in a training run, directional_reward
# was exactly zero for 374 of 720 steps (51.9%), nonzero only for
# steps 94-439 -- i.e. 09:34 to 15:19, matching the table above to
# the minute. Half of every episode carried no positional signal
# at all, and no harvest was possible during it.
#
# Only late May through mid-August gives a usable window that
# actually covers a 720-minute episode at this latitude. Change
# EPISODE_DATE to a winter date only if a long dark phase is
# deliberately part of the scenario.
###############################################################
# Canopy geometry. PAD and the lateral foliage half-width in
# get_obfuscation are both derived from these, so raising tree heights
# no longer requires hunting down hardcoded constants.
MAX_TERRAIN_HEIGHT = 50.0
MAX_FOLIAGE_HEIGHT = 32.0
MAX_HALF_WIDTH = math.ceil(MAX_FOLIAGE_HEIGHT / 4.0)

EPISODE_DATE = "2021-06-21"
EPISODE_TZ = "MST"

###############################################################
# Foliage attenuation calibration
#
# Closed-canopy forest transmits roughly 0.5-2% of incident light
# to the forest floor at dense/rainforest sites, with more open or
# mixed-temperate stands running higher -- commonly cited figures
# span roughly 2% up to 10%+ depending on species, season, and
# canopy closure. TARGET_CANOPY_TRANSMITTANCE picks a point in that
# range for a "wooded, not dense-rainforest" area; REFERENCE_ELEVATION_DEG
# and REFERENCE_CANOPY_HEIGHT define the specific crossing (sun
# angle, tree height, straight through the canopy center) that
# transmittance is calibrated against. MAX_FOLIAGE_ATTENUATION is
# then solved for below rather than chosen directly, so the actual
# tunable knob is a real transmittance percentage, not an opaque
# per-step constant. Lower TARGET_CANOPY_TRANSMITTANCE for denser/
# darker forest, raise it for sparser woodland.
###############################################################
TARGET_CANOPY_TRANSMITTANCE = 0.15
REFERENCE_ELEVATION_DEG = 30.0
REFERENCE_CANOPY_HEIGHT = 15.0
MAX_FOLIAGE_ATTENUATION = (
    -math.log(TARGET_CANOPY_TRANSMITTANCE)
    * 3.0
    * math.tan(math.radians(REFERENCE_ELEVATION_DEG))
    / REFERENCE_CANOPY_HEIGHT
)


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

            # Centre the episode on solar transit rather than starting
            # at a fixed clock hour. Starting at a hardcoded 08:00
            # wasted time at both ends: 96 minutes before the sun
            # cleared MIN_USABLE_ELEVATION and 4h41m after it dropped
            # back below.
            #
            # Deriving the start from transit keeps the window centred
            # automatically if max_num_steps, EPISODE_DATE, or the
            # scene coordinates change later, instead of silently
            # drifting off-centre again.
            transit = solarposition.sun_rise_set_transit_spa(
                pd.DatetimeIndex([
                    pd.Timestamp(f"{EPISODE_DATE} 12:00", tz=EPISODE_TZ)
                ]),
                self.lat_center,
                self.long_center,
            )["transit"].iloc[0]

            start = (
                transit - pd.Timedelta(minutes=self.max_num_steps // 2)
            ).round("min")

            self.times = pd.date_range(start, freq=self.stepSize,
                                       periods=self.max_num_steps, tz=EPISODE_TZ)
            random.seed(EPISODE_DATE)

        # Worst-case obstruction height is terrain (max 50) PLUS the
        # tallest foliage on top of it.
        #
        # Derived from MAX_FOLIAGE_HEIGHT rather than hardcoded: this
        # was the literal 62.0 (50 + 12) and would have silently
        # truncated shadows once heights rose to 32 m, since PAD sets
        # how far outside the map obstructions are still tracked. Too
        # small a PAD means distant tall trees stop casting shade into
        # the arena at low sun elevations -- exactly the condition the
        # taller canopy is meant to create.
        self.PAD = math.ceil(
            (MAX_TERRAIN_HEIGHT + MAX_FOLIAGE_HEIGHT)
            / math.tan(math.radians(MIN_USABLE_ELEVATION))
        )

        self.r_move = self.dim
        self.env_map = self.make_map()
        self.view_dist = self.dim

        self.topo_mask, self.foliage_mask = None, None
        self._obfuscation_cache = {}
        self.obfuscation_array = self.init_interference()

    def reset(self):
        self.ch.reset()
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        r = self.boundary_radius * np.sqrt(np.random.rand())
        new_x = self.boundary_center[0] + r * np.cos(theta)
        new_y = self.boundary_center[1] + r * np.sin(theta)
        self.ch.update_telemetry(new_x, new_y, np.random.uniform(-np.pi, np.pi))
        self.reset_terrain()
        self.reset_foliage()
        self.ch.init_solar_potential(self)

    def step_simulation(self, current_step: int, target_x: float, target_y: float):
        battery_before = self.ch.get_battery()
        ugv_x, ugv_y, ugv_yaw = self.ch.get_position()
        new_position, battery_after = self.ch.step(self, current_step, float(target_x), float(target_y))

        lat_offset = new_position[1] * self.stp
        long_offset = new_position[0] * self.stp
        solpos = solarposition.get_solarposition(self.times[current_step],
                                                 self.lat_center + lat_offset,
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

        self.boundary_center = np.array([self.dim / 2.0, self.dim / 2.0])
        self.boundary_radius = 250.0

        flg_done = self.place_devices()
        return flg_done

    def init_interference(self):
        self.reset_terrain()
        self.reset_foliage()
        return True

    def reset_terrain(self):
        # topo_mask is about to change -- any cached obfuscation
        # results were computed against the OLD terrain and are no
        # longer valid.
        self._obfuscation_cache = {}

        # topo_mask must be padded on BOTH sides (2*PAD), matching
        # foliage_mask in reset_foliage() -- get_obfuscation()'s ray
        # march can travel up to `self.dim` cells from the sample
        # point and converts coordinates with a single `+ self.PAD`
        # offset, which only stays in-bounds if there's PAD of room
        # on both the negative AND positive side.
        target_shape = (self.dim + 2 * self.PAD, self.dim + 2 * self.PAD)

        data = self._generate_woodland_terrain(target_shape)
        data = data - np.min(data)
        max_val = np.max(data)
        if max_val > 0:
            data = (data / max_val) * 50.0

        self.topo_mask = data
        return True

    def _generate_woodland_terrain(self, shape, base_res=64, num_octaves=5, persistence=0.55):
        """
        Procedurally generates rolling, natural-looking terrain --
        multiple octaves of isotropic Gaussian-smoothed noise
        (fractional-Brownian-motion style) -- instead of loading a
        real DEM. This specifically avoids the kind of large, one-
        directional slope a real topo_data.tif turned out to have
        (~97m west-to-east across the whole map, ~3m north-to-south),
        which was confounding the RL task: it made "always move this
        way" a trivially reward-optimal strategy on its own, largely
        independent of the foliage dynamics the scenario is actually
        meant to be about. Isotropic Gaussian smoothing has no
        preferred direction by construction, so any residual slope
        here is finite-sample noise, not a structural feature.

        No fixed seed -- uses the ambient numpy random state, same
        as reset_foliage(), so a fresh draw is generated every call
        rather than reusing one fixed terrain the whole run. Called
        every reset() (like foliage), not just once at __init__, so
        no single draw's structure can be memorized or exploited as
        a persistent bias across training.

        Generated at a small base_res and upsampled (zoom) to the
        full target shape rather than smoothed directly at full
        resolution -- the broad, low-frequency octaves need large
        Gaussian sigmas, which are cheap on a small grid and very
        expensive (multiple seconds) directly on a ~1273x1273 array.
        Terrain doesn't need cell-level detail for this to look and
        behave like natural rolling terrain once upsampled.
        """
        result = np.zeros((base_res, base_res), dtype=float)
        amplitude = 1.0
        total_amplitude = 0.0

        for octave in range(num_octaves):
            sigma = max(base_res / (4.0 * (2 ** octave)), 1.0)
            noise = np.random.standard_normal((base_res, base_res))
            smoothed = gaussian_filter(noise, sigma=sigma)
            result += amplitude * smoothed
            total_amplitude += amplitude
            amplitude *= persistence

        result /= total_amplitude

        zoom_factors = (shape[0] / base_res, shape[1] / base_res)
        return zoom(result, zoom_factors, order=1)

    def reset_foliage(self):
        # foliage_mask is about to change -- same invalidation
        # reasoning as reset_terrain().
        self._obfuscation_cache = {}

        # Max height reduced from 20 to 12: more realistic for sparse
        # open woodland (mature/old-growth trees aren't the target
        # here), and directly cheaper -- smaller PAD, smaller
        # max_half_width in get_obfuscation's lateral foliage check
        # (ceil(12/4)=3 vs ceil(20/4)=5, cutting that inner loop from
        # 11 iterations to 7), and rays reach the foliage-band
        # termination sooner on average.
        # Density raised; HEIGHTS DELIBERATELY UNCHANGED.
        #
        #                mean height   canopy cover   PAD   inner loop
        #   previous        1.68 m         35%        292       7
        #   this            5.25 m         80%        292       7
        #   (with heights   14.0 m         80%        386      17)
        #
        # PAD and max_half_width are set by the MAXIMUM foliage
        # height, not by density, so raising density alone triples
        # effective canopy at no structural cost: the padded arrays
        # keep their size and get_obfuscation's lateral foliage check
        # keeps its 7-iteration inner loop. Raising heights to
        # [0,8,16,24,32] would grow PAD to 386 and that loop to 17,
        # roughly doubling runtime -- and with the environment
        # measured at 94.2% of wall clock, that is the expensive
        # direction. Revisit only if 80% cover proves insufficient.
        # Heights raised [0,3,6,9,12] -> [0,8,16,24,32].
        #
        # This was deferred in round 2 on compute grounds, and the last
        # run showed why it can no longer be deferred: `mean_solar_w`
        # across 11 episodes with different random start positions
        # spanned only 22.7-26.7 W, a +-8% spread. Over a 12-hour day
        # the sun sweeps and shadows move, so with 12 m trees almost
        # any fixed position averaged the same harvest and there was no
        # spatial decision to make. An untrained policy barely moving
        # (mean_abs_action 0.007) ended every episode at ~90% battery.
        #
        # Shadow length is height/tan(elevation):
        #
        #   elevation   12 m tree    32 m tree
        #     69 deg      4.6 m       12.3 m     (solar noon)
        #     45 deg     12.0 m       32.0 m
        #     30 deg     20.8 m       55.4 m
        #
        # At noon a 12 m tree casts a 4.6 m shadow -- shorter than the
        # clearing scale, so essentially every clearing was lit and all
        # positions looked alike. A 32 m tree casts 12.3 m, comparable
        # to the clearing scale, so only some clearings receive sun and
        # WHICH one the robot sits in starts to matter.
        #
        # Compute cost, since the environment is ~94% of wall clock:
        # PAD grows 292 -> 386 (padded array 1.92M -> 2.47M cells) and
        # get_obfuscation's lateral foliage loop goes 7 -> 17
        # iterations. Expect roughly a doubling of episode time.
        choices = np.array([0, 8, 16, 24, 32])
        probs = [0.20, 0.25, 0.25, 0.20, 0.10]

        dim_padded = self.dim + 2 * self.PAD

        # Smooth-then-quantile-map, replacing smooth-then-digitize.
        #
        # The previous approach drew heights from `probs` directly and
        # then ran gaussian_filter over them before binning with fixed
        # edges. Smoothing collapses variance -- with sigma=1.5 the
        # field's std fell to ~0.7 -- so the fixed bin edges no longer
        # matched the intended distribution. Measured on the raised
        # densities that produced 85% of cells at exactly 6 m and 100%
        # canopy cover: a uniform ceiling with no clearings at all,
        # which is a WORSE training environment than the sparse one,
        # not a harder one. (The same effect was already distorting
        # the old settings, turning an intended 35% cover into 61%
        # with only two heights ever appearing.)
        #
        # Mapping smoothed uniform noise through its own rank instead
        # decouples the two concerns: `probs` sets the marginal
        # distribution exactly, at any sigma, while sigma alone
        # controls patch size. Clearings come out ~5-12 m across,
        # comfortably larger than the ~4.6 m shadow a 12 m tree casts
        # at June-noon elevation, so open ground genuinely receives
        # sun, and smaller than the 20 m view distance, so the agent
        # can perceive one when adjacent to it.
        # Raised 2.5 -> 5.0, together with the taller canopy above.
        #
        # These two have to move together. Taller trees alone would
        # overshoot: a 12.3 m noon shadow against 6.3 m mean clearings
        # would leave NOTHING lit, which is as uninformative as
        # everything being lit. Clearings must straddle the shadow
        # length so that some are sunlit and some are not.
        #
        #   sigma   mean clearing   p90    % wider than 12.3 m shadow
        #    2.5        6.3 m      12.0 m          8.2%
        #    4.0       10.0 m      18.0 m         25.3%
        #    5.0       12.7 m      25.0 m         37.7%
        #    6.5       15.4 m      29.0 m         53.7%
        #
        # 5.0 puts the mean clearing (12.7 m) right at the noon shadow
        # length, with 38% of clearings wide enough to hold sun. That
        # is the spread the policy can actually learn to exploit --
        # roughly a third of open ground is worth occupying, rather
        # than all of it or none of it.
        FOLIAGE_PATCH_SIGMA = 5.0

        field = gaussian_filter(
            np.random.rand(dim_padded, dim_padded), sigma=FOLIAGE_PATCH_SIGMA
        )

        # Rank each cell within the field, then cut at the cumulative
        # target probabilities.
        ranks = field.ravel().argsort().argsort().reshape(field.shape)
        quantiles = ranks / (field.size - 1)
        self.foliage_mask = choices[np.digitize(quantiles, np.cumsum(probs)[:-1])]

        return True

    def place_devices(self) -> list:
        sensor_pts = np.array([[0, 0]] * self.total_sensors, np.int32)

        print("Placing Sensors")
        for sensor in range(self.total_sensors):
            position = random.randint(0, self.dim * self.dim - 1)
            sensor_pts[sensor] = [int(position % self.dim), int(position / self.dim)]

        # KMeans with a single cluster's centroid is, by definition,
        # just the arithmetic mean of the points -- sklearn's n_init=10
        # restarts and iterative convergence check have nothing to
        # search over when there's only one possible partition, so
        # this was pure overhead for the same answer, re-paid every
        # episode.
        cluster = sensor_pts.mean(axis=0)

        self.r_move = 500
        self.sensor_pts = sensor_pts
        self.ch_pt = cluster.astype(int).tolist()  # FIXED: Removed undefined 'centroids' reference

        # FIXED: Pass r_move parameter directly to initialization safely
        self.ch = ugv_simulator.UGVSimulator(
            self,
            x=float(self.ch_pt[0]),
            y=float(self.ch_pt[1]),
            yaw=np.pi / 4,
            battery_level=100,
            r_move=self.r_move
        )
        return True

    def get_spectrum(self, x, y, tilt, azimuth, step):
        lat_offset = y * self.stp
        long_offset = x * self.stp

        solpos = solarposition.get_solarposition(self.times[step],
                                                 self.lat_center + lat_offset,
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
        # get_obfuscation is one of the most expensive calls in the
        # whole simulation (a per-pixel ray march over the whole
        # patch), and the same (position, step) query is frequently
        # made multiple times per environment step: obs() (for the
        # sequence history), the reward's before/after computation in
        # main.py, step_simulation()'s own next_local_observation, and
        # harvest_energy()'s internal call (via find_power()) all
        # often land on the identical final position and step -- up
        # to 4-6 full recomputations for what's really only 2 distinct
        # queries (before-position, after-position) per step. Caching
        # here, rather than restructuring every call site, fixes the
        # redundancy in one place regardless of who's calling it.
        #
        # This also incidentally fixes the "computes a whole 41x41
        # patch just to read one center pixel" waste in
        # find_power()/harvest_energy(): if that query was already
        # computed elsewhere this step (very likely, per above), it's
        # now a cache hit instead of a second full ray march.
        key = (
            int(x), int(y), int(step),
            round(float(azimuth), 6), round(float(zenith), 6),
            int(self.view_dist),
        )
        cached = self._obfuscation_cache.get(key)
        if cached is not None:
            return cached.copy()

        result = self._compute_obfuscation(x, y, step, azimuth, zenith)

        # Small bound, not unlimited growth -- the redundancy this
        # matters for is almost entirely within a single step; across
        # steps, positions differ, so old entries are rarely reused
        # anyway and don't need to be kept around.
        if len(self._obfuscation_cache) >= 8:
            self._obfuscation_cache.pop(next(iter(self._obfuscation_cache)))
        self._obfuscation_cache[key] = result

        return result.copy()

    def _compute_obfuscation(self, x: int, y: int, step, azimuth: float, zenith: float):
        # Vectorized over the whole patch at once, rather than a
        # per-pixel Python loop. The key property that makes this
        # tractable: d's loop bound depends only on tan_elevation,
        # fixed for the whole patch, not per-pixel; and k's loop bound
        # (half_width = ceil(canopy_radius)) only takes 5 possible
        # values since foliage_height is drawn from {0,5,10,15,20} by
        # construction (reset_foliage()). So both "variable-length"
        # loops are actually fixed-size, and each (d,k) iteration is
        # done as one numpy operation across the whole patch, with a
        # persistent `terminated` mask replacing the original's
        # per-pixel `break` statements -- once a pixel is marked
        # terminated (by terrain block, going out of bounds, or
        # dropping below the transmittance threshold), it stops being
        # updated for all later iterations, exactly mirroring where
        # the original's break would have stopped it.
        #
        # Terrain/foliage heights are compared relative to each
        # patch pixel's OWN terrain elevation (observer_terrain), not
        # an absolute zero. Without this, terrain averaging ~25 (after
        # min-max normalization to [0,50]) almost always exceeded the
        # near-zero sun-clearance height needed at short ray
        # distances, self-shadowing nearly every pixel regardless of
        # true line-of-sight -- this was the actual reason
        # directional_reward computed to exactly zero every step in
        # real training data, not a scale/weighting issue.
        v_dist = int(self.view_dist)
        patch_size = 2 * v_dist + 1

        if zenith >= 90.0 - MIN_USABLE_ELEVATION:
            return np.ones((patch_size, patch_size), dtype=np.float32)

        az_rad = math.radians(90.0 - azimuth)
        el_rad = math.radians(90.0 - zenith)
        tan_elevation = math.tan(el_rad)
        step_x = math.cos(az_rad)
        step_y = math.sin(az_rad)
        perp_x = -step_y
        perp_y = step_x

        center_x, center_y = int(x), int(y)

        jj, ii = np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='ij')
        global_x = center_x - v_dist + ii
        global_y = center_y - v_dist + jj
        in_map = (global_x >= 0) & (global_x < self.dim) & (global_y >= 0) & (global_y < self.dim)

        transmittance = np.ones((patch_size, patch_size), dtype=np.float64)
        terminated = ~in_map

        topo_h, topo_w = self.topo_mask.shape
        fol_h, fol_w = self.foliage_mask.shape

        # Each patch pixel's own terrain elevation, fetched once --
        # the baseline the sunline-clearance height is measured from.
        obs_x_safe = np.clip(global_x + self.PAD, 0, topo_w - 1)
        obs_y_safe = np.clip(global_y + self.PAD, 0, topo_h - 1)
        observer_terrain = self.topo_mask[obs_y_safe, obs_x_safe]

        # Worst case: terrain (max 50) + foliage on top of it (max 20)
        # = 70 possible obstruction height above an observer at 0.
        d_max = int(math.floor(62.0 / tan_elevation)) + 2
        # ceil(MAX_FOLIAGE_HEIGHT / 4) -- the largest canopy_radius any
        # tree can produce. Derived rather than hardcoded: at 3 (the
        # value for 12 m trees) a 32 m tree's canopy would be clipped
        # to a quarter of its true width.
        max_half_width = MAX_HALF_WIDTH

        for d in range(1, d_max):
            h_min = d * tan_elevation
            if h_min > 62.0:
                break

            active = ~terminated
            if not active.any():
                break

            sunline_height = observer_terrain + h_min

            ray_x = np.round(global_x + d * step_x).astype(np.int64)
            ray_y = np.round(global_y + d * step_y).astype(np.int64)
            ray_x_arr = ray_x + self.PAD
            ray_y_arr = ray_y + self.PAD

            in_bounds = (ray_x_arr >= 0) & (ray_x_arr < topo_w) & (ray_y_arr >= 0) & (ray_y_arr < topo_h)
            terminated = terminated | (active & (~in_bounds))
            active = ~terminated

            ray_x_safe = np.clip(ray_x_arr, 0, topo_w - 1)
            ray_y_safe = np.clip(ray_y_arr, 0, topo_h - 1)

            terrain_height = self.topo_mask[ray_y_safe, ray_x_safe]
            blocked = active & (terrain_height >= sunline_height)
            transmittance = np.where(blocked, 0.0, transmittance)
            terminated = terminated | blocked
            active = ~terminated

            # Foliage sits on top of the ray point's own local terrain,
            # not at an absolute height -- canopy_start/canopy_top are
            # therefore terrain_height + (a height above that ground).
            foliage_height = self.foliage_mask[ray_y_safe, ray_x_safe].astype(np.float64)
            has_foliage = active & (foliage_height > 0)
            canopy_start_abs = terrain_height + foliage_height / 3.0
            canopy_top_abs = terrain_height + foliage_height
            canopy_radius = foliage_height / 4.0
            in_band = has_foliage & (sunline_height >= canopy_start_abs) & (sunline_height <= canopy_top_abs)

            local_attenuation = np.zeros_like(transmittance)

            if in_band.any():
                for k in range(-max_half_width, max_half_width + 1):
                    check_x = np.round(ray_x + k * perp_x).astype(np.int64)
                    check_y = np.round(ray_y + k * perp_y).astype(np.int64)
                    check_x_arr = check_x + self.PAD
                    check_y_arr = check_y + self.PAD

                    k_in_bounds = (
                        (check_x_arr >= 0) & (check_x_arr < fol_w)
                        & (check_y_arr >= 0) & (check_y_arr < fol_h)
                    )
                    check_x_safe = np.clip(check_x_arr, 0, fol_w - 1)
                    check_y_safe = np.clip(check_y_arr, 0, fol_h - 1)
                    check_foliage = self.foliage_mask[check_y_safe, check_x_safe]

                    r = abs(k)
                    valid_k = in_band & k_in_bounds & (check_foliage > 0) & (r <= canopy_radius)

                    safe_radius = np.where(canopy_radius > 0, canopy_radius, 1.0)
                    r_norm = r / safe_radius
                    density = np.exp(-2.0 * r_norm * r_norm)
                    safe_denom = np.where(canopy_top_abs > canopy_start_abs, canopy_top_abs - canopy_start_abs, 1.0)
                    depth_frac = (sunline_height - canopy_start_abs) / safe_denom
                    attenuation = density * MAX_FOLIAGE_ATTENUATION * depth_frac
                    attenuation = np.where(valid_k, attenuation, 0.0)
                    local_attenuation = np.maximum(local_attenuation, attenuation)

            new_transmittance = transmittance * (1.0 - local_attenuation)
            transmittance = np.where(in_band, new_transmittance, transmittance)
            drop_below = in_band & (transmittance < 0.01)
            transmittance = np.where(drop_below, 0.0, transmittance)
            terminated = terminated | drop_below

        return np.where(in_map, 1.0 - transmittance, 1.0).astype(np.float32)