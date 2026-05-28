import numpy as np
from scipy.integrate import trapezoid
import math

class UGVSimulator:
    """
    Simulator for an Unmanned Ground Vehicle with movement control and telemetry capabilities.
    """
    def __init__(self, x=0.0, y=0.0, yaw=3.14159/4, battery_level=75, r_move=800.0):
        self.origin = [x, y, yaw]
        self.x = x
        self.y = y
        self.yaw = yaw
        self.movement_times = [0.0, 0.0]
        self.status = 'operational'

        self._comms = {
            "max_dist_lora": 5_000,  # LoRa maximum distance (m)
            "max_bitrate_lora": 24_975,  # LoRa maximum bitrate (bps)
            "current_active_lora": 4_200,  # LoRa active current (muA)
            "current_sleep_lora": 1_200,  # LoRa dormant current (muA)
            "voltage_lora": 3.7,  # LoRa voltage requirements (V)
            "pow_active_lora": 20,  # LoRa active power (muW)
            "pow_sleep_lora": 4,  # LoRa dormant power (muW)

            "max_dist_ambc": 800,  # AmBC maximum distance (m)
            "max_bitrate_ambc": 1_592,  # AmBC maximum bitrate (bps)
            "pow_ambc": 3,  # AmBC upkeep power (muW)
            "voltage_ambc": 3.3,  # AmBC voltage requirements (V)
            "current_ambc": 785  # AmBC upkeep current (muA)
        }

        # Battery Specs
        self.battery_level = min(max(battery_level, 0), 100)
        self.max_energy = 12_000  # Maximum battery capacity (mAh)
        self.charge_rate = 1 / 3  # Charge rate of battery (A/h)
        self.discharge_rate = 0.655  # Discharge rate of batter (A/h)
        self.battery_voltage = 15
        self.uav_charge_amp = 10  # A
        self.total_amp_spent = 0.0
        self.step_charge = 100 / (self.charge_rate * 60)

        # Movement Specs
        self.r_max = r_move
        self.speed = 20 / 60  # m / s
        self.yaw_speed = math.pi / 10
        self.acceleration = 300  # m / min**2
        self.u_rr = 0.1
        self.w = 21  # N

        # CPU Specs
        self.pow_cpu = 3.7  # CPU power (muW)
        self.current_cpu = 1_000  # CPU current drain (muA)

        # Solar Specs
        self.spectral_low = 0  # Lower bandwidth bound
        self.spectral_high = np.inf  # Upper bandwidth bound
        self.is_solar = True  # Flag for powered
        self.azimuth = 180
        self.tilt = 45
        self.h = 0
        self.solar_area = 1.020 * 0.520  # 1020 mm  x 520 mm
        self.solar_voltage = 18  # V
        self.solar_current = 6  # A
        self.solar_power = 100  # W

        self.spectral_response = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.23], [0.25], [0.27], [0.29], [0.31], [0.33], [0.3376], [0.3452], [0.3529],
                                           [0.3605], [0.3681], [0.3757], [0.3833], [0.3910], [0.3986], [0.4062],
                                           [0.4138], [0.4214], [0.4290], [0.4367], [0.4443], [0.4595], [0.4694],
                                           [0.4824], [0.4976], [0.5174], [0.5263], [0.5433], [0.5586], [0.5647],
                                           [0.5695], [0.5814], [0.5910], [0.5948], [0.5986], [0.6024], [0.6119],
                                           [0.6271], [0.6393], [0.6427], [0.6274], [0.6107], [0.5714], [0.5321],
                                           [0.4830], [0.4634], [0.4437], [0.4339], [0.4202], [0.3986], [0.3652],
                                           [0.3357], [0.3092], [0.2179], [0.1589], [10.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])

    def reset(self):
        self.battery_level = 100
        self.total_amp_spent = 0.0
        self.x, self.y, self.yaw = self.origin
        self.movement_times = [0.0, 0.0]

    def get_position(self):
        return self.x, self.y, self.yaw

    def get_battery(self):
        return self.battery_level

    def move(self, target_x, target_y, duration):
        if self.status == 'maintenance':
            raise Exception("UGV in maintenance mode - cannot move")
        # Vector to target
        dx = target_x - self.x
        dy = target_y - self.y
        target_angle = math.atan2(dy, dx)
        # Current orientation
        current_yaw = self.yaw
        self.movement_times = [0.0, 0.0]

        # ---------- Rotation phase ----------
        # Shortest angular difference
        delta = target_angle - current_yaw
        delta = math.atan2(math.sin(delta), math.cos(delta))
        # Time needed to rotate to the target angle at the given yaw_speed
        time_to_turn = abs(delta) / self.yaw_speed
        # Remaining time for travel
        travel_time = max(0.0, duration - time_to_turn)
        self.movement_times[0] = time_to_turn
        # If we spent all or more than the budget on turning, just orient and exit
        if travel_time <= 0:
            self.yaw = target_angle
            return self.x, self.y, self.yaw

        # ---------- Translation phase ----------
        # Total Euclidean distance to the target
        total_distance = math.hypot(dx, dy)
        # Distance we can cover in the remaining time
        distance_to_cover = self.speed * travel_time
        self.yaw = target_angle  # Face the target after turning

        if total_distance == 0:
            # Already at the target
            self.x = target_x
            self.y = target_y
        # If we can reach the target within the remaining time, go there directly
        elif distance_to_cover >= total_distance:
            self.x = target_x
            self.y = target_y
            self.movement_times[1] = total_distance / self.speed
        else:
            # Otherwise move only a fraction of the way
            fraction = distance_to_cover / total_distance
            self.x += dx * fraction
            self.y += dy * fraction
            self.movement_times[1] = travel_time
        return self.x, self.y, self.yaw

    def update_telemetry(self, new_x, new_y, new_yaw):
        self.x = new_x
        self.y = new_y
        self.yaw = new_yaw

    def set_status(self, new_status):
        new_status = new_status.lower()
        if new_status not in ['operational', 'maintenance', 'charging']:
            raise ValueError("Invalid status")
        self.status = new_status

    def __str__(self):
        return (f"UGV Position: ({self.x:.2f}m, {self.y:.2f}m)\n"
                f"Orientation: {self.yaw:.1f} rad\n"
                f"Speed: {self.speed:.2f} m/s\n"
                f"Battery: {self.battery_level:.1f}%\n"
                f"Status: {self.status.capitalize()}")

    def find_power(self, env, x: int, y: int, step: int,
                   sol_area: int, tilt: int, azimuth: int):
        # 1. Fetch the spectrum dictionary from pvlib
        spectra = env.get_spectrum(self.x, self.y, tilt, azimuth, step)
        interference = env.get_obfuscation(x, y, step)

        # 2. Extract and force arrays to be flat 1D vectors
        wavelengths = np.atleast_1d(spectra['wavelength']).flatten()
        poa_global = np.atleast_1d(spectra['poa_global']).flatten()

        # CRITICAL: If poa_global was generated as a 2D grid matrix,
        # extract only the first point's worth of data to match the wavelength dimension
        if len(poa_global) != len(wavelengths):
            # Slice poa_global down to match the exact length of the wavelength array
            poa_global = poa_global[:len(wavelengths)]

        # 3. Dynamically adapt your spectral_response vector to match target wavelengths
        raw_response = self.spectral_response.flatten()

        if len(raw_response) == len(wavelengths):
            response_1d = raw_response
        else:
            # Interpolate or slice your hardcoded spectral response dynamically
            # so it perfectly aligns with whatever shape pvlib outputs (e.g., 121 vs 122)
            response_1d = np.interp(
                wavelengths,
                np.linspace(wavelengths[0], wavelengths[-1], len(raw_response)),
                raw_response
            )

        # 4. Calculate the integrand (Shapes are now guaranteed to match)
        integrand = poa_global * response_1d

        # 5. Integrate along the wavelength axis using modern scipy
        from scipy.integrate import trapezoid
        cell_current = trapezoid(integrand, wavelengths)

        if np.isnan(cell_current):
            cell_current = 0.0

        # Apply your empirical scaling factors
        a = step / 60 + 2
        alpha = abs(104 - 65 * a + 47 * pow(a, 2) - 12 * pow(a, 3) + pow(a, 4))
        power = abs(alpha / 100) * (1-interference) * cell_current * sol_area
        return power

    def harvest_energy(self, env, step):
        curr_x = int(max(0, min(self.x, env.dim - 1)))
        curr_y = int(max(0, min(self.y, env.dim - 1)))

        power = self.find_power(env, curr_x, curr_y, step, self.solar_area, self.tilt, self.azimuth)
        # Check if the machine is powered
        if power / self.solar_current > self.solar_voltage * 0.6:
            self.is_solar = True
        else:
            self.is_solar = False
        amp_upkeep = (self.current_cpu + self._comms.get("current_active_lora")) / 1_000  # to mA
        self.total_amp_spent += amp_upkeep

    def battery_step(self):
        self.battery_level -= (self.total_amp_spent / self.max_energy) * 100  # Convert ratio to capacity %
        if self.is_solar:
            self.battery_level = min(100.0, max(0.0, (self.battery_level + self.step_charge)))
        return self.battery_level

    def step(self, env, step, target_x, target_y):
        self.total_amp_spent = 0.0
        self.move(float(target_x), float(target_y), 60.0)
        self.total_amp_spent += self.w * self.speed * self.u_rr * sum(self.movement_times) / 3600 # seconds
        self.harvest_energy(env, step)
        self.battery_step()
        return self.get_position(), self.get_battery()