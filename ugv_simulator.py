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
        self.status = "operational"

        # -----------------------------
        # Communications
        # -----------------------------
        self._comms = {
            "current_active_lora": 4200,  # uA
            "current_sleep_lora": 1200,  # uA
        }

        # -----------------------------
        # Battery Model
        # -----------------------------
        self.max_capacity_mAh = 12000.0
        self.battery_mAh = self.max_capacity_mAh * (battery_level / 100.0)

        self.battery_voltage = 15.0

        # energy accumulated during current step
        self.energy_used_mAh = 0.0
        self.energy_gained_mAh = 0.0

        # -----------------------------
        # Motion
        # -----------------------------
        self.r_max = r_move

        self.speed = 20 / 60.0  # m/s
        self.yaw_speed = math.pi / 10

        # Replace rolling-resistance approximation
        # with realistic motor power draw
        self.motor_power_w = 150.0

        # -----------------------------
        # CPU
        # -----------------------------
        self.current_cpu = 1000  # uA

        # -----------------------------
        # Solar
        # -----------------------------
        self.is_solar = True
        self.azimuth = 180
        self.tilt = 45

        self.solar_area = 1.020 * 0.520
        self.solar_voltage = 18
        self.solar_current = 6

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
        self.battery_mAh = self.max_capacity_mAh
        self.x, self.y, self.yaw = self.origin
        self.energy_used_mAh = 0.0
        self.energy_gained_mAh = 0.0

    def get_position(self):
        return self.x, self.y, self.yaw

    def get_battery(self):
        return 100.0 * self.battery_mAh / self.max_capacity_mAh

    def move(self, target_x, target_y, duration):

        dx = target_x - self.x
        dy = target_y - self.y

        target_angle = math.atan2(dy, dx)

        delta = target_angle - self.yaw
        delta = math.atan2(math.sin(delta), math.cos(delta))

        turn_time = abs(delta) / self.yaw_speed

        travel_time = max(0.0, duration - turn_time)

        self.movement_times = [turn_time, 0.0]

        if travel_time <= 0:
            self.yaw = target_angle
            return self.x, self.y, self.yaw

        total_distance = math.hypot(dx, dy)
        distance_possible = self.speed * travel_time

        self.yaw = target_angle

        if total_distance <= distance_possible:
            self.x = target_x
            self.y = target_y
            self.movement_times[1] = total_distance / self.speed
        else:
            frac = distance_possible / total_distance
            self.x += dx * frac
            self.y += dy * frac
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

    def find_power(self, env, x, y, step,
                   sol_area, tilt, azimuth):

        spectra, solpos = env.get_spectrum(
            self.x, self.y, tilt, azimuth, step
        )

        sun_azimuth = solpos["azimuth"].iloc[0]
        sun_zenith = solpos["apparent_zenith"].iloc[0]

        obfuscation_patch = env.get_obfuscation(
            x, y, step, sun_azimuth, sun_zenith
        )

        interference = obfuscation_patch[
            int(env.view_dist),
            int(env.view_dist)
        ]

        wavelengths = np.atleast_1d(
            spectra["wavelength"]
        ).flatten()

        poa_global = np.atleast_1d(
            spectra["poa_global"]
        ).flatten()

        if len(poa_global) != len(wavelengths):
            poa_global = poa_global[:len(wavelengths)]

        raw_response = self.spectral_response.flatten()

        if len(raw_response) == len(wavelengths):
            response_1d = raw_response
        else:
            response_1d = np.interp(
                wavelengths,
                np.linspace(
                    wavelengths[0],
                    wavelengths[-1],
                    len(raw_response)
                ),
                raw_response
            )

        integrand = poa_global * response_1d

        cell_current = trapezoid(
            integrand,
            wavelengths
        )

        if np.isnan(cell_current):
            cell_current = 0.0

        a = step / 60 + 2
        alpha = abs(
            104 - 65 * a + 47 * a ** 2 - 12 * a ** 3 + a ** 4
        )

        power = (
                abs(alpha / 100.0)
                * (1.0 - interference)
                * cell_current
                * sol_area
        )

        return max(0.0, power)

    def harvest_energy(self, env, step):

        curr_x = int(max(0, min(self.x, env.dim - 1)))
        curr_y = int(max(0, min(self.y, env.dim - 1)))

        solar_power_w = self.find_power(
            env,
            curr_x,
            curr_y,
            step,
            self.solar_area,
            self.tilt,
            self.azimuth
        )

        charge_current_mA = (solar_power_w / self.battery_voltage) * 1000.0

        self.energy_gained_mAh += (charge_current_mA * (1.0 / 60.0))

    def battery_step(self):

        self.battery_mAh += self.energy_gained_mAh
        self.battery_mAh -= self.energy_used_mAh

        self.battery_mAh = max(
            0.0,
            min(
                self.max_capacity_mAh,
                self.battery_mAh
            )
        )

    def consume_idle_energy(self):

        current_mA = (
            self.current_cpu +
            self._comms["current_active_lora"]
        ) / 1000.0

        self.energy_used_mAh += (
            current_mA * (1.0 / 60.0)
        )

    def consume_motion_energy(self):

        move_seconds = sum(self.movement_times)

        energy_wh = (
            self.motor_power_w *
            move_seconds / 3600.0
        )

        current_mAh = (
            energy_wh * 1000.0
        ) / self.battery_voltage

        self.energy_used_mAh += current_mAh

    def step(self, env, step, target_x, target_y):

        self.energy_used_mAh = 0.0
        self.energy_gained_mAh = 0.0

        self.move(
            float(target_x),
            float(target_y),
            60.0
        )

        self.consume_motion_energy()
        self.consume_idle_energy()
        self.harvest_energy(env, step)

        self.battery_step()

        return (
            self.get_position(),
            self.get_battery()
        )