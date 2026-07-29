import numpy as np
from scipy.integrate import trapezoid
import math
import random

class UGVSimulator:
    """
    Simulator for an Unmanned Ground Vehicle with movement control and telemetry capabilities.
    """
    def __init__(self, env, x=0.0, y=0.0, yaw=3.14159/4, battery_level=75, r_move=800.0):
        self.r_max = r_move
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

        # --------------------------------------------------
        # Battery Model
        # --------------------------------------------------

        self.capacity_Ah = 12.0
        self.max_capacity_mAh = self.capacity_Ah * 1000.0

        self.battery_mAh = (
                self.max_capacity_mAh *
                battery_level / 100.0
        )

        # State of Charge
        self.soc = self.battery_mAh / self.max_capacity_mAh

        # Battery parameters
        self.nominal_voltage = 14.8
        self.cutoff_voltage = 12.0
        self.full_voltage = 16.8

        # Internal resistance (Ohms)
        self.internal_resistance = 0.045

        # Coulombic efficiencies
        self.charge_efficiency = 0.97
        self.discharge_efficiency = 0.98

        # Capacity fade
        self.capacity_loss = 0.0

        # Peak charging current
        self.max_charge_current = 8.0  # Amps

        # Peak discharge current
        self.max_discharge_current = 10.0

        # --------------------------------------------------
        # Solar Charge Controller
        # --------------------------------------------------

        # MPPT efficiency
        self.mppt_efficiency = 0.96

        # Controller voltage
        self.charge_voltage = 16.8  # V

        # Wiring losses
        self.wiring_efficiency = 0.99

        # Dust/aging on panel
        self.panel_efficiency_factor = 0.98

        # --------------------------------------------------
        # Vehicle Dynamics
        # --------------------------------------------------

        # Vehicle Properties
        self.mass = 40.0  # kg
        self.gravity = 9.81

        # Forest terrain
        # Both increased from their original values (0.04, 18.0):
        # final_battery was landing at ~73.5% with a coefficient of
        # variation of only ~0.4% across very different policies at
        # very different training stages (episode 1 through 293) --
        # idle draw alone accounts for <1% of a full episode's
        # capacity, so motion cost was almost entirely determining
        # outcome, yet barely varying it. rolling_coeff (a fixed cost
        # of any forward movement, regardless of pattern) makes
        # constant movement itself more expensive, encouraging
        # settling into a good position rather than continuously
        # moving regardless of benefit. turn_drag_coeff (scales with
        # angular_velocity * velocity) specifically targets erratic,
        # turning-heavy movement -- directly relevant, since the
        # diagnosed boundary-riding behavior (>96% of an episode
        # spent within 2 units of the boundary radius) requires
        # continuously turning to follow the boundary's curve.
        self.rolling_coeff = 0.07
        self.turn_drag_coeff = 40.0

        # Drivetrain
        self.motor_efficiency = 0.90
        self.gearbox_efficiency = 0.95
        self.controller_efficiency = 0.97

        self.drivetrain_efficiency = (
                self.motor_efficiency *
                self.gearbox_efficiency *
                self.controller_efficiency
        )

        # Velocity State
        self.velocity = 0.0  # m/s
        self.target_velocity = 0.0

        self.max_velocity = 0.50  # m/s
        self.max_acceleration = 0.25  # m/s²
        self.max_deceleration = 0.35  # m/s²

        # Turning
        self.angular_velocity = 0.0
        self.max_turn_rate = math.pi / 8
        self.max_turn_accel = math.pi / 12

        # Simulation timestep
        self.dt = 1.0  # seconds

        # Number of dt-second physics ticks each call to step() advances
        # through. Referenced explicitly here AND in step()'s inner loop
        # AND in harvest_energy()'s duration conversion, so all three
        # stay in sync by construction rather than by two hardcoded
        # numbers (a `range(60)` and a `/60.0`) happening to agree.
        self.ticks_per_step = 60

        # -----------------------------
        # CPU
        # -----------------------------
        self.current_cpu = 1000  # uA

        # -----------------------------
        # -----------------------------
        self.current_payload = 960_000  # uA (960 mA)

        # -----------------------------
        # Solar
        # -----------------------------
        self.is_solar = True
        self.azimuth = 180
        self.tilt = 45

        # Reduced to ~65% of the original panel area (1.020*0.520):
        # harvesting was generous enough that final_battery landed
        # around ~73.5% almost regardless of policy quality (see
        # rolling_coeff/turn_drag_coeff comment above for the full
        # diagnosis). A smaller panel means harvest is more directly
        # limited by actual sun exposure -- good positioning (now
        # that directional_reward rewards it directly) should matter
        # more to the battery outcome, not just to a mostly-decorative
        # reward term riding on top of an outcome the panel size
        # already guaranteed.
        self.solar_area = (1.020 * 0.520) * 0.45

        self.panel_fill_factor = 0.75

        self.cell_efficiency = 0.20

        self.max_cell_efficiency = 0.22

        self.reference_response_ratio = 0.202
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
                                           [0.3357], [0.3092], [0.2179], [0.1589], [0.05], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])

        self.solar_potential = 0.0

    START_SOC_MIN = 0.30
    START_SOC_MAX = 0.40

    def reset(self):
        self.battery_mAh = self.max_capacity_mAh * random.uniform(
            self.START_SOC_MIN, self.START_SOC_MAX
        )
        self.update_soc()
        self.x, self.y, self.yaw = self.origin
        self.velocity = 0.0
        self.angular_velocity = 0.0
        self.target_velocity = 0.0
        self.energy_used_mAh = 0.0
        self.energy_gained_mAh = 0.0
        self.step_idle_mAh = 0.0
        self.step_motion_mAh = 0.0
        self.step_path_m = 0.0
        self.step_turn_integral = 0.0

    def init_solar_potential(self, env):
        curr_x = int(max(0, min(self.x, env.dim - 1)))
        curr_y = int(max(0, min(self.y, env.dim - 1)))
        safe_step = min(int(math.floor(0.0)), len(env.times) - 1)

        panel_power = self.find_power(
            env,
            curr_x,
            curr_y,
            safe_step,
            self.solar_area,
            self.tilt,
            self.azimuth
        )

        panel_power *= self.panel_efficiency_factor * self.wiring_efficiency * self.mppt_efficiency
        self.solar_potential = panel_power
        return True

    def get_position(self):
        return self.x, self.y, self.yaw

    def get_battery(self):
        return self.soc * 100.0

    def get_solar_potential(self):
        return self.solar_potential

    def update_soc(self):

        self.soc = max(0.0, min(1.0, self.battery_mAh / self.max_capacity_mAh))

    def compute_open_circuit_voltage(self):
        s = self.soc
        voltage = (12.0 + 3.0 * s + 1.2 * s ** 2 + 0.6 * s ** 3)

        return min(self.full_voltage, max(self.cutoff_voltage, voltage))

    def compute_internal_resistance(self):
        s = self.soc
        return self.internal_resistance * (1.0 + 1.8 * (1 - s) ** 2)

    def compute_terminal_voltage(self, current_A=0.0):
        ocv = self.compute_open_circuit_voltage()
        resistance = self.compute_internal_resistance()
        terminal = ocv - current_A * resistance

        return max(self.cutoff_voltage, terminal)

    def compute_charge_acceptance(self):
        s = self.soc

        if s < 0.80:
            return 1.00
        elif s < 0.90:
            return 0.90
        elif s < 0.95:
            return 0.65
        elif s < 0.98:
            return 0.35
        else:
            return 0.15

    def update_vehicle_dynamics(self, target_x, target_y):
        dx = target_x - self.x
        dy = target_y - self.y

        distance = math.hypot(dx, dy)

        arrival_tolerance = self.max_velocity * self.dt

        desired_heading = math.atan2(dy, dx)
        heading_error = desired_heading - self.yaw
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        if distance < arrival_tolerance:
            self.target_velocity = 0.0
        else:
            stopping_distance = (
                (self.velocity * self.velocity) / (2.0 * self.max_deceleration)
                + self.max_velocity * self.dt
            )
            approach = min(1.0, distance / max(stopping_distance, 1e-6))
            alignment = max(0.0, math.cos(heading_error))
            self.target_velocity = self.max_velocity * approach * alignment

        desired_turn_rate = max(-self.max_turn_rate, min(self.max_turn_rate, heading_error))
        delta_turn = (desired_turn_rate - self.angular_velocity)
        max_turn_change = self.max_turn_accel * self.dt
        delta_turn = max(-max_turn_change, min(max_turn_change, delta_turn))

        self.angular_velocity += delta_turn
        self.yaw += (self.angular_velocity * self.dt)

        speed_error = self.target_velocity - self.velocity

        if speed_error >= 0:
            accel = min(self.max_acceleration, speed_error)

        else:
            accel = max(-self.max_deceleration, speed_error)

        self.velocity += accel * self.dt
        self.velocity = max(0.0, min(self.velocity, self.max_velocity))
        self.x += self.velocity * math.cos(self.yaw) * self.dt
        self.y += self.velocity * math.sin(self.yaw) * self.dt

    def compute_drive_forces(self):
        rolling_force = self.rolling_coeff * self.mass * self.gravity
        acceleration = self.target_velocity - self.velocity
        acceleration_force = self.mass * acceleration
        turning_force = self.turn_drag_coeff * abs(self.angular_velocity) * self.velocity
        total_force = rolling_force + max(0.0, acceleration_force) + turning_force

        return {"rolling": rolling_force,
                "acceleration": acceleration_force,
                "turning": turning_force,
                "total": total_force}

    def compute_motor_power(self):
        forces = self.compute_drive_forces()
        mechanical_power = forces["total"] * self.velocity
        battery_power = mechanical_power / self.drivetrain_efficiency

        return max(0.0, battery_power)

    def update_battery_state(self):
        charge = self.energy_gained_mAh * self.compute_charge_acceptance() * self.charge_efficiency
        discharge = self.energy_used_mAh / self.discharge_efficiency

        self.battery_mAh += charge
        self.battery_mAh -= discharge
        self.battery_mAh = max(0.0, min(self.max_capacity_mAh, self.battery_mAh))
        self.update_soc()

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
                f"Velocity: {self.velocity:.2f} m/s\n"
                f"Battery: {self.get_battery():.1f}%\n"
                f"Status: {self.status.capitalize()}")

    def find_power(self, env, x, y, step, sol_area, tilt, azimuth):
        spectra, solpos = env.get_spectrum(x, y, tilt, azimuth, step)

        sun_azimuth = solpos["azimuth"].iloc[0]
        sun_zenith = solpos["apparent_zenith"].iloc[0]
        obfuscation_patch = env.get_obfuscation(x, y, step, sun_azimuth, sun_zenith)
        interference = obfuscation_patch[int(env.view_dist), int(env.view_dist)]
        wavelengths = np.atleast_1d(spectra["wavelength"]).flatten()
        poa_global = np.atleast_1d(spectra["poa_global"]).flatten()

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

        poa_total = trapezoid(poa_global, wavelengths)          # W/m^2
        weighted = trapezoid(poa_global * response_1d, wavelengths)

        if np.isnan(poa_total) or poa_total <= 0.0:
            return 0.0
        if np.isnan(weighted):
            weighted = 0.0

        response_ratio = weighted / poa_total
        spectral_factor = response_ratio / self.reference_response_ratio

        spectral_factor = float(np.clip(spectral_factor, 0.0, 1.25))

        panel_power = (
            poa_total
            * sol_area
            * self.cell_efficiency
            * self.panel_fill_factor
            * spectral_factor
            * (1.0 - interference)
        )

        physical_cap = poa_total * sol_area * self.max_cell_efficiency * (1.0 - interference)

        return max(0.0, min(panel_power, physical_cap))

    def harvest_energy(self, env, step):
        curr_x = int(max(0, min(self.x, env.dim - 1)))
        curr_y = int(max(0, min(self.y, env.dim - 1)))
        safe_step = min(int(math.floor(step)), len(env.times)-1)

        panel_power = self.find_power(
            env,
            curr_x,
            curr_y,
            safe_step,
            self.solar_area,
            self.tilt,
            self.azimuth
        )

        panel_power *= self.panel_efficiency_factor * self.wiring_efficiency * self.mppt_efficiency
        self.solar_potential = panel_power
        terminal_voltage = self.compute_terminal_voltage()
        charge_current = panel_power / max(terminal_voltage, 0.1)

        charge_current = min(charge_current, self.max_charge_current)

        if self.soc >= 0.999:
            charge_current = 0.0

        # This call represents the whole step() window, not a single
        # dt tick -- derive that duration explicitly (ticks_per_step *
        # dt seconds) and convert seconds->hours with /3600, exactly
        # like consume_motion_energy/consume_idle_energy do. The
        # previous `self.dt / 60.0` only worked because dt==1.0 and
        # step()'s loop happened to run 60 times; this ties the two
        # together instead of relying on that coincidence.
        seconds_per_step = self.ticks_per_step * self.dt
        self.energy_gained_mAh += charge_current * (seconds_per_step / 3600.0) * 1000.0

    def battery_step(self):
        self.update_battery_state()

    def consume_idle_energy(self):
        current_mA = (
            self.current_cpu +
            self.current_payload +
            self._comms["current_active_lora"]
        ) / 1000.0

        used = current_mA * (1.0 / 3600.0)
        self.energy_used_mAh += used
        self.step_idle_mAh += used

    def consume_motion_energy(self):
        battery_power = self.compute_motor_power()
        voltage = self.compute_open_circuit_voltage()

        current_A = battery_power / max(voltage, 1e-6)
        current_A = min(current_A, self.max_discharge_current)

        terminal_voltage = self.compute_terminal_voltage(current_A)
        battery_power = terminal_voltage * current_A
        energy_Wh = battery_power * self.dt / 3600.0

        used = energy_Wh * 1000.0 / terminal_voltage
        self.energy_used_mAh += used
        self.step_motion_mAh += used

        self.step_path_m += abs(self.velocity) * self.dt
        self.step_turn_integral += abs(self.angular_velocity) * self.dt

    def step(self, env, sim_step, target_x, target_y):
        self.energy_used_mAh = 0.0
        self.energy_gained_mAh = 0.0
        self.step_idle_mAh = 0.0
        self.step_motion_mAh = 0.0
        self.step_path_m = 0.0
        self.step_turn_integral = 0.0

        dx = target_x - env.boundary_center[0]
        dy = target_y - env.boundary_center[1]
        dist = math.hypot(dx, dy)

        if dist > env.boundary_radius:
            scale = env.boundary_radius / dist
            target_x = env.boundary_center[0] + dx * scale
            target_y = env.boundary_center[1] + dy * scale

        for second in range(self.ticks_per_step):
            self.update_vehicle_dynamics(target_x, target_y)
            self.consume_motion_energy()
            self.consume_idle_energy()

        self.harvest_energy(env, sim_step)
        self.update_battery_state()

        return self.get_position(), self.get_battery()
