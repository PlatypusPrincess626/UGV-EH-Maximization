import numpy as np
from scipy.integrate import trapezoid
import math
import os
import random

# Battery fidelity at low state of charge.
#
#   "honest" (default) -- loads are CONSTANT POWER and the BMS cutoff
#                         actually terminates the mission.
#   "legacy"           -- the previous behaviour, preserved so runs
#                         collected before this change remain
#                         reproducible.
#
#   LTAC_BATTERY_MODEL=legacy python main.py
#
# WHAT WAS WRONG WITH LEGACY
#
# Two things, and the second is the one that mattered.
#
# 1. `consume_idle_energy` drew a FIXED 965.2 mA regardless of state
#    of charge. Idle is 11582 mAh per episode against 236-575 mAh of
#    motion -- 95-98% of the total draw -- so the dominant term had no
#    voltage dependence at all. Real regulated electronics (CPU,
#    payload, LoRa) are constant-POWER loads: as terminal voltage
#    sags, current rises to hold power. Legacy applied that effect
#    only to motion, i.e. to 2-5% of the draw. The idle budget here is
#    calibrated to match the legacy current at SOC 0.90 (see
#    calibrated_idle_power), so the healthy band is as difficult as
#    before and only the tail gets harder -- 1.32x the legacy draw at
#    10% SOC, 1.34x at 5%.
#
# 2. `compute_terminal_voltage` returned max(cutoff_voltage, terminal),
#    which CLAMPED the cutoff instead of tripping it. Unclamped, at 15%
#    SOC under a 5 A motion load the terminal voltage is already 11.96 V
#    -- below the 12.0 V cutoff -- and at 10% it is 11.76 V. Real
#    hardware disconnects there. The simulated vehicle instead coasted
#    down toward 0% and was scored as having survived.
#
# The consequence for results collected under legacy: minimum state of
# charge below roughly 15% is optimistic, and an episode bottoming near
# 7% would very likely have ended on hardware. Episodes are the unit to
# count, not the mean.
#
# THIS INVALIDATES PRIOR RUNS. Every arm must be re-run before its
# numbers are comparable to anything produced under "honest".
BATTERY_MODEL = os.environ.get("LTAC_BATTERY_MODEL", "legacy").strip().lower()
if BATTERY_MODEL not in ("honest", "legacy"):
    raise ValueError(
        f"LTAC_BATTERY_MODEL must be 'honest' or 'legacy', got {BATTERY_MODEL!r}"
    )
HONEST_BATTERY = BATTERY_MODEL == "honest"

# ZIP load blend for the housekeeping draw: "aZ,aI,aP", summing to 1.
# See UGVSimulator.ZIP_Z for what each component means and why the
# default is pure constant power.
_zip_raw = os.environ.get("LTAC_ZIP", "0.0,0.0,1.0")
try:
    ZIP_COEFFS = tuple(float(x) for x in _zip_raw.split(","))
except ValueError:
    raise ValueError(f"LTAC_ZIP must be three floats 'aZ,aI,aP', got {_zip_raw!r}")
if len(ZIP_COEFFS) != 3:
    raise ValueError(f"LTAC_ZIP needs exactly three values, got {_zip_raw!r}")
if abs(sum(ZIP_COEFFS) - 1.0) > 1e-9:
    raise ValueError(f"LTAC_ZIP must sum to 1.0, got {sum(ZIP_COEFFS)}")
if any(x < 0.0 for x in ZIP_COEFFS):
    raise ValueError(f"LTAC_ZIP coefficients must be non-negative, got {_zip_raw!r}")
ZIP_EXPLICIT = "LTAC_ZIP" in os.environ

# Discharge-efficiency wall: where the accelerating low-SOC penalty
# begins, how deep it goes, and how sharply it climbs.
#
#   u    = max(0, WALL_SOC - soc)
#   knee = DEPTH * (exp(u/TAU) - 1) / (exp(WALL_SOC/TAU) - 1)
#   eta  = eta_0 * (1 - knee)
#
# so the drain MULTIPLIER is 1/(1 - knee): exactly 1.000 at and above
# WALL_SOC, then exponential below it, reaching 1/(1 - DEPTH) at empty.
#
# WHY HINGED RATHER THAN A PLAIN EXPONENTIAL
#
# A bare exp(-soc/TAU) has no hard start -- pushing its knee up to 20%
# also puts a 7-11% penalty at 30-35% SOC. That band is where every
# episode's opening deficit plays out (solar is 3-6 W against a 15.5 W
# idle draw, and minimum state of charge is set around step 40-54), and
# taxing it is exactly the mistake that took deaths from ~7 per 50
# episodes to 45. The hinge normalises the exponential so it is
# identically zero at and above the wall and carries the entire
# penalty below it.
#
# DEPTH 0.85, TAU 0.05, WALL 0.20 gives:
#
#   SOC    30%   25%   20%   15%   10%    7%    5%    2%    0%
#   mult  1.00  1.00  1.00  1.03  1.11  1.25  1.43  2.30  6.67
#
# Raise DEPTH for a harder floor (0.90 -> 10x at empty, 0.95 -> 20x);
# lower TAU to concentrate the penalty nearer empty; move WALL_SOC to
# shift where it starts. Applies only under LTAC_BATTERY_MODEL=honest.
#
# A NOTE ON WHERE 0.20 SITS
#
# The converged deterministic policy bottoms out around 26% state of
# charge, so a wall at 20% is below it and should not tax normal
# operation. Training samples the policy rather than taking its mean,
# and the measurement in transformer.py's LOG_STD_MAX note has the
# sampled policy reaching 7.5% against 21% deterministic -- so this
# WILL fire during exploration. That is arguably the point, but it is
# also the shape of the earlier COST_SOC_SAFE=0.25 failure, so watch
# the early-episode death rate on the first short run.
DISCHARGE_WALL_SOC = float(os.environ.get("LTAC_WALL_SOC", "0.20"))
DISCHARGE_KNEE_DEPTH = float(os.environ.get("LTAC_KNEE_DEPTH", "0.85"))
DISCHARGE_KNEE_TAU = float(os.environ.get("LTAC_KNEE_TAU", "0.05"))
if not 0.0 < DISCHARGE_WALL_SOC <= 1.0:
    raise ValueError("LTAC_WALL_SOC must be in (0, 1]")
if not 0.0 <= DISCHARGE_KNEE_DEPTH < 1.0:
    raise ValueError("LTAC_KNEE_DEPTH must be in [0, 1)")
if DISCHARGE_KNEE_TAU <= 0.0:
    raise ValueError("LTAC_KNEE_TAU must be > 0")
_WALL_NORM = math.exp(DISCHARGE_WALL_SOC / DISCHARGE_KNEE_TAU) - 1.0

# Consecutive steps below cutoff_voltage before the BMS latches.
# See UGVSimulator.note_undervoltage. dt is 1 s, so this is seconds.
UNDERVOLTAGE_TRIP_STEPS = max(1, int(os.environ.get("LTAC_UV_TRIP_STEPS", "3")))

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

        self.capacity_Ah = 5.0
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

        # BMS protection cutoff. Latched; see trip_bms().
        self._bms_tripped = False
        self._bms_reason = None
        self._undervoltage_steps = 0
        self._tick_current_A = 0.0
        self._tick_collapsed = False

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
        self._bms_tripped = False
        self._bms_reason = None
        self._undervoltage_steps = 0
        self._tick_current_A = 0.0
        self._tick_collapsed = False
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
        """
        State of charge in percent, 0.0 once the BMS has tripped.

        Reporting 0.0 rather than adding a separate flag is deliberate:
        main.py already ends the episode on `battery <= 0`, in both the
        training rollout and the validation loop, and already charges
        the death cost there. A protection disconnect and a full
        depletion are the same outcome for the mission, so they should
        travel the same code path -- no new branch, no second condition
        to keep in sync between the two loops.
        """
        if self._bms_tripped:
            return 0.0
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
        """
        Terminal voltage under load, CLAMPED at the cutoff.

        The clamp is why legacy never failed: it reports 12.0 V when
        the true terminal voltage is below that, so nothing downstream
        can notice. Kept for the legacy path and for callers that only
        want a voltage to divide by. Use
        `compute_terminal_voltage_raw` to ask whether the pack has
        actually collapsed.
        """
        return max(self.cutoff_voltage,
                   self.compute_terminal_voltage_raw(current_A))

    def compute_terminal_voltage_raw(self, current_A=0.0):
        """Unclamped. May legitimately fall below cutoff_voltage."""
        ocv = self.compute_open_circuit_voltage()
        resistance = self.compute_internal_resistance()
        return ocv - current_A * resistance

    # ZIP load coefficients for the housekeeping load: the fractions
    # of the draw that behave as constant-impedance, constant-current
    # and constant-power. Must sum to 1.0.
    #
    # This is the standard ZIP form from power-systems load modelling.
    # A real bus is a mixture, not any one of the three:
    #
    #   Z  resistive elements -- heaters, pull-ups, terminations.
    #      Current FALLS as the pack sags.
    #   I  current-regulated elements -- LED drivers, some sensor
    #      front ends, anything behind a current source.
    #      Current is flat, which is exactly the legacy assumption.
    #   P  buck-regulated digital elements -- CPU, radio, and most of
    #      a modern payload. Current RISES as the pack sags.
    #
    # Defaults are 0/0/1, i.e. pure constant power, which is the
    # conservative reading of this vehicle's load list: the payload is
    # 960 mA of the 965 mA total and sits behind a regulator. Set
    # (0, 1, 0) to recover the legacy fixed-current behaviour exactly
    # at every state of charge, or blend if the real hardware is
    # mixed:
    #
    #   LTAC_ZIP=0.0,0.2,0.8 python main.py
    ZIP_Z, ZIP_I, ZIP_P = ZIP_COEFFS

    # State of charge below which the housekeeping load starts
    # behaving as constant power. At or above it the load is pure
    # constant current -- bit-identical to legacy.
    #
    # WHY A KNEE AND NOT A GLOBAL BLEND
    #
    # The first version anchored a pure constant-power load at SOC
    # 0.90 on the reasoning that this is where the vehicle spends most
    # of its time (median 0.926). That was the wrong anchor. Time
    # spent is not where difficulty is decided: every episode opens in
    # deficit -- solar is 3-6 W against a 15.5 W idle draw -- and the
    # minimum state of charge is set around step 40-54, at 26-35% SOC.
    # Anchoring at 0.90 left the phase that decides nothing unchanged
    # and made the phase that decides everything 22-24% harder. Deaths
    # went from roughly 7 per 50 episodes to 45.
    #
    # The stated requirement was that the healthy region stay as
    # difficult as before and only the DANGER region get harder. A
    # knee delivers that literally: above SOC_CC_KNEE the load is
    # constant current and the arithmetic is legacy's, so the opening
    # deficit is untouched.
    SOC_CC_KNEE = float(os.environ.get("LTAC_SOC_CC_KNEE", "0.25"))

    def idle_zip_weights(self):
        """
        ZIP weights for the housekeeping load at the current state of
        charge.

        Ramps linearly from pure constant-current at SOC_CC_KNEE to
        pure constant-power at empty. Above the knee this returns
        (0, 1, 0), which reproduces the legacy fixed current exactly
        -- not approximately, since the constant-current branch of the
        ZIP form has no voltage dependence at all.

        If LTAC_ZIP was set explicitly, that fixed blend wins and this
        ramp is bypassed; the knee is the default behaviour, not a
        constraint on experimentation.
        """
        if ZIP_EXPLICIT:
            return (self.ZIP_Z, self.ZIP_I, self.ZIP_P)
        knee = self.SOC_CC_KNEE
        if knee <= 0.0:
            return (0.0, 1.0, 0.0)
        a_p = min(1.0, max(0.0, (knee - self.soc) / knee))
        return (0.0, 1.0 - a_p, a_p)

    def solve_load_current(self, current_A_ref, zip_coeffs=None):
        """
        Current drawn by a ZIP load, solved exactly against the pack's
        I-V characteristic.

        The load's own I-V curve, referenced to (I_ref, V_ref):

            I(V) = I_ref * [ aZ*(V/V_ref) + aI + aP*(V_ref/V) ]

        and the pack's, from its Thevenin model:

            V = OCV(soc) - I*R(soc)

        Two curves, one operating point: their intersection. Note the
        load curve passes through (V_ref, I_ref) for ANY choice of
        weights, since aZ + aI + aP = 1. That is what makes the
        calibration exact by construction rather than by a derived
        power budget -- at V_ref this draws the legacy current no
        matter how the blend is set, and the weights only shape how it
        departs as the pack sags.

        Substituting and multiplying through by V gives a quadratic in
        the terminal voltage:

            A*V^2 + B*V + C = 0
            A = 1 + R*I_ref*aZ/V_ref
            B = -(OCV - R*I_ref*aI)
            C = R*I_ref*aP*V_ref

        The physical root is the larger one, the operating point
        nearest open circuit; the smaller is the high-current branch a
        real supply never settles on. Closed form, so this costs the
        same as the old fixed-current path -- it runs every step of
        every episode.

        A negative discriminant means the two curves do not intersect:
        no operating point exists and the pack has collapsed under
        this load. That is a real failure, not a numerical edge case.

        Returns (current_A, terminal_V, ok).
        """
        if current_A_ref <= 0.0:
            return 0.0, self.compute_open_circuit_voltage(), "ok"

        aZ, aI, aP = (zip_coeffs if zip_coeffs is not None
                      else self.idle_zip_weights())

        ocv = self.compute_open_circuit_voltage()
        resistance = self.compute_internal_resistance()
        v_ref = self.reference_terminal_voltage(current_A_ref)

        a = 1.0 + resistance * current_A_ref * aZ / v_ref
        b = -(ocv - resistance * current_A_ref * aI)
        c = resistance * current_A_ref * aP * v_ref

        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return (ocv / (2.0 * resistance), ocv / 2.0, "collapsed")

        terminal = (-b + math.sqrt(discriminant)) / (2.0 * a)
        if terminal <= 0.0:
            return (ocv / (2.0 * resistance), ocv / 2.0, "collapsed")

        current = (ocv - terminal) / resistance
        if current > self.max_discharge_current:
            # SATURATION, not failure. The pack cannot source what was
            # asked for, so it delivers what it can -- exactly what
            # legacy's min(current, max_discharge_current) did. Whether
            # that constitutes a fault is decided by the resulting
            # terminal voltage, not by the clamp itself.
            return (self.max_discharge_current,
                    self.compute_terminal_voltage_raw(
                        self.max_discharge_current), "limited")

        return current, terminal, "ok"

    def solve_constant_power_current(self, power_W):
        """
        Constant-power draw, for loads specified in watts rather than
        as a reference current -- the drivetrain, whose demand is a
        mechanical wattage with no datasheet current to reference.

        Equivalent to solve_load_current with weights (0, 0, 1):
        R*I^2 - OCV*I + P = 0, physical root the smaller one.

        Returns (current_A, terminal_V, ok).
        """
        ocv = self.compute_open_circuit_voltage()
        resistance = self.compute_internal_resistance()

        if power_W <= 0.0:
            return 0.0, ocv, "ok"

        discriminant = ocv * ocv - 4.0 * resistance * power_W
        if discriminant <= 0.0:
            # Maximum deliverable power is OCV^2/(4R), at I = OCV/(2R).
            return ocv / (2.0 * resistance), ocv / 2.0, "collapsed"

        current = (ocv - math.sqrt(discriminant)) / (2.0 * resistance)
        if current > self.max_discharge_current:
            return (self.max_discharge_current,
                    self.compute_terminal_voltage_raw(
                        self.max_discharge_current), "limited")

        return current, ocv - current * resistance, "ok"

    # State of charge at which the constant-power idle budget is
    # calibrated to draw exactly the legacy fixed current.
    #
    # 0.90, matching SOC_TARGET in main.py, so the anchor is a state
    # the task already treats as the healthy setpoint rather than an
    # arbitrary number. It is also where the vehicle actually lives:
    # across all eight collected runs the time-weighted median state
    # of charge is 0.926 and 76% of steps sit above 0.60.
    IDLE_CALIBRATION_SOC = float(os.environ.get("LTAC_IDLE_CALIB_SOC", "0.25"))

    def reference_terminal_voltage(self, current_A_ref):
        """
        Power budget for the housekeeping load.

        Referencing the datasheet currents to the NOMINAL rail made
        the healthy region cheaper than legacy -- 0.852 A against
        0.965 A at full charge, because open-circuit voltage there
        (16.8 V) is well above nominal (14.8 V). That is defensible
        physics but it changes task difficulty everywhere, not just in
        the failure region, and the point of this change was to fix
        the tail without moving the rest.

        So the load curve is instead anchored at the TERMINAL voltage
        the pack actually presents at IDLE_CALIBRATION_SOC while
        supplying the legacy current:

            V_ref = OCV(s_ref) - I_ref * R(s_ref)

        Terminal, not open-circuit, so the match is exact rather than
        off by the I*R drop. Because the ZIP load curve passes through
        (V_ref, I_ref) for any weighting, this anchors every blend at
        the same healthy operating point.

        Resulting draw against the legacy fixed current:

            SOC     100%   90%    75%    50%    25%    10%     5%
            ratio   0.96   1.00   1.06   1.16   1.26   1.32   1.34

        Within 4% of legacy across the healthy band, rising to 1.34x
        as the pack collapses. That is the intended shape: same task
        above the setpoint, honest penalty below it.
        """
        s_ref = self.IDLE_CALIBRATION_SOC
        ocv_ref = min(self.full_voltage, max(
            self.cutoff_voltage,
            12.0 + 3.0 * s_ref + 1.2 * s_ref ** 2 + 0.6 * s_ref ** 3))
        r_ref = self.internal_resistance * (1.0 + 1.8 * (1.0 - s_ref) ** 2)
        return ocv_ref - current_A_ref * r_ref

    def effective_discharge_efficiency(self):
        """
        Coulombic efficiency, degraded near empty.

            u    = max(0, WALL_SOC - soc)
            knee = DEPTH * (exp(u/TAU) - 1) / (exp(WALL_SOC/TAU) - 1)
            eta  = eta_0 * (1 - knee)

        Charge removed from the pack is energy_used / eta, so the
        drain multiplier is 1/(1 - knee): identically 1.000 at and
        above WALL_SOC, then exponential below it. See
        DISCHARGE_WALL_SOC for why the accelerating penalty belongs
        here and not in the load model, and why it is hinged rather
        than a bare exponential.

        Evaluated at the state of charge at the START of the step,
        since update_battery_state runs once after the tick loop. That
        makes the penalty lag by one step -- 60 s -- which is
        conservative: the pack is always charged slightly less than
        the instantaneous state would justify, never more.
        """
        if not HONEST_BATTERY:
            return self.discharge_efficiency

        below = DISCHARGE_WALL_SOC - self.soc
        if below <= 0.0:
            return self.discharge_efficiency

        knee = DISCHARGE_KNEE_DEPTH * (
            math.exp(below / DISCHARGE_KNEE_TAU) - 1.0) / _WALL_NORM
        return self.discharge_efficiency * max(1.0 - knee, 1e-3)

    def evaluate_pack_state(self):
        """
        Decide whether this step's operating point constitutes a fault.

        THE BUG THIS REPLACES

        The first version tripped the BMS whenever the solver returned
        not-ok, and the solver returned not-ok for TWO different
        things: no operating point exists, and the demand exceeded
        max_discharge_current. The second is saturation, which legacy
        handled with a plain min() and which happens routinely --
        early exploration commands large accelerations, the drivetrain
        asks for more than 10 A, and the mission was being killed for
        it. That is why almost every episode was dying within the
        first 50 steps regardless of state of charge.

        A current clamp is not a fault. Only the resulting VOLTAGE
        decides: if the pack still holds above cutoff while delivering
        its maximum, the vehicle is merely underpowered, which is a
        control problem and not a protection event.

        PERSISTENCE

        Real undervoltage protection has a delay -- typically a few
        hundred milliseconds to a few seconds -- precisely so a
        transient does not disconnect the pack. At dt = 1 s a single
        step below cutoff is within that window, so tripping on it
        would be modelling the protection as faster than any real BMS.
        UNDERVOLTAGE_TRIP_STEPS consecutive steps are required; the
        counter resets on any step that recovers, since the latch is
        meant for a pack that stays down, not one that dips.

        Set LTAC_UV_TRIP_STEPS=1 to trip on the first step.
        """
        current = min(self._tick_current_A, self.max_discharge_current)
        terminal = self.compute_terminal_voltage_raw(current)
        self._tick_collapsed = False

        if terminal < self.cutoff_voltage:
            self._undervoltage_steps += 1
            if self._undervoltage_steps >= UNDERVOLTAGE_TRIP_STEPS:
                self.trip_bms(
                    f"pack at {terminal:.2f} V under {current:.2f} A for "
                    f"{self._undervoltage_steps} s (soc {self.soc:.3f})")
        else:
            self._undervoltage_steps = 0

    def trip_bms(self, reason):
        """
        Latch the protection cutoff.

        Latched rather than momentary because a real BMS disconnect
        does not clear when the load drops -- the vehicle is down until
        someone intervenes, which is not something that happens inside
        an episode. `get_battery()` reports 0.0 once this is set, which
        is what every existing `battery <= 0` check in main.py already
        tests, so the mission ends through the SAME path as a full
        depletion with no new branch anywhere in the training loop.
        """
        if not self._bms_tripped:
            self._bms_tripped = True
            self._bms_reason = reason

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
        discharge = self.energy_used_mAh / self.effective_discharge_efficiency()

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
        """
        Housekeeping draw: CPU, payload and radio.

        The component currents are datasheet figures at the nominal
        pack voltage, so the honest reading of them is a POWER budget,
        not a current budget: the regulators hold power and let current
        rise as the pack sags. Legacy took them as a fixed current,
        which made 95-98% of the draw independent of state of charge.
        """
        current_mA = (
            self.current_cpu +
            self.current_payload +
            self._comms["current_active_lora"]
        ) / 1000.0

        if HONEST_BATTERY:
            current_A, terminal, status = self.solve_load_current(
                current_mA / 1000.0)
            self._tick_current_A += current_A
            if status == "collapsed":
                self._tick_collapsed = True
            current_mA = current_A * 1000.0

        used = current_mA * (self.dt / 3600.0)
        self.energy_used_mAh += used
        self.step_idle_mAh += used

    def consume_motion_energy(self):
        """
        Drivetrain draw.

        Also a constant-power load: the mechanical demand is what it
        is, and the drivetrain pulls whatever current delivers it.
        Legacy derived current from the OPEN-CIRCUIT voltage, which
        ignores the sag the current itself causes and so understates
        the draw -- mildly at high charge, most at low charge, which is
        exactly where it matters. Routed through the same solver as
        idle so the two cannot drift apart.
        """
        battery_power = self.compute_motor_power()

        if HONEST_BATTERY:
            current_A, terminal_voltage, status = self.solve_constant_power_current(
                battery_power)
            self._tick_current_A += current_A
            if status == "collapsed":
                self._tick_collapsed = True
            terminal_voltage = max(terminal_voltage, 1e-6)
        else:
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
            self._tick_current_A = 0.0
            self.consume_motion_energy()
            self.consume_idle_energy()
            # ONE check per tick, on the SUM of the loads.
            #
            # Previously each consume_* called note_undervoltage with
            # its own load in isolation, so a motion sag was cleared by
            # the healthy idle reading that followed it and the counter
            # could never reach its threshold except from idle alone.
            # The pack sees the sum, not each load separately.
            self.evaluate_pack_state()

        self.harvest_energy(env, sim_step)
        self.update_battery_state()

        return self.get_position(), self.get_battery()
