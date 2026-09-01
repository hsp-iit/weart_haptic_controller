#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""ROS 2: WEART TouchDIVER index -> q1,q2,q3 -> LEAP Hand v1.
 
ZERO LeRobot dependency.
 
Dependencies:
    rclpy
    std_msgs
    numpy
    scipy
    leap_cpp              # the pybind controller already used in your project
 
Expected ROS input type: std_msgs/msg/String
Example payload:
    ts=1788171631772 | closure=0.000 | opening=1.000 | abduction=0.525 |
    adduction=0.475 | ACC[g]=(0.087, 0.928, -0.286) |
    GYRO[deg/s]=(2.030, -7.000, -5.740) | ToF[mm]=83
 
Estimator model:
    index = planar 3R chain
    q1 = MCP forward/flexion
    q2 = PIP flexion
    q3 = DIP flexion
 
Objective:
    sensor consistency + temporal continuity + DIP/PIP coupling.
 
LEAP Hand v1 convention used here (official API):
    motor 0 = index MCP side
    motor 1 = index MCP forward
    motor 2 = index PIP
    motor 3 = index DIP
 
    real motor angle = LEAPsim angle + pi
 
The commanded index motors are resolved from the URDF at startup (they are
currently 1,2,3 for the supplied model). Every other joint remains at its
current commanded neutral value unless you change the code.
 
IMPORTANT:
    1. Hardware output is OFF by default.
    2. Keep the human index OPEN and hand STILL during startup calibration.
    3. Verify IMU axis/sign and ToF geometry for your WEART mounting.
    4. Start dry-run first and inspect estimated q before enabling motors.
"""
 
from __future__ import annotations
 
import ctypes
import glob
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
 
import numpy as np
from scipy.optimize import least_squares
 
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
 
 
# =============================================================================
# HUMAN INDEX MODEL
# =============================================================================
 
# Approximate phalanx lengths [mm]. Measure your own index and replace these.
FINGER_LENGTHS_MM = np.array([53.1, 36.1, 25.0], dtype=float)
 
# Simplified point on the palm used by the ToF range model in the planar frame.
# Startup calibration removes a constant offset, not a wrong geometric direction.
PALM_TOF_POINT_MM = np.array([0.0, 0.0], dtype=float)
 
# Human joint limits [rad].
# q1=MCP forward, q2=PIP, q3=DIP.
HUMAN_Q_LOWER = np.deg2rad(np.array([-10.0, 0.0, 0.0], dtype=float))
HUMAN_Q_UPPER = np.deg2rad(np.array([95.0, 120.0, 90.0], dtype=float))
 
 
# =============================================================================
# WEART IMU MOUNTING -- MUST MATCH YOUR SENSOR FRAME
# =============================================================================
 
# Dominant gyro axis for planar flexion: 0=x, 1=y, 2=z.
GYRO_FLEX_AXIS = 2
GYRO_FLEX_SIGN = +1.0
 
# Accelerometer gravity angle used to slowly correct gyro drift:
# theta_acc = atan2(sign_num * acc[num_axis], sign_den * acc[den_axis])
ACC_ANGLE_NUM_AXIS = 0
ACC_ANGLE_DEN_AXIS = 1
ACC_ANGLE_NUM_SIGN = +1.0
ACC_ANGLE_DEN_SIGN = +1.0
 
ACC_CORRECTION_GAIN = 0.04
ACC_NORM_SIGMA_G = 0.12
MIN_DT_S = 0.002
MAX_DT_S = 0.25
 
 
# =============================================================================
# OPTIMIZATION WEIGHTS
# =============================================================================
 
THETA_SIGMA_RAD = np.deg2rad(4.0)
TOF_SIGMA_MM = 4.0
SMOOTH_SIGMA_RAD = np.deg2rad(22.0)
 
# Human DIP flexion normally follows PIP flexion. This resolves the otherwise
# ambiguous q2/q3 split from a single IMU orientation and ToF range.
DIP_TO_PIP_RATIO = 0.67
DIP_COUPLING_SIGMA_RAD = np.deg2rad(12.0)
 
# =============================================================================
# STARTUP CALIBRATION
# =============================================================================
 
CALIBRATION_SAMPLES = 30
CALIBRATION_MAX_CLOSURE = 0.10
CALIBRATION_MAX_GYRO_NORM_DEG_S = 15.0
 
 
# =============================================================================
# LEAP HAND V1 -- NO LEROBOT
# =============================================================================
 
# The robot model is the single source of truth for joint ordering, travel and
# speed limits. The supplied LEAP URDF exposes the primary index flexion chain
# as palm_lower --(1)--> mcp_joint ... pip --(2)--> dip --(3)--> fingertip.
DEFAULT_URDF_PATH = str(Path(__file__).with_name("model.urdf"))
 
 
@dataclass(frozen=True)
class LeapUrdfModel:
    """LEAP joint metadata read from the robot URDF."""
 
    joint_names: tuple[str, ...]
    sim_lower: np.ndarray
    sim_upper: np.ndarray
    velocity_limits: np.ndarray
    index_flexion_motor_indices: tuple[int, int, int]
 
    @property
    def motor_count(self) -> int:
        return len(self.joint_names)
 
 
def load_leap_urdf(urdf_path: str) -> LeapUrdfModel:
    """Load numerical LEAP joints and locate the index flexion chain.
 
    Numerical joint names are the hardware motor indices expected by the LEAP
    controller. Rejecting holes prevents commands being sent to a motor other
    than the one described by the URDF.
    """
    path = Path(urdf_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"LEAP URDF not found: {path}")
 
    root = ET.parse(path).getroot()
    joints: dict[int, ET.Element] = {}
    for joint in root.findall("joint"):
        name = joint.get("name", "")
        if name.isdecimal():
            joints[int(name)] = joint
 
    if not joints:
        raise ValueError(f"No numerical actuated joints found in {path}")
    indices = list(range(max(joints) + 1))
    if sorted(joints) != indices:
        raise ValueError(
            "LEAP numerical joint names must be contiguous from 0; found "
            f"{sorted(joints)}"
        )
 
    lower: list[float] = []
    upper: list[float] = []
    velocity: list[float] = []
    child_to_joint: dict[str, int] = {}
    parent_child_to_joint: dict[tuple[str, str], int] = {}
    for index in indices:
        joint = joints[index]
        limit = joint.find("limit")
        parent = joint.find("parent")
        child = joint.find("child")
        if (
            limit is None
            or parent is None
            or child is None
            or limit.get("lower") is None
            or limit.get("upper") is None
            or limit.get("velocity") is None
            or parent.get("link") is None
            or child.get("link") is None
        ):
            raise ValueError(f"Joint {index} has incomplete URDF metadata")
        lower.append(float(limit.get("lower")))
        upper.append(float(limit.get("upper")))
        velocity.append(float(limit.get("velocity")))
        parent_link = str(parent.get("link"))
        child_link = str(child.get("link"))
        child_to_joint[child_link] = index
        parent_child_to_joint[(parent_link, child_link)] = index
 
    # Joint 0 is the MCP side joint. The selected links identify the three
    # forward/flexion motors without duplicating their numerical IDs in code.
    try:
        index_mcp = child_to_joint["mcp_joint"]
        index_pip = parent_child_to_joint[("pip", "dip")]
        index_dip = parent_child_to_joint[("dip", "fingertip")]
    except KeyError as exc:
        raise ValueError(
            "URDF does not contain the expected primary index links "
            "(mcp_joint, pip, dip, fingertip)"
        ) from exc
 
    return LeapUrdfModel(
        joint_names=tuple(str(i) for i in indices),
        sim_lower=np.asarray(lower, dtype=float),
        sim_upper=np.asarray(upper, dtype=float),
        velocity_limits=np.asarray(velocity, dtype=float),
        index_flexion_motor_indices=(index_mcp, index_pip, index_dip),
    )
 
# Smooth/rate-limit command before sending it to hardware.
COMMAND_LOW_PASS_ALPHA = 0.2
COMMAND_MAX_SPEED_RAD_S = 2.0
 
# Official example gains for LEAP Hand v1. Override with ROS parameters if your
# C++ controller/configuration uses different values.
DEFAULT_KP = 600.0
DEFAULT_KI = 0.0
DEFAULT_KD = 200.0
 
DEFAULT_ENABLE_HARDWARE = False
DEFAULT_LEAP_PORT = ""  # empty = auto-detect /dev/serial/by-id/*, then ttyUSB0/1
DEFAULT_INPUT_TOPIC = "/weart/index/raw"
DEFAULT_OUTPUT_TOPIC = "/weart/index/estimated_joints"
 
 
# =============================================================================
# WEART STRING PARSER
# =============================================================================
 
SAMPLE_RE = re.compile(
    r"ts=(?P<ts>\d+)\s*\|\s*"
    r"closure=(?P<closure>[-+0-9.eE]+)\s*\|\s*"
    r"opening=(?P<opening>[-+0-9.eE]+)\s*\|\s*"
    r"abduction=(?P<abduction>[-+0-9.eE]+)\s*\|\s*"
    r"adduction=(?P<adduction>[-+0-9.eE]+)\s*\|\s*"
    r"ACC\[g\]=\(\s*(?P<ax>[-+0-9.eE]+)\s*,\s*(?P<ay>[-+0-9.eE]+)\s*,\s*(?P<az>[-+0-9.eE]+)\s*\)\s*\|\s*"
    r"GYRO\[deg/s\]=\(\s*(?P<gx>[-+0-9.eE]+)\s*,\s*(?P<gy>[-+0-9.eE]+)\s*,\s*(?P<gz>[-+0-9.eE]+)\s*\)\s*\|\s*"
    r"ToF\[mm\]=(?P<tof>[-+0-9.eE]+)"
)
 
 
@dataclass
class WeartSample:
    ts_ms: int
    closure: float
    opening: float
    abduction: float
    adduction: float
    acc_g: np.ndarray
    gyro_deg_s: np.ndarray
    tof_mm: float
 
 
def parse_weart_line(line: str) -> WeartSample:
    m = SAMPLE_RE.search(line.strip())
    if m is None:
        raise ValueError(f"Cannot parse WEART line: {line!r}")
 
    x = m.groupdict()
    return WeartSample(
        ts_ms=int(x["ts"]),
        closure=float(x["closure"]),
        opening=float(x["opening"]),
        abduction=float(x["abduction"]),
        adduction=float(x["adduction"]),
        acc_g=np.array([float(x["ax"]), float(x["ay"]), float(x["az"])], dtype=float),
        gyro_deg_s=np.array([float(x["gx"]), float(x["gy"]), float(x["gz"])], dtype=float),
        tof_mm=float(x["tof"]),
    )
 
 
# =============================================================================
# PLANAR 3R MODEL
# =============================================================================
 
 
def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi
 
 
def angle_error(a: float, b: float) -> float:
    return wrap_pi(a - b)
 
 
def fingertip_xy_mm(q: np.ndarray) -> np.ndarray:
    q1, q2, q3 = np.asarray(q, dtype=float)
    l1, l2, l3 = FINGER_LENGTHS_MM
 
    t1 = q1
    t2 = q1 + q2
    t3 = q1 + q2 + q3
 
    return np.array(
        [
            l1 * math.cos(t1) + l2 * math.cos(t2) + l3 * math.cos(t3),
            l1 * math.sin(t1) + l2 * math.sin(t2) + l3 * math.sin(t3),
        ],
        dtype=float,
    )
 
 
def tof_forward_model_mm(q: np.ndarray) -> float:
    """Simplified fingertip/palm range model.
 
    Replace this function once the exact WEART ToF emitter/receiver geometry relative
    to the palm is known. The rest of the estimator does not need to change.
    """
    p_tip = fingertip_xy_mm(q)
    return float(np.linalg.norm(p_tip - PALM_TOF_POINT_MM))
 
 
class IndexEstimator:
    def __init__(self) -> None:
        self.q = np.zeros(3, dtype=float)
        self.theta = 0.0
        self.last_ts_ms: int | None = None
 
        self.acc_open_angle: float | None = None
        self.gyro_bias_deg_s = 0.0
        self.tof_offset_mm = 0.0
 
        self._calib_acc_angles: list[float] = []
        self._calib_gyro_flex: list[float] = []
        self._calib_tof: list[float] = []
        self.calibrated = False
 
    def _raw_acc_angle(self, acc: np.ndarray) -> float:
        num = ACC_ANGLE_NUM_SIGN * float(acc[ACC_ANGLE_NUM_AXIS])
        den = ACC_ANGLE_DEN_SIGN * float(acc[ACC_ANGLE_DEN_AXIS])
        return math.atan2(num, den)
 
# calibration samples 30
    def _sample_is_good_for_calibration(self, s: WeartSample) -> bool:
        return (
            s.closure <= CALIBRATION_MAX_CLOSURE
            and float(np.linalg.norm(s.gyro_deg_s)) <= CALIBRATION_MAX_GYRO_NORM_DEG_S
            and 0.75 <= float(np.linalg.norm(s.acc_g)) <= 1.25
        )
 
    def _calibrate(self, s: WeartSample) -> float:
        if not self._sample_is_good_for_calibration(s):
            return len(self._calib_tof) / CALIBRATION_SAMPLES
 
        self._calib_acc_angles.append(self._raw_acc_angle(s.acc_g))
        self._calib_gyro_flex.append(
            GYRO_FLEX_SIGN * float(s.gyro_deg_s[GYRO_FLEX_AXIS])
        )
        self._calib_tof.append(float(s.tof_mm))
 
        if len(self._calib_tof) >= CALIBRATION_SAMPLES:
            sin_mean = float(np.mean(np.sin(self._calib_acc_angles)))
            cos_mean = float(np.mean(np.cos(self._calib_acc_angles)))
            self.acc_open_angle = math.atan2(sin_mean, cos_mean)
 
            self.gyro_bias_deg_s = float(np.mean(self._calib_gyro_flex))
 
            open_tof = float(np.median(self._calib_tof))
            self.tof_offset_mm = open_tof - tof_forward_model_mm(np.zeros(3))
 
            self.q[:] = 0.0
            self.theta = 0.0
            self.calibrated = True
 
        return min(1.0, len(self._calib_tof) / CALIBRATION_SAMPLES)
 
    def _compute_dt(self, ts_ms: int) -> float:
        if self.last_ts_ms is None:
            self.last_ts_ms = ts_ms
            return 0.01
 
        dt = (ts_ms - self.last_ts_ms) * 1e-3
        self.last_ts_ms = ts_ms
 
        if not np.isfinite(dt) or dt <= 0.0:
            return 0.01
 
        return float(np.clip(dt, MIN_DT_S, MAX_DT_S))
# Integral of the gyro flexion axis, corrected by the accelerometer when near 1 g.
    def _update_theta(self, s: WeartSample, dt: float) -> float:
        gyro_flex_deg_s = (
            GYRO_FLEX_SIGN * float(s.gyro_deg_s[GYRO_FLEX_AXIS])
            - self.gyro_bias_deg_s
        )
 
        theta_gyro = wrap_pi(
            self.theta + math.radians(gyro_flex_deg_s) * dt
        )
 
        # Trust the accelerometer as a gravity reference only near 1 g.
        acc_norm = float(np.linalg.norm(s.acc_g))
        trust = math.exp(
            -0.5 * ((acc_norm - 1.0) / ACC_NORM_SIGMA_G) ** 2
        )
 
        assert self.acc_open_angle is not None
        theta_acc = wrap_pi(self._raw_acc_angle(s.acc_g) - self.acc_open_angle)
 
        correction = (
            ACC_CORRECTION_GAIN
            * trust
            * angle_error(theta_acc, theta_gyro)
        )
 
        self.theta = wrap_pi(theta_gyro + correction)
        return self.theta
# least square error
    def _residual(
        self,
        q: np.ndarray,
        theta: float,
        tof_mm: float,
        q_prev: np.ndarray,
    ) -> np.ndarray:
        # Distal phalanx orientation in the planar 3R model.
        r_theta = angle_error(float(np.sum(q)), theta) / THETA_SIGMA_RAD
 
        # Time-of-Flight geometric consistency.
        tof_pred = tof_forward_model_mm(q) + self.tof_offset_mm
        r_tof = (tof_pred - tof_mm) / TOF_SIGMA_MM
 
        # Kinematic branch / temporal continuity.
        r_smooth = (q - q_prev) / SMOOTH_SIGMA_RAD
 
        # Prefer physiological PIP/DIP sharing, rather than assigning all
        # flexion to q1 and q2 while leaving q3 at zero.
        r_dip_coupling = (
            float(q[2]) - DIP_TO_PIP_RATIO * float(q[1])
        ) / DIP_COUPLING_SIGMA_RAD
 
        return np.concatenate(
            (
                np.array([r_theta, r_tof, r_dip_coupling], dtype=float),
                r_smooth,
            )
        )
 
    def update(
        self,
        s: WeartSample,
    ) -> tuple[np.ndarray | None, dict[str, float]]:
        dt = self._compute_dt(s.ts_ms)
 
        if not self.calibrated:
            progress = self._calibrate(s)
            return None, {"calibration": progress, "dt": dt}
 
        theta = self._update_theta(s, dt)
        q_prev = self.q.copy()
 
        # Multi-start because the range model can be poorly conditioned near a
        # perfectly straight finger.
        seeds = [
            q_prev,
            np.array([theta, 0.0, 0.0]),
            np.array([theta - 0.15, 0.15, 0.0]),
            np.array([theta - 0.30, 0.30, 0.0]),
            np.array([theta - 0.50, 0.50, 0.0]),
            np.array([theta - 0.35, 0.25, 0.10]),
        ]
 
        best = None
        eps = 1e-8
 
        for seed in seeds:
            x0 = np.clip(seed, HUMAN_Q_LOWER + eps, HUMAN_Q_UPPER - eps)
 
            result = least_squares(
                self._residual,
                x0=x0,
                bounds=(HUMAN_Q_LOWER, HUMAN_Q_UPPER),
                args=(theta, s.tof_mm, q_prev),
                method="trf",
                max_nfev=100,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
 
            if best is None or result.cost < best.cost:
                best = result
 
        assert best is not None
        self.q = best.x.astype(float, copy=True)
 
        return self.q.copy(), {
            "dt": dt,
            "theta_deg": math.degrees(theta),
            "tof_meas_mm": float(s.tof_mm),
            "tof_pred_mm": tof_forward_model_mm(self.q) + self.tof_offset_mm,
            "cost": float(best.cost),
            "closure_weart": float(s.closure),
        }
 
 
# =============================================================================
# LEAP C++ CONTROLLER -- DIRECT, ZERO LEROBOT
# =============================================================================
 
 
def _load_leap_cpp():
    """Load the same pybind LEAP controller used by your existing setup."""
    pybind_dir = os.path.expanduser("~/.local/lib/python/leap_cpp")
    lib_dir = os.path.expanduser("~/.local/lib")
 
    if os.path.isdir(pybind_dir) and pybind_dir not in sys.path:
        sys.path.insert(0, pybind_dir)
 
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in existing_ld:
        os.environ["LD_LIBRARY_PATH"] = (
            lib_dir + (os.pathsep + existing_ld if existing_ld else "")
        )
 
    for lib_name in ("libdxl_x64_cpp.so", "libdynamixel_client.so"):
        try:
            ctypes.CDLL(
                os.path.join(lib_dir, lib_name),
                ctypes.RTLD_GLOBAL,
            )
        except OSError:
            # The libraries may already be linked through leap_cpp.
            pass
 
    import leap_cpp
 
    return leap_cpp
 
 
def autodetect_leap_port(requested_port: str) -> str:
    """Use an explicit port if provided, otherwise find a likely LEAP USB port."""
    requested_port = requested_port.strip()
    if requested_port:
        return requested_port
 
    candidates: list[str] = []
    candidates.extend(sorted(glob.glob("/dev/serial/by-id/*")))
    candidates.extend(["/dev/ttyUSB0", "/dev/ttyUSB1"])
 
    for path in candidates:
        if os.path.exists(path):
            return path
 
    raise RuntimeError(
        "No LEAP serial port found. Pass e.g. "
        "--ros-args -p leap_port:=/dev/serial/by-id/<your-device>"
    )
 
 
def leap_sim_to_motor(q_sim: np.ndarray, motor_count: int) -> np.ndarray:
    """Official LEAP v1 sim -> real convention: add pi to every joint."""
    q_sim = np.asarray(q_sim, dtype=float)
    if q_sim.shape != (motor_count,):
        raise ValueError(
            f"Expected {motor_count} LEAPsim joints, got {q_sim.shape}"
        )
    return q_sim + math.pi
 
 
class DirectLeapIndexDriver:
    def __init__(
        self,
        *,
        enabled: bool,
        port: str,
        kp: float,
        ki: float,
        kd: float,
        model: LeapUrdfModel,
    ) -> None:
        self.enabled = bool(enabled)
        self.requested_port = str(port)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.model = model
 
        # LEAPsim zero means physically open/neutral -> real motors at pi.
        self.current_sim_qpos = np.zeros(self.model.motor_count, dtype=float)
        self.filtered_index_q = np.zeros(3, dtype=float)
 
        self._ctrl = None
        self.port: str | None = None
 
    def connect(self) -> None:
        print(
            "[LEAP] URDF mapping: index flexion motors "
            f"{self.model.index_flexion_motor_indices}; "
            f"loaded {self.model.motor_count} joints.",
            flush=True,
        )
        print(
            "[LEAP] This node commands ONLY the index flexion joints.",
            flush=True,
        )
 
        if not self.enabled:
            print("[LEAP] hardware output DISABLED (dry-run).", flush=True)
            return
 
        self.port = autodetect_leap_port(self.requested_port)
        leap_cpp = _load_leap_cpp()
 
        self._ctrl = leap_cpp.LeapController(self.port)
        self._ctrl.connect()
        self._ctrl.setGains(int(self.kp), int(self.ki), int(self.kd))
 
        # Physical open/neutral pose: all LEAP real joints at pi.
        home_motor_q = leap_sim_to_motor(
            np.zeros(self.model.motor_count, dtype=float),
            self.model.motor_count,
        )
        self._ctrl.set_leap(home_motor_q)
 
        print(
            f"[LEAP] connected on {self.port}; gains="
            f"({self.kp:g}, {self.ki:g}, {self.kd:g}); sent open pose.",
            flush=True,
        )
 
    def disconnect(self) -> None:
        if self._ctrl is not None:
            self._ctrl.disconnect()
            self._ctrl = None
            print("[LEAP] disconnected.", flush=True)
# retargeting from human to LEAP limits
    def _human_to_leap_index(
        self,
        q_human: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        # The URDF is authoritative for the mechanical/software travel limits.
        indices = list(self.model.index_flexion_motor_indices)
        lower = self.model.sim_lower[indices]
        upper = self.model.sim_upper[indices]
        q_human = np.clip(
            np.asarray(q_human, dtype=float), HUMAN_Q_LOWER, HUMAN_Q_UPPER
        )
 
        # Keep q=0 as the LEAP neutral pose and map positive human flexion to
        # the URDF upper stop. MCP extension, the only negative human range,
        # maps to its URDF lower stop. This avoids hard-coded LEAP scales.
        positive_scale = upper / HUMAN_Q_UPPER
        negative_scale = positive_scale.copy()
        has_extension = HUMAN_Q_LOWER < -np.finfo(float).eps
        negative_scale[has_extension] = (
            lower[has_extension] / HUMAN_Q_LOWER[has_extension]
        )
        q_target = q_human * np.where(
            q_human < 0.0, negative_scale, positive_scale
        )
        q_target = np.clip(q_target, lower, upper)
 
        # Low-pass filter.
        q_lp = self.filtered_index_q + COMMAND_LOW_PASS_ALPHA * (
            q_target - self.filtered_index_q
        )
 
        # Per-sample speed limit.
        # Never exceed either the configured safety speed or the URDF velocity.
        max_speed = np.minimum(
            COMMAND_MAX_SPEED_RAD_S,
            self.model.velocity_limits[indices],
        )
        max_step = max_speed * max(dt, MIN_DT_S)
        delta = np.clip(
            q_lp - self.filtered_index_q,
            -max_step,
            +max_step,
        )
 
        self.filtered_index_q = self.filtered_index_q + delta
        return self.filtered_index_q.copy()
 
    def command(
        self,
        q_human: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        q_index_sim = self._human_to_leap_index(q_human, dt)
 
        # Preserve every other LEAP joint in the current command and update only
        # index MCP-forward, PIP, DIP.
        for motor_idx, value in zip(
            self.model.index_flexion_motor_indices,
            q_index_sim,
            strict=True,
        ):
            self.current_sim_qpos[motor_idx] = float(value)
 
        motor_qpos = leap_sim_to_motor(
            self.current_sim_qpos,
            self.model.motor_count,
        )
 
        if self.enabled:
            if self._ctrl is None:
                raise RuntimeError(
                    "LEAP output enabled but controller is not connected"
                )
            self._ctrl.set_leap(motor_qpos)
 
        return q_index_sim.copy(), motor_qpos.copy()
 
 
# =============================================================================
# ROS 2 NODE
# =============================================================================
 
 
class WeartIndexLeapNode(Node):
    def __init__(self) -> None:
        super().__init__("weart_index_to_leap")
 
        self.declare_parameter("input_topic", DEFAULT_INPUT_TOPIC)
        self.declare_parameter("output_topic", DEFAULT_OUTPUT_TOPIC)
        self.declare_parameter("enable_hardware", DEFAULT_ENABLE_HARDWARE)
        self.declare_parameter("leap_port", DEFAULT_LEAP_PORT)
        self.declare_parameter("urdf_path", DEFAULT_URDF_PATH)
        self.declare_parameter("kp", DEFAULT_KP)
        self.declare_parameter("ki", DEFAULT_KI)
        self.declare_parameter("kd", DEFAULT_KD)
 
        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        enable_hardware = bool(self.get_parameter("enable_hardware").value)
        leap_port = str(self.get_parameter("leap_port").value)
        urdf_path = str(self.get_parameter("urdf_path").value)
        kp = float(self.get_parameter("kp").value)
        ki = float(self.get_parameter("ki").value)
        kd = float(self.get_parameter("kd").value)
 
        self.estimator = IndexEstimator()
        self.model = load_leap_urdf(urdf_path)
        self.leap = DirectLeapIndexDriver(
            enabled=enable_hardware,
            port=leap_port,
            kp=kp,
            ki=ki,
            kd=kd,
            model=self.model,
        )
        self.leap.connect()
 
        self.sub = self.create_subscription(
            String,
            input_topic,
            self._on_weart,
            20,
        )
        self.pub = self.create_publisher(
            Float64MultiArray,
            output_topic,
            20,
        )
 
        self.counter = 0
 
        self.get_logger().info(f"WEART input: {input_topic}")
        self.get_logger().info(f"Estimated q output: {output_topic}")
        self.get_logger().info(
            f"LEAP URDF: {Path(urdf_path).expanduser()} | "
            f"index motors={self.model.index_flexion_motor_indices}"
        )
        self.get_logger().warning(
            "Keep INDEX OPEN and HAND STILL until calibration reaches 100%."
        )
 
        if enable_hardware:
            self.get_logger().warning("LEAP HARDWARE OUTPUT ENABLED.")
        else:
            self.get_logger().warning(
                "LEAP dry-run. Enable only after checking q with "
                "--ros-args -p enable_hardware:=true"
            )
 
    def destroy_node(self):
        try:
            self.leap.disconnect()
        finally:
            return super().destroy_node()
 
    def _on_weart(self, msg: String) -> None:
        try:
            sample = parse_weart_line(msg.data)
        except Exception as exc:
            self.get_logger().warning(f"WEART parse error: {exc}")
            return
 
        try:
            q_human, diag = self.estimator.update(sample)
        except Exception as exc:
            self.get_logger().error(f"Estimator failed: {exc}")
            return
 
        if q_human is None:
            if self.counter % 5 == 0:
                self.get_logger().info(
                    f"Calibration: {100.0 * diag['calibration']:.0f}% "
                    "(finger open + hand still)"
                )
            self.counter += 1
            return
 
        # Publish estimated human q=[MCP-forward, PIP, DIP] in radians.
        out = Float64MultiArray()
        out.data = [float(v) for v in q_human]
        self.pub.publish(out)
 
        try:
            q_index_sim, motor_qpos = self.leap.command(
                q_human,
                float(diag["dt"]),
            )
        except Exception as exc:
            self.get_logger().error(f"LEAP command failed: {exc}")
            return
 
        if self.counter % 5 == 0:
            q_human_deg = np.rad2deg(q_human)
            q_leap_deg = np.rad2deg(q_index_sim)
            motor_selected = motor_qpos[
                list(self.model.index_flexion_motor_indices)
            ]
 
            self.get_logger().info(
                "human q=[%.1f, %.1f, %.1f] deg | "
                "LEAPsim=[%.1f, %.1f, %.1f] deg | "
                "theta=%.1f deg | "
                "ToF=%.1f -> %.1f mm | index motor=%s"
                % (
                    q_human_deg[0],
                    q_human_deg[1],
                    q_human_deg[2],
                    q_leap_deg[0],
                    q_leap_deg[1],
                    q_leap_deg[2],
                    diag["theta_deg"],
                    diag["tof_meas_mm"],
                    diag["tof_pred_mm"],
                    np.round(motor_selected, 3).tolist(),
                )
            )
 
        self.counter += 1
 
 
def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
 
    try:
        node = WeartIndexLeapNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
 
 
if __name__ == "__main__":
    main()