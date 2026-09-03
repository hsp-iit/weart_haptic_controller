#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""ROS 2: WEART TouchDIVER index -> direct LEAP index q1,q2,q3 via full URDF FK.

ZERO LeRobot dependency.

Dependencies:
    rclpy
    std_msgs
    numpy
    scipy
    leap_cpp

Expected ROS input type: std_msgs/msg/String
Example payload:
    ts=1788171631772 | closure=0.000 | opening=1.000 | abduction=0.525 |
    adduction=0.475 | ACC[g]=(0.087, 0.928, -0.286) |
    GYRO[deg/s]=(2.030, -7.000, -5.740) | ToF[mm]=83

Estimator state / output:
    q1 = LEAP joint 1 = index MCP forward/flexion
    q2 = LEAP joint 2 = index PIP flexion
    q3 = LEAP joint 3 = index DIP flexion

Planar assumption:
    LEAP joint 0 (index MCP side/adduction-abduction) is fixed to 0.

Unlike the older version, there is NO intermediate "human q" and NO linear
human->LEAP joint scaling. The optimizer solves directly for LEAP q1,q2,q3.
The fingertip position and distal orientation are computed with the complete
URDF transform chain:

    palm_lower --joint 1--> mcp_joint --joint 0(q=0)--> pip
               --joint 2--> dip --joint 3--> fingertip
               --index_tip(fixed)--> index_tip_head

Cost function:
    IMU fingertip orientation consistency
  + WEART ToF consistency through calibrated URDF range variation
  + temporal continuity
  + preference for minimum |q3|

IMPORTANT:
    1. Hardware output is OFF by default.
    2. Keep the HUMAN index OPEN and HAND STILL during startup calibration.
    3. The LEAP URDF does NOT define the WEART sensor mounting. Therefore the
       IMU axis/sign and the palm ToF reference point remain calibration params.
    4. Run dry first and verify estimated q before enabling hardware.
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
# GENERAL / ROS DEFAULTS
# =============================================================================

DEFAULT_INPUT_TOPIC = "/weart/index/raw"
DEFAULT_OUTPUT_TOPIC = "/weart/index/estimated_leap_joints"
DEFAULT_URDF_PATH = str(Path(__file__).with_name("model.urdf"))

DEFAULT_ENABLE_HARDWARE = False
DEFAULT_LEAP_PORT = ""

DEFAULT_KP = 600.0
DEFAULT_KI = 0.0
DEFAULT_KD = 200.0

MIN_DT_S = 0.002
MAX_DT_S = 0.25


# =============================================================================
# WEART IMU MOUNTING -- CANNOT BE INFERRED FROM THE LEAP URDF
# =============================================================================

# Dominant gyro axis for planar fingertip rotation: 0=x, 1=y, 2=z.
GYRO_FLEX_AXIS = 2
GYRO_FLEX_SIGN = +1.0

# Gravity angle used to slowly correct gyro drift:
# theta_acc = atan2(sign_num * acc[num_axis], sign_den * acc[den_axis])
ACC_ANGLE_NUM_AXIS = 0
ACC_ANGLE_DEN_AXIS = 1
ACC_ANGLE_NUM_SIGN = +1.0
ACC_ANGLE_DEN_SIGN = +1.0

ACC_CORRECTION_GAIN = 0.04
ACC_NORM_SIGMA_G = 0.12


# =============================================================================
# ESTIMATOR WEIGHTS
# =============================================================================

THETA_SIGMA_RAD = np.deg2rad(4.0)
TOF_SIGMA_MM = 4.0
SMOOTH_SIGMA_RAD = np.deg2rad(18.0)

# Smaller -> stronger preference for DIP q3 ~= 0.
Q3_PREFERENCE_SIGMA_RAD = np.deg2rad(10.0)


# =============================================================================
# STARTUP CALIBRATION
# =============================================================================

CALIBRATION_SAMPLES = 30
CALIBRATION_MAX_CLOSURE = 0.10
CALIBRATION_MAX_GYRO_NORM_DEG_S = 15.0


# =============================================================================
# TOF MODEL DEFAULTS
# =============================================================================

# The ToF reference point is expressed in the URDF palm_lower frame.
# Default = palm_lower origin. Change these ROS params if you know a better
# correspondence for the WEART palm-side ranging point.
DEFAULT_TOF_PALM_X_MM = 0.0
DEFAULT_TOF_PALM_Y_MM = 0.0
DEFAULT_TOF_PALM_Z_MM = 0.0

# Human-hand ToF millimetres are not assumed equal to LEAP geometry millimetres.
# We use:
#   tof_pred = tof_open_human
#              + scale * (range_robot(q) - range_robot(q_open))
# scale may also be negative if the chosen robot palm reference has the opposite
# distance trend from the WEART measurement.
DEFAULT_TOF_DISTANCE_SCALE = 1.0


# =============================================================================
# COMMAND FILTER
# =============================================================================

COMMAND_LOW_PASS_ALPHA = 0.30
COMMAND_MAX_SPEED_RAD_S = 2.0


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
        acc_g=np.array(
            [float(x["ax"]), float(x["ay"]), float(x["az"])],
            dtype=float,
        ),
        gyro_deg_s=np.array(
            [float(x["gx"]), float(x["gy"]), float(x["gz"])],
            dtype=float,
        ),
        tof_mm=float(x["tof"]),
    )


# =============================================================================
# URDF MATH
# =============================================================================


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def angle_error(a: float, b: float) -> float:
    return wrap_pi(a - b)


def _parse_vec3(text: str | None) -> np.ndarray:
    if text is None or not text.strip():
        return np.zeros(3, dtype=float)
    values = [float(v) for v in text.split()]
    if len(values) != 3:
        raise ValueError(f"Expected xyz/rpy vector of length 3, got {text!r}")
    return np.asarray(values, dtype=float)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = [float(v) for v in rpy]

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        dtype=float,
    )
    ry = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        dtype=float,
    )
    rz = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return rz @ ry @ rx


def _origin_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=float)
    t[:3, :3] = _rpy_matrix(rpy)
    t[:3, 3] = xyz
    return t


def _axis_rotation(axis: np.ndarray, q: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise ValueError("URDF revolute joint has zero axis")
    x, y, z = axis / norm

    c = math.cos(q)
    s = math.sin(q)
    C = 1.0 - c

    r = np.array(
        [
            [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
            [z*x*C - y*s, z*y*C + x*s, z*z*C + c],
        ],
        dtype=float,
    )

    t = np.eye(4, dtype=float)
    t[:3, :3] = r
    return t


def _signed_angle_in_plane(
    v_ref: np.ndarray,
    v: np.ndarray,
    plane_normal: np.ndarray,
) -> float:
    """Signed angle after projecting both vectors into the motion plane."""
    n = np.asarray(plane_normal, dtype=float)
    n /= np.linalg.norm(n)

    a = np.asarray(v_ref, dtype=float)
    b = np.asarray(v, dtype=float)

    a = a - n * float(np.dot(a, n))
    b = b - n * float(np.dot(b, n))

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-10 or nb < 1e-10:
        raise RuntimeError("Cannot define projected distal fingertip orientation")

    a /= na
    b /= nb

    return math.atan2(
        float(np.dot(n, np.cross(a, b))),
        float(np.dot(a, b)),
    )


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_T: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    velocity: float


class LeapIndexKinematics:
    """Full URDF forward kinematics for the primary LEAP index finger."""

    def __init__(
        self,
        urdf_path: str,
        tof_palm_point_mm: np.ndarray,
    ) -> None:
        self.urdf_path = str(Path(urdf_path).expanduser().resolve())
        path = Path(self.urdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"LEAP URDF not found: {path}")

        root = ET.parse(path).getroot()
        self.joints: dict[str, UrdfJoint] = {}

        numerical_joint_indices: list[int] = []

        for elem in root.findall("joint"):
            name = elem.attrib.get("name", "")
            joint_type = elem.attrib.get("type", "fixed")

            parent_e = elem.find("parent")
            child_e = elem.find("child")
            if parent_e is None or child_e is None:
                continue

            origin_e = elem.find("origin")
            xyz = _parse_vec3(
                origin_e.attrib.get("xyz") if origin_e is not None else None
            )
            rpy = _parse_vec3(
                origin_e.attrib.get("rpy") if origin_e is not None else None
            )

            axis_e = elem.find("axis")
            axis = _parse_vec3(
                axis_e.attrib.get("xyz") if axis_e is not None else "0 0 1"
            )

            limit_e = elem.find("limit")
            lower = 0.0
            upper = 0.0
            velocity = math.inf
            if limit_e is not None:
                lower = float(limit_e.attrib.get("lower", "0"))
                upper = float(limit_e.attrib.get("upper", "0"))
                velocity = float(limit_e.attrib.get("velocity", "inf"))

            self.joints[name] = UrdfJoint(
                name=name,
                joint_type=joint_type,
                parent=str(parent_e.attrib["link"]),
                child=str(child_e.attrib["link"]),
                origin_T=_origin_transform(xyz, rpy),
                axis=axis,
                lower=lower,
                upper=upper,
                velocity=velocity,
            )

            if name.isdecimal():
                numerical_joint_indices.append(int(name))

        # Expected topology for the supplied LEAP right-hand URDF.
        required = ("1", "0", "2", "3", "index_tip")
        missing = [name for name in required if name not in self.joints]
        if missing:
            raise ValueError(f"Missing expected index joints in URDF: {missing}")

        j1 = self.joints["1"]
        j0 = self.joints["0"]
        j2 = self.joints["2"]
        j3 = self.joints["3"]
        jtip = self.joints["index_tip"]

        expected_topology = (
            j1.parent == "palm_lower"
            and j1.child == "mcp_joint"
            and j0.parent == "mcp_joint"
            and j0.child == "pip"
            and j2.parent == "pip"
            and j2.child == "dip"
            and j3.parent == "dip"
            and j3.child == "fingertip"
            and jtip.parent == "fingertip"
            and jtip.child == "index_tip_head"
        )
        if not expected_topology:
            raise ValueError(
                "URDF index topology differs from expected "
                "palm_lower -> 1 -> 0 -> 2 -> 3 -> index_tip"
            )

        self.index_motor_indices = (1, 2, 3)
        self.motor_count = max(numerical_joint_indices) + 1

        if sorted(numerical_joint_indices) != list(range(self.motor_count)):
            raise ValueError(
                "Numerical LEAP joint names are not contiguous from zero: "
                f"{sorted(numerical_joint_indices)}"
            )

        self.q_lower = np.array([j1.lower, j2.lower, j3.lower], dtype=float)
        self.q_upper = np.array([j1.upper, j2.upper, j3.upper], dtype=float)
        self.q_velocity = np.array(
            [j1.velocity, j2.velocity, j3.velocity],
            dtype=float,
        )

        self.tof_palm_point_m = np.asarray(tof_palm_point_mm, dtype=float) * 1e-3

        open_fk = self.forward(np.zeros(3, dtype=float), debug=True)
        self.open_tip_m = open_fk["tip_m"]
        self.open_distal_vector = open_fk["distal_vector"]
        self.plane_normal = open_fk["axis1_palm"].copy()
        self.plane_normal /= np.linalg.norm(self.plane_normal)

        # Verify the three active flexion axes are effectively parallel.
        self.flex_axis_parallel_cos = np.array(
            [
                abs(float(np.dot(self.plane_normal, open_fk["axis2_palm"]))),
                abs(float(np.dot(self.plane_normal, open_fk["axis3_palm"]))),
            ],
            dtype=float,
        )

        self.open_robot_range_mm = self.robot_range_mm(np.zeros(3, dtype=float))

        # Diagnostics: center-to-center / center-to-tip 3D distances at q=0.
        self.effective_segment_lengths_mm = 1000.0 * np.array(
            [
                np.linalg.norm(open_fk["joint2_pos_m"] - open_fk["joint1_pos_m"]),
                np.linalg.norm(open_fk["joint3_pos_m"] - open_fk["joint2_pos_m"]),
                np.linalg.norm(open_fk["tip_m"] - open_fk["joint3_pos_m"]),
            ],
            dtype=float,
        )

    @staticmethod
    def _apply_joint(
        T_parent: np.ndarray,
        joint: UrdfJoint,
        q: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # URDF: parent -> joint origin -> joint rotation -> child.
        T_joint = T_parent @ joint.origin_T
        joint_pos = T_joint[:3, 3].copy()

        axis_parent_frame = T_joint[:3, :3] @ joint.axis
        axis_parent_frame /= np.linalg.norm(axis_parent_frame)

        if joint.joint_type == "fixed":
            return T_joint, joint_pos, axis_parent_frame

        T_child = T_joint @ _axis_rotation(joint.axis, q)
        return T_child, joint_pos, axis_parent_frame

    def forward(
        self,
        q: np.ndarray,
        *,
        debug: bool = False,
    ):
        q1, q2, q3 = np.asarray(q, dtype=float)

        # The reference frame is palm_lower, so start from identity.
        T = np.eye(4, dtype=float)

        # palm_lower -- joint 1 --> mcp_joint
        T, p1, a1 = self._apply_joint(T, self.joints["1"], q1)

        # mcp_joint -- joint 0 --> pip
        # Planar assumption: MCP-side/adduction-abduction is fixed at zero.
        T, p0, a0 = self._apply_joint(T, self.joints["0"], 0.0)

        # pip -- joint 2 --> dip
        T, p2, a2 = self._apply_joint(T, self.joints["2"], q2)

        # dip -- joint 3 --> fingertip
        T_fingertip, p3, a3 = self._apply_joint(T, self.joints["3"], q3)

        # fingertip -- fixed index_tip --> index_tip_head
        T_tip = T_fingertip @ self.joints["index_tip"].origin_T
        p_tip = T_tip[:3, 3].copy()

        # Use the joint-3-center -> index_tip_head vector as distal direction.
        distal_vector = p_tip - p3

        if not debug:
            return p_tip

        return {
            "tip_m": p_tip,
            "distal_vector": distal_vector,
            "joint1_pos_m": p1,
            "joint0_pos_m": p0,
            "joint2_pos_m": p2,
            "joint3_pos_m": p3,
            "axis1_palm": a1,
            "axis0_palm": a0,
            "axis2_palm": a2,
            "axis3_palm": a3,
            "T_fingertip": T_fingertip,
            "T_tip": T_tip,
        }

    def fingertip_orientation_rad(self, q: np.ndarray) -> float:
        """Distal fingertip orientation relative to the URDF open pose."""
        fk = self.forward(q, debug=True)
        return wrap_pi(
            _signed_angle_in_plane(
                self.open_distal_vector,
                fk["distal_vector"],
                self.plane_normal,
            )
        )

    def robot_range_mm(self, q: np.ndarray) -> float:
        p_tip = self.forward(q)
        return 1000.0 * float(np.linalg.norm(p_tip - self.tof_palm_point_m))

    def describe(self) -> str:
        return (
            f"URDF: {self.urdf_path}\n"
            f"  direct optimized joints: (1=MCP-forward, 2=PIP, 3=DIP)\n"
            f"  joint 0 MCP-side fixed at 0 rad\n"
            f"  q lower [deg]: {np.round(np.rad2deg(self.q_lower), 2).tolist()}\n"
            f"  q upper [deg]: {np.round(np.rad2deg(self.q_upper), 2).tolist()}\n"
            f"  q velocity [rad/s]: {np.round(self.q_velocity, 3).tolist()}\n"
            f"  effective zero-pose 3D segment distances [mm]: "
            f"{np.round(self.effective_segment_lengths_mm, 2).tolist()}\n"
            f"  open index_tip_head in palm_lower [m]: "
            f"{np.round(self.open_tip_m, 5).tolist()}\n"
            f"  configured ToF palm point [m]: "
            f"{np.round(self.tof_palm_point_m, 5).tolist()}\n"
            f"  open robot range [mm]: {self.open_robot_range_mm:.2f}\n"
            f"  flex-axis parallel cosines: "
            f"{np.round(self.flex_axis_parallel_cos, 6).tolist()}"
        )


# =============================================================================
# INDEX ESTIMATOR -- SOLVES DIRECTLY FOR LEAP q1,q2,q3
# =============================================================================


class IndexEstimator:
    def __init__(
        self,
        kinematics: LeapIndexKinematics,
        *,
        tof_distance_scale: float,
    ) -> None:
        self.kin = kinematics
        self.tof_distance_scale = float(tof_distance_scale)

        self.q = np.zeros(3, dtype=float)
        self.theta = 0.0
        self.last_ts_ms: int | None = None

        self.acc_open_angle: float | None = None
        self.gyro_bias_deg_s = 0.0
        self.open_tof_human_mm: float | None = None

        self._calib_acc_angles: list[float] = []
        self._calib_gyro_flex: list[float] = []
        self._calib_tof: list[float] = []

        self.calibrated = False

    def _raw_acc_angle(self, acc: np.ndarray) -> float:
        num = ACC_ANGLE_NUM_SIGN * float(acc[ACC_ANGLE_NUM_AXIS])
        den = ACC_ANGLE_DEN_SIGN * float(acc[ACC_ANGLE_DEN_AXIS])
        return math.atan2(num, den)

    def _sample_is_good_for_calibration(self, s: WeartSample) -> bool:
        return (
            s.closure <= CALIBRATION_MAX_CLOSURE
            and float(np.linalg.norm(s.gyro_deg_s))
            <= CALIBRATION_MAX_GYRO_NORM_DEG_S
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
            # Circular mean of open-hand gravity angle.
            sin_mean = float(np.mean(np.sin(self._calib_acc_angles)))
            cos_mean = float(np.mean(np.cos(self._calib_acc_angles)))
            self.acc_open_angle = math.atan2(sin_mean, cos_mean)

            # Gyro bias while stationary.
            self.gyro_bias_deg_s = float(np.mean(self._calib_gyro_flex))

            # Absolute human ToF at open hand.
            self.open_tof_human_mm = float(np.median(self._calib_tof))

            # q=0 is explicitly the LEAP open/neutral reference in this model.
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

    def _update_theta(self, s: WeartSample, dt: float) -> float:
        # 1) Fast angular motion from gyro.
        gyro_flex_deg_s = (
            GYRO_FLEX_SIGN * float(s.gyro_deg_s[GYRO_FLEX_AXIS])
            - self.gyro_bias_deg_s
        )
        theta_gyro = wrap_pi(
            self.theta + math.radians(gyro_flex_deg_s) * dt
        )

        # 2) Slow gravity correction from accelerometer only when |a| ~= 1 g.
        acc_norm = float(np.linalg.norm(s.acc_g))
        trust = math.exp(
            -0.5 * ((acc_norm - 1.0) / ACC_NORM_SIGMA_G) ** 2
        )

        assert self.acc_open_angle is not None
        theta_acc = wrap_pi(self._raw_acc_angle(s.acc_g) - self.acc_open_angle)

        self.theta = wrap_pi(
            theta_gyro
            + ACC_CORRECTION_GAIN
            * trust
            * angle_error(theta_acc, theta_gyro)
        )
        return self.theta

    def _predict_weart_tof_mm(self, q: np.ndarray) -> float:
        """Map URDF range CHANGE to the human WEART ToF reference."""
        assert self.open_tof_human_mm is not None

        robot_range_delta_mm = (
            self.kin.robot_range_mm(q) - self.kin.open_robot_range_mm
        )

        return (
            self.open_tof_human_mm
            + self.tof_distance_scale * robot_range_delta_mm
        )

    def _residual(
        self,
        q: np.ndarray,
        theta_meas: float,
        tof_meas_mm: float,
        q_prev: np.ndarray,
    ) -> np.ndarray:
        # A) Distal orientation predicted by FULL URDF FK.
        theta_pred = self.kin.fingertip_orientation_rad(q)
        r_theta = angle_error(theta_pred, theta_meas) / THETA_SIGMA_RAD

        # B) ToF consistency using FULL URDF fingertip position.
        tof_pred = self._predict_weart_tof_mm(q)
        r_tof = (tof_pred - tof_meas_mm) / TOF_SIGMA_MM

        # C) Prefer the same kinematic branch from one sample to the next.
        r_smooth = (q - q_prev) / SMOOTH_SIGMA_RAD

        # D) Requested secondary objective: minimize |DIP q3|.
        r_q3 = float(q[2]) / Q3_PREFERENCE_SIGMA_RAD

        return np.concatenate(
            (
                np.array([r_theta, r_tof, r_q3], dtype=float),
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

        lo = self.kin.q_lower
        hi = self.kin.q_upper

        # Multiple initial guesses reduce the chance of selecting a bad local
        # solution when different joint distributions fit similar measurements.
        seeds = [
            q_prev,
            np.array([theta, 0.0, 0.0], dtype=float),
            np.array([0.75 * theta, 0.25 * theta, 0.0], dtype=float),
            np.array([0.55 * theta, 0.45 * theta, 0.0], dtype=float),
            np.array([0.50 * theta, 0.40 * theta, 0.10 * theta], dtype=float),
            0.5 * (lo + hi),
        ]

        best = None
        eps = 1e-8

        for seed in seeds:
            x0 = np.clip(seed, lo + eps, hi - eps)

            result = least_squares(
                self._residual,
                x0=x0,
                bounds=(lo, hi),
                args=(theta, s.tof_mm, q_prev),
                method="trf",
                max_nfev=120,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
            )

            if best is None or result.cost < best.cost:
                best = result

        assert best is not None
        self.q = best.x.astype(float, copy=True)

        theta_pred = self.kin.fingertip_orientation_rad(self.q)
        tof_pred = self._predict_weart_tof_mm(self.q)

        return self.q.copy(), {
            "dt": dt,
            "theta_meas_deg": math.degrees(theta),
            "theta_pred_deg": math.degrees(theta_pred),
            "tof_meas_mm": float(s.tof_mm),
            "tof_pred_mm": float(tof_pred),
            "robot_range_mm": self.kin.robot_range_mm(self.q),
            "cost": float(best.cost),
            "closure_weart": float(s.closure),
        }


# =============================================================================
# DIRECT LEAP C++ CONTROLLER -- ZERO LEROBOT
# =============================================================================


def _load_leap_cpp():
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
            pass

    import leap_cpp
    return leap_cpp


def autodetect_leap_port(requested_port: str) -> str:
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


def leap_sim_to_motor(q_sim: np.ndarray) -> np.ndarray:
    """LEAP v1 convention used by the existing setup: motor = sim + pi."""
    return np.asarray(q_sim, dtype=float) + math.pi


class DirectLeapIndexDriver:
    def __init__(
        self,
        kinematics: LeapIndexKinematics,
        *,
        enabled: bool,
        port: str,
        kp: float,
        ki: float,
        kd: float,
    ) -> None:
        self.kin = kinematics
        self.enabled = bool(enabled)
        self.requested_port = str(port)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

        # Every motor starts at LEAPsim q=0. Only 1,2,3 are updated here.
        self.current_sim_qpos = np.zeros(self.kin.motor_count, dtype=float)
        self.filtered_index_q = np.zeros(3, dtype=float)

        self._ctrl = None
        self.port: str | None = None

    def connect(self) -> None:
        print(
            "[LEAP] Direct index control: joint 0 fixed, joints 1/2/3 commanded.",
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

        # Open/neutral command.
        self._ctrl.set_leap(
            leap_sim_to_motor(np.zeros(self.kin.motor_count, dtype=float))
        )

        print(
            f"[LEAP] connected on {self.port}; gains="
            f"({self.kp:g}, {self.ki:g}, {self.kd:g})",
            flush=True,
        )

    def disconnect(self) -> None:
        if self._ctrl is not None:
            self._ctrl.disconnect()
            self._ctrl = None
            print("[LEAP] disconnected.", flush=True)

    def read_actual_sim_qpos(self) -> np.ndarray | None:
        """Read all physical joints and convert motor coordinates to URDF q."""
        if self._ctrl is None:
            return None
        motor_qpos = np.asarray(self._ctrl.read_pos(), dtype=float).reshape(-1)
        if motor_qpos.size != self.kin.motor_count:
            raise ValueError(
                f"LEAP returned {motor_qpos.size} joints, "
                f"expected {self.kin.motor_count}"
            )
        if not np.all(np.isfinite(motor_qpos)):
            raise ValueError("LEAP returned non-finite joint positions")
        return motor_qpos - math.pi

    def command(
        self,
        q_leap: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        # q_leap is ALREADY in LEAP URDF coordinates. No human->robot scaling.
        q_target = np.clip(
            np.asarray(q_leap, dtype=float),
            self.kin.q_lower,
            self.kin.q_upper,
        )

        # Low-pass smoothing.
        q_lp = self.filtered_index_q + COMMAND_LOW_PASS_ALPHA * (
            q_target - self.filtered_index_q
        )

        # Respect both our safety speed and the URDF velocity limit.
        max_speed = np.minimum(
            COMMAND_MAX_SPEED_RAD_S,
            self.kin.q_velocity,
        )
        max_step = max_speed * max(dt, MIN_DT_S)
        delta = np.clip(
            q_lp - self.filtered_index_q,
            -max_step,
            +max_step,
        )
        self.filtered_index_q = self.filtered_index_q + delta

        # Planar index: keep side joint 0 at exactly zero.
        self.current_sim_qpos[0] = 0.0
        self.current_sim_qpos[1] = self.filtered_index_q[0]
        self.current_sim_qpos[2] = self.filtered_index_q[1]
        self.current_sim_qpos[3] = self.filtered_index_q[2]

        motor_qpos = leap_sim_to_motor(self.current_sim_qpos)

        if self.enabled:
            if self._ctrl is None:
                raise RuntimeError("LEAP output enabled but controller is not connected")
            self._ctrl.set_leap(motor_qpos)

        return self.filtered_index_q.copy(), motor_qpos.copy()


# =============================================================================
# ROS 2 NODE
# =============================================================================


class WeartIndexLeapUrdfNode(Node):
    def __init__(self) -> None:
        super().__init__("weart_index_to_leap_full_urdf")

        self.declare_parameter("input_topic", DEFAULT_INPUT_TOPIC)
        self.declare_parameter("output_topic", DEFAULT_OUTPUT_TOPIC)
        self.declare_parameter("urdf_path", DEFAULT_URDF_PATH)

        self.declare_parameter("enable_hardware", DEFAULT_ENABLE_HARDWARE)
        self.declare_parameter("leap_port", DEFAULT_LEAP_PORT)
        self.declare_parameter("kp", DEFAULT_KP)
        self.declare_parameter("ki", DEFAULT_KI)
        self.declare_parameter("kd", DEFAULT_KD)

        self.declare_parameter("tof_distance_scale", DEFAULT_TOF_DISTANCE_SCALE)
        self.declare_parameter("tof_palm_x_mm", DEFAULT_TOF_PALM_X_MM)
        self.declare_parameter("tof_palm_y_mm", DEFAULT_TOF_PALM_Y_MM)
        self.declare_parameter("tof_palm_z_mm", DEFAULT_TOF_PALM_Z_MM)

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        urdf_path = str(self.get_parameter("urdf_path").value)

        enable_hardware = bool(self.get_parameter("enable_hardware").value)
        leap_port = str(self.get_parameter("leap_port").value)

        tof_palm_point_mm = np.array(
            [
                float(self.get_parameter("tof_palm_x_mm").value),
                float(self.get_parameter("tof_palm_y_mm").value),
                float(self.get_parameter("tof_palm_z_mm").value),
            ],
            dtype=float,
        )

        self.kin = LeapIndexKinematics(
            urdf_path,
            tof_palm_point_mm=tof_palm_point_mm,
        )

        self.get_logger().info("\n" + self.kin.describe())

        if np.any(self.kin.flex_axis_parallel_cos < 0.999):
            self.get_logger().warning(
                "URDF flexion axes 1/2/3 are not perfectly parallel; full 3D FK "
                "is still used, but check the planar interpretation."
            )

        self.estimator = IndexEstimator(
            self.kin,
            tof_distance_scale=float(
                self.get_parameter("tof_distance_scale").value
            ),
        )

        self.leap = DirectLeapIndexDriver(
            self.kin,
            enabled=enable_hardware,
            port=leap_port,
            kp=float(self.get_parameter("kp").value),
            ki=float(self.get_parameter("ki").value),
            kd=float(self.get_parameter("kd").value),
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
        self.get_logger().info(f"Direct LEAP q output: {output_topic}")
        self.get_logger().warning(
            "Calibration: keep HUMAN INDEX OPEN and HAND STILL until 100%."
        )

        if enable_hardware:
            self.get_logger().warning("LEAP HARDWARE OUTPUT ENABLED.")
        else:
            self.get_logger().warning("LEAP dry-run mode: no motor commands sent.")

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
            q_leap, diag = self.estimator.update(sample)
        except Exception as exc:
            self.get_logger().error(f"Estimator failed: {exc}")
            return

        if q_leap is None:
            if self.counter % 5 == 0:
                self.get_logger().info(
                    f"Calibration: {100.0 * diag['calibration']:.0f}% "
                    "(finger open + hand still)"
                )
            self.counter += 1
            return

        # Publish DIRECT LEAP URDF q=[joint1, joint2, joint3] in radians.
        out = Float64MultiArray()
        out.data = [float(v) for v in q_leap]
        self.pub.publish(out)

        try:
            q_cmd, motor_qpos = self.leap.command(
                q_leap,
                float(diag["dt"]),
            )
        except Exception as exc:
            self.get_logger().error(f"LEAP command failed: {exc}")
            return

        if self.counter % 5 == 0:
            self.get_logger().info(
                "q direct=[%.1f, %.1f, %.1f] deg | "
                "cmd=[%.1f, %.1f, %.1f] deg | "
                "q3 minimized | theta IMU/FK=%.1f/%.1f deg | "
                "ToF meas/pred=%.1f/%.1f mm | robot_range=%.1f mm | "
                "cost=%.3f | motor123=%s"
                % (
                    *np.rad2deg(q_leap),
                    *np.rad2deg(q_cmd),
                    diag["theta_meas_deg"],
                    diag["theta_pred_deg"],
                    diag["tof_meas_mm"],
                    diag["tof_pred_mm"],
                    diag["robot_range_mm"],
                    diag["cost"],
                    np.round(motor_qpos[[1, 2, 3]], 3).tolist(),
                )
            )

        self.counter += 1


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None

    try:
        node = WeartIndexLeapUrdfNode()
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
