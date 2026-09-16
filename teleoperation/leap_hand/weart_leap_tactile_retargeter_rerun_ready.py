#!/usr/bin/env python3
"""
WEART -> LEAP Hand retargeting ROS 2 node.

Idea:
  1) Use WEART closure/adduction as the PRIMARY task.
  2) Keep the LEAP configuration physically valid and temporally smooth.
  3) Use a SECONDARY "tactile-ready" cost:
       - for each fingertip, compute the outward normal of the tactile pad;
       - define a task/object-independent grasp center in the palm frame;
       - penalize only when the pad normal points too far away from that grasp center.
     Inside a configurable cone there is ZERO tactile penalty.

This node intentionally does NOT use the raw accelerometer, gyroscope, or ToF yet:
  - gyro is angular velocity, not absolute orientation;
  - accelerometer integration drifts quickly;
  - ToF semantics/geometry must be known before converting it into a robot constraint.

Dependencies:
  pip install numpy scipy
  Pinocchio: install according to your ROS / system setup.

Input topics (std_msgs/String, compatible with the user's current publisher):
  /weart/thumb/raw
  /weart/index/raw
  /weart/middle/raw

Output:
  /leap/target_joint_positions     std_msgs/Float64MultiArray

IMPORTANT:
  The URDF strongly suggests that the tactile contact surface of each fingertip
  is approximately normal to local -Y (thin collision box along Y, displaced
  toward negative Y). This is used as a DEFAULT and must be verified in RViz
  or from the actual sensor CAD.
"""

import math
import re
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.optimize import least_squares

import pinocchio as pin

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

# Optional direct LEAP hardware API.
# These are the same low-level helpers used by the user's previous
# three-finger/full-hand controller.
try:
    from weart_leap_index_ik import (
        COMMAND_LOW_PASS_ALPHA,
        COMMAND_MAX_SPEED_RAD_S,
        DEFAULT_KD,
        DEFAULT_KI,
        DEFAULT_KP,
        DEFAULT_LEAP_BAUDRATE,
        DEFAULT_LEAP_PORT,
        MIN_DT_S,
        _load_leap_cpp,
        autodetect_leap_port,
        leap_sim_to_motor,
    )
    HARDWARE_DRIVER_AVAILABLE = True
except Exception as exc:
    print(
        f"[LEAP] hardware helpers unavailable ({exc!r}); "
        "falling back to dry-run defaults.",
        flush=True,
    )
    COMMAND_LOW_PASS_ALPHA = 1.0
    COMMAND_MAX_SPEED_RAD_S = 4.0
    DEFAULT_KD = 0.0
    DEFAULT_KI = 0.0
    DEFAULT_KP = 0.0
    DEFAULT_LEAP_BAUDRATE = 4000000
    DEFAULT_LEAP_PORT = ""
    MIN_DT_S = 0.002
    _load_leap_cpp = None
    autodetect_leap_port = None
    leap_sim_to_motor = None
    HARDWARE_DRIVER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def soft_hinge(x: float) -> float:
    """Equivalent to max(0, x), kept as a named function for readability."""
    return max(0.0, x)


def normalized_joint_value(q: float, q_open: float, q_closed: float) -> float:
    """
    Returns approximately:
      0 -> open
      1 -> closed
    Works even if the joint closes in the negative direction.
    """
    denom = q_closed - q_open
    if abs(denom) < 1e-9:
        return 0.0
    return (q - q_open) / denom


# ---------------------------------------------------------------------------
# WEART parsing
# ---------------------------------------------------------------------------

# Temporary input override; set False to restore live middle-finger input.
ASSUME_MIDDLE_OPEN = False

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

CLOSURE_RE = re.compile(rf"closure=({FLOAT})")
ADDUCTION_RE = re.compile(rf"adduction=({FLOAT})")
TOF_RE = re.compile(r"ToF\[mm\]=([^\s|]+)")


@dataclass
class WeartSample:
    closure: float = 0.0
    adduction: float = 1.0
    tof_mm: Optional[float] = None


def parse_weart_string(text: str) -> WeartSample:
    closure_match = CLOSURE_RE.search(text)
    adduction_match = ADDUCTION_RE.search(text)
    tof_match = TOF_RE.search(text)

    closure = float(closure_match.group(1)) if closure_match else 0.0
    adduction = float(adduction_match.group(1)) if adduction_match else 1.0

    tof = None
    if tof_match:
        raw = tof_match.group(1)
        try:
            tof = float(raw)
        except ValueError:
            tof = None

    return WeartSample(
        closure=clamp(closure, 0.0, 1.0),
        adduction=clamp(adduction, 0.0, 1.0),
        tof_mm=tof,
    )


# ---------------------------------------------------------------------------
# Retargeting configuration
# ---------------------------------------------------------------------------

@dataclass
class FingerConfig:
    name: str
    frame_name: str

    # LEAP flexion joints used to represent closure.
    flexion_joint_names: tuple

    # Optional joint used for adduction/abduction.
    adduction_joint_name: Optional[str]

    # Open/closed calibration for each flexion joint.
    q_open: np.ndarray
    q_closed: np.ndarray

    # Nominal range for the adduction joint.
    q_adduct_min: Optional[float] = None
    q_adduct_max: Optional[float] = None

    # Tactile sensor outward normal in the LOCAL fingertip frame.
    sensor_normal_local: np.ndarray = None

    # Allowed angular deviation from "faces grasp center".
    tactile_cone_deg: float = 75.0


# ---------------------------------------------------------------------------
# Main retargeter
# ---------------------------------------------------------------------------

class LeapRetargeter:
    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        if self.model.nq != 16:
            raise RuntimeError(
                f"Expected a 16-DoF LEAP hand, but Pinocchio model.nq={self.model.nq}"
            )

        self.joint_q_index: Dict[str, int] = {}
        for joint_id, joint_name in enumerate(self.model.names):
            if joint_id == 0:  # universe
                continue
            joint = self.model.joints[joint_id]
            if joint.nq == 1:
                self.joint_q_index[joint_name] = joint.idx_q

        # ------------------------------------------------------------------
        # IMPORTANT CALIBRATION ASSUMPTIONS
        #
        # Standard LEAP-like convention used here:
        #   index:  0 = MCP lateral, 1/2/3 = flexion
        #   middle: 4 = MCP lateral, 5/6/7 = flexion
        #   ring:   8 = MCP lateral, 9/10/11 = flexion
        #   thumb: 12 = base opposition/adduction-like DoF,
        #          13/14/15 = flexion/opposition chain
        #
        # Verify this on your real hand before commanding hardware.
        # ------------------------------------------------------------------

        self.fingers = {
            "index": FingerConfig(
                name="index",
                frame_name="index_tip_head",
                flexion_joint_names=("1", "2", "3"),
                adduction_joint_name=None,
                q_open=np.array([0.0, 0.0, 0.0]),
                q_closed=np.array([1.75, 1.45, 1.55]),
                sensor_normal_local=np.array([-1.0, 0.0, 0.0]),
                tactile_cone_deg=75.0,
            ),
            "middle": FingerConfig(
                name="middle",
                frame_name="middle_tip_head",
                flexion_joint_names=("5", "6", "7"),
                adduction_joint_name=None,
                q_open=np.array([0.0, 0.0, 0.0]),
                q_closed=np.array([1.75, 1.45, 1.55]),
                sensor_normal_local=np.array([-1.0, 0.0, 0.0]),
                tactile_cone_deg=75.0,
            ),
            "thumb": FingerConfig(
                name="thumb",
                frame_name="thumb_tip_head",
                flexion_joint_names=("13", "14", "15"),
                adduction_joint_name="12",
                q_open=np.array([0.0, 0.0, 0.0]),
                q_closed=np.array([1.65, 1.25, 1.20]),
                q_adduct_min=-0.15,
                q_adduct_max=1.55,
                sensor_normal_local=np.array([-1.0, 0.0, 0.0]),
                tactile_cone_deg=85.0,
            ),
        }

        # A point in the palm frame representing the generic grasp workspace.
        # This is NOT an object position.
        #
        # Tune in RViz. The only role of this point is to define "inward /
        # grasp-facing" for the tactile pads.
        self.grasp_center_palm = np.array([-0.045, -0.035, 0.045])

        self.palm_frame_id = self.model.getFrameId("palm_lower")
        if self.palm_frame_id >= len(self.model.frames):
            raise RuntimeError("Frame 'palm_lower' not found in URDF.")

        for cfg in self.fingers.values():
            frame_id = self.model.getFrameId(cfg.frame_name)
            if frame_id >= len(self.model.frames):
                raise RuntimeError(f"Frame '{cfg.frame_name}' not found in URDF.")

        # Start close to a neutral/open pose.
        self.q_prev = pin.neutral(self.model)

        # Cache joint bounds from URDF.
        self.lower = np.asarray(self.model.lowerPositionLimit, dtype=float).copy()
        self.upper = np.asarray(self.model.upperPositionLimit, dtype=float).copy()

        # Some Pinocchio models can contain "infinite" bounds.
        self.lower[~np.isfinite(self.lower)] = -math.pi
        self.upper[~np.isfinite(self.upper)] = math.pi

        # Weights act on RESIDUALS, therefore their squares contribute to cost.
        self.w_closure = 8.0
        self.w_adduction = 3.0 # 3.0 #4.0
        # Keep this OFF while visually verifying the tactile-pad normal in Rerun.
        # After identifying the correct local normal for each fingertip, set
        # this back to e.g. 1.5.
        self.w_tactile = 0.5 #0.5 #1.5
        self.w_posture = 0.35 # 0.350 #0.35
        self.w_smooth = 1.0 #0/.5 #0.50
        self.w_joint_margin = 0.0 #0.20
        # Keep unobserved lateral DoFs near neutral so the tactile term
        # cannot move index/middle sideways just to improve pad orientation.
        self.w_lateral = 1.0

        # Thumb-index tactile pinch geometry.
        # Separate weights make geometric fine tuning explicit.
        self.w_axis_alignment = 0.0 #0.5
        self.w_facing_normals = 0.0 #2.0
        self.w_gap = 0.0 #0.2

        self.index_lateral_neutral = 0.0
        self.middle_lateral_neutral = 0.0

        # Do not let optimizer take huge frame-to-frame jumps.
        self.max_nfev = 25

    def q_index(self, joint_name: str) -> int:
        if joint_name not in self.joint_q_index:
            raise KeyError(f"Joint '{joint_name}' not found in Pinocchio model.")
        return self.joint_q_index[joint_name]

    # -----------------------------------------------------------------------
    # Human-intent representation
    # -----------------------------------------------------------------------

    def robot_closure(self, q: np.ndarray, cfg: FingerConfig) -> float:
        values = []
        for j_name, q_open, q_closed in zip(
            cfg.flexion_joint_names, cfg.q_open, cfg.q_closed
        ):
            qi = q[self.q_index(j_name)]
            values.append(normalized_joint_value(qi, q_open, q_closed))

        return float(np.mean(values))

    def weart_thumb_adduction_to_leap(self, adduction: float) -> float:
        """
        Map WEART adduction to LEAP joint 12.

        On the user's setup the semantic direction is reversed:
          WEART adduction increases  -> LEAP joint 12 must decrease.

        Keeping this conversion in ONE function prevents q_nom and the
        optimization residual from accidentally using opposite conventions.
        """
        cfg = self.fingers["thumb"]
        a_robot = 1.0 - clamp(float(adduction), 0.0, 1.0)

        return (
            cfg.q_adduct_min
            + a_robot * (cfg.q_adduct_max - cfg.q_adduct_min)
        )

    def nominal_q_from_weart(
        self, samples: Dict[str, WeartSample]
    ) -> np.ndarray:
        """
        A weak nominal posture.

        This is NOT the final retargeting result.
        It simply gives the optimizer a natural distribution of flexion across
        joints. The closure loss itself only cares about the aggregate closure,
        leaving some freedom for the tactile term.
        """
        q_nom = self.q_prev.copy()

        for finger_name, sample in samples.items():
            cfg = self.fingers[finger_name]
            c = sample.closure

            for j_name, q_open, q_closed in zip(
                cfg.flexion_joint_names, cfg.q_open, cfg.q_closed
            ):
                idx = self.q_index(j_name)
                closure_gain = 0.80
                closure_exponent = 1.4

                c_shaped = np.clip(
                    closure_gain * (sample.closure ** closure_exponent),
                    0.0,
                    1.0,
                )
                q_nom[idx] = q_open + c_shaped * (q_closed - q_open)

            if cfg.adduction_joint_name is not None:
                idx = self.q_index(cfg.adduction_joint_name)

                # Your publisher defines:
                #   adduction = 1 - abduction
                #
                # Here adduction=0 maps to q_adduct_min
                #      adduction=1 maps to q_adduct_max
                #
                # If the real thumb moves in the opposite direction, swap
                # q_adduct_min and q_adduct_max.
                q_nom[idx] = self.weart_thumb_adduction_to_leap(
                    sample.adduction
                )

        return np.clip(q_nom, self.lower, self.upper)

    # -----------------------------------------------------------------------
    # Tactile geometry
    # -----------------------------------------------------------------------

    def shape_closure(self, c: float) -> float:
        gain = 0.80
        exponent = 1.4

        return float(np.clip(
            gain * (c ** exponent),
            0.0,
            1.0,
        ))

    def tactile_activation(self, closure: float) -> float:
        """Gate the tactile prior using WEART closure."""
        c = clamp(float(closure), 0.0, 1.0)
        c0 = 0.15  # tactile prior fully OFF below this closure
        c1 = 0.40  # tactile prior fully ON above this closure

        if c <= c0:
            return 0.0
        if c >= c1:
            return 1.0

        x = (c - c0) / (c1 - c0)
        return float(x * x * (3.0 - 2.0 * x))

    def tactile_angle_rad(self, q: np.ndarray, cfg: FingerConfig) -> float:
        """
        Angle between:
          sensor outward normal
        and
          direction from fingertip toward a generic grasp center.

        Small angle = pad faces into the generic grasp workspace.
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        tip_id = self.model.getFrameId(cfg.frame_name)
        tip_pose = self.data.oMf[tip_id]

        palm_pose = self.data.oMf[self.palm_frame_id]

        sensor_normal_world = normalize(
            tip_pose.rotation @ cfg.sensor_normal_local
        )

        grasp_center_world = (
            palm_pose.translation
            + palm_pose.rotation @ self.grasp_center_palm
        )

        direction_to_grasp_center = normalize(
            grasp_center_world - tip_pose.translation
        )

        dot = clamp(
            float(np.dot(sensor_normal_world, direction_to_grasp_center)),
            -1.0,
            1.0,
        )
        return math.acos(dot)

    def tactile_cone_residual(
        self, q: np.ndarray, cfg: FingerConfig
    ) -> float:
        """
        Zero inside the allowed cone.
        Grows only when sensor orientation becomes clearly unfavorable.
        """
        theta = self.tactile_angle_rad(q, cfg)
        theta_max = math.radians(cfg.tactile_cone_deg)
        return soft_hinge(theta - theta_max)

    def fingertip_position_and_normal(
        self,
        q: np.ndarray,
        cfg: FingerConfig,
    ):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        tip_id = self.model.getFrameId(cfg.frame_name)
        tip_pose = self.data.oMf[tip_id]

        p = tip_pose.translation.copy()

        n = normalize(
            tip_pose.rotation @ cfg.sensor_normal_local
        )

        return p, n


    def thumb_index_pinch_residuals(
        self,
        q: np.ndarray,
        thumb_closure: float,
        index_closure: float,
    ):
        """
        Thumb-index tactile pinch geometry.

        The index tactile pad defines the reference axis.

        Returns:
            axis_res: 3D normalized lateral/off-axis displacement.
            facing_res: scalar orientation residual, zero for opposite normals.
            gap_res: scalar soft minimum-gap violation.
        """
        thumb_cfg = self.fingers["thumb"]
        index_cfg = self.fingers["index"]

        p_thumb, n_thumb = self.fingertip_position_and_normal(q, thumb_cfg)
        p_index, n_index = self.fingertip_position_and_normal(q, index_cfg)

        # 1) AXIS ALIGNMENT
        # v_perp is the component of thumb-index displacement perpendicular
        # to the index tactile normal. Zero means both pad centers lie on the
        # same tactile-normal axis.
        v = p_thumb - p_index
        v_parallel = float(np.dot(v, n_index)) * n_index
        v_perp = v - v_parallel
        axis_scale_m = 0.030  # normalization scale, not a desired gap
        axis_res = v_perp / axis_scale_m

        # 2) FACING NORMALS
        # Ideal: n_thumb = -n_index -> dot = -1 -> residual = 0.
        facing_res = 1.0 + float(np.dot(n_thumb, n_index))

        # 3) TARGET GAP DRIVEN BY HUMAN CLOSURE
        # Distance between the pad centers measured along the index normal.
        gap_m = abs(float(np.dot(v, n_index)))

        # Average human closure of thumb and index.
        c_pair = clamp(
            0.5 * (float(thumb_closure) + float(index_closure)),
            0.0,
            1.0,
        )

        # Desired separation between tactile pads:
        #   fully open pair   -> about 100 mm
        #   fully closed pair -> about 10 mm
        #
        # This is a generic geometry prior, not an object-size estimate.
        gap_open_m = 0.100
        gap_closed_m = 0.010

        target_gap_m = (
            gap_open_m * (1.0 - c_pair)
            + gap_closed_m * c_pair
        )

        # Signed normalized error:
        #   > 0 : pads are farther apart than desired
        #   < 0 : pads are closer than desired
        #   = 0 : desired gap reached
        gap_scale_m = 0.050
        gap_res = (gap_m - target_gap_m) / gap_scale_m

        return axis_res, facing_res, gap_res

    # -----------------------------------------------------------------------
    # Joint-limit safety
    # -----------------------------------------------------------------------

    def joint_margin_residuals(self, q: np.ndarray) -> np.ndarray:
        """
        Adds a soft cost when a joint is very close to a limit.

        The hard bounds are already enforced by scipy.
        This residual just discourages living exactly on the boundaries.
        """
        span = np.maximum(self.upper - self.lower, 1e-6)
        margin = 0.08 * span

        low_distance = q - self.lower
        high_distance = self.upper - q

        low_penalty = np.maximum(0.0, margin - low_distance) / margin
        high_penalty = np.maximum(0.0, margin - high_distance) / margin

        return low_penalty + high_penalty

    # -----------------------------------------------------------------------
    # Complete optimization
    # -----------------------------------------------------------------------

    def residual_vector(
        self,
        q: np.ndarray,
        samples: Dict[str, WeartSample],
        q_nom: np.ndarray,
    ) -> np.ndarray:

        residuals = []

        # ---------------------------------------------------------------
        # PRIMARY: WEART closure
        # ---------------------------------------------------------------
        for finger_name, sample in samples.items():
            cfg = self.fingers[finger_name]
            c_target = self.shape_closure(sample.closure)

            c_robot = self.robot_closure(q, cfg)

            residuals.append(
                self.w_closure * (c_robot - c_target)
            )

        # ---------------------------------------------------------------
        # PRIMARY-ish: WEART thumb adduction
        # ---------------------------------------------------------------
        thumb_sample = samples.get("thumb")
        thumb_cfg = self.fingers["thumb"]

        if thumb_sample is not None and thumb_cfg.adduction_joint_name is not None:
            idx = self.q_index(thumb_cfg.adduction_joint_name)

            desired = self.weart_thumb_adduction_to_leap(
                thumb_sample.adduction
            )

            scale = max(
                abs(thumb_cfg.q_adduct_max - thumb_cfg.q_adduct_min),
                1e-6,
            )
            residuals.append(
                self.w_adduction * ((q[idx] - desired) / scale)
            )

        # ---------------------------------------------------------------
        # SECONDARY: tactile-ready orientation
        # ---------------------------------------------------------------
        for finger_name, sample in samples.items():
            cfg = self.fingers[finger_name]
            tactile_gain = self.tactile_activation(sample.closure)
            residuals.append(
                tactile_gain
                * self.w_tactile
                * self.tactile_cone_residual(q, cfg)
            )
        # ---------------------------------------------------------------
        # SECONDARY: thumb-index tactile pinch geometry
        # ---------------------------------------------------------------
        thumb_sample = samples.get("thumb")
        index_sample = samples.get("index")

        if thumb_sample is not None and index_sample is not None:
            g_thumb = self.tactile_activation(thumb_sample.closure)
            g_index = self.tactile_activation(index_sample.closure)
            g_pair = math.sqrt(g_thumb * g_index)

            axis_res, facing_res, gap_res = self.thumb_index_pinch_residuals(
                q,
                thumb_sample.closure,
                index_sample.closure,
            )

            residuals.extend(
                (g_pair * self.w_axis_alignment * axis_res).tolist()
            )
            residuals.append(
                g_pair * self.w_facing_normals * facing_res
            )
            residuals.append(
                g_pair * self.w_gap * gap_res
            )

        # ---------------------------------------------------------------
        # SECONDARY: neutral lateral posture for unobserved DoFs
        # ---------------------------------------------------------------
        # WEART does not provide index/middle adduction in this setup.
        # Keep LEAP joints 0 and 4 near neutral so the tactile term cannot
        # move the fingers sideways when closure is small or zero.
        index_lat_idx = self.q_index("0")
        middle_lat_idx = self.q_index("4")

        index_lat_scale = max(
            self.upper[index_lat_idx] - self.lower[index_lat_idx], 1e-6
        )
        middle_lat_scale = max(
            self.upper[middle_lat_idx] - self.lower[middle_lat_idx], 1e-6
        )

        residuals.append(
            self.w_lateral
            * (q[index_lat_idx] - self.index_lateral_neutral)
            / index_lat_scale
        )
        residuals.append(
            self.w_lateral
            * (q[middle_lat_idx] - self.middle_lateral_neutral)
            / middle_lat_scale
        )

        # ---------------------------------------------------------------
        # SECONDARY: weak human-like / nominal posture regularizer
        #
        # This prevents closure alone from producing arbitrary joint
        # distributions, but is deliberately much weaker than closure.
        # ---------------------------------------------------------------
        span = np.maximum(self.upper - self.lower, 1e-6)
        posture_res = (q - q_nom) / span
        residuals.extend(
            (self.w_posture * posture_res).tolist()
        )

        # ---------------------------------------------------------------
        # SECONDARY: temporal smoothness
        # ---------------------------------------------------------------
        smooth_res = (q - self.q_prev) / span
        residuals.extend(
            (self.w_smooth * smooth_res).tolist()
        )

        # ---------------------------------------------------------------
        # SAFETY: stay away from hard joint limits
        # ---------------------------------------------------------------
        residuals.extend(
            (
                self.w_joint_margin
                * self.joint_margin_residuals(q)
            ).tolist()
        )

        return np.asarray(residuals, dtype=float)

    def solve(self, samples: Dict[str, WeartSample]):
        q_nom = self.nominal_q_from_weart(samples)

        result = least_squares(
            fun=self.residual_vector,
            x0=self.q_prev,
            bounds=(self.lower, self.upper),
            args=(samples, q_nom),
            method="trf",
            max_nfev=self.max_nfev,
            xtol=1e-4,
            ftol=1e-4,
            gtol=1e-4,
        )

        q_solution = np.asarray(result.x, dtype=float)
        self.q_prev = q_solution

        diagnostics = {}
        for finger_name in samples:
            cfg = self.fingers[finger_name]
            diagnostics[f"{finger_name}_closure_robot"] = (
                self.robot_closure(q_solution, cfg)
            )
            diagnostics[f"{finger_name}_tactile_angle_deg"] = math.degrees(
                self.tactile_angle_rad(q_solution, cfg)
            )
            diagnostics[f"{finger_name}_tactile_activation"] = (
                self.tactile_activation(samples[finger_name].closure)
            )

        p_thumb, n_thumb = self.fingertip_position_and_normal(
            q_solution, self.fingers["thumb"]
        )
        p_index, n_index = self.fingertip_position_and_normal(
            q_solution, self.fingers["index"]
        )
        # Use fingertip frame origins as pad positions (URDF units: metres).
        # Positive gap: thumb lies on the outward-normal side of the index.
        delta = p_thumb - p_index
        gap = float(np.dot(delta, n_index))
        diagnostics["axis_err_mm"] = 1000.0 * float(
            np.linalg.norm(delta - gap * n_index)
        )
        diagnostics["facing_err_deg"] = math.degrees(
            math.acos(clamp(float(np.dot(n_thumb, -n_index)), -1.0, 1.0))
        )
        diagnostics["gap_mm"] = 1000.0 * gap


        thumb_sample = samples["thumb"]
        index_sample = samples["index"]

        # ---------------------------------------------------------------
        # Pair gating
        # ---------------------------------------------------------------
        g_thumb = self.tactile_activation(thumb_sample.closure)
        g_index = self.tactile_activation(index_sample.closure)

        g_pair = math.sqrt(g_thumb * g_index)

        # Weighted pinch-geometry residual magnitudes used by the solver.
        axis_res, facing_res, gap_res = self.thumb_index_pinch_residuals(
            q_solution,
            thumb_sample.closure,
            index_sample.closure,
        )
        diagnostics["axis_cost"] = (
            g_pair * self.w_axis_alignment * float(np.linalg.norm(axis_res))
        )
        diagnostics["facing_cost"] = (
            g_pair * self.w_facing_normals * abs(float(facing_res))
        )
        diagnostics["gap_cost"] = (
            g_pair * self.w_gap * abs(float(gap_res))
        )

        for finger_name in ("thumb", "index", "middle"):
            cfg = self.fingers[finger_name]
            sample = samples[finger_name]

            c_target = self.shape_closure(sample.closure)
            c_robot = self.robot_closure(q_solution, cfg)

            diagnostics[f"{finger_name}_closure_target"] = c_target

            diagnostics[f"{finger_name}_closure_cost"] = (
                self.w_closure
                * abs(c_robot - c_target)
            )

        thumb_cfg = self.fingers["thumb"]
        idx = self.q_index(thumb_cfg.adduction_joint_name)

        desired_add = self.weart_thumb_adduction_to_leap(
            thumb_sample.adduction
        )

        add_scale = max(
            abs(
                thumb_cfg.q_adduct_max
                - thumb_cfg.q_adduct_min
            ),
            1e-6,
        )

        diagnostics["thumb_adduction_cost"] = (
            self.w_adduction
            * abs(
                (q_solution[idx] - desired_add)
                / add_scale
            )
        )
        return q_solution, diagnostics



class DirectLeapFullHandDriver:
    """
    Small full-hand LEAP driver.

    Input/output joint coordinates are the same 16 URDF/simulation coordinates
    published to /leap/target_joint_positions.

    Before sending to the physical hand, the vector is converted with
    leap_sim_to_motor(), exactly like in the user's previous LEAP controller.
    """

    def __init__(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        velocity_limits: np.ndarray,
        *,
        enabled: bool,
        port: str,
        baudrate: int,
        kp: float,
        ki: float,
        kd: float,
    ):
        self.lower = np.asarray(lower, dtype=float).copy()
        self.upper = np.asarray(upper, dtype=float).copy()
        self.velocity_limits = np.asarray(velocity_limits, dtype=float).copy()

        self.enabled = bool(enabled)
        self.requested_port = str(port)
        self.baudrate = int(baudrate)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

        self.current_sim_qpos = np.zeros(16, dtype=float)
        self._ctrl = None
        self.port = None

    def connect(self):
        if not self.enabled:
            print("[LEAP] hardware output DISABLED (dry-run).", flush=True)
            return

        if not HARDWARE_DRIVER_AVAILABLE:
            raise RuntimeError(
                "LEAP hardware helpers could not be imported from "
                "weart_leap_index_ik.py"
            )

        self.port = autodetect_leap_port(self.requested_port)
        leap_cpp = _load_leap_cpp()
        self._ctrl = leap_cpp.LeapController(self.port)
        self._ctrl.connect()
        self._ctrl.setGains(int(self.kp), int(self.ki), int(self.kd))

        # Start from the current neutral/open command.
        self._ctrl.set_leap(leap_sim_to_motor(self.current_sim_qpos))
        print(
            f"[LEAP] connected on {self.port} at {self.baudrate} baud; "
            "sent initial open pose.",
            flush=True,
        )

    def disconnect(self):
        if self._ctrl is not None:
            self._ctrl.disconnect()
            self._ctrl = None
            print("[LEAP] disconnected.", flush=True)

    def command(self, q_target: np.ndarray, dt: float) -> np.ndarray:
        """
        Send a complete 16-joint target.

        A low-pass + velocity limiter is applied before commanding hardware.
        This is intentionally the same safety pattern used by the older code.
        """
        target = np.asarray(q_target, dtype=float).reshape(-1)
        if target.size != 16:
            raise ValueError(
                f"Expected 16 LEAP joint targets, got {target.size}"
            )
        if not np.all(np.isfinite(target)):
            raise ValueError("Non-finite LEAP target received")

        target = np.clip(target, self.lower, self.upper)

        low_pass = (
            self.current_sim_qpos
            + COMMAND_LOW_PASS_ALPHA * (target - self.current_sim_qpos)
        )

        safe_velocity = np.minimum(
            np.asarray(self.velocity_limits, dtype=float),
            float(COMMAND_MAX_SPEED_RAD_S),
        )
        safe_velocity[~np.isfinite(safe_velocity)] = float(
            COMMAND_MAX_SPEED_RAD_S
        )

        max_step = safe_velocity * max(float(dt), float(MIN_DT_S))
        filtered = self.current_sim_qpos + np.clip(
            low_pass - self.current_sim_qpos,
            -max_step,
            max_step,
        )

        self.current_sim_qpos = filtered

        if self.enabled:
            if self._ctrl is None:
                raise RuntimeError(
                    "LEAP output enabled but controller is not connected"
                )

            motor_qpos = leap_sim_to_motor(self.current_sim_qpos)
            self._ctrl.set_leap(motor_qpos)

        return self.current_sim_qpos.copy()

    def read_actual_sim_qpos(self):
        """Read physical motor positions and convert them back to URDF q."""
        if self._ctrl is None:
            return None

        motor_qpos = np.asarray(self._ctrl.read_pos(), dtype=float).reshape(-1)
        if motor_qpos.size != 16:
            raise ValueError(
                f"LEAP returned {motor_qpos.size} joints, expected 16"
            )
        if not np.all(np.isfinite(motor_qpos)):
            raise ValueError("LEAP returned non-finite joint positions")

        # Same inverse convention used by the user's previous driver.
        return motor_qpos - math.pi


# ---------------------------------------------------------------------------
# ROS 2 node
# ---------------------------------------------------------------------------

class WeartLeapRetargetingNode(Node):
    def __init__(self):
        super().__init__("weart_leap_tactile_retargeter")

        self.declare_parameter("urdf_path", "")
        self.declare_parameter("control_rate_hz", 30.0)
        self.declare_parameter("enable_hardware", False)
        self.declare_parameter("leap_port", DEFAULT_LEAP_PORT)
        self.declare_parameter("leap_baudrate", DEFAULT_LEAP_BAUDRATE)
        self.declare_parameter("kp", DEFAULT_KP)
        self.declare_parameter("ki", DEFAULT_KI)
        self.declare_parameter("kd", DEFAULT_KD)
        self.declare_parameter(
            "actual_feedback_period_s", 0.05
        )

        urdf_path = (
            self.get_parameter("urdf_path")
            .get_parameter_value()
            .string_value
        )
        if not urdf_path:
            raise RuntimeError(
                "Set ROS parameter 'urdf_path' to the LEAP URDF file."
            )

        rate_hz = (
            self.get_parameter("control_rate_hz")
            .get_parameter_value()
            .double_value
        )

        self.retargeter = LeapRetargeter(urdf_path)

        self.enable_hardware = bool(self.get_parameter("enable_hardware").value)

        # IMPORTANT:
        # Pinocchio's internal q order is NOT guaranteed to be numeric joint
        # order 0..15. The physical LEAP API / leap_sim_to_motor expects the
        # vector in LEAP numeric joint order. Build hardware-side arrays
        # explicitly in that order.
        hardware_lower = np.array(
            [
                self.retargeter.lower[self.retargeter.q_index(str(i))]
                for i in range(16)
            ],
            dtype=float,
        )
        hardware_upper = np.array(
            [
                self.retargeter.upper[self.retargeter.q_index(str(i))]
                for i in range(16)
            ],
            dtype=float,
        )

        velocity_limits = np.zeros(16, dtype=float)
        for i in range(16):
            joint_name = str(i)
            joint_id = self.retargeter.model.getJointId(joint_name)
            joint = self.retargeter.model.joints[joint_id]
            velocity_limits[i] = float(
                self.retargeter.model.velocityLimit[joint.idx_v]
            )

        self.leap = DirectLeapFullHandDriver(
            hardware_lower,
            hardware_upper,
            velocity_limits,
            enabled=self.enable_hardware,
            port=str(self.get_parameter("leap_port").value),
            baudrate=int(self.get_parameter("leap_baudrate").value),
            kp=float(self.get_parameter("kp").value),
            ki=float(self.get_parameter("ki").value),
            kd=float(self.get_parameter("kd").value),
        )
        self.leap.connect()

        self.last_control_time_s = (
            self.get_clock().now().nanoseconds * 1e-9
        )

        self.samples: Dict[str, WeartSample] = {}
        self.last_rx_time: Dict[str, float] = {}

        self.create_subscription(
            String,
            "/weart/thumb/raw",
            lambda msg: self.weart_callback("thumb", msg),
            20,
        )
        self.create_subscription(
            String,
            "/weart/index/raw",
            lambda msg: self.weart_callback("index", msg),
            20,
        )
        self.create_subscription(
            String,
            "/weart/middle/raw",
            lambda msg: self.weart_callback("middle", msg),
            20,
        )

        self.joint_pub = self.create_publisher(
            Float64MultiArray,
            "/leap/target_joint_positions",
            20,
        )

        self.actual_joint_pub = self.create_publisher(
            Float64MultiArray,
            "/leap/actual_joint_positions",
            20,
        )

        self.actual_timer = None
        if self.enable_hardware:
            feedback_period_s = float(
                self.get_parameter("actual_feedback_period_s").value
            )
            self.actual_timer = self.create_timer(
                feedback_period_s,
                self.publish_actual_pose,
            )

        self.timer = self.create_timer(
            1.0 / max(rate_hz, 1.0),
            self.control_step,
        )

        self.diag_counter = 0

        self.get_logger().info(
            "WEART -> LEAP tactile-ready retargeter started."
        )
        if self.enable_hardware:
            self.get_logger().warn("LEAP HARDWARE OUTPUT ENABLED.")
            self.get_logger().info(
                "Hardware command order: explicit LEAP numeric joints [0..15]."
            )
        else:
            self.get_logger().warn(
                "Dry-run mode: publishing targets for Rerun/ROS only; "
                "no motor commands are sent."
            )
        self.get_logger().warn(
            "Before commanding hardware, verify joint directions, "
            "q_open/q_closed values, thumb adduction direction, and the "
            "local tactile sensor normal in RViz."
        )

    def publish_actual_pose(self):
        try:
            actual_q = self.leap.read_actual_sim_qpos()
            if actual_q is None:
                return

            msg = Float64MultiArray()
            msg.data = [float(v) for v in actual_q]
            self.actual_joint_pub.publish(msg)
        except Exception as exc:
            self.get_logger().warning(
                f"LEAP actual feedback failed: {exc}",
                throttle_duration_sec=2.0,
            )

    def destroy_node(self):
        try:
            if self.leap is not None:
                try:
                    self.leap.disconnect()
                except Exception:
                    pass
        finally:
            return super().destroy_node()

    def weart_callback(self, finger_name: str, msg: String):
        self.samples[finger_name] = parse_weart_string(msg.data)
        self.last_rx_time[finger_name] = self.get_clock().now().nanoseconds * 1e-9

    def data_is_fresh(self, finger_name: str, max_age_s: float = 0.25) -> bool:
        if finger_name not in self.last_rx_time:
            return False

        now_s = self.get_clock().now().nanoseconds * 1e-9
        return (now_s - self.last_rx_time[finger_name]) <= max_age_s

    def control_step(self):
        required = ("thumb", "index") if ASSUME_MIDDLE_OPEN else ("thumb", "index", "middle")

        if not all(name in self.samples for name in required):
            return

        if not all(self.data_is_fresh(name) for name in required):
            self.get_logger().warn(
                "WEART data stale; not publishing a new LEAP target.",
                throttle_duration_sec=2.0,
            )
            return

        active_samples = {
            name: self.samples[name]
            for name in required
        }
        if ASSUME_MIDDLE_OPEN:
            active_samples["middle"] = WeartSample(closure=0.0)

        try:
            q, diag = self.retargeter.solve(active_samples)
        except Exception as exc:
            self.get_logger().error(
                f"Retargeting optimization failed: {exc}",
                throttle_duration_sec=1.0,
            )
            return

        # Rerun viewer expects the complete LEAP 16-joint vector as a
        # Float64MultiArray on /leap/target_joint_positions.
        msg = Float64MultiArray()
        msg.data = [
            float(q[self.retargeter.q_index(str(i))])
            for i in range(16)
        ]
        self.joint_pub.publish(msg)

        # Optional direct command to the physical LEAP Hand.
        # ROS publication stays active in both modes so Rerun always sees the
        # same target pose.
        if self.enable_hardware:
            try:
                now_s = self.get_clock().now().nanoseconds * 1e-9
                dt = max(now_s - self.last_control_time_s, MIN_DT_S)
                self.last_control_time_s = now_s

                # q is in Pinocchio internal order. msg.data was already
                # reordered explicitly to LEAP joints [0, 1, ..., 15].
                q_leap_numeric = np.asarray(msg.data, dtype=float)
                q_commanded = self.leap.command(q_leap_numeric, dt)
            except Exception as exc:
                self.get_logger().error(
                    f"LEAP hardware command failed: {exc}",
                    throttle_duration_sec=1.0,
                )
                return

        self.diag_counter += 1
        if self.diag_counter % 30 == 0:
            self.get_logger().info(
                " | ".join(
                    [
                        f"thumb: c={diag['thumb_closure_robot']:.2f}, "
                        f"tact={diag['thumb_tactile_angle_deg']:.1f}deg, "
                        f"gate={diag['thumb_tactile_activation']:.2f}",
                        f"index: c={diag['index_closure_robot']:.2f}, "
                        f"tact={diag['index_tactile_angle_deg']:.1f}deg, "
                        f"gate={diag['index_tactile_activation']:.2f}",
                        f"middle: c={diag['middle_closure_robot']:.2f}, "
                        f"tact={diag['middle_tactile_angle_deg']:.1f}deg, "
                        f"gate={diag['middle_tactile_activation']:.2f}",
                        f"thumb-index: axis={diag['axis_err_mm']:.1f} mm | "
                        f"facing={diag['facing_err_deg']:.1f} deg | "
                        f"gap={diag['gap_mm']:.1f} mm",
                    ]
                )
            )
            self.get_logger().info(
                 "COSTS | "
                 f"axis={diag['axis_cost']:.3f} | "
                 f"facing={diag['facing_cost']:.3f} | "
                 f"gap={diag['gap_cost']:.3f} | "
                 f"cl_thumb={diag['thumb_closure_cost']:.3f} | "
                 f"cl_index={diag['index_closure_cost']:.3f} | "
                 f"add_thumb={diag['thumb_adduction_cost']:.3f}"
             )


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = WeartLeapRetargetingNode()
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
