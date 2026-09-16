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
import time
import re
from dataclasses import dataclass
from typing import Dict, Optional
from collections import deque

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

        # Determine the natural index-middle lateral ordering from the neutral pose.
        # We measure the relative fingertip displacement in the PALM frame and
        # use its dominant lateral component sign as the reference ordering.
        pin.forwardKinematics(self.model, self.data, self.q_prev)
        pin.updateFramePlacements(self.model, self.data)

        palm_pose_neutral = self.data.oMf[self.palm_frame_id]
        p_index_neutral = self.data.oMf[
            self.model.getFrameId(self.fingers["index"].frame_name)
        ].translation
        p_middle_neutral = self.data.oMf[
            self.model.getFrameId(self.fingers["middle"].frame_name)
        ].translation

        delta_im_world = p_middle_neutral - p_index_neutral
        delta_im_palm = palm_pose_neutral.rotation.T @ delta_im_world

        # For the LEAP hand, the palm-frame Y axis is used as the lateral axis.
        # The sign is inferred automatically from the neutral pose.
        self.index_middle_lateral_axis_palm = np.array([0.0, 1.0, 0.0])
        neutral_signed_sep = float(
            np.dot(delta_im_palm, self.index_middle_lateral_axis_palm)
        )
        self.index_middle_order_sign = 1.0 if neutral_signed_sep >= 0.0 else -1.0

        # Cache joint bounds from URDF.
        self.lower = np.asarray(self.model.lowerPositionLimit, dtype=float).copy()
        self.upper = np.asarray(self.model.upperPositionLimit, dtype=float).copy()

        # Some Pinocchio models can contain "infinite" bounds.
        self.lower[~np.isfinite(self.lower)] = -math.pi
        self.upper[~np.isfinite(self.upper)] = math.pi

        # TASK MODE: OPEN A JAR
        # Define these BEFORE computing the task-specific q0/q4 targets.
        self.w_jar_wrap_index = 2.5
        self.w_jar_wrap_middle = 2.5
        self.jar_wrap_fraction_index = 0.55
        self.jar_wrap_fraction_middle = 0.55
        self.jar_task_enabled = False

        # --------------------------------------------------------------
        # AUTOMATIC UNSCREW CYCLE
        # --------------------------------------------------------------
        # Normal retargeting is used until a sufficiently closed three-finger
        # grasp is detected. Then the grasp pose is captured and the controller
        # takes over:
        #
        #   1) SWEEP:    freeze every joint except q0/q4 and move both negative.
        #   2) RELEASE:  open the flexion joints slightly while q0/q4 reset.
        #   3) RECLOSE:  return flexion joints to the captured grasp.
        #   4) repeat SWEEP -> RELEASE -> RECLOSE.
        #
        # The cycle is aborted ONLY when the human hand is essentially fully open.
        self.unscrew_cycle_enabled = True

        # Entry / exit thresholds from raw WEART closure.
        # Entry is intentionally permissive ("large tolerance").
        self.unscrew_lock_mean_closure = 0.50
        self.unscrew_lock_min_finger_closure = 0.20
        self.unscrew_fully_open_threshold = 0.08

        # q0/q4 final target. "At least -0.5" means that if the captured joint
        # is already more negative than -0.5, we do not move it back positive.
        self.unscrew_q0_target = -0.50
        self.unscrew_q4_target = -0.50
        self.unscrew_q12_target = 2.00

        # Timing [s].
        self.unscrew_sweep_time_s = 1.00
        self.unscrew_release_time_s = 0.45
        self.unscrew_reclose_time_s = 0.45
        self.unscrew_hold_time_s = 0.10

        # How much of the captured FLEXION is released between sweeps.
        # 0.15 means ~15% toward the open posture.
        self.unscrew_release_fraction = 0.5

        # State machine.
        self.unscrew_state = "WAIT_GRASP"
        self.unscrew_phase_start_s = time.monotonic()
        self.unscrew_grasp_q = None
        self.unscrew_reset_q0 = None
        self.unscrew_reset_q4 = None
        self.unscrew_reset_q12 = None
        self.unscrew_cycle_count = 0

        # --------------------------------------------------------------
        # TWIST MODE: keep the three-finger pinch, move only adduction DoFs
        # --------------------------------------------------------------
        # At 30 Hz, 15 samples correspond to about 0.5 s.
        self.twist_history_len = 15
        self.twist_closure_tol = 0.20
        self.twist_min_mean_closure = 0.25
        self.twist_tof_trigger_mm = 5.0

        # ToF -> coordinated adduction sweep tuning.
        # IMPORTANT:
        # During JAR-TWIST, thumb/index/middle no longer use three independent
        # ToF commands. A SINGLE shared ToF progress drives q0, q4 and q12 so
        # that all three fingertips sweep in the SAME palm-frame direction.
        self.twist_tof_scale_mm = 35.0

        # Reject implausible ToF samples/spikes.
        self.twist_tof_min_valid_mm = 5.0
        self.twist_tof_max_valid_mm = 150.0
        self.twist_tof_max_jump_mm = 40.0

        # Low-pass filtering coefficient for ToF.
        # 0 -> frozen, 1 -> no filtering.
        self.twist_tof_filter_alpha = 0.25

        # Small deadband around zero delta-ToF to suppress jitter.
        self.twist_tof_deadband_mm = 2.0

        # Coordinated lateral sweep direction in the PALM frame.
        # +1 -> palm +Y, -1 -> palm -Y.
        # Flip this ONE value if the hand moves in the tightening direction.
        self.twist_sweep_direction = +1.0

        # Per-finger sign corrections.
        # From the real-hand test, INDEX was moving in the opposite physical
        # direction and crossing over MIDDLE, so flip INDEX only.
        self.twist_index_sign_multiplier = -1.0
        self.twist_middle_sign_multiplier = +1.0
        self.twist_thumb_sign_multiplier = +1.0

        # Maximum fraction of each joint range used by the shared twist sweep.
        self.twist_adduction_fraction_index = 0.18
        self.twist_adduction_fraction_middle = 0.18
        self.twist_adduction_fraction_thumb = 0.10

        # State.
        self.twist_active = False
        self.twist_hold_q = None
        self.twist_reference_tof = {}
        self.twist_filtered_tof = {}
        self.twist_last_raw_tof = {}
        self.closure_history = {
            "thumb": deque(maxlen=self.twist_history_len),
            "index": deque(maxlen=self.twist_history_len),
            "middle": deque(maxlen=self.twist_history_len),
        }
        self.tof_history = {
            "thumb": deque(maxlen=self.twist_history_len),
            "index": deque(maxlen=self.twist_history_len),
            "middle": deque(maxlen=self.twist_history_len),
        }

        q_neutral = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, q_neutral)
        pin.updateFramePlacements(self.model, self.data)
        palm_pose = self.data.oMf[self.palm_frame_id]

        def inward_axis_palm_for(finger_name: str):
            tip = self.data.oMf[
                self.model.getFrameId(self.fingers[finger_name].frame_name)
            ].translation
            tip_palm = palm_pose.rotation.T @ (tip - palm_pose.translation)
            d = self.grasp_center_palm - tip_palm
            d[2] = 0.0
            if np.linalg.norm(d) < 1e-8:
                d = np.array([0.0, 1.0, 0.0])
            return normalize(d)

        self.jar_index_inward_axis_palm = inward_axis_palm_for("index")
        self.jar_middle_inward_axis_palm = inward_axis_palm_for("middle")

        self.jar_index_joint_sign = self.infer_joint_sign_for_tip_motion(
            "0", "index", self.jar_index_inward_axis_palm
        )
        self.jar_middle_joint_sign = self.infer_joint_sign_for_tip_motion(
            "4", "middle", self.jar_middle_inward_axis_palm
        )

        # For the actual unscrewing sweep, all three fingertips must move in
        # the SAME palm-frame lateral direction. Because q0, q4 and q12 can have
        # different joint sign conventions, infer the correct motor/joint sign
        # separately for each digit but against the SAME desired palm direction.
        self.twist_sweep_axis_palm = np.array(
            [0.0, float(self.twist_sweep_direction), 0.0],
            dtype=float,
        )

        self.twist_index_joint_sign = self.infer_joint_sign_for_tip_motion(
            "0", "index", self.twist_sweep_axis_palm
        )
        self.twist_middle_joint_sign = self.infer_joint_sign_for_tip_motion(
            "4", "middle", self.twist_sweep_axis_palm
        )
        self.twist_thumb_joint_sign = self.infer_joint_sign_for_tip_motion(
            "12", "thumb", self.twist_sweep_axis_palm
        )

        # Apply empirical sign correction from the physical LEAP test.
        self.twist_index_joint_sign *= self.twist_index_sign_multiplier
        self.twist_middle_joint_sign *= self.twist_middle_sign_multiplier
        self.twist_thumb_joint_sign *= self.twist_thumb_sign_multiplier

        # Keep the old inward signs/targets available for the generic jar-wrap
        # setup, but the JAR-TWIST controller below uses the coordinated sweep
        # signs above.
        self.jar_thumb_inward_axis_palm = inward_axis_palm_for("thumb")
        self.jar_thumb_joint_sign = self.infer_joint_sign_for_tip_motion(
            "12", "thumb", self.jar_thumb_inward_axis_palm
        )

        self.jar_index_target = self.target_from_signed_fraction(
            "0", self.jar_index_joint_sign, self.jar_wrap_fraction_index
        )
        self.jar_middle_target = self.target_from_signed_fraction(
            "4", self.jar_middle_joint_sign, self.jar_wrap_fraction_middle
        )

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
        # Index/middle lateral DoFs (q0, q4) are not directly observed by WEART.
        # Keep them near neutral when the fingers are open, but progressively
        # release them during grasp so robot-specific tactile geometry can use them.
        self.w_lateral_open = 1.0
        self.w_lateral_grasp = 0.20

        # Tri-digit tactile grasp geometry.
        # Thumb-index is the primary pair; thumb-middle is a softer stabilizing pair.
        self.w_axis_TI = 0.15
        self.w_face_TI = 0.30
        self.w_gap_TI = 0.20

        self.w_axis_TM = 0.10
        self.w_face_TM = 0.25
        self.w_gap_TM = 0.15

        # Prevent index and middle from collapsing onto each other.
        self.w_index_middle_spread = 0.50

        # Preserve the anatomical lateral ordering of index and middle.
        # q0/q4 remain free to move, but the fingertips should not cross/swap sides.
        self.w_index_middle_order = 2.0
        self.index_middle_order_margin_m = 0.010  # 10 mm signed lateral margin
        self.index_middle_order_scale_m = 0.020   # normalization scale

        # Axis alignment is a soft envelope, not a "perfect collinearity" target.
        self.axis_tolerance_TI_m = 0.020   # 20 mm
        self.axis_tolerance_TM_m = 0.030   # 30 mm
        self.axis_scale_m = 0.030          # normalization scale

        # Soft minimum index-middle fingertip spacing.
        self.index_middle_min_spread_m = 0.025  # 25 mm
        self.index_middle_spread_scale_m = 0.025

        self.index_lateral_neutral = 0.0
        self.middle_lateral_neutral = 0.0

        # Do not let optimizer take huge frame-to-frame jumps.
        self.max_nfev = 25

    def q_index(self, joint_name: str) -> int:
        if joint_name not in self.joint_q_index:
            raise KeyError(f"Joint '{joint_name}' not found in Pinocchio model.")
        return self.joint_q_index[joint_name]


    def _finger_tip_position_world(self, q: np.ndarray, finger_name: str) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        cfg = self.fingers[finger_name]
        return self.data.oMf[self.model.getFrameId(cfg.frame_name)].translation.copy()

    def _palm_axis_world(self, q: np.ndarray, axis_palm: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        palm_pose = self.data.oMf[self.palm_frame_id]
        return normalize(palm_pose.rotation @ np.asarray(axis_palm, dtype=float))

    def infer_joint_sign_for_tip_motion(
        self,
        joint_name: str,
        finger_name: str,
        desired_axis_palm: np.ndarray,
        eps: float = 0.05,
    ) -> float:
        q0 = pin.neutral(self.model)
        idx = self.q_index(joint_name)

        q_plus = q0.copy()
        q_minus = q0.copy()
        q_plus[idx] = np.clip(q_plus[idx] + eps, self.lower[idx], self.upper[idx])
        q_minus[idx] = np.clip(q_minus[idx] - eps, self.lower[idx], self.upper[idx])

        axis_world = self._palm_axis_world(q0, desired_axis_palm)
        p_plus = self._finger_tip_position_world(q_plus, finger_name)
        p_minus = self._finger_tip_position_world(q_minus, finger_name)

        score_plus = float(np.dot(p_plus, axis_world))
        score_minus = float(np.dot(p_minus, axis_world))
        return 1.0 if score_plus >= score_minus else -1.0

    def target_from_signed_fraction(
        self,
        joint_name: str,
        sign: float,
        fraction: float,
    ) -> float:
        idx = self.q_index(joint_name)
        q0 = float(pin.neutral(self.model)[idx])
        fraction = clamp(float(fraction), 0.0, 1.0)

        if sign >= 0.0:
            return q0 + fraction * (self.upper[idx] - q0)
        return q0 - fraction * (q0 - self.lower[idx])

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


    def finger_pair_grasp_residuals(
        self,
        q: np.ndarray,
        reference_name: str,
        other_name: str,
        reference_closure: float,
        other_closure: float,
        axis_tolerance_m: float,
    ):
        """
        Generic tactile pair geometry.

        The reference finger defines the tactile-normal axis.

        Returns:
            axis_res:
                scalar soft off-axis residual. Zero inside a tolerance tube.
            facing_res:
                scalar orientation residual. Zero when pad normals are opposite.
            gap_res:
                scalar target-gap residual driven by the two human closures.
            raw_axis_m:
                raw off-axis error in metres.
            raw_gap_m:
                raw separation along the reference tactile normal in metres.
        """
        ref_cfg = self.fingers[reference_name]
        other_cfg = self.fingers[other_name]

        p_ref, n_ref = self.fingertip_position_and_normal(q, ref_cfg)
        p_other, n_other = self.fingertip_position_and_normal(q, other_cfg)

        # ---------------------------------------------------------------
        # 1) SOFT AXIS ALIGNMENT WITH DEAD-ZONE
        # ---------------------------------------------------------------
        v = p_other - p_ref

        v_parallel = float(np.dot(v, n_ref)) * n_ref
        v_perp = v - v_parallel
        raw_axis_m = float(np.linalg.norm(v_perp))

        # No penalty while the other fingertip lies inside a tolerance tube
        # around the reference tactile-normal axis.
        axis_res = soft_hinge(
            raw_axis_m - float(axis_tolerance_m)
        ) / self.axis_scale_m

        # ---------------------------------------------------------------
        # 2) FACING NORMALS
        # ---------------------------------------------------------------
        # Ideal: n_other = -n_ref -> dot = -1 -> residual = 0.
        facing_res = 1.0 + float(np.dot(n_other, n_ref))

        # ---------------------------------------------------------------
        # 3) CLOSURE-DRIVEN TARGET GAP
        # ---------------------------------------------------------------
        raw_gap_m = abs(float(np.dot(v, n_ref)))

        c_pair = clamp(
            0.5 * (float(reference_closure) + float(other_closure)),
            0.0,
            1.0,
        )

        gap_open_m = 0.100
        gap_closed_m = 0.010

        target_gap_m = (
            gap_open_m * (1.0 - c_pair)
            + gap_closed_m * c_pair
        )

        gap_scale_m = 0.050
        gap_res = (raw_gap_m - target_gap_m) / gap_scale_m

        return axis_res, facing_res, gap_res, raw_axis_m, raw_gap_m


    def index_middle_spread_residual(
        self,
        q: np.ndarray,
    ):
        """
        Soft anti-collapse residual for index and middle fingertip centers.

        Zero while their Euclidean separation is above the configured minimum.
        """
        p_index, _ = self.fingertip_position_and_normal(
            q, self.fingers["index"]
        )
        p_middle, _ = self.fingertip_position_and_normal(
            q, self.fingers["middle"]
        )

        distance_m = float(np.linalg.norm(p_index - p_middle))

        spread_res = soft_hinge(
            self.index_middle_min_spread_m - distance_m
        ) / self.index_middle_spread_scale_m

        return spread_res, distance_m

    def index_middle_order_residual(
        self,
        q: np.ndarray,
    ):
        """
        Preserve index-middle lateral ordering in the palm frame.

        The neutral pose determines which finger lies on which lateral side.
        The residual is zero while the signed lateral separation stays above
        a configurable margin. It grows only when the fingers approach crossing
        or actually swap order.
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        palm_pose = self.data.oMf[self.palm_frame_id]

        p_index = self.data.oMf[
            self.model.getFrameId(self.fingers["index"].frame_name)
        ].translation
        p_middle = self.data.oMf[
            self.model.getFrameId(self.fingers["middle"].frame_name)
        ].translation

        delta_world = p_middle - p_index
        delta_palm = palm_pose.rotation.T @ delta_world

        signed_sep_m = (
            self.index_middle_order_sign
            * float(np.dot(delta_palm, self.index_middle_lateral_axis_palm))
        )

        order_res = soft_hinge(
            self.index_middle_order_margin_m - signed_sep_m
        ) / self.index_middle_order_scale_m

        return order_res, signed_sep_m


    def _unscrew_set_state(self, state: str):
        self.unscrew_state = str(state)
        self.unscrew_phase_start_s = time.monotonic()


    @staticmethod
    def _smoothstep01(x: float) -> float:
        x = clamp(float(x), 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)


    def _unscrew_phase_alpha(self, duration_s: float) -> float:
        duration_s = max(float(duration_s), 1e-6)
        elapsed = time.monotonic() - self.unscrew_phase_start_s
        return self._smoothstep01(elapsed / duration_s)


    def _unscrew_is_fully_open(self, samples: Dict[str, WeartSample]) -> bool:
        vals = [
            float(samples[name].closure)
            for name in ("thumb", "index", "middle")
            if name in samples
        ]
        return (
            len(vals) == 3
            and max(vals) <= self.unscrew_fully_open_threshold
        )


    def _unscrew_grasp_reached(self, samples: Dict[str, WeartSample]) -> bool:
        vals = np.asarray(
            [
                float(samples[name].closure)
                for name in ("thumb", "index", "middle")
            ],
            dtype=float,
        )
        return (
            float(np.mean(vals)) >= self.unscrew_lock_mean_closure
            and float(np.min(vals)) >= self.unscrew_lock_min_finger_closure
        )


    def _unscrew_open_flexion_pose(self) -> np.ndarray:
        """
        Build the small-release pose from the captured grasp.

        ONLY flexion joints are opened slightly. q0/q4 are handled separately.
        Thumb q12 and every other non-flexion joint stay frozen.
        """
        q_release = self.unscrew_grasp_q.copy()
        f = clamp(self.unscrew_release_fraction, 0.0, 1.0)

        for finger_name in ("thumb", "index", "middle"):
            cfg = self.fingers[finger_name]
            for j_name, q_open in zip(cfg.flexion_joint_names, cfg.q_open):
                idx = self.q_index(j_name)
                q_grasp = float(self.unscrew_grasp_q[idx])
                q_release[idx] = q_grasp + f * (float(q_open) - q_grasp)

        return np.clip(q_release, self.lower, self.upper)


    def solve_unscrew_cycle(self, samples: Dict[str, WeartSample]):
        """
        Automatic ratcheting-style jar unscrewing controller.

        Returns:
            None
                while WAIT_GRASP and normal retargeting should run.

            (q, diagnostics)
                while the automatic cycle owns the hand.

        Once the grasp is captured, all motion is deterministic:
          SWEEP:
              q0 and q4 move simultaneously toward <= -0.5.
              Every other joint is frozen.

          RELEASE:
              q0/q4 return to their captured reset values while the flexion
              chains open slightly.

          RECLOSE:
              flexion returns to the captured grasp, q0/q4 remain reset.

          Then another negative q0/q4 sweep begins.

        Opening the HUMAN glove completely aborts the automatic cycle and
        restores normal retargeting.
        """
        if not self.unscrew_cycle_enabled:
            return None

        # Human fully opens -> release the state machine immediately.
        if self.unscrew_state != "WAIT_GRASP" and self._unscrew_is_fully_open(samples):
            self.unscrew_grasp_q = None
            self.unscrew_reset_q0 = None
            self.unscrew_reset_q4 = None
            self.unscrew_reset_q12 = None
            self.unscrew_cycle_count = 0
            self._unscrew_set_state("WAIT_GRASP")
            return None

        # Normal retargeting until a stable-enough closed grasp is reached.
        if self.unscrew_state == "WAIT_GRASP":
            if not self._unscrew_grasp_reached(samples):
                return None

            self.unscrew_grasp_q = self.q_prev.copy()

            idx0 = self.q_index("0")
            idx4 = self.q_index("4")
            idx12 = self.q_index("12")

            self.unscrew_reset_q0 = float(self.unscrew_grasp_q[idx0])
            self.unscrew_reset_q4 = float(self.unscrew_grasp_q[idx4])
            self.unscrew_reset_q12 = float(self.unscrew_grasp_q[idx12])

            self.unscrew_cycle_count = 1
            self._unscrew_set_state("SWEEP")

        idx0 = self.q_index("0")
        idx4 = self.q_index("4")
        idx12 = self.q_index("12")

        # The targets are "at least -0.5":
        # do not move a joint positive if it was already below -0.5.
        q0_target = max(
            float(self.lower[idx0]),
            min(float(self.unscrew_q0_target), float(self.unscrew_reset_q0)),
        )
        q4_target = max(
            float(self.lower[idx4]),
            min(float(self.unscrew_q4_target), float(self.unscrew_reset_q4)),
        )
        q12_target = min(
            float(self.upper[idx12]),
            max(float(self.unscrew_q12_target), float(self.unscrew_reset_q12)),
        )

        now_elapsed = time.monotonic() - self.unscrew_phase_start_s

        if self.unscrew_state == "SWEEP":
            a = self._unscrew_phase_alpha(self.unscrew_sweep_time_s)

            # Freeze the captured grasp. ONLY q0 and q4 move.
            q = self.unscrew_grasp_q.copy()

            q[idx0] = (
                float(self.unscrew_reset_q0)
                + a * (q0_target - float(self.unscrew_reset_q0))
            )
            q[idx4] = (
                float(self.unscrew_reset_q4)
                + a * (q4_target - float(self.unscrew_reset_q4))
            )
            q[idx12] = (
                float(self.unscrew_reset_q12)
                + a * (q12_target - float(self.unscrew_reset_q12))
            )

            if now_elapsed >= self.unscrew_sweep_time_s:
                self._unscrew_set_state("HOLD_NEGATIVE")

        elif self.unscrew_state == "HOLD_NEGATIVE":
            q = self.unscrew_grasp_q.copy()
            q[idx0] = q0_target
            q[idx4] = q4_target
            q[idx12] = q12_target

            if now_elapsed >= self.unscrew_hold_time_s:
                self._unscrew_set_state("RELEASE")

        elif self.unscrew_state == "RELEASE":
            a = self._unscrew_phase_alpha(self.unscrew_release_time_s)

            q_closed = self.unscrew_grasp_q.copy()
            q_openish = self._unscrew_open_flexion_pose()

            # Slightly open flexion.
            q = q_closed + a * (q_openish - q_closed)

            # At the same time return the two adduction joints to the
            # captured start side, ready for the next ratchet stroke.
            q[idx0] = q0_target + a * (float(self.unscrew_reset_q0) - q0_target)
            q[idx4] = q4_target + a * (float(self.unscrew_reset_q4) - q4_target)
            q[idx12] = q12_target + a * (float(self.unscrew_reset_q12) - q12_target)

            if now_elapsed >= self.unscrew_release_time_s:
                self._unscrew_set_state("RECLOSE")

        elif self.unscrew_state == "RECLOSE":
            a = self._unscrew_phase_alpha(self.unscrew_reclose_time_s)

            q_openish = self._unscrew_open_flexion_pose()
            q_closed = self.unscrew_grasp_q.copy()

            # Reclose to exactly the captured grasp.
            q = q_openish + a * (q_closed - q_openish)

            # q0/q4 stay reset while reclosing.
            q[idx0] = float(self.unscrew_reset_q0)
            q[idx4] = float(self.unscrew_reset_q4)
            q[idx12] = float(self.unscrew_reset_q12)

            if now_elapsed >= self.unscrew_reclose_time_s:
                self.unscrew_cycle_count += 1
                self._unscrew_set_state("SWEEP")

        else:
            # Safety fallback.
            self._unscrew_set_state("WAIT_GRASP")
            return None

        q = np.clip(q, self.lower, self.upper)
        self.q_prev = q.copy()

        diagnostics = {
            "unscrew_active": 1.0,
            "unscrew_state": self.unscrew_state,
            "unscrew_cycle_count": float(self.unscrew_cycle_count),
            "unscrew_q0": float(q[idx0]),
            "unscrew_q4": float(q[idx4]),
            "unscrew_q12": float(q[idx12]),
            "unscrew_q0_target": float(q0_target),
            "unscrew_q4_target": float(q4_target),
            "unscrew_q12_target": float(q12_target),
            "unscrew_mean_human_closure": float(
                np.mean([
                    samples["thumb"].closure,
                    samples["index"].closure,
                    samples["middle"].closure,
                ])
            ),
        }
        return q, diagnostics


    def filter_tof_sample(
        self,
        finger_name: str,
        raw_tof_mm: Optional[float],
    ) -> Optional[float]:
        """
        Validate + despike + low-pass filter a ToF sample.

        Invalid values are ignored and the previous filtered value is kept.
        """
        if raw_tof_mm is None or not np.isfinite(raw_tof_mm):
            return self.twist_filtered_tof.get(finger_name)

        raw = float(raw_tof_mm)

        if raw < self.twist_tof_min_valid_mm or raw > self.twist_tof_max_valid_mm:
            return self.twist_filtered_tof.get(finger_name)

        last_raw = self.twist_last_raw_tof.get(finger_name)
        if last_raw is not None and abs(raw - last_raw) > self.twist_tof_max_jump_mm:
            return self.twist_filtered_tof.get(finger_name)

        self.twist_last_raw_tof[finger_name] = raw

        prev = self.twist_filtered_tof.get(finger_name)
        if prev is None:
            filt = raw
        else:
            a = clamp(self.twist_tof_filter_alpha, 0.0, 1.0)
            filt = prev + a * (raw - prev)

        self.twist_filtered_tof[finger_name] = float(filt)
        return float(filt)


    def tof_progress_from_reference(
        self,
        reference_tof_mm: Optional[float],
        current_tof_mm: Optional[float],
    ) -> float:
        """
        Convert filtered ToF displacement into a smooth [-1, 1] command.

        Positive when ToF decreases relative to the reference.
        Uses a deadband and tanh saturation.
        """
        if reference_tof_mm is None or current_tof_mm is None:
            return 0.0

        delta_mm = float(reference_tof_mm - current_tof_mm)

        if abs(delta_mm) <= self.twist_tof_deadband_mm:
            return 0.0

        # Remove the deadband continuously.
        delta_eff = math.copysign(
            abs(delta_mm) - self.twist_tof_deadband_mm,
            delta_mm,
        )

        scale = max(float(self.twist_tof_scale_mm), 1e-6)
        return float(math.tanh(delta_eff / scale))


    def update_twist_state(self, samples: Dict[str, WeartSample]) -> bool:
        """
        Detect the jar-twisting phase.

        Normal mode:
            use the full three-finger retargeting.

        Twist mode:
            closure has stayed approximately constant while ToF changes.
            The current pinch pose is frozen and only q0, q4, q12 are allowed
            to change.

        Returns True while twist mode is active.
        """
        fingers = ("thumb", "index", "middle")

        # Update short histories.
        for name in fingers:
            sample = samples.get(name)
            if sample is None:
                continue
            self.closure_history[name].append(float(sample.closure))
            filtered_tof = self.filter_tof_sample(name, sample.tof_mm)
            if filtered_tof is not None:
                self.tof_history[name].append(float(filtered_tof))

        # Require enough closure history to make a temporal decision.
        if any(len(self.closure_history[name]) < self.twist_history_len for name in fingers):
            return self.twist_active

        closure_stable = True
        mean_closures = []
        for name in fingers:
            h = np.asarray(self.closure_history[name], dtype=float)
            mean_closures.append(float(np.mean(h)))
            if float(np.max(h) - np.min(h)) > self.twist_closure_tol:
                closure_stable = False

        grasp_present = float(np.mean(mean_closures)) >= self.twist_min_mean_closure

        # Look for a meaningful ToF excursion in at least two digits.
        tof_excursions = {}
        for name in fingers:
            h = self.tof_history[name]
            if len(h) >= self.twist_history_len:
                arr = np.asarray(h, dtype=float)
                tof_excursions[name] = float(np.max(arr) - np.min(arr))

        tof_motion_count = sum(
            exc >= self.twist_tof_trigger_mm
            for exc in tof_excursions.values()
        )
        tof_motion = tof_motion_count >= 2

        # Enter twist mode only from a valid, already established grasp.
        if (not self.twist_active) and closure_stable and grasp_present and tof_motion:
            self.twist_active = True
            self.twist_hold_q = self.q_prev.copy()

            # Use the oldest available ToF in the temporal window as reference,
            # so the current excursion is retained instead of being reset to zero.
            self.twist_reference_tof = {}
            for name in fingers:
                h = self.tof_history[name]
                if len(h) > 0:
                    self.twist_reference_tof[name] = float(h[0])

            # Reset raw jump memory at mode entry so the next valid samples
            # are compared against the current motion, not an old pre-grasp spike.
            self.twist_last_raw_tof = {
                name: self.twist_last_raw_tof.get(name)
                for name in fingers
                if name in self.twist_last_raw_tof
            }

        # Exit when the grasp itself changes substantially.
        if self.twist_active and ((not closure_stable) or (not grasp_present)):
            self.twist_active = False
            self.twist_hold_q = None
            self.twist_reference_tof = {}

        return self.twist_active


    def shared_twist_progress(self) -> float:
        """
        Build ONE twist command from the three filtered ToF signals.

        Each finger contributes its displacement relative to the ToF reference.
        We use the median so one noisy finger cannot dominate the command.

        Positive progress means the fingertips moved toward the palm relative
        to the captured grasp reference; negative means they moved away again.
        """
        progresses = []

        for name in ("thumb", "index", "middle"):
            ref = self.twist_reference_tof.get(name)
            cur = self.twist_filtered_tof.get(name)
            if ref is None or cur is None:
                continue

            progresses.append(
                self.tof_progress_from_reference(ref, cur)
            )

        if not progresses:
            return 0.0

        return float(np.median(np.asarray(progresses, dtype=float)))


    def tof_driven_adduction_target(
        self,
        q_hold: np.ndarray,
        finger_name: str,
        sample: WeartSample,
        reference_tof_mm: Optional[float],
        joint_name: str,
        motion_sign: float,
        max_fraction: float,
    ) -> float:
        """
        Convert FILTERED ToF displacement into an adduction-joint target.

        The command uses:
          - validity/outlier rejection,
          - low-pass filtered ToF,
          - deadband,
          - tanh saturation,
          - a limited fraction of the joint range.
        """
        idx = self.q_index(joint_name)

        current_tof = self.filter_tof_sample(finger_name, sample.tof_mm)
        progress = self.tof_progress_from_reference(
            reference_tof_mm,
            current_tof,
        )

        span = float(self.upper[idx] - self.lower[idx])
        target = (
            float(q_hold[idx])
            + float(motion_sign)
            * float(max_fraction)
            * span
            * progress
        )

        return float(np.clip(target, self.lower[idx], self.upper[idx]))


    def solve_twist_mode(self, samples: Dict[str, WeartSample]):
        """
        Jar-opening twist controller.

        The three-finger pinch pose is frozen when twist mode starts.

        ONLY:
            q0  -> index lateral/adduction-like DoF
            q4  -> middle lateral/adduction-like DoF
            q12 -> thumb adduction/opposition DoF

        are changed.

        Crucially, all three are driven by ONE shared ToF progress and all three
        are mapped to the SAME palm-frame lateral sweep direction. Therefore,
        when the thumb sweeps to its "right", index and middle sweep to their
        corresponding "right" as well, even if their joint signs are different.
        """
        if self.twist_hold_q is None:
            self.twist_hold_q = self.q_prev.copy()

        # Update all filtered ToF values first.
        for name in ("thumb", "index", "middle"):
            self.filter_tof_sample(name, samples[name].tof_mm)

        shared_u = self.shared_twist_progress()

        q = self.twist_hold_q.copy()

        # INDEX q0
        idx0 = self.q_index("0")
        span0 = float(self.upper[idx0] - self.lower[idx0])
        q[idx0] = np.clip(
            float(self.twist_hold_q[idx0])
            + float(self.twist_index_joint_sign)
            * float(self.twist_adduction_fraction_index)
            * span0
            * shared_u,
            self.lower[idx0],
            self.upper[idx0],
        )

        # MIDDLE q4
        idx4 = self.q_index("4")
        span4 = float(self.upper[idx4] - self.lower[idx4])
        q[idx4] = np.clip(
            float(self.twist_hold_q[idx4])
            + float(self.twist_middle_joint_sign)
            * float(self.twist_adduction_fraction_middle)
            * span4
            * shared_u,
            self.lower[idx4],
            self.upper[idx4],
        )

        # THUMB q12
        idx12 = self.q_index("12")
        span12 = float(self.upper[idx12] - self.lower[idx12])
        q[idx12] = np.clip(
            float(self.twist_hold_q[idx12])
            + float(self.twist_thumb_joint_sign)
            * float(self.twist_adduction_fraction_thumb)
            * span12
            * shared_u,
            self.lower[idx12],
            self.upper[idx12],
        )

        q = np.clip(q, self.lower, self.upper)
        self.q_prev = q.copy()

        diagnostics = {
            "twist_active": 1.0,
            "twist_q0": float(q[idx0]),
            "twist_q4": float(q[idx4]),
            "twist_q12": float(q[idx12]),
            "twist_tof_thumb": float(self.twist_filtered_tof.get("thumb", math.nan)),
            "twist_tof_index": float(self.twist_filtered_tof.get("index", math.nan)),
            "twist_tof_middle": float(self.twist_filtered_tof.get("middle", math.nan)),
            "twist_progress_shared": float(shared_u),
            "twist_sign_index": float(self.twist_index_joint_sign),
            "twist_sign_middle": float(self.twist_middle_joint_sign),
            "twist_sign_thumb": float(self.twist_thumb_joint_sign),
            "twist_sweep_direction": float(self.twist_sweep_direction),
            "twist_index_sign_multiplier": float(self.twist_index_sign_multiplier),
            "twist_middle_sign_multiplier": float(self.twist_middle_sign_multiplier),
            "twist_thumb_sign_multiplier": float(self.twist_thumb_sign_multiplier),
        }

        return q, diagnostics


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
        # SECONDARY: tri-digit tactile grasp geometry
        # ---------------------------------------------------------------
        thumb_sample = samples.get("thumb")
        index_sample = samples.get("index")
        middle_sample = samples.get("middle")

        if thumb_sample is not None and index_sample is not None:
            g_thumb = self.tactile_activation(thumb_sample.closure)
            g_index = self.tactile_activation(index_sample.closure)
            g_TI = math.sqrt(g_thumb * g_index)

            axis_TI, face_TI, gap_TI, _, _ = self.finger_pair_grasp_residuals(
                q,
                reference_name="index",
                other_name="thumb",
                reference_closure=index_sample.closure,
                other_closure=thumb_sample.closure,
                axis_tolerance_m=self.axis_tolerance_TI_m,
            )

            residuals.append(g_TI * self.w_axis_TI * axis_TI)
            residuals.append(g_TI * self.w_face_TI * face_TI)
            residuals.append(g_TI * self.w_gap_TI * gap_TI)

        if thumb_sample is not None and middle_sample is not None:
            g_thumb = self.tactile_activation(thumb_sample.closure)
            g_middle = self.tactile_activation(middle_sample.closure)
            g_TM = math.sqrt(g_thumb * g_middle)

            axis_TM, face_TM, gap_TM, _, _ = self.finger_pair_grasp_residuals(
                q,
                reference_name="middle",
                other_name="thumb",
                reference_closure=middle_sample.closure,
                other_closure=thumb_sample.closure,
                axis_tolerance_m=self.axis_tolerance_TM_m,
            )

            residuals.append(g_TM * self.w_axis_TM * axis_TM)
            residuals.append(g_TM * self.w_face_TM * face_TM)
            residuals.append(g_TM * self.w_gap_TM * gap_TM)

        if index_sample is not None and middle_sample is not None:
            g_index = self.tactile_activation(index_sample.closure)
            g_middle = self.tactile_activation(middle_sample.closure)
            g_IM = math.sqrt(g_index * g_middle)

            spread_res, _ = self.index_middle_spread_residual(q)
            order_res, _ = self.index_middle_order_residual(q)

            residuals.append(
                g_IM
                * self.w_index_middle_spread
                * spread_res
            )
            residuals.append(
                g_IM
                * self.w_index_middle_order
                * order_res
            )

        # ---------------------------------------------------------------
        # TASK-SPECIFIC: jar-opening wrap using q0/q4
        # ---------------------------------------------------------------
        if self.jar_task_enabled:
            index_sample = samples.get("index")
            middle_sample = samples.get("middle")

            if index_sample is not None:
                g_idx = self.tactile_activation(index_sample.closure)
                idx0 = self.q_index("0")
                span0 = max(self.upper[idx0] - self.lower[idx0], 1e-6)
                residuals.append(
                    g_idx * self.w_jar_wrap_index
                    * (q[idx0] - self.jar_index_target) / span0
                )

            if middle_sample is not None:
                g_mid = self.tactile_activation(middle_sample.closure)
                idx4 = self.q_index("4")
                span4 = max(self.upper[idx4] - self.lower[idx4], 1e-6)
                residuals.append(
                    g_mid * self.w_jar_wrap_middle
                    * (q[idx4] - self.jar_middle_target) / span4
                )

        # ---------------------------------------------------------------
        # SECONDARY: adaptive neutral posture for unobserved lateral DoFs
        # ---------------------------------------------------------------
        # WEART does not provide index/middle adduction in this setup.
        #
        # Open fingers: keep q0/q4 near neutral.
        # Closing fingers: reduce the neutral penalty so tactile/contact-ready
        # objectives can exploit these robot-specific DoFs.
        index_lat_idx = self.q_index("0")
        middle_lat_idx = self.q_index("4")

        index_lat_scale = max(
            self.upper[index_lat_idx] - self.lower[index_lat_idx], 1e-6
        )
        middle_lat_scale = max(
            self.upper[middle_lat_idx] - self.lower[middle_lat_idx], 1e-6
        )

        index_sample = samples.get("index")
        middle_sample = samples.get("middle")

        g_index_lat = (
            self.tactile_activation(index_sample.closure)
            if index_sample is not None else 0.0
        )
        g_middle_lat = (
            self.tactile_activation(middle_sample.closure)
            if middle_sample is not None else 0.0
        )

        w_index_lat = (
            self.w_lateral_open * (1.0 - g_index_lat)
            + self.w_lateral_grasp * g_index_lat
        )
        w_middle_lat = (
            self.w_lateral_open * (1.0 - g_middle_lat)
            + self.w_lateral_grasp * g_middle_lat
        )

        residuals.append(
            w_index_lat
            * (q[index_lat_idx] - self.index_lateral_neutral)
            / index_lat_scale
        )
        residuals.append(
            w_middle_lat
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
        # Automatic unscrewing cycle has priority once a grasp is captured.
        cycle_result = self.solve_unscrew_cycle(samples)
        if cycle_result is not None:
            return cycle_result

        # Normal retargeting while waiting for the grasp.
        # The older ToF-twist controller is intentionally bypassed here so it
        # cannot interfere with the deterministic q0/q4 ratcheting cycle.
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

        diagnostics = {"twist_active": 0.0, "unscrew_active": 0.0}
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
        # Tri-digit grasp diagnostics
        # ---------------------------------------------------------------
        thumb_sample = samples["thumb"]
        index_sample = samples["index"]
        middle_sample = samples["middle"]

        g_thumb = self.tactile_activation(thumb_sample.closure)
        g_index = self.tactile_activation(index_sample.closure)
        g_middle = self.tactile_activation(middle_sample.closure)

        g_TI = math.sqrt(g_thumb * g_index)
        g_TM = math.sqrt(g_thumb * g_middle)
        g_IM = math.sqrt(g_index * g_middle)

        axis_TI, face_TI, gap_TI, raw_axis_TI_m, raw_gap_TI_m = (
            self.finger_pair_grasp_residuals(
                q_solution,
                reference_name="index",
                other_name="thumb",
                reference_closure=index_sample.closure,
                other_closure=thumb_sample.closure,
                axis_tolerance_m=self.axis_tolerance_TI_m,
            )
        )

        axis_TM, face_TM, gap_TM, raw_axis_TM_m, raw_gap_TM_m = (
            self.finger_pair_grasp_residuals(
                q_solution,
                reference_name="middle",
                other_name="thumb",
                reference_closure=middle_sample.closure,
                other_closure=thumb_sample.closure,
                axis_tolerance_m=self.axis_tolerance_TM_m,
            )
        )

        spread_res, spread_m = self.index_middle_spread_residual(q_solution)
        order_res, signed_order_sep_m = self.index_middle_order_residual(q_solution)

        # Raw geometry.
        diagnostics["axis_TI_mm"] = 1000.0 * raw_axis_TI_m
        diagnostics["gap_TI_mm"] = 1000.0 * raw_gap_TI_m
        diagnostics["axis_TM_mm"] = 1000.0 * raw_axis_TM_m
        diagnostics["gap_TM_mm"] = 1000.0 * raw_gap_TM_m
        diagnostics["spread_IM_mm"] = 1000.0 * spread_m
        diagnostics["order_IM_signed_mm"] = 1000.0 * signed_order_sep_m

        # Facing angles in degrees.
        _, n_thumb = self.fingertip_position_and_normal(
            q_solution, self.fingers["thumb"]
        )
        _, n_index = self.fingertip_position_and_normal(
            q_solution, self.fingers["index"]
        )
        _, n_middle = self.fingertip_position_and_normal(
            q_solution, self.fingers["middle"]
        )

        diagnostics["facing_TI_deg"] = math.degrees(
            math.acos(
                clamp(float(np.dot(n_thumb, -n_index)), -1.0, 1.0)
            )
        )
        diagnostics["facing_TM_deg"] = math.degrees(
            math.acos(
                clamp(float(np.dot(n_thumb, -n_middle)), -1.0, 1.0)
            )
        )

        # Weighted residual magnitudes.
        diagnostics["axis_TI_cost"] = g_TI * self.w_axis_TI * abs(float(axis_TI))
        diagnostics["face_TI_cost"] = g_TI * self.w_face_TI * abs(float(face_TI))
        diagnostics["gap_TI_cost"] = g_TI * self.w_gap_TI * abs(float(gap_TI))

        diagnostics["axis_TM_cost"] = g_TM * self.w_axis_TM * abs(float(axis_TM))
        diagnostics["face_TM_cost"] = g_TM * self.w_face_TM * abs(float(face_TM))
        diagnostics["gap_TM_cost"] = g_TM * self.w_gap_TM * abs(float(gap_TM))

        diagnostics["spread_IM_cost"] = (
            g_IM * self.w_index_middle_spread * abs(float(spread_res))
        )
        diagnostics["order_IM_cost"] = (
            g_IM * self.w_index_middle_order * abs(float(order_res))
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
        # Diagnostics for adaptive lateral freedom.
        index_lat_idx = self.q_index("0")
        middle_lat_idx = self.q_index("4")

        g_index_lat = self.tactile_activation(samples["index"].closure)
        g_middle_lat = self.tactile_activation(samples["middle"].closure)

        diagnostics["index_lateral_q"] = float(q_solution[index_lat_idx])
        diagnostics["middle_lateral_q"] = float(q_solution[middle_lat_idx])

        diagnostics["index_lateral_weight"] = (
            self.w_lateral_open * (1.0 - g_index_lat)
            + self.w_lateral_grasp * g_index_lat
        )
        diagnostics["middle_lateral_weight"] = (
            self.w_lateral_open * (1.0 - g_middle_lat)
            + self.w_lateral_grasp * g_middle_lat
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
            if diag.get("unscrew_active", 0.0) > 0.5:
                self.get_logger().info(
                    "UNSCREW CYCLE | "
                    f"state={diag['unscrew_state']} | "
                    f"cycle={int(diag['unscrew_cycle_count'])} | "
                    f"human_closure={diag['unscrew_mean_human_closure']:.2f} | "
                    f"q0={diag['unscrew_q0']:.3f} "
                    f"(target={diag['unscrew_q0_target']:.3f}) | "
                    f"q4={diag['unscrew_q4']:.3f} "
                    f"(target={diag['unscrew_q4_target']:.3f}) | "
                    f"q12_inst={diag['unscrew_q12']:.3f} | "
                    f"q12_target={diag['unscrew_q12_target']:.3f}"
                )
                return

            if diag.get("twist_active", 0.0) > 0.5:
                self.get_logger().info(
                    "JAR-TWIST ACTIVE | "
                    f"ToF thumb={diag['twist_tof_thumb']:.1f} mm, "
                    f"index={diag['twist_tof_index']:.1f} mm, "
                    f"middle={diag['twist_tof_middle']:.1f} mm | "
                    f"q0={diag['twist_q0']:.3f}, "
                    f"q4={diag['twist_q4']:.3f}, "
                    f"q12={diag['twist_q12']:.3f} | "
                    f"u_shared={diag['twist_progress_shared']:+.2f} | "
                    f"signs[i={diag['twist_sign_index']:+.0f}, "
                    f"m={diag['twist_sign_middle']:+.0f}, "
                    f"t={diag['twist_sign_thumb']:+.0f}] | "
                    f"sweep={diag['twist_sweep_direction']:+.0f}"
                )
                return

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
                        f"T-I: axis={diag['axis_TI_mm']:.1f} mm, "
                        f"face={diag['facing_TI_deg']:.1f} deg, "
                        f"gap={diag['gap_TI_mm']:.1f} mm | "
                        f"T-M: axis={diag['axis_TM_mm']:.1f} mm, "
                        f"face={diag['facing_TM_deg']:.1f} deg, "
                        f"gap={diag['gap_TM_mm']:.1f} mm | "
                        f"I-M spread={diag['spread_IM_mm']:.1f} mm, "
                        f"order={diag['order_IM_signed_mm']:.1f} mm",
                    ]
                )
            )
            self.get_logger().info(
                 "COSTS | "
                 f"TI[a={diag['axis_TI_cost']:.3f}, "
                 f"f={diag['face_TI_cost']:.3f}, "
                 f"g={diag['gap_TI_cost']:.3f}] | "
                 f"TM[a={diag['axis_TM_cost']:.3f}, "
                 f"f={diag['face_TM_cost']:.3f}, "
                 f"g={diag['gap_TM_cost']:.3f}] | "
                 f"IMspread={diag['spread_IM_cost']:.3f} | "
                 f"IMorder={diag['order_IM_cost']:.3f} | "
                 f"cl_thumb={diag['thumb_closure_cost']:.3f} | "
                 f"cl_index={diag['index_closure_cost']:.3f} | "
                 f"add_thumb={diag['thumb_adduction_cost']:.3f}"
             )
            self.get_logger().info(
                "LATERAL | "
                f"q0_index={diag['index_lateral_q']:.3f} rad, "
                f"w0={diag['index_lateral_weight']:.2f} | "
                f"q4_middle={diag['middle_lateral_q']:.3f} rad, "
                f"w4={diag['middle_lateral_weight']:.2f}"
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
