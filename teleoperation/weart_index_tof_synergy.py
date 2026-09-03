#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""ROS 2: continuously blend DIP-only, synergy and MCP-only index poses.

The normalized WEART ToF is used as a continuous shape coordinate:

    small ToF   -> DIP-only pose
    middle ToF  -> conventional three-joint synergy
    large ToF   -> MCP-only pose

Three configurable ToF anchors define the exact poses. Smoothstep interpolation
between anchors preserves the two useful extremes without discontinuities in
the intermediate configurations. Raw ToF is normalized by the open-finger ToF
learned during startup calibration.

Hardware output is disabled by default. Run this controller instead of, not
together with, the other LEAP controllers.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

if __package__:
    from .weart_index_tof_branch import (
        DEFAULT_CLOSURE_ANGLE_RANGE_DEG,
        DEFAULT_LARGE_TOF_SELECTS_MCP,
        TofBranchConfig,
        TofBranchEstimator,
    )
    from .weart_leap_index_ik import (
        DEFAULT_ENABLE_HARDWARE,
        DEFAULT_KD,
        DEFAULT_KI,
        DEFAULT_KP,
        DEFAULT_LEAP_PORT,
        DirectLeapIndexDriver,
        LeapIndexKinematics,
        WeartSample,
        parse_weart_line,
    )
else:
    from weart_index_tof_branch import (
        DEFAULT_CLOSURE_ANGLE_RANGE_DEG,
        DEFAULT_LARGE_TOF_SELECTS_MCP,
        TofBranchConfig,
        TofBranchEstimator,
    )
    from weart_leap_index_ik import (
        DEFAULT_ENABLE_HARDWARE,
        DEFAULT_KD,
        DEFAULT_KI,
        DEFAULT_KP,
        DEFAULT_LEAP_PORT,
        DirectLeapIndexDriver,
        LeapIndexKinematics,
        WeartSample,
        parse_weart_line,
    )


DEFAULT_INPUT_TOPIC = "/weart/index/raw"
DEFAULT_OUTPUT_TOPIC = "/weart/index/tof_synergy_leap_joints"
DEFAULT_WEIGHTS_OUTPUT_TOPIC = "/weart/index/tof_synergy_weights"
DEFAULT_TARGET_OUTPUT_TOPIC = "/leap/target_joint_positions"
DEFAULT_ACTUAL_OUTPUT_TOPIC = "/leap/actual_joint_positions"
DEFAULT_URDF_PATH = str(Path(__file__).with_name("model.urdf"))
DEFAULT_ACTUAL_FEEDBACK_PERIOD_S = 0.05

DEFAULT_CLOSED_POSE_DEG = np.array([85.0, 95.0, 65.0], dtype=float)
DEFAULT_SYNERGY_EXPONENTS = np.array([0.85, 1.05, 1.25], dtype=float)
DEFAULT_TOF_DIP_ANCHOR = 0.35
DEFAULT_TOF_SYNERGY_ANCHOR = 0.50
DEFAULT_TOF_MCP_ANCHOR = 0.65
DEFAULT_TOF_FILTER_ALPHA = 0.25
DEFAULT_TOF_MEDIAN_WINDOW = 5
DEFAULT_TOF_DEADBAND = 0.008
DEFAULT_TOF_MAX_RATE_PER_S = 1.0
DEFAULT_CLOSURE_FILTER_ALPHA = 0.35
DEFAULT_CLOSURE_MAX_RATE_PER_S = 3.0
DEFAULT_POSE_FILTER_FREQUENCY_HZ = 2.0
DEFAULT_POSE_MAX_SPEED_DEG_S = 120.0
DEFAULT_POSE_MAX_ACCELERATION_DEG_S2 = 500.0

# Alpha values are interpreted at this nominal period, then converted using dt.
FILTER_REFERENCE_PERIOD_S = 0.01


@dataclass(frozen=True)
class TofSynergyConfig:
    closure_angle_range_rad: float
    closed_pose_rad: np.ndarray
    synergy_exponents: np.ndarray
    tof_dip_anchor: float = DEFAULT_TOF_DIP_ANCHOR
    tof_synergy_anchor: float = DEFAULT_TOF_SYNERGY_ANCHOR
    tof_mcp_anchor: float = DEFAULT_TOF_MCP_ANCHOR
    tof_filter_alpha: float = DEFAULT_TOF_FILTER_ALPHA
    tof_median_window: int = DEFAULT_TOF_MEDIAN_WINDOW
    tof_deadband: float = DEFAULT_TOF_DEADBAND
    tof_max_rate_per_s: float = DEFAULT_TOF_MAX_RATE_PER_S
    closure_filter_alpha: float = DEFAULT_CLOSURE_FILTER_ALPHA
    closure_max_rate_per_s: float = DEFAULT_CLOSURE_MAX_RATE_PER_S
    pose_filter_frequency_hz: float = DEFAULT_POSE_FILTER_FREQUENCY_HZ
    pose_max_speed_rad_s: float = math.radians(DEFAULT_POSE_MAX_SPEED_DEG_S)
    pose_max_acceleration_rad_s2: float = math.radians(
        DEFAULT_POSE_MAX_ACCELERATION_DEG_S2
    )
    large_tof_selects_mcp: bool = DEFAULT_LARGE_TOF_SELECTS_MCP

    def validate(self) -> None:
        if (
            not math.isfinite(self.closure_angle_range_rad)
            or self.closure_angle_range_rad <= 0.0
        ):
            raise ValueError("closure_angle_range_rad must be positive")
        if self.closed_pose_rad.shape != (3,):
            raise ValueError("closed_pose_rad must contain MCP, PIP and DIP")
        if self.synergy_exponents.shape != (3,) or np.any(
            self.synergy_exponents <= 0.0
        ):
            raise ValueError("synergy_exponents must contain three positive values")
        if not (
            0.0
            <= self.tof_dip_anchor
            < self.tof_synergy_anchor
            < self.tof_mcp_anchor
            <= 1.0
        ):
            raise ValueError("ToF anchors must satisfy 0 <= dip < synergy < mcp <= 1")
        if not 0.0 < self.tof_filter_alpha <= 1.0:
            raise ValueError("tof_filter_alpha must be in (0, 1]")
        if self.tof_median_window < 1 or self.tof_median_window % 2 == 0:
            raise ValueError("tof_median_window must be a positive odd integer")
        if not 0.0 <= self.tof_deadband < 0.1:
            raise ValueError("tof_deadband must be in [0, 0.1)")
        if self.tof_max_rate_per_s <= 0.0:
            raise ValueError("tof_max_rate_per_s must be positive")
        if not 0.0 < self.closure_filter_alpha <= 1.0:
            raise ValueError("closure_filter_alpha must be in (0, 1]")
        if self.closure_max_rate_per_s <= 0.0:
            raise ValueError("closure_max_rate_per_s must be positive")
        if (
            min(
                self.pose_filter_frequency_hz,
                self.pose_max_speed_rad_s,
                self.pose_max_acceleration_rad_s2,
            )
            <= 0.0
        ):
            raise ValueError(
                "pose filter frequency, speed and acceleration must be positive"
            )


class RateLimitedLowPass:
    """Sample-rate-independent low-pass with deadband and rate limit."""

    def __init__(
        self,
        *,
        alpha: float,
        max_rate_per_s: float,
        deadband: float = 0.0,
    ) -> None:
        self.alpha = float(alpha)
        self.max_rate_per_s = float(max_rate_per_s)
        self.deadband = float(deadband)
        self.value: float | None = None

    def update(self, target: float, dt: float) -> float:
        target = float(np.clip(target, 0.0, 1.0))
        if self.value is None:
            self.value = target
            return target

        error = target - self.value
        if abs(error) <= self.deadband:
            return self.value

        # Remove the deadband without introducing a jump at its edge.
        target_after_deadband = target - math.copysign(self.deadband, error)
        exponent = max(float(dt), 0.0) / FILTER_REFERENCE_PERIOD_S
        alpha_dt = 1.0 - (1.0 - self.alpha) ** exponent
        step = alpha_dt * (target_after_deadband - self.value)
        max_step = self.max_rate_per_s * max(float(dt), 0.0)
        self.value += float(np.clip(step, -max_step, max_step))
        self.value = float(np.clip(self.value, 0.0, 1.0))
        return self.value


class SmoothPoseFilter:
    """Three-stage filter with acceleration limiting and URDF-aware braking."""

    def __init__(
        self,
        *,
        frequency_hz: float,
        max_speed_rad_s: float,
        max_acceleration_rad_s2: float,
        q_lower: np.ndarray,
        q_upper: np.ndarray,
    ) -> None:
        self.omega = 2.0 * math.pi * float(frequency_hz)
        self.max_speed_rad_s = float(max_speed_rad_s)
        self.max_acceleration_rad_s2 = float(max_acceleration_rad_s2)
        self.q_lower = np.asarray(q_lower, dtype=float)
        self.q_upper = np.asarray(q_upper, dtype=float)
        if self.q_lower.shape != self.q_upper.shape or self.q_lower.ndim != 1:
            raise ValueError("q_lower and q_upper must be equal-length vectors")
        self.stage_1 = np.zeros_like(self.q_lower)
        self.stage_2 = np.zeros_like(self.q_lower)
        self.q = np.zeros_like(self.q_lower)
        self.velocity = np.zeros_like(self.q_lower)

    def update(self, target: np.ndarray, dt: float) -> np.ndarray:
        target = np.asarray(target, dtype=float)
        if target.shape != self.q.shape:
            raise ValueError(
                f"Expected target shape {self.q.shape}, got {target.shape}"
            )
        remaining = max(float(dt), 1e-6)

        # Small integration steps keep the response stable after delayed samples.
        while remaining > 0.0:
            step_dt = min(remaining, FILTER_REFERENCE_PERIOD_S)
            alpha = 1.0 - math.exp(-self.omega * step_dt)
            max_position_step = self.max_speed_rad_s * step_dt

            self.stage_1 += np.clip(
                alpha * (target - self.stage_1),
                -max_position_step,
                max_position_step,
            )
            self.stage_2 += np.clip(
                alpha * (self.stage_1 - self.stage_2),
                -max_position_step,
                max_position_step,
            )

            proposed_q = self.q + np.clip(
                alpha * (self.stage_2 - self.q),
                -max_position_step,
                max_position_step,
            )
            desired_velocity = (proposed_q - self.q) / step_dt
            max_velocity_change = self.max_acceleration_rad_s2 * step_dt
            self.velocity += np.clip(
                desired_velocity - self.velocity,
                -max_velocity_change,
                max_velocity_change,
            )
            self.velocity = np.clip(
                self.velocity,
                -self.max_speed_rad_s,
                self.max_speed_rad_s,
            )

            # Start braking early enough to stop at the URDF limits.
            lower_distance = np.maximum(self.q - self.q_lower, 0.0)
            upper_distance = np.maximum(self.q_upper - self.q, 0.0)
            lower_safe_speed = np.maximum(
                np.sqrt(2.0 * self.max_acceleration_rad_s2 * lower_distance)
                - self.max_acceleration_rad_s2 * step_dt,
                0.0,
            )
            upper_safe_speed = np.maximum(
                np.sqrt(2.0 * self.max_acceleration_rad_s2 * upper_distance)
                - self.max_acceleration_rad_s2 * step_dt,
                0.0,
            )
            self.velocity = np.clip(
                self.velocity,
                -lower_safe_speed,
                upper_safe_speed,
            )

            self.q += self.velocity * step_dt
            self.q = np.clip(self.q, self.q_lower, self.q_upper)
            remaining -= step_dt

        return self.q.copy()


class TofSynergyBlendEstimator:
    """Generate a continuous posture from closure and normalized ToF."""

    def __init__(
        self,
        kinematics: LeapIndexKinematics,
        config: TofSynergyConfig,
    ) -> None:
        config.validate()
        self.kin = kinematics
        self.config = config
        self.closed_pose_rad = np.clip(
            np.asarray(config.closed_pose_rad, dtype=float),
            self.kin.q_lower,
            self.kin.q_upper,
        )

        self.tof_tracker = TofBranchEstimator(
            kinematics,
            TofBranchConfig(
                closure_angle_range_rad=config.closure_angle_range_rad,
                tof_threshold=config.tof_synergy_anchor,
                tof_hysteresis=0.0,
                # Filtering is applied below after the median outlier rejection.
                tof_filter_alpha=1.0,
                branch_update_min_closure=0.0,
                large_tof_selects_mcp=config.large_tof_selects_mcp,
            ),
        )
        self.tof_samples: deque[float] = deque(maxlen=config.tof_median_window)
        self.tof_filter = RateLimitedLowPass(
            alpha=config.tof_filter_alpha,
            max_rate_per_s=config.tof_max_rate_per_s,
            deadband=config.tof_deadband,
        )
        self.closure_filter = RateLimitedLowPass(
            alpha=config.closure_filter_alpha,
            max_rate_per_s=config.closure_max_rate_per_s,
        )
        self.pose_filter = SmoothPoseFilter(
            frequency_hz=config.pose_filter_frequency_hz,
            max_speed_rad_s=config.pose_max_speed_rad_s,
            max_acceleration_rad_s2=config.pose_max_acceleration_rad_s2,
            q_lower=np.maximum(self.kin.q_lower, 0.0),
            q_upper=self.kin.q_upper,
        )

    @staticmethod
    def _smoothstep(value: float) -> float:
        x = float(np.clip(value, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def blend_weights(self, tof_normalized: float) -> np.ndarray:
        t = float(np.clip(tof_normalized, 0.0, 1.0))
        if not self.config.large_tof_selects_mcp:
            t = 1.0 - t

        dip = self.config.tof_dip_anchor
        synergy = self.config.tof_synergy_anchor
        mcp = self.config.tof_mcp_anchor

        if t <= dip:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        if t < synergy:
            u = self._smoothstep((t - dip) / (synergy - dip))
            return np.array([1.0 - u, u, 0.0], dtype=float)
        if t < mcp:
            u = self._smoothstep((t - synergy) / (mcp - synergy))
            return np.array([0.0, 1.0 - u, u], dtype=float)
        return np.array([0.0, 0.0, 1.0], dtype=float)

    def candidate_poses(self, closure: float) -> tuple[np.ndarray, ...]:
        c = float(np.clip(closure, 0.0, 1.0))
        branch_angle = self.config.closure_angle_range_rad * c

        q_dip = np.array([0.0, 0.0, branch_angle], dtype=float)
        q_synergy = self.closed_pose_rad * np.power(c, self.config.synergy_exponents)
        q_mcp = np.array([branch_angle, 0.0, 0.0], dtype=float)

        return tuple(
            np.clip(q, self.kin.q_lower, self.kin.q_upper)
            for q in (q_dip, q_synergy, q_mcp)
        )

    def update(
        self,
        sample: WeartSample,
        *,
        tof_normalized_offset: float = 0.0,
    ) -> tuple[np.ndarray | None, dict[str, float | str]]:
        tracker_q, tracker_diag = self.tof_tracker.update(sample)
        if tracker_q is None:
            return None, tracker_diag

        dt = float(tracker_diag["dt"])
        closure_sensor = float(tracker_diag["closure_used"])
        tof_normalized_sensor = float(tracker_diag["tof_normalized"])
        tof_normalized = float(
            np.clip(tof_normalized_sensor + tof_normalized_offset, 0.0, 1.0)
        )

        self.tof_samples.append(tof_normalized)
        tof_median = float(np.median(self.tof_samples))
        tof_filtered = self.tof_filter.update(tof_median, dt)
        closure = self.closure_filter.update(closure_sensor, dt)

        weights = self.blend_weights(tof_filtered)
        candidates = self.candidate_poses(closure)
        q_goal = sum(
            (weight * candidate for weight, candidate in zip(weights, candidates)),
            start=np.zeros(3, dtype=float),
        )
        q = self.pose_filter.update(q_goal, dt)
        q = np.clip(q, self.kin.q_lower, self.kin.q_upper)

        dominant = ("dip", "synergy", "mcp")[int(np.argmax(weights))]
        return q, {
            **tracker_diag,
            "closure_sensor": closure_sensor,
            "closure_used": closure,
            "tof_normalized_sensor": tof_normalized_sensor,
            "tof_normalized": tof_normalized,
            "tof_median": tof_median,
            "tof_filtered": tof_filtered,
            "weight_dip": float(weights[0]),
            "weight_synergy": float(weights[1]),
            "weight_mcp": float(weights[2]),
            "dominant": dominant,
            "q_goal_mcp": float(q_goal[0]),
            "q_goal_pip": float(q_goal[1]),
            "q_goal_dip": float(q_goal[2]),
        }


class WeartIndexTofSynergyNode(Node):
    def __init__(self) -> None:
        super().__init__("weart_index_to_leap_tof_synergy")

        self.declare_parameter("input_topic", DEFAULT_INPUT_TOPIC)
        self.declare_parameter("output_topic", DEFAULT_OUTPUT_TOPIC)
        self.declare_parameter("weights_output_topic", DEFAULT_WEIGHTS_OUTPUT_TOPIC)
        self.declare_parameter("target_output_topic", DEFAULT_TARGET_OUTPUT_TOPIC)
        self.declare_parameter("actual_output_topic", DEFAULT_ACTUAL_OUTPUT_TOPIC)
        self.declare_parameter(
            "actual_feedback_period_s", DEFAULT_ACTUAL_FEEDBACK_PERIOD_S
        )
        self.declare_parameter("urdf_path", DEFAULT_URDF_PATH)
        self.declare_parameter("enable_hardware", DEFAULT_ENABLE_HARDWARE)
        self.declare_parameter("leap_port", DEFAULT_LEAP_PORT)
        self.declare_parameter("kp", DEFAULT_KP)
        self.declare_parameter("ki", DEFAULT_KI)
        self.declare_parameter("kd", DEFAULT_KD)

        self.declare_parameter(
            "closure_angle_range_deg", DEFAULT_CLOSURE_ANGLE_RANGE_DEG
        )
        self.declare_parameter("closed_pose_deg", DEFAULT_CLOSED_POSE_DEG.tolist())
        self.declare_parameter("synergy_exponents", DEFAULT_SYNERGY_EXPONENTS.tolist())
        self.declare_parameter("tof_dip_anchor", DEFAULT_TOF_DIP_ANCHOR)
        self.declare_parameter("tof_synergy_anchor", DEFAULT_TOF_SYNERGY_ANCHOR)
        self.declare_parameter("tof_mcp_anchor", DEFAULT_TOF_MCP_ANCHOR)
        self.declare_parameter("tof_filter_alpha", DEFAULT_TOF_FILTER_ALPHA)
        self.declare_parameter("tof_median_window", DEFAULT_TOF_MEDIAN_WINDOW)
        self.declare_parameter("tof_deadband", DEFAULT_TOF_DEADBAND)
        self.declare_parameter("tof_max_rate_per_s", DEFAULT_TOF_MAX_RATE_PER_S)
        self.declare_parameter("closure_filter_alpha", DEFAULT_CLOSURE_FILTER_ALPHA)
        self.declare_parameter("closure_max_rate_per_s", DEFAULT_CLOSURE_MAX_RATE_PER_S)
        self.declare_parameter(
            "pose_filter_frequency_hz", DEFAULT_POSE_FILTER_FREQUENCY_HZ
        )
        self.declare_parameter("pose_max_speed_deg_s", DEFAULT_POSE_MAX_SPEED_DEG_S)
        self.declare_parameter(
            "pose_max_acceleration_deg_s2",
            DEFAULT_POSE_MAX_ACCELERATION_DEG_S2,
        )
        self.declare_parameter("large_tof_selects_mcp", DEFAULT_LARGE_TOF_SELECTS_MCP)

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        weights_output_topic = str(self.get_parameter("weights_output_topic").value)
        target_output_topic = str(self.get_parameter("target_output_topic").value)
        actual_output_topic = str(self.get_parameter("actual_output_topic").value)

        self.kin = LeapIndexKinematics(
            str(self.get_parameter("urdf_path").value),
            tof_palm_point_mm=np.zeros(3, dtype=float),
        )
        config = TofSynergyConfig(
            closure_angle_range_rad=math.radians(
                float(self.get_parameter("closure_angle_range_deg").value)
            ),
            closed_pose_rad=np.deg2rad(
                np.asarray(
                    self.get_parameter("closed_pose_deg").value,
                    dtype=float,
                )
            ),
            synergy_exponents=np.asarray(
                self.get_parameter("synergy_exponents").value,
                dtype=float,
            ),
            tof_dip_anchor=float(self.get_parameter("tof_dip_anchor").value),
            tof_synergy_anchor=float(self.get_parameter("tof_synergy_anchor").value),
            tof_mcp_anchor=float(self.get_parameter("tof_mcp_anchor").value),
            tof_filter_alpha=float(self.get_parameter("tof_filter_alpha").value),
            tof_median_window=int(self.get_parameter("tof_median_window").value),
            tof_deadband=float(self.get_parameter("tof_deadband").value),
            tof_max_rate_per_s=float(self.get_parameter("tof_max_rate_per_s").value),
            closure_filter_alpha=float(
                self.get_parameter("closure_filter_alpha").value
            ),
            closure_max_rate_per_s=float(
                self.get_parameter("closure_max_rate_per_s").value
            ),
            pose_filter_frequency_hz=float(
                self.get_parameter("pose_filter_frequency_hz").value
            ),
            pose_max_speed_rad_s=math.radians(
                float(self.get_parameter("pose_max_speed_deg_s").value)
            ),
            pose_max_acceleration_rad_s2=math.radians(
                float(self.get_parameter("pose_max_acceleration_deg_s2").value)
            ),
            large_tof_selects_mcp=bool(
                self.get_parameter("large_tof_selects_mcp").value
            ),
        )
        self.estimator = TofSynergyBlendEstimator(self.kin, config)

        self.leap = DirectLeapIndexDriver(
            self.kin,
            enabled=bool(self.get_parameter("enable_hardware").value),
            port=str(self.get_parameter("leap_port").value),
            kp=float(self.get_parameter("kp").value),
            ki=float(self.get_parameter("ki").value),
            kd=float(self.get_parameter("kd").value),
        )
        self.leap.connect()

        self.subscription = self.create_subscription(
            String, input_topic, self._on_weart, 20
        )
        self.output_pub = self.create_publisher(Float64MultiArray, output_topic, 20)
        self.weights_pub = self.create_publisher(
            Float64MultiArray, weights_output_topic, 20
        )
        self.target_pub = self.create_publisher(
            Float64MultiArray, target_output_topic, 20
        )
        self.actual_pub = self.create_publisher(
            Float64MultiArray, actual_output_topic, 20
        )

        self.actual_feedback_failures = 0
        self.actual_timer = None
        if self.leap.enabled:
            feedback_period_s = float(
                self.get_parameter("actual_feedback_period_s").value
            )
            if feedback_period_s <= 0.0:
                raise ValueError("actual_feedback_period_s must be positive")
            self.actual_timer = self.create_timer(
                feedback_period_s, self._publish_actual_pose
            )

        self.counter = 0
        self.get_logger().info(f"WEART input: {input_topic}")
        self.get_logger().info(f"ToF-synergy q output: {output_topic}")
        self.get_logger().info(f"Blend weights output: {weights_output_topic}")
        self.get_logger().info(f"Full commanded LEAP q: {target_output_topic}")
        self.get_logger().info(
            "ToF anchors DIP/synergy/MCP=[%.2f, %.2f, %.2f] | "
            "closure 0.5 -> branch angle %.1f deg"
            % (
                config.tof_dip_anchor,
                config.tof_synergy_anchor,
                config.tof_mcp_anchor,
                0.5 * math.degrees(config.closure_angle_range_rad),
            )
        )
        self.get_logger().info(
            "Smoothing: ToF median=%d alpha=%.2f deadband=%.3f rate=%.2f/s | "
            "pose speed/accel=%.0f deg/s / %.0f deg/s^2"
            % (
                config.tof_median_window,
                config.tof_filter_alpha,
                config.tof_deadband,
                config.tof_max_rate_per_s,
                math.degrees(config.pose_max_speed_rad_s),
                math.degrees(config.pose_max_acceleration_rad_s2),
            )
        )
        self.get_logger().warning(
            "Calibration: keep HUMAN INDEX OPEN until ToF calibration reaches 100%."
        )
        if self.leap.enabled:
            self.get_logger().warning("LEAP HARDWARE OUTPUT ENABLED.")
            self.get_logger().info(f"Actual LEAP q output: {actual_output_topic}")
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
            q_target, diag = self.estimator.update(sample)
        except Exception as exc:
            self.get_logger().warning(f"WEART/estimator error: {exc}")
            return

        if q_target is None:
            if self.counter % 5 == 0:
                self.get_logger().info(
                    f"ToF calibration: {100.0 * diag['calibration']:.0f}% "
                    "(index open)"
                )
            self.counter += 1
            return

        output = Float64MultiArray()
        output.data = [float(value) for value in q_target]
        self.output_pub.publish(output)

        weights = Float64MultiArray()
        weights.data = [
            float(diag["weight_dip"]),
            float(diag["weight_synergy"]),
            float(diag["weight_mcp"]),
            float(diag["tof_filtered"]),
        ]
        self.weights_pub.publish(weights)

        try:
            q_command, motor_qpos = self.leap.command(q_target, float(diag["dt"]))
        except Exception as exc:
            self.get_logger().error(f"LEAP command failed: {exc}")
            return

        commanded = Float64MultiArray()
        commanded.data = [float(value) for value in self.leap.current_sim_qpos]
        self.target_pub.publish(commanded)

        if self.counter % 5 == 0:
            self.get_logger().info(
                "closure sensor/filtered=%.3f/%.3f | "
                "ToF=%.1f mm raw/median/filtered=%.3f/%.3f/%.3f | "
                "weights DIP/SYN/MCP=[%.2f, %.2f, %.2f] dominant=%s | "
                "goal=[%.1f, %.1f, %.1f] deg | smooth=[%.1f, %.1f, %.1f] deg | "
                "cmd=[%.1f, %.1f, %.1f] deg | motor123=%s"
                % (
                    diag["closure_sensor"],
                    diag["closure_used"],
                    diag["tof_mm"],
                    diag["tof_normalized"],
                    diag["tof_median"],
                    diag["tof_filtered"],
                    diag["weight_dip"],
                    diag["weight_synergy"],
                    diag["weight_mcp"],
                    diag["dominant"],
                    math.degrees(diag["q_goal_mcp"]),
                    math.degrees(diag["q_goal_pip"]),
                    math.degrees(diag["q_goal_dip"]),
                    *np.rad2deg(q_target),
                    *np.rad2deg(q_command),
                    np.round(motor_qpos[[1, 2, 3]], 3).tolist(),
                )
            )
        self.counter += 1

    def _publish_actual_pose(self) -> None:
        try:
            actual_q = self.leap.read_actual_sim_qpos()
            if actual_q is None:
                return
            output = Float64MultiArray()
            output.data = [float(value) for value in actual_q]
            self.actual_pub.publish(output)
            self.actual_feedback_failures = 0
        except Exception as exc:
            self.actual_feedback_failures += 1
            if (
                self.actual_feedback_failures == 1
                or self.actual_feedback_failures % 20 == 0
            ):
                self.get_logger().warning(f"LEAP actual feedback failed: {exc}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = WeartIndexTofSynergyNode()
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
