#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""ROS 2: select an MCP-only or DIP-only LEAP pose using WEART ToF.

The heuristic intentionally represents two ambiguous configurations:

    closure=0.5, large ToF -> [MCP=90 deg, PIP=0, DIP=0]
    closure=0.5, small ToF -> [MCP=0, PIP=0, DIP=90 deg]

Raw ToF is published in millimetres. Startup calibration learns the open-finger
distance and normalizes each sample as ``tof_mm / open_tof_mm``. A Schmitt
trigger around the configurable 0.5 threshold prevents branch chatter.

Hardware output is disabled by default. Run this controller instead of, not
together with, the other LEAP controllers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

if __package__:
    from .weart_leap_index_ik import (
        DEFAULT_ENABLE_HARDWARE,
        DEFAULT_KD,
        DEFAULT_KI,
        DEFAULT_KP,
        DEFAULT_LEAP_PORT,
        MAX_DT_S,
        MIN_DT_S,
        DirectLeapIndexDriver,
        LeapIndexKinematics,
        WeartSample,
        parse_weart_line,
    )
else:
    from weart_leap_index_ik import (
        DEFAULT_ENABLE_HARDWARE,
        DEFAULT_KD,
        DEFAULT_KI,
        DEFAULT_KP,
        DEFAULT_LEAP_PORT,
        MAX_DT_S,
        MIN_DT_S,
        DirectLeapIndexDriver,
        LeapIndexKinematics,
        WeartSample,
        parse_weart_line,
    )


DEFAULT_INPUT_TOPIC = "/weart/index/raw"
DEFAULT_OUTPUT_TOPIC = "/weart/index/tof_branch_leap_joints"
DEFAULT_BRANCH_OUTPUT_TOPIC = "/weart/index/tof_branch"
DEFAULT_TARGET_OUTPUT_TOPIC = "/leap/target_joint_positions"
DEFAULT_ACTUAL_OUTPUT_TOPIC = "/leap/actual_joint_positions"
DEFAULT_URDF_PATH = str(Path(__file__).with_name("model.urdf"))
DEFAULT_ACTUAL_FEEDBACK_PERIOD_S = 0.05

DEFAULT_CLOSURE_ANGLE_RANGE_DEG = 180.0
DEFAULT_TOF_THRESHOLD = 0.50
DEFAULT_TOF_HYSTERESIS = 0.05
DEFAULT_TOF_FILTER_ALPHA = 0.25
DEFAULT_BRANCH_UPDATE_MIN_CLOSURE = 0.15
DEFAULT_LARGE_TOF_SELECTS_MCP = True

CALIBRATION_SAMPLES = 30
CALIBRATION_MAX_CLOSURE = 0.10

MCP_BRANCH = "mcp"
DIP_BRANCH = "dip"


@dataclass(frozen=True)
class TofBranchConfig:
    closure_angle_range_rad: float
    tof_threshold: float = DEFAULT_TOF_THRESHOLD
    tof_hysteresis: float = DEFAULT_TOF_HYSTERESIS
    tof_filter_alpha: float = DEFAULT_TOF_FILTER_ALPHA
    branch_update_min_closure: float = DEFAULT_BRANCH_UPDATE_MIN_CLOSURE
    large_tof_selects_mcp: bool = DEFAULT_LARGE_TOF_SELECTS_MCP

    def validate(self) -> None:
        if (
            not math.isfinite(self.closure_angle_range_rad)
            or self.closure_angle_range_rad <= 0.0
        ):
            raise ValueError("closure_angle_range_rad must be positive")
        if not 0.0 < self.tof_threshold < 1.0:
            raise ValueError("tof_threshold must be in (0, 1)")
        if (
            not 0.0
            <= self.tof_hysteresis
            < min(self.tof_threshold, 1.0 - self.tof_threshold)
        ):
            raise ValueError("tof_hysteresis is incompatible with tof_threshold")
        if not 0.0 < self.tof_filter_alpha <= 1.0:
            raise ValueError("tof_filter_alpha must be in (0, 1]")
        if not 0.0 <= self.branch_update_min_closure <= 1.0:
            raise ValueError("branch_update_min_closure must be in [0, 1]")


class TofBranchEstimator:
    """Map closure to angle and use normalized ToF as a branch selector."""

    def __init__(
        self,
        kinematics: LeapIndexKinematics,
        config: TofBranchConfig,
    ) -> None:
        config.validate()
        self.kin = kinematics
        self.config = config

        self.open_closure = 0.0
        self.open_tof_mm: float | None = None
        self.filtered_tof_normalized: float | None = None
        self.branch = MCP_BRANCH
        self.last_ts_ms: int | None = None

        self._calib_closure: list[float] = []
        self._calib_tof: list[float] = []
        self.calibrated = False

    def _sample_is_good_for_calibration(self, sample: WeartSample) -> bool:
        return (
            sample.closure <= CALIBRATION_MAX_CLOSURE
            and math.isfinite(sample.tof_mm)
            and sample.tof_mm > 0.0
        )

    def _calibrate(self, sample: WeartSample) -> float:
        if not self._sample_is_good_for_calibration(sample):
            return len(self._calib_tof) / CALIBRATION_SAMPLES

        self._calib_closure.append(float(sample.closure))
        self._calib_tof.append(float(sample.tof_mm))
        if len(self._calib_tof) >= CALIBRATION_SAMPLES:
            self.open_closure = float(np.median(self._calib_closure))
            self.open_tof_mm = float(np.median(self._calib_tof))
            self.filtered_tof_normalized = 1.0
            self.branch = MCP_BRANCH
            self.calibrated = True

        return min(1.0, len(self._calib_tof) / CALIBRATION_SAMPLES)

    def _normalized_closure(self, closure: float) -> float:
        denominator = max(1.0 - self.open_closure, 1e-6)
        normalized = (float(closure) - self.open_closure) / denominator
        return float(np.clip(normalized, 0.0, 1.0))

    def _compute_dt(self, ts_ms: int) -> float:
        if self.last_ts_ms is None:
            self.last_ts_ms = ts_ms
            return 0.01
        dt = (ts_ms - self.last_ts_ms) * 1e-3
        self.last_ts_ms = ts_ms
        if not math.isfinite(dt) or dt <= 0.0:
            return 0.01
        return float(np.clip(dt, MIN_DT_S, MAX_DT_S))

    def _normalized_tof(self, tof_mm: float) -> float:
        assert self.open_tof_mm is not None
        if not math.isfinite(tof_mm) or tof_mm <= 0.0:
            if self.filtered_tof_normalized is None:
                return 1.0
            return self.filtered_tof_normalized
        return float(np.clip(float(tof_mm) / self.open_tof_mm, 0.0, 1.0))

    def _update_filtered_tof(self, tof_normalized: float) -> float:
        if self.filtered_tof_normalized is None:
            self.filtered_tof_normalized = tof_normalized
        else:
            alpha = self.config.tof_filter_alpha
            self.filtered_tof_normalized += alpha * (
                tof_normalized - self.filtered_tof_normalized
            )
        return self.filtered_tof_normalized

    def _select_branch(self, tof_normalized: float, closure: float) -> str:
        if closure < self.config.branch_update_min_closure:
            return self.branch

        low = self.config.tof_threshold - self.config.tof_hysteresis
        high = self.config.tof_threshold + self.config.tof_hysteresis

        if self.config.large_tof_selects_mcp:
            if self.branch == MCP_BRANCH and tof_normalized < low:
                self.branch = DIP_BRANCH
            elif self.branch == DIP_BRANCH and tof_normalized > high:
                self.branch = MCP_BRANCH
        else:
            if self.branch == MCP_BRANCH and tof_normalized > high:
                self.branch = DIP_BRANCH
            elif self.branch == DIP_BRANCH and tof_normalized < low:
                self.branch = MCP_BRANCH
        return self.branch

    def update(
        self,
        sample: WeartSample,
    ) -> tuple[np.ndarray | None, dict[str, float | str]]:
        dt = self._compute_dt(sample.ts_ms)
        if not self.calibrated:
            progress = self._calibrate(sample)
            return None, {"calibration": progress, "dt": dt}

        closure = self._normalized_closure(sample.closure)
        tof_normalized = self._normalized_tof(sample.tof_mm)
        tof_filtered = self._update_filtered_tof(tof_normalized)
        branch = self._select_branch(tof_filtered, closure)

        angle = self.config.closure_angle_range_rad * closure
        q = np.zeros(3, dtype=float)
        if branch == MCP_BRANCH:
            q[0] = angle
        else:
            q[2] = angle
        q = np.clip(q, self.kin.q_lower, self.kin.q_upper)

        return q, {
            "dt": dt,
            "closure_raw": float(sample.closure),
            "closure_used": closure,
            "angle_deg": math.degrees(angle),
            "tof_mm": float(sample.tof_mm),
            "tof_normalized": tof_normalized,
            "tof_filtered": tof_filtered,
            "branch": branch,
        }


class WeartIndexTofBranchNode(Node):
    def __init__(self) -> None:
        super().__init__("weart_index_to_leap_tof_branch")

        self.declare_parameter("input_topic", DEFAULT_INPUT_TOPIC)
        self.declare_parameter("output_topic", DEFAULT_OUTPUT_TOPIC)
        self.declare_parameter("branch_output_topic", DEFAULT_BRANCH_OUTPUT_TOPIC)
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
        self.declare_parameter("tof_threshold", DEFAULT_TOF_THRESHOLD)
        self.declare_parameter("tof_hysteresis", DEFAULT_TOF_HYSTERESIS)
        self.declare_parameter("tof_filter_alpha", DEFAULT_TOF_FILTER_ALPHA)
        self.declare_parameter(
            "branch_update_min_closure", DEFAULT_BRANCH_UPDATE_MIN_CLOSURE
        )
        self.declare_parameter("large_tof_selects_mcp", DEFAULT_LARGE_TOF_SELECTS_MCP)

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        branch_output_topic = str(self.get_parameter("branch_output_topic").value)
        target_output_topic = str(self.get_parameter("target_output_topic").value)
        actual_output_topic = str(self.get_parameter("actual_output_topic").value)

        self.kin = LeapIndexKinematics(
            str(self.get_parameter("urdf_path").value),
            tof_palm_point_mm=np.zeros(3, dtype=float),
        )
        config = TofBranchConfig(
            closure_angle_range_rad=math.radians(
                float(self.get_parameter("closure_angle_range_deg").value)
            ),
            tof_threshold=float(self.get_parameter("tof_threshold").value),
            tof_hysteresis=float(self.get_parameter("tof_hysteresis").value),
            tof_filter_alpha=float(self.get_parameter("tof_filter_alpha").value),
            branch_update_min_closure=float(
                self.get_parameter("branch_update_min_closure").value
            ),
            large_tof_selects_mcp=bool(
                self.get_parameter("large_tof_selects_mcp").value
            ),
        )
        self.estimator = TofBranchEstimator(self.kin, config)

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
        self.branch_pub = self.create_publisher(String, branch_output_topic, 20)
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
        self.get_logger().info(f"ToF-branch q output: {output_topic}")
        self.get_logger().info(f"Selected branch output: {branch_output_topic}")
        self.get_logger().info(f"Full commanded LEAP q: {target_output_topic}")
        self.get_logger().info(
            "closure 0.5 -> %.1f deg | ToF threshold=%.2f +/- %.2f | "
            "large ToF selects %s"
            % (
                0.5 * math.degrees(config.closure_angle_range_rad),
                config.tof_threshold,
                config.tof_hysteresis,
                MCP_BRANCH if config.large_tof_selects_mcp else DIP_BRANCH,
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
        self.branch_pub.publish(String(data=str(diag["branch"])))

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
                "closure=%.3f -> angle=%.1f deg | "
                "ToF=%.1f mm norm/raw-filtered=%.3f/%.3f | branch=%s | "
                "target=[%.1f, %.1f, %.1f] deg | "
                "cmd=[%.1f, %.1f, %.1f] deg | motor123=%s"
                % (
                    diag["closure_used"],
                    diag["angle_deg"],
                    diag["tof_mm"],
                    diag["tof_normalized"],
                    diag["tof_filtered"],
                    diag["branch"],
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
        node = WeartIndexTofBranchNode()
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
