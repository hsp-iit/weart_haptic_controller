#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""ROS 2: WEART thumb -> smooth four-DoF LEAP thumb synergy.

WEART closure and ToF continuously shape the three flexion joints 13, 14 and
15. WEART adduction independently controls joint 12. Since thumb adduction can
also change the measured ToF, an optional linear compensation is available:

    T_corrected = T_sensor + k_adduction * (a - a_open)

The default compensation is zero until it can be identified from real data.
Hardware output is disabled by default.
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
    from .weart_index_tof_synergy import (
        DEFAULT_CLOSURE_FILTER_ALPHA,
        DEFAULT_CLOSURE_MAX_RATE_PER_S,
        DEFAULT_POSE_FILTER_FREQUENCY_HZ,
        DEFAULT_POSE_MAX_ACCELERATION_DEG_S2,
        DEFAULT_POSE_MAX_SPEED_DEG_S,
        DEFAULT_TOF_DEADBAND,
        DEFAULT_TOF_DIP_ANCHOR,
        DEFAULT_TOF_FILTER_ALPHA,
        DEFAULT_TOF_MAX_RATE_PER_S,
        DEFAULT_TOF_MCP_ANCHOR,
        DEFAULT_TOF_MEDIAN_WINDOW,
        DEFAULT_TOF_SYNERGY_ANCHOR,
        RateLimitedLowPass,
        SmoothPoseFilter,
        TofSynergyBlendEstimator,
        TofSynergyConfig,
    )
    from .weart_leap_index_ik import (
        DEFAULT_ENABLE_HARDWARE,
        DEFAULT_KD,
        DEFAULT_KI,
        DEFAULT_KP,
        DEFAULT_LEAP_PORT,
        _load_leap_cpp,
        autodetect_leap_port,
        leap_sim_to_motor,
        parse_weart_line,
    )
    from .weart_three_fingers_synergy import (
        FINGER_SPECS,
        LeapFingerKinematics,
        LeapHandModel,
        load_leap_hand_model,
    )
else:
    from weart_index_tof_synergy import (
        DEFAULT_CLOSURE_FILTER_ALPHA,
        DEFAULT_CLOSURE_MAX_RATE_PER_S,
        DEFAULT_POSE_FILTER_FREQUENCY_HZ,
        DEFAULT_POSE_MAX_ACCELERATION_DEG_S2,
        DEFAULT_POSE_MAX_SPEED_DEG_S,
        DEFAULT_TOF_DEADBAND,
        DEFAULT_TOF_DIP_ANCHOR,
        DEFAULT_TOF_FILTER_ALPHA,
        DEFAULT_TOF_MAX_RATE_PER_S,
        DEFAULT_TOF_MCP_ANCHOR,
        DEFAULT_TOF_MEDIAN_WINDOW,
        DEFAULT_TOF_SYNERGY_ANCHOR,
        RateLimitedLowPass,
        SmoothPoseFilter,
        TofSynergyBlendEstimator,
        TofSynergyConfig,
    )
    from weart_leap_index_ik import (
        DEFAULT_ENABLE_HARDWARE,
        DEFAULT_KD,
        DEFAULT_KI,
        DEFAULT_KP,
        DEFAULT_LEAP_PORT,
        _load_leap_cpp,
        autodetect_leap_port,
        leap_sim_to_motor,
        parse_weart_line,
    )
    from weart_three_fingers_synergy import (
        FINGER_SPECS,
        LeapFingerKinematics,
        LeapHandModel,
        load_leap_hand_model,
    )


DEFAULT_INPUT_TOPIC = "/weart/thumb/raw"
DEFAULT_OUTPUT_TOPIC = "/weart/thumb/synergy_leap_joints"
DEFAULT_STATE_OUTPUT_TOPIC = "/weart/thumb/synergy_state"
DEFAULT_TARGET_OUTPUT_TOPIC = "/leap/target_joint_positions"
DEFAULT_ACTUAL_OUTPUT_TOPIC = "/leap/actual_joint_positions"
DEFAULT_URDF_PATH = str(Path(__file__).with_name("model.urdf"))
DEFAULT_ACTUAL_FEEDBACK_PERIOD_S = 0.05

THUMB_MOTOR_INDICES = np.array([12, 13, 14, 15], dtype=int)
DEFAULT_CLOSED_POSE_DEG = np.array([75.0, 70.0, 55.0], dtype=float)
DEFAULT_SYNERGY_EXPONENTS = np.array([0.80, 1.00, 1.25], dtype=float)
DEFAULT_CLOSURE_ANGLE_RANGE_DEG = 180.0

# adduction=0 -> first endpoint; adduction=1 -> second endpoint.
DEFAULT_ADDUCTION_RANGE_DEG = np.array([0.0, 60.0], dtype=float)
DEFAULT_INVERT_ADDUCTION = False
DEFAULT_ADDUCTION_FILTER_ALPHA = 0.30
DEFAULT_ADDUCTION_DEADBAND = 0.005
DEFAULT_ADDUCTION_MAX_RATE_PER_S = 1.5
DEFAULT_ADDUCTION_POSE_FILTER_FREQUENCY_HZ = 2.0
DEFAULT_ADDUCTION_MAX_SPEED_DEG_S = 90.0
DEFAULT_ADDUCTION_MAX_ACCELERATION_DEG_S2 = 360.0

# Shift of normalized ToF for a full unit of adduction relative to calibration.
DEFAULT_TOF_ADDUCTION_COMPENSATION = 0.0


@dataclass(frozen=True)
class ThumbSynergyConfig:
    flexion: TofSynergyConfig
    adduction_range_rad: np.ndarray
    invert_adduction: bool = DEFAULT_INVERT_ADDUCTION
    adduction_filter_alpha: float = DEFAULT_ADDUCTION_FILTER_ALPHA
    adduction_deadband: float = DEFAULT_ADDUCTION_DEADBAND
    adduction_max_rate_per_s: float = DEFAULT_ADDUCTION_MAX_RATE_PER_S
    adduction_pose_filter_frequency_hz: float = (
        DEFAULT_ADDUCTION_POSE_FILTER_FREQUENCY_HZ
    )
    adduction_max_speed_rad_s: float = math.radians(DEFAULT_ADDUCTION_MAX_SPEED_DEG_S)
    adduction_max_acceleration_rad_s2: float = math.radians(
        DEFAULT_ADDUCTION_MAX_ACCELERATION_DEG_S2
    )
    tof_adduction_compensation: float = DEFAULT_TOF_ADDUCTION_COMPENSATION

    def validate(self) -> None:
        self.flexion.validate()
        if self.adduction_range_rad.shape != (2,):
            raise ValueError("adduction_range_rad must contain two endpoints")
        if not np.all(np.isfinite(self.adduction_range_rad)):
            raise ValueError("adduction_range_rad must be finite")
        if not 0.0 < self.adduction_filter_alpha <= 1.0:
            raise ValueError("adduction_filter_alpha must be in (0, 1]")
        if not 0.0 <= self.adduction_deadband < 0.1:
            raise ValueError("adduction_deadband must be in [0, 0.1)")
        if (
            min(
                self.adduction_max_rate_per_s,
                self.adduction_pose_filter_frequency_hz,
                self.adduction_max_speed_rad_s,
                self.adduction_max_acceleration_rad_s2,
            )
            <= 0.0
        ):
            raise ValueError("adduction smoothing limits must be positive")
        if not math.isfinite(self.tof_adduction_compensation):
            raise ValueError("tof_adduction_compensation must be finite")


class ThumbSynergyEstimator:
    """Fuse thumb flexion and adduction while keeping them observable."""

    def __init__(
        self,
        kinematics: LeapFingerKinematics,
        model: LeapHandModel,
        config: ThumbSynergyConfig,
    ) -> None:
        config.validate()
        self.kin = kinematics
        self.model = model
        self.config = config
        self.flexion = TofSynergyBlendEstimator(kinematics, config.flexion)

        side_index = kinematics.side_motor_index
        self.adduction_range_rad = np.clip(
            np.asarray(config.adduction_range_rad, dtype=float),
            model.sim_lower[side_index],
            model.sim_upper[side_index],
        )
        self.adduction_filter = RateLimitedLowPass(
            alpha=config.adduction_filter_alpha,
            max_rate_per_s=config.adduction_max_rate_per_s,
            deadband=config.adduction_deadband,
        )
        self.adduction_pose_filter = SmoothPoseFilter(
            frequency_hz=config.adduction_pose_filter_frequency_hz,
            max_speed_rad_s=config.adduction_max_speed_rad_s,
            max_acceleration_rad_s2=config.adduction_max_acceleration_rad_s2,
            q_lower=np.array([model.sim_lower[side_index]], dtype=float),
            q_upper=np.array([model.sim_upper[side_index]], dtype=float),
        )

        self.last_ts_ms: int | None = None
        self.adduction_reference: float | None = None
        self._calibration_adduction: list[float] = []

    def _compute_dt(self, ts_ms: int) -> float:
        if self.last_ts_ms is None:
            self.last_ts_ms = int(ts_ms)
            return 0.01
        dt = (int(ts_ms) - self.last_ts_ms) * 1e-3
        self.last_ts_ms = int(ts_ms)
        if not math.isfinite(dt) or dt <= 0.0:
            return 0.01
        return float(np.clip(dt, 0.002, 0.25))

    def _normalized_adduction(self, raw_adduction: float) -> float:
        if not math.isfinite(raw_adduction):
            raise ValueError("WEART adduction is not finite")
        value = float(np.clip(raw_adduction, 0.0, 1.0))
        return 1.0 - value if self.config.invert_adduction else value

    def _adduction_goal(self, adduction: float) -> float:
        q_at_zero, q_at_one = self.adduction_range_rad
        return float(q_at_zero + adduction * (q_at_one - q_at_zero))

    def update(self, sample):
        dt = self._compute_dt(sample.ts_ms)
        adduction_raw = float(sample.adduction)
        adduction_sensor = self._normalized_adduction(sample.adduction)
        adduction = self.adduction_filter.update(adduction_sensor, dt)
        q_adduction_goal = self._adduction_goal(adduction)
        q_adduction = float(
            self.adduction_pose_filter.update(
                np.array([q_adduction_goal], dtype=float), dt
            )[0]
        )

        if not self.flexion.tof_tracker.calibrated:
            if sample.closure <= 0.10:
                self._calibration_adduction.append(adduction_sensor)
            q_flexion, diag = self.flexion.update(sample)
            if self.flexion.tof_tracker.calibrated:
                values = self._calibration_adduction or [adduction_sensor]
                self.adduction_reference = float(np.median(values))
            return None, {
                **diag,
                "adduction_raw": adduction_raw,
                "adduction_sensor": adduction_sensor,
                "adduction_filtered": adduction,
                "q_adduction_goal": q_adduction_goal,
                "q_adduction": q_adduction,
            }

        if self.adduction_reference is None:
            self.adduction_reference = adduction_sensor
        tof_offset = self.config.tof_adduction_compensation * (
            adduction - self.adduction_reference
        )
        q_flexion, diag = self.flexion.update(
            sample,
            tof_normalized_offset=tof_offset,
        )
        assert q_flexion is not None

        q_thumb = np.array([q_adduction, *q_flexion], dtype=float)
        return q_thumb, {
            **diag,
            "adduction_raw": adduction_raw,
            "adduction_sensor": adduction_sensor,
            "adduction_filtered": adduction,
            "adduction_reference": self.adduction_reference,
            "q_adduction_goal": q_adduction_goal,
            "q_adduction": q_adduction,
            "tof_adduction_offset": tof_offset,
        }


class DirectLeapThumbDriver:
    def __init__(
        self,
        model: LeapHandModel,
        *,
        enabled: bool,
        port: str,
        kp: float,
        ki: float,
        kd: float,
    ) -> None:
        self.model = model
        self.enabled = bool(enabled)
        self.requested_port = str(port)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.current_sim_qpos = np.zeros(model.motor_count, dtype=float)
        self._ctrl = None
        self.port: str | None = None

    def connect(self) -> None:
        print("[LEAP] thumb control: joints 12,13,14,15.", flush=True)
        if not self.enabled:
            print("[LEAP] hardware output DISABLED (dry-run).", flush=True)
            return

        self.port = autodetect_leap_port(self.requested_port)
        leap_cpp = _load_leap_cpp()
        self._ctrl = leap_cpp.LeapController(self.port)
        self._ctrl.connect()
        self._ctrl.setGains(int(self.kp), int(self.ki), int(self.kd))
        self._ctrl.set_leap(leap_sim_to_motor(self.current_sim_qpos))
        print(f"[LEAP] connected on {self.port}; sent neutral pose.", flush=True)

    def disconnect(self) -> None:
        if self._ctrl is not None:
            self._ctrl.disconnect()
            self._ctrl = None
            print("[LEAP] disconnected.", flush=True)

    def command(self, q_thumb: np.ndarray) -> np.ndarray:
        q_thumb = np.asarray(q_thumb, dtype=float)
        if q_thumb.shape != (4,):
            raise ValueError("q_thumb must contain joints 12,13,14,15")
        lower = self.model.sim_lower[THUMB_MOTOR_INDICES]
        upper = self.model.sim_upper[THUMB_MOTOR_INDICES]
        q_thumb = np.clip(q_thumb, lower, upper)
        self.current_sim_qpos[THUMB_MOTOR_INDICES] = q_thumb
        motor_qpos = leap_sim_to_motor(self.current_sim_qpos)

        if self.enabled:
            if self._ctrl is None:
                raise RuntimeError(
                    "LEAP output enabled but controller is not connected"
                )
            self._ctrl.set_leap(motor_qpos)
        return motor_qpos.copy()

    def read_actual_sim_qpos(self) -> np.ndarray | None:
        if self._ctrl is None:
            return None
        motor_qpos = np.asarray(self._ctrl.read_pos(), dtype=float).reshape(-1)
        if motor_qpos.size != self.model.motor_count:
            raise ValueError(
                f"LEAP returned {motor_qpos.size} joints, "
                f"expected {self.model.motor_count}"
            )
        if not np.all(np.isfinite(motor_qpos)):
            raise ValueError("LEAP returned non-finite joint positions")
        return motor_qpos - math.pi


class WeartThumbSynergyNode(Node):
    def __init__(self) -> None:
        super().__init__("weart_thumb_to_leap_synergy")

        self.declare_parameter("input_topic", DEFAULT_INPUT_TOPIC)
        self.declare_parameter("output_topic", DEFAULT_OUTPUT_TOPIC)
        self.declare_parameter("state_output_topic", DEFAULT_STATE_OUTPUT_TOPIC)
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
        self.declare_parameter("large_tof_selects_mcp", True)

        self.declare_parameter(
            "adduction_range_deg", DEFAULT_ADDUCTION_RANGE_DEG.tolist()
        )
        self.declare_parameter("invert_adduction", DEFAULT_INVERT_ADDUCTION)
        self.declare_parameter("adduction_filter_alpha", DEFAULT_ADDUCTION_FILTER_ALPHA)
        self.declare_parameter("adduction_deadband", DEFAULT_ADDUCTION_DEADBAND)
        self.declare_parameter(
            "adduction_max_rate_per_s", DEFAULT_ADDUCTION_MAX_RATE_PER_S
        )
        self.declare_parameter(
            "adduction_pose_filter_frequency_hz",
            DEFAULT_ADDUCTION_POSE_FILTER_FREQUENCY_HZ,
        )
        self.declare_parameter(
            "adduction_max_speed_deg_s", DEFAULT_ADDUCTION_MAX_SPEED_DEG_S
        )
        self.declare_parameter(
            "adduction_max_acceleration_deg_s2",
            DEFAULT_ADDUCTION_MAX_ACCELERATION_DEG_S2,
        )
        self.declare_parameter(
            "tof_adduction_compensation", DEFAULT_TOF_ADDUCTION_COMPENSATION
        )

        self.model = load_leap_hand_model(str(self.get_parameter("urdf_path").value))
        self.kinematics = LeapFingerKinematics(self.model, FINGER_SPECS["thumb"])

        flexion_config = TofSynergyConfig(
            closure_angle_range_rad=math.radians(
                float(self.get_parameter("closure_angle_range_deg").value)
            ),
            closed_pose_rad=np.deg2rad(
                np.asarray(self.get_parameter("closed_pose_deg").value, dtype=float)
            ),
            synergy_exponents=np.asarray(
                self.get_parameter("synergy_exponents").value, dtype=float
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
        config = ThumbSynergyConfig(
            flexion=flexion_config,
            adduction_range_rad=np.deg2rad(
                np.asarray(
                    self.get_parameter("adduction_range_deg").value,
                    dtype=float,
                )
            ),
            invert_adduction=bool(self.get_parameter("invert_adduction").value),
            adduction_filter_alpha=float(
                self.get_parameter("adduction_filter_alpha").value
            ),
            adduction_deadband=float(self.get_parameter("adduction_deadband").value),
            adduction_max_rate_per_s=float(
                self.get_parameter("adduction_max_rate_per_s").value
            ),
            adduction_pose_filter_frequency_hz=float(
                self.get_parameter("adduction_pose_filter_frequency_hz").value
            ),
            adduction_max_speed_rad_s=math.radians(
                float(self.get_parameter("adduction_max_speed_deg_s").value)
            ),
            adduction_max_acceleration_rad_s2=math.radians(
                float(self.get_parameter("adduction_max_acceleration_deg_s2").value)
            ),
            tof_adduction_compensation=float(
                self.get_parameter("tof_adduction_compensation").value
            ),
        )
        self.estimator = ThumbSynergyEstimator(self.kinematics, self.model, config)

        self.leap = DirectLeapThumbDriver(
            self.model,
            enabled=bool(self.get_parameter("enable_hardware").value),
            port=str(self.get_parameter("leap_port").value),
            kp=float(self.get_parameter("kp").value),
            ki=float(self.get_parameter("ki").value),
            kd=float(self.get_parameter("kd").value),
        )
        self.leap.connect()

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        state_output_topic = str(self.get_parameter("state_output_topic").value)
        target_output_topic = str(self.get_parameter("target_output_topic").value)
        actual_output_topic = str(self.get_parameter("actual_output_topic").value)

        self.subscription = self.create_subscription(
            String, input_topic, self._on_weart, 20
        )
        self.output_pub = self.create_publisher(Float64MultiArray, output_topic, 20)
        self.state_pub = self.create_publisher(
            Float64MultiArray, state_output_topic, 20
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
            period = float(self.get_parameter("actual_feedback_period_s").value)
            if period <= 0.0:
                raise ValueError("actual_feedback_period_s must be positive")
            self.actual_timer = self.create_timer(period, self._publish_actual_pose)

        self.counter = 0
        self.get_logger().info(f"WEART thumb input: {input_topic}")
        self.get_logger().info(f"Thumb q12-15 output: {output_topic}")
        self.get_logger().info(f"Full commanded LEAP q: {target_output_topic}")
        self.get_logger().info(
            "mapping: adduction -> q12 [%.1f, %.1f] deg (inverted=%s) | "
            "closure+ToF -> q13-15 | ToF/add compensation=%.3f"
            % (
                *np.rad2deg(self.estimator.adduction_range_rad),
                config.invert_adduction,
                config.tof_adduction_compensation,
            )
        )
        self.get_logger().warning(
            "Calibration: keep THUMB OPEN and STILL until ToF reaches 100%."
        )
        if self.leap.enabled:
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
            q_thumb, diag = self.estimator.update(sample)
        except Exception as exc:
            self.get_logger().warning(f"WEART thumb/estimator error: {exc}")
            return

        if q_thumb is None:
            if self.counter % 5 == 0:
                self.get_logger().info(
                    f"Thumb ToF calibration: {100.0 * diag['calibration']:.0f}%"
                )
            self.counter += 1
            return

        output = Float64MultiArray()
        output.data = [float(value) for value in q_thumb]
        self.output_pub.publish(output)

        state = Float64MultiArray()
        state.data = [
            float(diag["weight_dip"]),
            float(diag["weight_synergy"]),
            float(diag["weight_mcp"]),
            float(diag["tof_filtered"]),
            float(diag["adduction_sensor"]),
            float(diag["adduction_filtered"]),
            float(diag["tof_adduction_offset"]),
        ]
        self.state_pub.publish(state)

        try:
            motor_qpos = self.leap.command(q_thumb)
        except Exception as exc:
            self.get_logger().error(f"LEAP thumb command failed: {exc}")
            return

        commanded = Float64MultiArray()
        commanded.data = [float(value) for value in self.leap.current_sim_qpos]
        self.target_pub.publish(commanded)

        if self.counter % 5 == 0:
            self.get_logger().info(
                "closure=%.3f | ToF raw/corrected/filtered=%.3f/%.3f/%.3f | "
                "weights DIP/SYN/MCP=[%.2f, %.2f, %.2f] | "
                "adduction ROS/used/filtered=%.3f/%.3f/%.3f q12=%.1f deg | "
                "q13-15=[%.1f, %.1f, %.1f] deg | motor12-15=%s"
                % (
                    diag["closure_used"],
                    diag["tof_normalized_sensor"],
                    diag["tof_normalized"],
                    diag["tof_filtered"],
                    diag["weight_dip"],
                    diag["weight_synergy"],
                    diag["weight_mcp"],
                    diag["adduction_raw"],
                    diag["adduction_sensor"],
                    diag["adduction_filtered"],
                    math.degrees(q_thumb[0]),
                    *np.rad2deg(q_thumb[1:]),
                    np.round(motor_qpos[THUMB_MOTOR_INDICES], 3).tolist(),
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
        node = WeartThumbSynergyNode()
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
