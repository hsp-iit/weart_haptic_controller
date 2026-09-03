#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""ROS 2: WEART index -> two-DoF adaptive synergy -> LEAP index.

The original one-DoF synergy cannot represent an MCP-only posture because one
closure value bends MCP, PIP and DIP together. This estimator instead solves
for two variables:

    x = [q_mcp, c_distal]
    q = [q_mcp,
         q_pip_closed * c_distal ** p_pip,
         q_dip_closed * c_distal ** p_dip]

The independent MCP coordinate makes [90 deg, 0 deg, 0 deg] representable.
Fingertip orientation from the IMU constrains total flexion; URDF forward
kinematics and ToF help distinguish MCP rotation from distal curl. Closure and
temporal continuity regularize the estimate.

The palm must remain still because only the fingertip IMU is available. The
ToF palm point and distance scale are calibration parameters, not URDF data.
Hardware output is disabled by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

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
        angle_error,
        parse_weart_line,
        wrap_pi,
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
        angle_error,
        parse_weart_line,
        wrap_pi,
    )


DEFAULT_INPUT_TOPIC = "/weart/index/raw"
DEFAULT_OUTPUT_TOPIC = "/weart/index/mcp_synergy_leap_joints"
DEFAULT_TARGET_OUTPUT_TOPIC = "/leap/target_joint_positions"
DEFAULT_ACTUAL_OUTPUT_TOPIC = "/leap/actual_joint_positions"
DEFAULT_URDF_PATH = str(Path(__file__).with_name("model.urdf"))
DEFAULT_ACTUAL_FEEDBACK_PERIOD_S = 0.05

DEFAULT_MCP_REFERENCE_DEG = 90.0
DEFAULT_DISTAL_CLOSED_POSE_DEG = np.array([95.0, 65.0], dtype=float)
DEFAULT_DISTAL_EXPONENTS = np.array([1.0, 1.2], dtype=float)

DEFAULT_CLOSURE_SIGMA = 0.10
DEFAULT_IMU_SIGMA_DEG = 8.0
DEFAULT_TOF_SIGMA_MM = 8.0
DEFAULT_SMOOTH_SIGMA_DEG = 20.0
DEFAULT_TOF_DISTANCE_SCALE = 1.0
DEFAULT_TOF_PALM_POINT_MM = np.zeros(3, dtype=float)

CALIBRATION_SAMPLES = 30
CALIBRATION_MAX_CLOSURE = 0.10
CALIBRATION_MAX_GYRO_NORM_DEG_S = 15.0

GYRO_FLEX_AXIS = 2
GYRO_FLEX_SIGN = +1.0
ACC_ANGLE_NUM_AXIS = 0
ACC_ANGLE_DEN_AXIS = 1
ACC_ANGLE_NUM_SIGN = +1.0
ACC_ANGLE_DEN_SIGN = +1.0
ACC_CORRECTION_GAIN = 0.04
ACC_NORM_SIGMA_G = 0.12


@dataclass(frozen=True)
class McpSynergyConfig:
    mcp_reference_rad: float
    distal_closed_pose_rad: np.ndarray
    distal_exponents: np.ndarray
    closure_sigma: float = DEFAULT_CLOSURE_SIGMA
    imu_sigma_rad: float = math.radians(DEFAULT_IMU_SIGMA_DEG)
    tof_sigma_mm: float = DEFAULT_TOF_SIGMA_MM
    smooth_sigma_rad: float = math.radians(DEFAULT_SMOOTH_SIGMA_DEG)
    tof_distance_scale: float = DEFAULT_TOF_DISTANCE_SCALE
    use_tof: bool = True

    def validate(self) -> None:
        if not math.isfinite(self.mcp_reference_rad) or self.mcp_reference_rad <= 0:
            raise ValueError("mcp_reference_rad must be positive")
        if self.distal_closed_pose_rad.shape != (2,):
            raise ValueError("distal_closed_pose_rad must contain PIP and DIP")
        if self.distal_exponents.shape != (2,) or np.any(self.distal_exponents <= 0.0):
            raise ValueError("distal_exponents must contain two positive values")
        if (
            min(
                self.closure_sigma,
                self.imu_sigma_rad,
                self.tof_sigma_mm,
                self.smooth_sigma_rad,
            )
            <= 0.0
        ):
            raise ValueError("all estimator sigmas must be positive")
        if not math.isfinite(self.tof_distance_scale):
            raise ValueError("tof_distance_scale must be finite")


class McpAdaptiveSynergyEstimator:
    """Estimate independent MCP flexion and one shared distal coordinate."""

    def __init__(
        self,
        kinematics: LeapIndexKinematics,
        config: McpSynergyConfig,
    ) -> None:
        config.validate()
        self.kin = kinematics
        self.config = config

        self.mcp_lower = max(0.0, float(self.kin.q_lower[0]))
        self.mcp_upper = float(self.kin.q_upper[0])
        self.mcp_reference_rad = float(
            np.clip(config.mcp_reference_rad, self.mcp_lower, self.mcp_upper)
        )
        self.distal_closed_pose_rad = np.clip(
            np.asarray(config.distal_closed_pose_rad, dtype=float),
            self.kin.q_lower[1:],
            self.kin.q_upper[1:],
        )

        self.state = np.array([self.mcp_lower, 0.0], dtype=float)
        self.q = self.q_from_state(self.state)
        self.theta = 0.0
        self.last_ts_ms: int | None = None

        self.acc_open_angle: float | None = None
        self.gyro_bias_deg_s = 0.0
        self.open_tof_human_mm: float | None = None
        self.open_closure = 0.0

        self._calib_acc_angles: list[float] = []
        self._calib_gyro_flex: list[float] = []
        self._calib_tof: list[float] = []
        self._calib_closure: list[float] = []
        self.calibrated = False

    def q_from_state(self, state: np.ndarray) -> np.ndarray:
        q_mcp = float(np.clip(state[0], self.mcp_lower, self.mcp_upper))
        c_distal = float(np.clip(state[1], 0.0, 1.0))
        q_distal = self.distal_closed_pose_rad * np.power(
            c_distal, self.config.distal_exponents
        )
        q = np.array([q_mcp, q_distal[0], q_distal[1]], dtype=float)
        return np.clip(q, self.kin.q_lower, self.kin.q_upper)

    def predicted_closure(self, state: np.ndarray) -> float:
        mcp_progress = float(np.clip(state[0] / self.mcp_reference_rad, 0.0, 1.0))
        distal_progress = float(np.clip(state[1], 0.0, 1.0))
        # A full MCP bend or a full distal curl can both represent closure=1.
        return max(mcp_progress, distal_progress)

    def _raw_acc_angle(self, acc: np.ndarray) -> float:
        numerator = ACC_ANGLE_NUM_SIGN * float(acc[ACC_ANGLE_NUM_AXIS])
        denominator = ACC_ANGLE_DEN_SIGN * float(acc[ACC_ANGLE_DEN_AXIS])
        return math.atan2(numerator, denominator)

    def _sample_is_good_for_calibration(self, sample: WeartSample) -> bool:
        return (
            sample.closure <= CALIBRATION_MAX_CLOSURE
            and float(np.linalg.norm(sample.gyro_deg_s))
            <= CALIBRATION_MAX_GYRO_NORM_DEG_S
            and 0.75 <= float(np.linalg.norm(sample.acc_g)) <= 1.25
            and np.isfinite(sample.tof_mm)
            and sample.tof_mm > 0.0
        )

    def _calibrate(self, sample: WeartSample) -> float:
        if not self._sample_is_good_for_calibration(sample):
            return len(self._calib_tof) / CALIBRATION_SAMPLES

        self._calib_acc_angles.append(self._raw_acc_angle(sample.acc_g))
        self._calib_gyro_flex.append(
            GYRO_FLEX_SIGN * float(sample.gyro_deg_s[GYRO_FLEX_AXIS])
        )
        self._calib_tof.append(float(sample.tof_mm))
        self._calib_closure.append(float(sample.closure))

        if len(self._calib_tof) >= CALIBRATION_SAMPLES:
            self.acc_open_angle = math.atan2(
                float(np.mean(np.sin(self._calib_acc_angles))),
                float(np.mean(np.cos(self._calib_acc_angles))),
            )
            self.gyro_bias_deg_s = float(np.mean(self._calib_gyro_flex))
            self.open_tof_human_mm = float(np.median(self._calib_tof))
            self.open_closure = float(np.median(self._calib_closure))
            self.state[:] = (self.mcp_lower, 0.0)
            self.q = self.q_from_state(self.state)
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

    def _normalized_closure(self, closure: float) -> float:
        denominator = max(1.0 - self.open_closure, 1e-6)
        normalized = (float(closure) - self.open_closure) / denominator
        return float(np.clip(normalized, 0.0, 1.0))

    def _update_theta(self, sample: WeartSample, dt: float) -> tuple[float, float]:
        gyro_flex_deg_s = (
            GYRO_FLEX_SIGN * float(sample.gyro_deg_s[GYRO_FLEX_AXIS])
            - self.gyro_bias_deg_s
        )
        theta_gyro = wrap_pi(self.theta + math.radians(gyro_flex_deg_s) * dt)

        acc_norm = float(np.linalg.norm(sample.acc_g))
        acc_trust = math.exp(-0.5 * ((acc_norm - 1.0) / ACC_NORM_SIGMA_G) ** 2)
        assert self.acc_open_angle is not None
        theta_acc = wrap_pi(self._raw_acc_angle(sample.acc_g) - self.acc_open_angle)
        self.theta = wrap_pi(
            theta_gyro
            + ACC_CORRECTION_GAIN * acc_trust * angle_error(theta_acc, theta_gyro)
        )
        return self.theta, acc_trust

    def _predict_weart_tof_mm(self, q: np.ndarray) -> float:
        assert self.open_tof_human_mm is not None
        robot_range_delta_mm = self.kin.robot_range_mm(q) - self.kin.open_robot_range_mm
        return (
            self.open_tof_human_mm
            + self.config.tof_distance_scale * robot_range_delta_mm
        )

    def _residual(
        self,
        state: np.ndarray,
        closure_meas: float,
        theta_meas: float,
        tof_meas_mm: float,
        q_previous: np.ndarray,
    ) -> np.ndarray:
        q = self.q_from_state(state)
        theta_pred = self.kin.fingertip_orientation_rad(q)

        residuals = [
            (self.predicted_closure(state) - closure_meas) / self.config.closure_sigma,
            angle_error(theta_pred, theta_meas) / self.config.imu_sigma_rad,
        ]

        if self.config.use_tof and np.isfinite(tof_meas_mm) and tof_meas_mm > 0.0:
            residuals.append(
                (self._predict_weart_tof_mm(q) - float(tof_meas_mm))
                / self.config.tof_sigma_mm
            )

        residuals.extend((q - q_previous) / self.config.smooth_sigma_rad)
        return np.asarray(residuals, dtype=float)

    def update(
        self,
        sample: WeartSample,
    ) -> tuple[np.ndarray | None, dict[str, float]]:
        dt = self._compute_dt(sample.ts_ms)
        if not self.calibrated:
            progress = self._calibrate(sample)
            return None, {"calibration": progress, "dt": dt}

        closure = self._normalized_closure(sample.closure)
        theta, imu_trust = self._update_theta(sample, dt)
        q_previous = self.q.copy()

        theta_mcp_seed = np.clip(theta, self.mcp_lower, self.mcp_upper)
        nominal_mcp = np.clip(
            closure * self.mcp_reference_rad,
            self.mcp_lower,
            self.mcp_upper,
        )
        seeds = (
            self.state,
            np.array([theta_mcp_seed, 0.0]),
            np.array([nominal_mcp, closure]),
            np.array([self.mcp_lower, closure]),
        )

        lower = np.array([self.mcp_lower, 0.0], dtype=float)
        upper = np.array([self.mcp_upper, 1.0], dtype=float)
        best = None
        eps = 1e-8
        for seed in seeds:
            result = least_squares(
                self._residual,
                x0=np.clip(np.asarray(seed, dtype=float), lower + eps, upper - eps),
                bounds=(lower, upper),
                args=(
                    closure,
                    theta,
                    sample.tof_mm,
                    q_previous,
                ),
                loss="soft_l1",
                f_scale=1.0,
                max_nfev=40,
                ftol=1e-6,
                xtol=1e-6,
                gtol=1e-6,
            )
            if best is None or result.cost < best.cost:
                best = result

        assert best is not None
        self.state = best.x.astype(float, copy=True)
        self.q = self.q_from_state(self.state)
        theta_pred = self.kin.fingertip_orientation_rad(self.q)
        tof_pred = self._predict_weart_tof_mm(self.q)

        return self.q.copy(), {
            "dt": dt,
            "closure_raw": float(sample.closure),
            "closure_used": closure,
            "closure_pred": self.predicted_closure(self.state),
            "distal_closure": float(self.state[1]),
            "theta_meas_deg": math.degrees(theta),
            "theta_pred_deg": math.degrees(theta_pred),
            "imu_trust": imu_trust,
            "tof_meas_mm": float(sample.tof_mm),
            "tof_pred_mm": float(tof_pred),
            "cost": float(best.cost),
        }


class WeartIndexMcpSynergyNode(Node):
    def __init__(self) -> None:
        super().__init__("weart_index_to_leap_mcp_synergy")

        self.declare_parameter("input_topic", DEFAULT_INPUT_TOPIC)
        self.declare_parameter("output_topic", DEFAULT_OUTPUT_TOPIC)
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

        self.declare_parameter("mcp_reference_deg", DEFAULT_MCP_REFERENCE_DEG)
        self.declare_parameter(
            "distal_closed_pose_deg", DEFAULT_DISTAL_CLOSED_POSE_DEG.tolist()
        )
        self.declare_parameter("distal_exponents", DEFAULT_DISTAL_EXPONENTS.tolist())
        self.declare_parameter("closure_sigma", DEFAULT_CLOSURE_SIGMA)
        self.declare_parameter("imu_sigma_deg", DEFAULT_IMU_SIGMA_DEG)
        self.declare_parameter("tof_sigma_mm", DEFAULT_TOF_SIGMA_MM)
        self.declare_parameter("smooth_sigma_deg", DEFAULT_SMOOTH_SIGMA_DEG)
        self.declare_parameter("tof_distance_scale", DEFAULT_TOF_DISTANCE_SCALE)
        self.declare_parameter("tof_palm_point_mm", DEFAULT_TOF_PALM_POINT_MM.tolist())
        self.declare_parameter("use_tof", True)

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        target_output_topic = str(self.get_parameter("target_output_topic").value)
        actual_output_topic = str(self.get_parameter("actual_output_topic").value)

        tof_palm_point_mm = np.asarray(
            self.get_parameter("tof_palm_point_mm").value, dtype=float
        )
        if tof_palm_point_mm.shape != (3,):
            raise ValueError("tof_palm_point_mm must contain x, y and z")

        self.kin = LeapIndexKinematics(
            str(self.get_parameter("urdf_path").value),
            tof_palm_point_mm=tof_palm_point_mm,
        )
        config = McpSynergyConfig(
            mcp_reference_rad=math.radians(
                float(self.get_parameter("mcp_reference_deg").value)
            ),
            distal_closed_pose_rad=np.deg2rad(
                np.asarray(
                    self.get_parameter("distal_closed_pose_deg").value,
                    dtype=float,
                )
            ),
            distal_exponents=np.asarray(
                self.get_parameter("distal_exponents").value, dtype=float
            ),
            closure_sigma=float(self.get_parameter("closure_sigma").value),
            imu_sigma_rad=math.radians(
                float(self.get_parameter("imu_sigma_deg").value)
            ),
            tof_sigma_mm=float(self.get_parameter("tof_sigma_mm").value),
            smooth_sigma_rad=math.radians(
                float(self.get_parameter("smooth_sigma_deg").value)
            ),
            tof_distance_scale=float(self.get_parameter("tof_distance_scale").value),
            use_tof=bool(self.get_parameter("use_tof").value),
        )
        self.estimator = McpAdaptiveSynergyEstimator(self.kin, config)

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
        self.get_logger().info(f"MCP-adaptive q output: {output_topic}")
        self.get_logger().info(f"Full commanded LEAP q: {target_output_topic}")
        self.get_logger().info(
            "state=[independent MCP, distal closure] | MCP reference=%.1f deg | "
            "distal closed=%s deg | IMU sigma=%.1f deg | ToF=%s"
            % (
                math.degrees(self.estimator.mcp_reference_rad),
                np.round(np.rad2deg(self.estimator.distal_closed_pose_rad), 1),
                math.degrees(config.imu_sigma_rad),
                "on" if config.use_tof else "off",
            )
        )
        self.get_logger().warning(
            "Calibration: keep HUMAN INDEX OPEN and PALM STILL until 100%."
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
                    f"Calibration: {100.0 * diag['calibration']:.0f}% "
                    "(index open + palm still)"
                )
            self.counter += 1
            return

        output = Float64MultiArray()
        output.data = [float(value) for value in q_target]
        self.output_pub.publish(output)

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
                "closure raw/used/pred=%.3f/%.3f/%.3f | distal=%.3f | "
                "q target=[%.1f, %.1f, %.1f] deg | "
                "cmd=[%.1f, %.1f, %.1f] deg | "
                "IMU meas/pred=%.1f/%.1f deg trust=%.2f | "
                "ToF meas/pred=%.1f/%.1f mm | cost=%.3f | motor123=%s"
                % (
                    diag["closure_raw"],
                    diag["closure_used"],
                    diag["closure_pred"],
                    diag["distal_closure"],
                    *np.rad2deg(q_target),
                    *np.rad2deg(q_command),
                    diag["theta_meas_deg"],
                    diag["theta_pred_deg"],
                    diag["imu_trust"],
                    diag["tof_meas_mm"],
                    diag["tof_pred_mm"],
                    diag["cost"],
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
        node = WeartIndexMcpSynergyNode()
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
