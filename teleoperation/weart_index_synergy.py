#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""ROS 2: WEART index -> one-DoF sensor fusion -> LEAP index synergy.

The WEART thimble does not provide enough independent measurements to recover
three human joint angles uniquely. This estimator therefore solves only for a
normalized finger closure ``c`` and converts it to a configurable LEAP posture:

    q_i(c) = q_closed_i * c ** exponent_i

WEART closure is the strong observation. The fingertip IMU and ToF are weak
corrections. The ToF-vs-closure slope is learned online because the LEAP URDF
does not contain the human sensor mounting geometry.

Keep the human index open and the hand still during startup calibration. Run in
dry mode first; hardware output is disabled by default.
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
        DirectLeapIndexDriver,
        LeapIndexKinematics,
        WeartSample,
        angle_error,
        parse_weart_line,
        wrap_pi,
    )


DEFAULT_INPUT_TOPIC = "/weart/index/raw"
DEFAULT_OUTPUT_TOPIC = "/weart/index/synergy_leap_joints"
DEFAULT_TARGET_OUTPUT_TOPIC = "/leap/target_joint_positions"
DEFAULT_ACTUAL_OUTPUT_TOPIC = "/leap/actual_joint_positions"
DEFAULT_URDF_PATH = str(Path(__file__).with_name("model.urdf"))
DEFAULT_ACTUAL_FEEDBACK_PERIOD_S = 0.05

# Conservative full-closure posture. Values are clipped to the URDF limits.
DEFAULT_CLOSED_POSE_DEG = np.array([85.0, 95.0, 65.0], dtype=float)
DEFAULT_SYNERGY_EXPONENTS = np.array([0.85, 1.05, 1.25], dtype=float)

CALIBRATION_SAMPLES = 30
CALIBRATION_MAX_CLOSURE = 0.10
CALIBRATION_MAX_GYRO_NORM_DEG_S = 15.0

MIN_DT_S = 0.002
MAX_DT_S = 0.25

# Sensor frame: keep aligned with the other two index estimators.
GYRO_FLEX_AXIS = 2
GYRO_FLEX_SIGN = +1.0
ACC_ANGLE_NUM_AXIS = 0
ACC_ANGLE_DEN_AXIS = 1
ACC_ANGLE_NUM_SIGN = +1.0
ACC_ANGLE_DEN_SIGN = +1.0
ACC_CORRECTION_GAIN = 0.04
ACC_NORM_SIGMA_G = 0.12

DEFAULT_CLOSURE_SIGMA = 0.055
DEFAULT_IMU_SIGMA_DEG = 25.0
DEFAULT_TOF_SIGMA_MM = 12.0
DEFAULT_SMOOTH_SIGMA = 0.18

# Automatic ToF regression becomes active only after meaningful finger motion.
TOF_FIT_MIN_CLOSURE = 0.15
TOF_FIT_REQUIRED_SAMPLES = 20
TOF_FIT_REQUIRED_SPAN = 0.60
TOF_FIT_MIN_DELTA_MM = 3.0
TOF_FIT_MAX_ABS_DELTA_MM = 250.0
TOF_FIT_FORGETTING = 0.995


@dataclass(frozen=True)
class SynergyConfig:
    closed_pose_rad: np.ndarray
    exponents: np.ndarray
    closure_sigma: float = DEFAULT_CLOSURE_SIGMA
    imu_sigma_rad: float = math.radians(DEFAULT_IMU_SIGMA_DEG)
    use_imu: bool = True
    tof_sigma_mm: float = DEFAULT_TOF_SIGMA_MM
    smooth_sigma: float = DEFAULT_SMOOTH_SIGMA
    tof_delta_closed_mm: float = 0.0

    def validate(self) -> None:
        if self.closed_pose_rad.shape != (3,):
            raise ValueError("closed_pose_rad must contain MCP, PIP and DIP")
        if self.exponents.shape != (3,) or np.any(self.exponents <= 0.0):
            raise ValueError("synergy_exponents must contain three positive values")
        if (
            min(
                self.closure_sigma,
                self.imu_sigma_rad,
                self.tof_sigma_mm,
                self.smooth_sigma,
            )
            <= 0.0
        ):
            raise ValueError("all estimator sigmas must be positive")


class ClosureSynergyEstimator:
    """Estimate one observable closure coordinate, then apply a LEAP synergy."""

    def __init__(
        self,
        kinematics: LeapIndexKinematics,
        config: SynergyConfig,
    ) -> None:
        config.validate()
        self.kin = kinematics
        self.config = config
        self.closed_pose_rad = np.clip(
            np.asarray(config.closed_pose_rad, dtype=float),
            self.kin.q_lower,
            self.kin.q_upper,
        )

        self.c = 0.0
        self.theta = 0.0
        self.last_ts_ms: int | None = None

        self.acc_open_angle: float | None = None
        self.gyro_bias_deg_s = 0.0
        self.open_tof_mm: float | None = None
        self.open_closure = 0.0

        self._calib_acc_angles: list[float] = []
        self._calib_gyro_flex: list[float] = []
        self._calib_tof: list[float] = []
        self._calib_closure: list[float] = []

        self._tof_fit_xx = 0.0
        self._tof_fit_xy = 0.0
        self._tof_fit_samples = 0
        self._tof_fit_max_closure = 0.0

        self.calibrated = False

    def q_from_closure(self, closure: float) -> np.ndarray:
        c = float(np.clip(closure, 0.0, 1.0))
        q = self.closed_pose_rad * np.power(c, self.config.exponents)
        return np.clip(q, self.kin.q_lower, self.kin.q_upper)

    def _raw_acc_angle(self, acc: np.ndarray) -> float:
        num = ACC_ANGLE_NUM_SIGN * float(acc[ACC_ANGLE_NUM_AXIS])
        den = ACC_ANGLE_DEN_SIGN * float(acc[ACC_ANGLE_DEN_AXIS])
        return math.atan2(num, den)

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
            self.open_tof_mm = float(np.median(self._calib_tof))
            self.open_closure = float(np.median(self._calib_closure))
            self.c = 0.0
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

    def _normalized_closure(self, raw_closure: float) -> float:
        denominator = max(1.0 - self.open_closure, 1e-6)
        value = (float(raw_closure) - self.open_closure) / denominator
        return float(np.clip(value, 0.0, 1.0))

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

    def _update_tof_fit(self, c_weart: float, tof_mm: float) -> None:
        if self.config.tof_delta_closed_mm != 0.0:
            return
        assert self.open_tof_mm is not None
        delta = float(tof_mm) - self.open_tof_mm
        if (
            c_weart < TOF_FIT_MIN_CLOSURE
            or float(tof_mm) <= 0.0
            or not np.isfinite(delta)
            or abs(delta) > TOF_FIT_MAX_ABS_DELTA_MM
        ):
            return

        self._tof_fit_xx = TOF_FIT_FORGETTING * self._tof_fit_xx + c_weart * c_weart
        self._tof_fit_xy = TOF_FIT_FORGETTING * self._tof_fit_xy + c_weart * delta
        self._tof_fit_samples += 1
        self._tof_fit_max_closure = max(self._tof_fit_max_closure, c_weart)

    def _tof_delta_closed(self) -> float | None:
        configured = float(self.config.tof_delta_closed_mm)
        if configured != 0.0:
            return configured
        if (
            self._tof_fit_samples < TOF_FIT_REQUIRED_SAMPLES
            or self._tof_fit_max_closure < TOF_FIT_REQUIRED_SPAN
            or self._tof_fit_xx <= 1e-9
        ):
            return None
        slope = self._tof_fit_xy / self._tof_fit_xx
        if abs(slope) < TOF_FIT_MIN_DELTA_MM:
            return None
        return float(slope)

    def _residual(
        self,
        c_array: np.ndarray,
        c_weart: float,
        theta_meas: float,
        imu_trust: float,
        tof_meas_mm: float,
        tof_delta_closed_mm: float | None,
        c_prev: float,
    ) -> np.ndarray:
        c = float(c_array[0])
        q = self.q_from_closure(c)

        residuals = [
            (c - c_weart) / self.config.closure_sigma,
            (c - c_prev) / self.config.smooth_sigma,
        ]

        if self.config.use_imu:
            # A rotating palm is indistinguishable from finger motion with one
            # IMU; low weighting limits this unavoidable source of error.
            theta_pred = self.kin.fingertip_orientation_rad(q)
            residuals.append(
                math.sqrt(imu_trust)
                * angle_error(theta_pred, theta_meas)
                / self.config.imu_sigma_rad
            )

        if tof_delta_closed_mm is not None:
            assert self.open_tof_mm is not None
            tof_pred = self.open_tof_mm + tof_delta_closed_mm * c
            residuals.append((tof_pred - float(tof_meas_mm)) / self.config.tof_sigma_mm)

        return np.asarray(residuals, dtype=float)

    def update(
        self,
        sample: WeartSample,
    ) -> tuple[np.ndarray | None, dict[str, float]]:
        dt = self._compute_dt(sample.ts_ms)
        if not self.calibrated:
            progress = self._calibrate(sample)
            return None, {"calibration": progress, "dt": dt}

        c_weart = self._normalized_closure(sample.closure)
        theta, imu_trust = self._update_theta(sample, dt)
        self._update_tof_fit(c_weart, sample.tof_mm)
        tof_delta_closed = self._tof_delta_closed()
        tof_valid = np.isfinite(sample.tof_mm) and sample.tof_mm > 0.0
        tof_delta_for_sample = tof_delta_closed if tof_valid else None
        c_prev = self.c

        seeds = (c_weart, c_prev, 0.5 * (c_weart + c_prev))
        best = None
        for seed in seeds:
            result = least_squares(
                self._residual,
                x0=np.array([np.clip(seed, 1e-8, 1.0 - 1e-8)]),
                bounds=(np.array([0.0]), np.array([1.0])),
                args=(
                    c_weart,
                    theta,
                    imu_trust,
                    sample.tof_mm,
                    tof_delta_for_sample,
                    c_prev,
                ),
                loss="soft_l1",
                f_scale=1.0,
                max_nfev=40,
            )
            if best is None or result.cost < best.cost:
                best = result

        assert best is not None
        self.c = float(best.x[0])
        q = self.q_from_closure(self.c)
        theta_pred = (
            self.kin.fingertip_orientation_rad(q) if self.config.use_imu else math.nan
        )

        tof_pred = math.nan
        if tof_delta_for_sample is not None:
            assert self.open_tof_mm is not None
            tof_pred = self.open_tof_mm + tof_delta_for_sample * self.c

        return q, {
            "dt": dt,
            "closure_raw": float(sample.closure),
            "closure_weart": c_weart,
            "closure_fused": self.c,
            "theta_meas_deg": math.degrees(theta),
            "theta_pred_deg": math.degrees(theta_pred),
            "imu_trust": imu_trust,
            "tof_meas_mm": float(sample.tof_mm),
            "tof_pred_mm": tof_pred,
            "tof_delta_closed_mm": (
                math.nan if tof_delta_closed is None else tof_delta_closed
            ),
            "cost": float(best.cost),
        }


class WeartIndexSynergyNode(Node):
    def __init__(self) -> None:
        super().__init__("weart_index_to_leap_synergy")

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

        self.declare_parameter("closed_pose_deg", DEFAULT_CLOSED_POSE_DEG.tolist())
        self.declare_parameter("synergy_exponents", DEFAULT_SYNERGY_EXPONENTS.tolist())
        self.declare_parameter("closure_sigma", DEFAULT_CLOSURE_SIGMA)
        self.declare_parameter("imu_sigma_deg", DEFAULT_IMU_SIGMA_DEG)
        self.declare_parameter("tof_sigma_mm", DEFAULT_TOF_SIGMA_MM)
        self.declare_parameter("smooth_sigma", DEFAULT_SMOOTH_SIGMA)
        self.declare_parameter("tof_delta_closed_mm", 0.0)

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        target_output_topic = str(self.get_parameter("target_output_topic").value)
        actual_output_topic = str(self.get_parameter("actual_output_topic").value)
        urdf_path = str(self.get_parameter("urdf_path").value)

        self.kin = LeapIndexKinematics(
            urdf_path,
            tof_palm_point_mm=np.zeros(3, dtype=float),
        )
        config = SynergyConfig(
            closed_pose_rad=np.deg2rad(
                np.asarray(self.get_parameter("closed_pose_deg").value, dtype=float)
            ),
            exponents=np.asarray(
                self.get_parameter("synergy_exponents").value, dtype=float
            ),
            closure_sigma=float(self.get_parameter("closure_sigma").value),
            imu_sigma_rad=math.radians(
                float(self.get_parameter("imu_sigma_deg").value)
            ),
            tof_sigma_mm=float(self.get_parameter("tof_sigma_mm").value),
            smooth_sigma=float(self.get_parameter("smooth_sigma").value),
            tof_delta_closed_mm=float(self.get_parameter("tof_delta_closed_mm").value),
        )
        self.estimator = ClosureSynergyEstimator(self.kin, config)

        enable_hardware = bool(self.get_parameter("enable_hardware").value)
        self.leap = DirectLeapIndexDriver(
            self.kin,
            enabled=enable_hardware,
            port=str(self.get_parameter("leap_port").value),
            kp=float(self.get_parameter("kp").value),
            ki=float(self.get_parameter("ki").value),
            kd=float(self.get_parameter("kd").value),
        )
        self.leap.connect()

        self.sub = self.create_subscription(String, input_topic, self._on_weart, 20)
        self.pub = self.create_publisher(Float64MultiArray, output_topic, 20)
        self.target_pub = self.create_publisher(
            Float64MultiArray, target_output_topic, 20
        )
        self.actual_pub = self.create_publisher(
            Float64MultiArray, actual_output_topic, 20
        )
        self.actual_feedback_failures = 0
        self.actual_timer = None
        if enable_hardware:
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
        self.get_logger().info(f"Synergy LEAP q output: {output_topic}")
        self.get_logger().info(f"Full commanded LEAP q: {target_output_topic}")
        self.get_logger().info(
            "closed pose [deg]=%s | exponents=%s"
            % (
                np.round(np.rad2deg(self.estimator.closed_pose_rad), 2).tolist(),
                np.round(config.exponents, 3).tolist(),
            )
        )
        self.get_logger().warning(
            "Calibration: keep HUMAN INDEX OPEN and HAND STILL until 100%."
        )
        if enable_hardware:
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
                    "(finger open + hand still)"
                )
            self.counter += 1
            return

        out = Float64MultiArray()
        out.data = [float(v) for v in q_target]
        self.pub.publish(out)

        try:
            q_cmd, motor_qpos = self.leap.command(q_target, float(diag["dt"]))
        except Exception as exc:
            self.get_logger().error(f"LEAP command failed: {exc}")
            return

        commanded = Float64MultiArray()
        commanded.data = [float(value) for value in self.leap.current_sim_qpos]
        self.target_pub.publish(commanded)

        if self.counter % 5 == 0:
            self.get_logger().info(
                "closure raw/used/fused=%.3f/%.3f/%.3f | "
                "q target=[%.1f, %.1f, %.1f] deg | "
                "cmd=[%.1f, %.1f, %.1f] deg | "
                "IMU meas/pred=%.1f/%.1f deg | "
                "ToF meas/pred=%.1f/%.1f mm | dToF=%.1f | motor123=%s"
                % (
                    diag["closure_raw"],
                    diag["closure_weart"],
                    diag["closure_fused"],
                    *np.rad2deg(q_target),
                    *np.rad2deg(q_cmd),
                    diag["theta_meas_deg"],
                    diag["theta_pred_deg"],
                    diag["tof_meas_mm"],
                    diag["tof_pred_mm"],
                    diag["tof_delta_closed_mm"],
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
        node = WeartIndexSynergyNode()
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
