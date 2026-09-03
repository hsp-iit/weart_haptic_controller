#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""ROS 2: WEART thumb/index/middle -> LEAP Hand synergy control.

Each finger uses the one-dimensional closure fusion implemented in
``weart_index_synergy.py``. The thumb additionally maps the published WEART
adduction value to LEAP joint 12:

    index closure  -> joints 1, 2, 3       (joint 0 fixed at zero)
    middle closure -> joints 5, 6, 7       (joint 4 fixed at zero)
    thumb adduction -> joint 12
    thumb closure   -> joints 13, 14, 15

Hardware output is disabled by default. Keep all three fingers open and still
during startup calibration.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

if __package__:
    from .weart_index_synergy import (
        DEFAULT_CLOSURE_SIGMA,
        DEFAULT_SMOOTH_SIGMA,
        DEFAULT_TOF_SIGMA_MM,
        ClosureSynergyEstimator,
        SynergyConfig,
    )
    from .weart_leap_index_ik import (
        COMMAND_LOW_PASS_ALPHA,
        COMMAND_MAX_SPEED_RAD_S,
        DEFAULT_ENABLE_HARDWARE,
        DEFAULT_KD,
        DEFAULT_KI,
        DEFAULT_KP,
        DEFAULT_LEAP_PORT,
        MIN_DT_S,
        UrdfJoint,
        _axis_rotation,
        _load_leap_cpp,
        _origin_transform,
        _parse_vec3,
        _signed_angle_in_plane,
        autodetect_leap_port,
        leap_sim_to_motor,
        parse_weart_line,
        wrap_pi,
    )
else:
    from weart_index_synergy import (
        DEFAULT_CLOSURE_SIGMA,
        DEFAULT_SMOOTH_SIGMA,
        DEFAULT_TOF_SIGMA_MM,
        ClosureSynergyEstimator,
        SynergyConfig,
    )
    from weart_leap_index_ik import (
        COMMAND_LOW_PASS_ALPHA,
        COMMAND_MAX_SPEED_RAD_S,
        DEFAULT_ENABLE_HARDWARE,
        DEFAULT_KD,
        DEFAULT_KI,
        DEFAULT_KP,
        DEFAULT_LEAP_PORT,
        MIN_DT_S,
        UrdfJoint,
        _axis_rotation,
        _load_leap_cpp,
        _origin_transform,
        _parse_vec3,
        _signed_angle_in_plane,
        autodetect_leap_port,
        leap_sim_to_motor,
        parse_weart_line,
        wrap_pi,
    )


DEFAULT_URDF_PATH = str(Path(__file__).with_name("model.urdf"))
DEFAULT_TARGET_OUTPUT_TOPIC = "/leap/target_joint_positions"
DEFAULT_ACTUAL_OUTPUT_TOPIC = "/leap/actual_joint_positions"
DEFAULT_ACTUAL_FEEDBACK_PERIOD_S = 0.05

DEFAULT_INPUT_TOPICS = {
    "index": "/weart/index/raw",
    "middle": "/weart/middle/raw",
    "thumb": "/weart/thumb/raw",
}
DEFAULT_OUTPUT_TOPICS = {
    "index": "/weart/index/synergy_leap_joints",
    "middle": "/weart/middle/synergy_leap_joints",
    "thumb": "/weart/thumb/synergy_leap_joints",
}

DEFAULT_CLOSED_POSE_DEG = {
    "index": np.array([85.0, 95.0, 65.0], dtype=float),
    "middle": np.array([85.0, 95.0, 65.0], dtype=float),
    "thumb": np.array([75.0, 70.0, 55.0], dtype=float),
}
DEFAULT_SYNERGY_EXPONENTS = {
    "index": np.array([0.85, 1.05, 1.25], dtype=float),
    "middle": np.array([0.85, 1.05, 1.25], dtype=float),
    "thumb": np.array([0.80, 1.00, 1.25], dtype=float),
}
DEFAULT_IMU_SIGMA_DEG = {
    "index": 25.0,
    "middle": 25.0,
    "thumb": 45.0,
}
DEFAULT_USE_IMU = {
    "index": True,
    "middle": True,
    # Thumb axes are not planar and adduction also changes IMU orientation.
    "thumb": False,
}

# adduction=0 -> first angle, adduction=1 -> second angle.
DEFAULT_THUMB_ADDUCTION_RANGE_DEG = np.array([0.0, 60.0], dtype=float)


@dataclass(frozen=True)
class FingerSpec:
    name: str
    chain_joint_names: tuple[str, ...]
    flexion_joint_names: tuple[str, str, str]
    side_joint_name: str
    tip_joint_name: str


FINGER_SPECS = {
    "index": FingerSpec(
        name="index",
        chain_joint_names=("1", "0", "2", "3"),
        flexion_joint_names=("1", "2", "3"),
        side_joint_name="0",
        tip_joint_name="index_tip",
    ),
    "middle": FingerSpec(
        name="middle",
        chain_joint_names=("5", "4", "6", "7"),
        flexion_joint_names=("5", "6", "7"),
        side_joint_name="4",
        tip_joint_name="middle_tip",
    ),
    "thumb": FingerSpec(
        name="thumb",
        chain_joint_names=("12", "13", "14", "15"),
        flexion_joint_names=("13", "14", "15"),
        side_joint_name="12",
        tip_joint_name="thumb_tip",
    ),
}


@dataclass(frozen=True)
class LeapHandModel:
    urdf_path: str
    joints: dict[str, UrdfJoint]
    sim_lower: np.ndarray
    sim_upper: np.ndarray
    velocity_limits: np.ndarray

    @property
    def motor_count(self) -> int:
        return len(self.sim_lower)


def load_leap_hand_model(urdf_path: str) -> LeapHandModel:
    path = Path(urdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LEAP URDF not found: {path}")

    root = ET.parse(path).getroot()
    joints: dict[str, UrdfJoint] = {}
    numerical: dict[int, UrdfJoint] = {}

    for elem in root.findall("joint"):
        name = str(elem.get("name", ""))
        parent = elem.find("parent")
        child = elem.find("child")
        if (
            not name
            or parent is None
            or child is None
            or parent.get("link") is None
            or child.get("link") is None
        ):
            continue

        origin = elem.find("origin")
        axis = elem.find("axis")
        limit = elem.find("limit")
        joint_type = str(elem.get("type", "fixed"))

        lower = 0.0
        upper = 0.0
        velocity = math.inf
        if limit is not None:
            lower = float(limit.get("lower", "0"))
            upper = float(limit.get("upper", "0"))
            velocity = float(limit.get("velocity", "inf"))

        joint = UrdfJoint(
            name=name,
            joint_type=joint_type,
            parent=str(parent.get("link")),
            child=str(child.get("link")),
            origin_T=_origin_transform(
                _parse_vec3(origin.get("xyz") if origin is not None else None),
                _parse_vec3(origin.get("rpy") if origin is not None else None),
            ),
            axis=_parse_vec3(axis.get("xyz") if axis is not None else "0 0 1"),
            lower=lower,
            upper=upper,
            velocity=velocity,
        )
        joints[name] = joint
        if name.isdecimal():
            numerical[int(name)] = joint

    required = {
        joint_name
        for spec in FINGER_SPECS.values()
        for joint_name in (*spec.chain_joint_names, spec.tip_joint_name)
    }
    missing = sorted(required.difference(joints))
    if missing:
        raise ValueError(f"Missing expected LEAP URDF joints: {missing}")

    expected_indices = list(range(max(numerical) + 1)) if numerical else []
    if sorted(numerical) != expected_indices:
        raise ValueError(
            "LEAP numerical joint names must be contiguous from zero; found "
            f"{sorted(numerical)}"
        )

    for spec in FINGER_SPECS.values():
        chain = [joints[name] for name in spec.chain_joint_names]
        if chain[0].parent != "palm_lower":
            raise ValueError(f"{spec.name} chain does not start at palm_lower")
        for previous, current in zip(chain, chain[1:], strict=False):
            if previous.child != current.parent:
                raise ValueError(
                    f"Broken {spec.name} URDF chain between "
                    f"{previous.name} and {current.name}"
                )
        tip = joints[spec.tip_joint_name]
        if chain[-1].child != tip.parent or tip.joint_type != "fixed":
            raise ValueError(f"Invalid fixed tip for {spec.name}")

    return LeapHandModel(
        urdf_path=str(path),
        joints=joints,
        sim_lower=np.array([numerical[i].lower for i in expected_indices], dtype=float),
        sim_upper=np.array([numerical[i].upper for i in expected_indices], dtype=float),
        velocity_limits=np.array(
            [numerical[i].velocity for i in expected_indices], dtype=float
        ),
    )


class LeapFingerKinematics:
    """URDF forward kinematics adapter used by ClosureSynergyEstimator."""

    def __init__(self, model: LeapHandModel, spec: FingerSpec) -> None:
        self.model = model
        self.spec = spec
        self.motor_indices = tuple(int(name) for name in spec.flexion_joint_names)
        self.side_motor_index = int(spec.side_joint_name)
        self.q_lower = self.model.sim_lower[list(self.motor_indices)].copy()
        self.q_upper = self.model.sim_upper[list(self.motor_indices)].copy()
        self.q_velocity = self.model.velocity_limits[list(self.motor_indices)].copy()

        open_fk = self.forward(np.zeros(3, dtype=float), debug=True)
        self.open_distal_vector = open_fk["distal_vector"]
        self.plane_normal = open_fk["first_flex_axis"]
        self.plane_normal /= np.linalg.norm(self.plane_normal)
        self.flex_axis_parallel_cos = np.array(
            [
                abs(float(np.dot(self.plane_normal, axis)))
                for axis in open_fk["flex_axes"][1:]
            ],
            dtype=float,
        )

    @staticmethod
    def _apply_joint(
        parent_T: np.ndarray,
        joint: UrdfJoint,
        q: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        joint_T = parent_T @ joint.origin_T
        position = joint_T[:3, 3].copy()
        axis = joint_T[:3, :3] @ joint.axis
        axis /= np.linalg.norm(axis)
        if joint.joint_type == "fixed":
            return joint_T, position, axis
        return joint_T @ _axis_rotation(joint.axis, q), position, axis

    def forward(self, q: np.ndarray, *, debug: bool = False):
        q_values = np.asarray(q, dtype=float)
        if q_values.shape != (3,):
            raise ValueError(f"Expected three {self.spec.name} flexion joints")
        active_q = dict(zip(self.spec.flexion_joint_names, q_values, strict=True))

        transform = np.eye(4, dtype=float)
        flex_axes: list[np.ndarray] = []
        last_flex_position: np.ndarray | None = None
        for name in self.spec.chain_joint_names:
            joint = self.model.joints[name]
            angle = float(active_q.get(name, 0.0))
            transform, position, axis = self._apply_joint(transform, joint, angle)
            if name in active_q:
                flex_axes.append(axis)
                last_flex_position = position

        tip_joint = self.model.joints[self.spec.tip_joint_name]
        tip_T = transform @ tip_joint.origin_T
        tip_position = tip_T[:3, 3].copy()
        assert last_flex_position is not None

        if not debug:
            return tip_position
        return {
            "tip_position": tip_position,
            "distal_vector": tip_position - last_flex_position,
            "first_flex_axis": flex_axes[0].copy(),
            "flex_axes": flex_axes,
        }

    def fingertip_orientation_rad(self, q: np.ndarray) -> float:
        fk = self.forward(q, debug=True)
        return wrap_pi(
            _signed_angle_in_plane(
                self.open_distal_vector,
                fk["distal_vector"],
                self.plane_normal,
            )
        )


@dataclass
class FingerRuntime:
    name: str
    kinematics: LeapFingerKinematics
    estimator: ClosureSynergyEstimator
    publisher: Any
    counter: int = 0


class DirectLeapThreeFingerDriver:
    def __init__(
        self,
        model: LeapHandModel,
        kinematics: dict[str, LeapFingerKinematics],
        *,
        thumb_adduction_range_rad: np.ndarray,
        enabled: bool,
        port: str,
        kp: float,
        ki: float,
        kd: float,
    ) -> None:
        self.model = model
        self.kinematics = kinematics
        self.enabled = bool(enabled)
        self.requested_port = str(port)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

        adduction_range = np.asarray(thumb_adduction_range_rad, dtype=float)
        if adduction_range.shape != (2,):
            raise ValueError("thumb_adduction_range_deg must contain two values")
        thumb_side = self.kinematics["thumb"].side_motor_index
        self.thumb_adduction_range_rad = np.clip(
            adduction_range,
            self.model.sim_lower[thumb_side],
            self.model.sim_upper[thumb_side],
        )

        self.current_sim_qpos = np.zeros(self.model.motor_count, dtype=float)
        self.filtered_flexion = {
            name: np.zeros(3, dtype=float) for name in self.kinematics
        }
        self.filtered_thumb_adduction = 0.0
        self._ctrl = None
        self.port: str | None = None

    def connect(self) -> None:
        mappings = []
        for name, kin in self.kinematics.items():
            mappings.append(f"{name}={kin.motor_indices}")
        mappings.append(f"thumb-adduction={self.kinematics['thumb'].side_motor_index}")
        print("[LEAP] synergy mappings: " + ", ".join(mappings), flush=True)

        if not self.enabled:
            print("[LEAP] hardware output DISABLED (dry-run).", flush=True)
            return

        self.port = autodetect_leap_port(self.requested_port)
        leap_cpp = _load_leap_cpp()
        self._ctrl = leap_cpp.LeapController(self.port)
        self._ctrl.connect()
        self._ctrl.setGains(int(self.kp), int(self.ki), int(self.kd))
        self._ctrl.set_leap(leap_sim_to_motor(self.current_sim_qpos))
        print(f"[LEAP] connected on {self.port}; sent open pose.", flush=True)

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
        if motor_qpos.size != self.model.motor_count:
            raise ValueError(
                f"LEAP returned {motor_qpos.size} joints, "
                f"expected {self.model.motor_count}"
            )
        if not np.all(np.isfinite(motor_qpos)):
            raise ValueError("LEAP returned non-finite joint positions")
        return motor_qpos - math.pi

    @staticmethod
    def _filtered_target(
        previous: np.ndarray,
        target: np.ndarray,
        max_speed: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        low_pass = previous + COMMAND_LOW_PASS_ALPHA * (target - previous)
        max_step = max_speed * max(float(dt), MIN_DT_S)
        return previous + np.clip(low_pass - previous, -max_step, max_step)

    def thumb_adduction_target(self, adduction: float) -> float:
        value = float(np.clip(adduction, 0.0, 1.0))
        q_min, q_max = self.thumb_adduction_range_rad
        return float(q_min + value * (q_max - q_min))

    def command(
        self,
        finger_name: str,
        q_flexion: np.ndarray,
        dt: float,
        *,
        thumb_adduction: float | None = None,
    ) -> tuple[np.ndarray, float | None, np.ndarray]:
        kin = self.kinematics[finger_name]
        target = np.clip(np.asarray(q_flexion, dtype=float), kin.q_lower, kin.q_upper)
        max_speed = np.minimum(COMMAND_MAX_SPEED_RAD_S, kin.q_velocity)
        filtered = self._filtered_target(
            self.filtered_flexion[finger_name], target, max_speed, dt
        )
        self.filtered_flexion[finger_name] = filtered

        for motor_index, value in zip(kin.motor_indices, filtered, strict=True):
            self.current_sim_qpos[motor_index] = float(value)

        side_value: float | None = None
        if finger_name == "thumb":
            if thumb_adduction is None:
                raise ValueError("thumb command requires the WEART adduction value")
            side_index = kin.side_motor_index
            side_target = self.thumb_adduction_target(thumb_adduction)
            side_speed = min(
                COMMAND_MAX_SPEED_RAD_S,
                float(self.model.velocity_limits[side_index]),
            )
            side_filtered = self._filtered_target(
                np.array([self.filtered_thumb_adduction]),
                np.array([side_target]),
                np.array([side_speed]),
                dt,
            )
            self.filtered_thumb_adduction = float(side_filtered[0])
            self.current_sim_qpos[side_index] = self.filtered_thumb_adduction
            side_value = self.filtered_thumb_adduction
        else:
            self.current_sim_qpos[kin.side_motor_index] = 0.0

        motor_qpos = leap_sim_to_motor(self.current_sim_qpos)
        if self.enabled:
            if self._ctrl is None:
                raise RuntimeError(
                    "LEAP output enabled but controller is not connected"
                )
            self._ctrl.set_leap(motor_qpos)
        return filtered.copy(), side_value, motor_qpos.copy()


class WeartThreeFingersSynergyNode(Node):
    def __init__(self) -> None:
        super().__init__("weart_three_fingers_to_leap_synergy")

        self.declare_parameter("urdf_path", DEFAULT_URDF_PATH)
        self.declare_parameter("enable_hardware", DEFAULT_ENABLE_HARDWARE)
        self.declare_parameter("leap_port", DEFAULT_LEAP_PORT)
        self.declare_parameter("target_output_topic", DEFAULT_TARGET_OUTPUT_TOPIC)
        self.declare_parameter("actual_output_topic", DEFAULT_ACTUAL_OUTPUT_TOPIC)
        self.declare_parameter(
            "actual_feedback_period_s", DEFAULT_ACTUAL_FEEDBACK_PERIOD_S
        )
        self.declare_parameter("kp", DEFAULT_KP)
        self.declare_parameter("ki", DEFAULT_KI)
        self.declare_parameter("kd", DEFAULT_KD)
        self.declare_parameter("closure_sigma", DEFAULT_CLOSURE_SIGMA)
        self.declare_parameter("tof_sigma_mm", DEFAULT_TOF_SIGMA_MM)
        self.declare_parameter("smooth_sigma", DEFAULT_SMOOTH_SIGMA)
        self.declare_parameter(
            "thumb_adduction_range_deg",
            DEFAULT_THUMB_ADDUCTION_RANGE_DEG.tolist(),
        )

        for name in FINGER_SPECS:
            self.declare_parameter(f"{name}_input_topic", DEFAULT_INPUT_TOPICS[name])
            self.declare_parameter(f"{name}_output_topic", DEFAULT_OUTPUT_TOPICS[name])
            self.declare_parameter(
                f"{name}_closed_pose_deg", DEFAULT_CLOSED_POSE_DEG[name].tolist()
            )
            self.declare_parameter(
                f"{name}_synergy_exponents",
                DEFAULT_SYNERGY_EXPONENTS[name].tolist(),
            )
            self.declare_parameter(f"{name}_imu_sigma_deg", DEFAULT_IMU_SIGMA_DEG[name])
            self.declare_parameter(f"{name}_use_imu", DEFAULT_USE_IMU[name])
            self.declare_parameter(f"{name}_tof_delta_closed_mm", 0.0)

        self.model = load_leap_hand_model(str(self.get_parameter("urdf_path").value))
        self.kinematics = {
            name: LeapFingerKinematics(self.model, spec)
            for name, spec in FINGER_SPECS.items()
        }

        self.leap = DirectLeapThreeFingerDriver(
            self.model,
            self.kinematics,
            thumb_adduction_range_rad=np.deg2rad(
                np.asarray(
                    self.get_parameter("thumb_adduction_range_deg").value,
                    dtype=float,
                )
            ),
            enabled=bool(self.get_parameter("enable_hardware").value),
            port=str(self.get_parameter("leap_port").value),
            kp=float(self.get_parameter("kp").value),
            ki=float(self.get_parameter("ki").value),
            kd=float(self.get_parameter("kd").value),
        )
        self.leap.connect()

        target_output_topic = str(self.get_parameter("target_output_topic").value)
        actual_output_topic = str(self.get_parameter("actual_output_topic").value)
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

        closure_sigma = float(self.get_parameter("closure_sigma").value)
        tof_sigma_mm = float(self.get_parameter("tof_sigma_mm").value)
        smooth_sigma = float(self.get_parameter("smooth_sigma").value)

        self.fingers: dict[str, FingerRuntime] = {}
        self.finger_subscriptions = []
        for name in FINGER_SPECS:
            config = SynergyConfig(
                closed_pose_rad=np.deg2rad(
                    np.asarray(
                        self.get_parameter(f"{name}_closed_pose_deg").value,
                        dtype=float,
                    )
                ),
                exponents=np.asarray(
                    self.get_parameter(f"{name}_synergy_exponents").value,
                    dtype=float,
                ),
                closure_sigma=closure_sigma,
                imu_sigma_rad=math.radians(
                    float(self.get_parameter(f"{name}_imu_sigma_deg").value)
                ),
                use_imu=bool(self.get_parameter(f"{name}_use_imu").value),
                tof_sigma_mm=tof_sigma_mm,
                smooth_sigma=smooth_sigma,
                tof_delta_closed_mm=float(
                    self.get_parameter(f"{name}_tof_delta_closed_mm").value
                ),
            )
            estimator = ClosureSynergyEstimator(self.kinematics[name], config)
            output_topic = str(self.get_parameter(f"{name}_output_topic").value)
            input_topic = str(self.get_parameter(f"{name}_input_topic").value)
            publisher = self.create_publisher(Float64MultiArray, output_topic, 20)
            subscription = self.create_subscription(
                String,
                input_topic,
                partial(self._on_weart, name),
                20,
            )
            self.finger_subscriptions.append(subscription)
            self.fingers[name] = FingerRuntime(
                name=name,
                kinematics=self.kinematics[name],
                estimator=estimator,
                publisher=publisher,
            )
            self.get_logger().info(
                f"{name}: {input_topic} -> {output_topic} | "
                f"flexion motors={self.kinematics[name].motor_indices} | "
                f"IMU={'on' if config.use_imu else 'off'}"
            )

        adduction_deg = np.rad2deg(self.leap.thumb_adduction_range_rad)
        self.get_logger().info(
            "thumb adduction [0,1] -> joint 12 [%.1f, %.1f] deg"
            % (adduction_deg[0], adduction_deg[1])
        )
        self.get_logger().info(f"Full commanded LEAP q: {target_output_topic}")
        self.get_logger().warning(
            "Calibration: keep thumb, index and middle OPEN and STILL until 100%."
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

    def _on_weart(self, finger_name: str, msg: String) -> None:
        runtime = self.fingers[finger_name]
        try:
            sample = parse_weart_line(msg.data)
            q_target, diag = runtime.estimator.update(sample)
        except Exception as exc:
            self.get_logger().warning(f"{finger_name} WEART/estimator error: {exc}")
            return

        if q_target is None:
            if runtime.counter % 5 == 0 or diag["calibration"] >= 1.0:
                self.get_logger().info(
                    f"{finger_name} calibration: " f"{100.0 * diag['calibration']:.0f}%"
                )
            runtime.counter += 1
            return

        thumb_adduction = sample.adduction if finger_name == "thumb" else None
        thumb_side_target = (
            self.leap.thumb_adduction_target(sample.adduction)
            if finger_name == "thumb"
            else None
        )

        output = Float64MultiArray()
        if thumb_side_target is None:
            output.data = [float(value) for value in q_target]
        else:
            # Thumb output order follows LEAP motors [12,13,14,15].
            output.data = [float(thumb_side_target), *map(float, q_target)]
        runtime.publisher.publish(output)

        try:
            q_command, side_command, motor_qpos = self.leap.command(
                finger_name,
                q_target,
                float(diag["dt"]),
                thumb_adduction=thumb_adduction,
            )
        except Exception as exc:
            self.get_logger().error(f"{finger_name} LEAP command failed: {exc}")
            return

        commanded = Float64MultiArray()
        commanded.data = [float(value) for value in self.leap.current_sim_qpos]
        self.target_pub.publish(commanded)

        if runtime.counter % 5 == 0:
            side_text = (
                ""
                if side_command is None
                else f" | adduction={sample.adduction:.3f} "
                f"q12={math.degrees(side_command):.1f} deg"
            )
            self.get_logger().info(
                "%s closure raw/fused=%.3f/%.3f | "
                "target=[%.1f, %.1f, %.1f] deg | "
                "cmd=[%.1f, %.1f, %.1f] deg%s | motors=%s"
                % (
                    finger_name,
                    diag["closure_raw"],
                    diag["closure_fused"],
                    *np.rad2deg(q_target),
                    *np.rad2deg(q_command),
                    side_text,
                    np.round(motor_qpos, 3).tolist(),
                )
            )
        runtime.counter += 1

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
        node = WeartThreeFingersSynergyNode()
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
