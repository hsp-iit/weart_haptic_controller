#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""ROS 2: WEART index/thumb/middle -> LEAP Hand.

Analogous to weart_index_subscriber.py, but listens to three WEART raw topics
and commands the corresponding LEAP finger chains together.

Assumptions:
    - /weart/index/raw controls LEAP index flexion motors 1,2,3.
    - /weart/middle/raw controls LEAP middle flexion motors 5,6,7.
    - /weart/thumb/raw controls LEAP thumb flexion motors 13,14,15.
      Thumb base joint 12 is kept at its current neutral command.
"""

from __future__ import annotations

import glob
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from weart_index_subscriber import (
    COMMAND_LOW_PASS_ALPHA,
    COMMAND_MAX_SPEED_RAD_S,
    DEFAULT_ENABLE_HARDWARE,
    DEFAULT_KD,
    DEFAULT_KI,
    DEFAULT_KP,
    DEFAULT_LEAP_PORT,
    DEFAULT_URDF_PATH,
    HUMAN_Q_LOWER,
    HUMAN_Q_UPPER,
    MIN_DT_S,
    IndexEstimator,
    _load_leap_cpp,
    leap_sim_to_motor,
    parse_weart_line,
)


DEFAULT_INDEX_INPUT_TOPIC = "/weart/index/raw"
DEFAULT_INDEX_OUTPUT_TOPIC = "/weart/index/estimated_joints"
DEFAULT_THUMB_INPUT_TOPIC = "/weart/thumb/raw"
DEFAULT_THUMB_OUTPUT_TOPIC = "/weart/thumb/estimated_joints"
DEFAULT_MIDDLE_INPUT_TOPIC = "/weart/middle/raw"
DEFAULT_MIDDLE_OUTPUT_TOPIC = "/weart/middle/estimated_joints"


@dataclass(frozen=True)
class LeapFingerModel:
    joint_names: tuple[str, ...]
    sim_lower: np.ndarray
    sim_upper: np.ndarray
    velocity_limits: np.ndarray
    finger_motor_indices: dict[str, tuple[int, int, int]]

    @property
    def motor_count(self) -> int:
        return len(self.joint_names)


@dataclass
class FingerRuntime:
    name: str
    estimator: IndexEstimator
    publisher: any
    counter: int = 0


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


def load_leap_finger_model(urdf_path: str) -> LeapFingerModel:
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

    try:
        finger_motor_indices = {
            "index": (
                child_to_joint["mcp_joint"],
                parent_child_to_joint[("pip", "dip")],
                parent_child_to_joint[("dip", "fingertip")],
            ),
            "middle": (
                child_to_joint["mcp_joint_2"],
                parent_child_to_joint[("pip_2", "dip_2")],
                parent_child_to_joint[("dip_2", "fingertip_2")],
            ),
            "thumb": (
                child_to_joint["thumb_pip"],
                parent_child_to_joint[("thumb_pip", "thumb_dip")],
                parent_child_to_joint[("thumb_dip", "thumb_fingertip")],
            ),
        }
    except KeyError as exc:
        raise ValueError(
            "URDF does not contain the expected index/middle/thumb finger chains"
        ) from exc

    return LeapFingerModel(
        joint_names=tuple(str(i) for i in indices),
        sim_lower=np.asarray(lower, dtype=float),
        sim_upper=np.asarray(upper, dtype=float),
        velocity_limits=np.asarray(velocity, dtype=float),
        finger_motor_indices=finger_motor_indices,
    )


class DirectLeapMultiFingerDriver:
    def __init__(
        self,
        *,
        enabled: bool,
        port: str,
        kp: float,
        ki: float,
        kd: float,
        model: LeapFingerModel,
    ) -> None:
        self.enabled = bool(enabled)
        self.requested_port = str(port)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.model = model

        self.current_sim_qpos = np.zeros(self.model.motor_count, dtype=float)
        self.filtered_q = {
            name: np.zeros(3, dtype=float)
            for name in self.model.finger_motor_indices
        }

        self._ctrl = None
        self.port: str | None = None

    def connect(self) -> None:
        print(
            "[LEAP] URDF mappings: "
            + ", ".join(
                f"{name}={indices}"
                for name, indices in self.model.finger_motor_indices.items()
            ),
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

    def _human_to_leap_finger(
        self,
        finger_name: str,
        q_human: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        indices = list(self.model.finger_motor_indices[finger_name])
        lower = self.model.sim_lower[indices]
        upper = self.model.sim_upper[indices]
        q_human = np.clip(
            np.asarray(q_human, dtype=float), HUMAN_Q_LOWER, HUMAN_Q_UPPER
        )

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

        q_prev = self.filtered_q[finger_name]
        q_lp = q_prev + COMMAND_LOW_PASS_ALPHA * (q_target - q_prev)

        max_speed = np.minimum(
            COMMAND_MAX_SPEED_RAD_S,
            self.model.velocity_limits[indices],
        )
        max_step = max_speed * max(dt, MIN_DT_S)
        delta = np.clip(q_lp - q_prev, -max_step, +max_step)

        self.filtered_q[finger_name] = q_prev + delta
        return self.filtered_q[finger_name].copy()

    def command(
        self,
        finger_name: str,
        q_human: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        q_finger_sim = self._human_to_leap_finger(finger_name, q_human, dt)

        for motor_idx, value in zip(
            self.model.finger_motor_indices[finger_name],
            q_finger_sim,
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

        return q_finger_sim.copy(), motor_qpos.copy()


class WeartThreeFingersLeapNode(Node):
    def __init__(self) -> None:
        super().__init__("weart_three_fingers_to_leap")

        self.declare_parameter("index_input_topic", DEFAULT_INDEX_INPUT_TOPIC)
        self.declare_parameter("index_output_topic", DEFAULT_INDEX_OUTPUT_TOPIC)
        self.declare_parameter("thumb_input_topic", DEFAULT_THUMB_INPUT_TOPIC)
        self.declare_parameter("thumb_output_topic", DEFAULT_THUMB_OUTPUT_TOPIC)
        self.declare_parameter("middle_input_topic", DEFAULT_MIDDLE_INPUT_TOPIC)
        self.declare_parameter("middle_output_topic", DEFAULT_MIDDLE_OUTPUT_TOPIC)
        self.declare_parameter("enable_hardware", DEFAULT_ENABLE_HARDWARE)
        self.declare_parameter("leap_port", DEFAULT_LEAP_PORT)
        self.declare_parameter("urdf_path", DEFAULT_URDF_PATH)
        self.declare_parameter("kp", DEFAULT_KP)
        self.declare_parameter("ki", DEFAULT_KI)
        self.declare_parameter("kd", DEFAULT_KD)

        enable_hardware = bool(self.get_parameter("enable_hardware").value)
        leap_port = str(self.get_parameter("leap_port").value)
        urdf_path = str(self.get_parameter("urdf_path").value)
        kp = float(self.get_parameter("kp").value)
        ki = float(self.get_parameter("ki").value)
        kd = float(self.get_parameter("kd").value)

        self.model = load_leap_finger_model(urdf_path)
        self.leap = DirectLeapMultiFingerDriver(
            enabled=enable_hardware,
            port=leap_port,
            kp=kp,
            ki=ki,
            kd=kd,
            model=self.model,
        )
        self.leap.connect()

        self.fingers: dict[str, FingerRuntime] = {}
        self.finger_subscriptions = []

        for finger_name, input_topic, output_topic in (
            (
                "index",
                str(self.get_parameter("index_input_topic").value),
                str(self.get_parameter("index_output_topic").value),
            ),
            (
                "thumb",
                str(self.get_parameter("thumb_input_topic").value),
                str(self.get_parameter("thumb_output_topic").value),
            ),
            (
                "middle",
                str(self.get_parameter("middle_input_topic").value),
                str(self.get_parameter("middle_output_topic").value),
            ),
        ):
            pub = self.create_publisher(Float64MultiArray, output_topic, 20)
            sub = self.create_subscription(
                String,
                input_topic,
                partial(self._on_weart, finger_name),
                20,
            )
            self.finger_subscriptions.append(sub)
            self.fingers[finger_name] = FingerRuntime(
                name=finger_name,
                estimator=IndexEstimator(),
                publisher=pub,
            )
            self.get_logger().info(
                f"{finger_name}: input={input_topic} output={output_topic} "
                f"motors={self.model.finger_motor_indices[finger_name]}"
            )

        if enable_hardware:
            self.get_logger().warning("LEAP HARDWARE OUTPUT ENABLED.")
        else:
            self.get_logger().warning(
                "LEAP dry-run. Enable with "
                "--ros-args -p enable_hardware:=true"
            )

        self.get_logger().warning(
            "Each finger calibrates independently: keep index, thumb and middle "
            "open and still until all three reach 100%."
        )

    def destroy_node(self):
        try:
            self.leap.disconnect()
        finally:
            return super().destroy_node()

    def _on_weart(self, finger_name: str, msg: String) -> None:
        runtime = self.fingers[finger_name]

        try:
            sample = parse_weart_line(msg.data)
        except Exception as exc:
            self.get_logger().warning(f"{finger_name} parse error: {exc}")
            return

        try:
            q_human, diag = runtime.estimator.update(sample)
        except Exception as exc:
            self.get_logger().error(f"{finger_name} estimator failed: {exc}")
            return

        if q_human is None:
            if runtime.counter % 5 == 0:
                self.get_logger().info(
                    f"{finger_name} calibration: "
                    f"{100.0 * diag['calibration']:.0f}%"
                )
            runtime.counter += 1
            return

        out = Float64MultiArray()
        out.data = [float(v) for v in q_human]
        runtime.publisher.publish(out)

        try:
            q_finger_sim, motor_qpos = self.leap.command(
                finger_name,
                q_human,
                float(diag["dt"]),
            )
        except Exception as exc:
            self.get_logger().error(f"{finger_name} LEAP command failed: {exc}")
            return

        if runtime.counter % 5 == 0:
            motor_selected = motor_qpos[
                list(self.model.finger_motor_indices[finger_name])
            ]
            self.get_logger().info(
                "%s human q=[%.1f, %.1f, %.1f] deg | "
                "LEAPsim=[%.1f, %.1f, %.1f] deg | motors=%s"
                % (
                    finger_name,
                    *np.rad2deg(q_human),
                    *np.rad2deg(q_finger_sim),
                    np.round(motor_selected, 3).tolist(),
                )
            )

        runtime.counter += 1


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None

    try:
        node = WeartThreeFingersLeapNode()
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
