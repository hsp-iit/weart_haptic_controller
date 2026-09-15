#!/usr/bin/env python3
"""
WEART -> Robotera XHAND direct retargeting ROS 2 node.

This is the first, hardware-oriented bridge:
  - subscribe to the existing WEART raw String topics;
  - map normalized WEART closure/adduction to the 12 XHAND joints;
  - publish the commanded 12-joint target on /xhand/target_joint_positions;
  - optionally send the command to the real hand through Robotera's
    xhand_controller Python binding.

Hardware output is disabled by default. Enable it only after checking the
published targets.
"""

from __future__ import annotations

import math
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

try:
    from xhand_controller import xhand_control

    XHAND_CONTROLLER_AVAILABLE = True
except Exception:
    robotera_site_packages = Path("/usr/local/lib/python3.12/dist-packages")
    if robotera_site_packages.exists():
        sys.path.append(str(robotera_site_packages))
    try:
        from xhand_controller import xhand_control

        XHAND_CONTROLLER_AVAILABLE = True
    except Exception:
        xhand_control = None
        XHAND_CONTROLLER_AVAILABLE = False

try:
    from weartsdk import (
        TouchEffect,
        WeArtClient,
        WeArtCommon,
        WeArtForce,
        WeArtHapticObject,
        WeArtMessages,
        WeArtTemperature,
        WeArtTexture,
    )

    WEART_HAPTICS_AVAILABLE = True
except Exception:
    WEART_HAPTICS_AVAILABLE = False


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
CLOSURE_RE = re.compile(rf"closure=({FLOAT})")
ADDUCTION_RE = re.compile(rf"adduction=({FLOAT})")


JOINT_NAMES = (
    "right_hand_thumb_bend_joint",
    "right_hand_thumb_rota_joint1",
    "right_hand_thumb_rota_joint2",
    "right_hand_index_bend_joint",
    "right_hand_index_joint1",
    "right_hand_index_joint2",
    "right_hand_mid_joint1",
    "right_hand_mid_joint2",
    "right_hand_ring_joint1",
    "right_hand_ring_joint2",
    "right_hand_pinky_joint1",
    "right_hand_pinky_joint2",
)

# Limits from robotera/xhand1/urdf/Xhand-urdf/xhand_right/urdf/xhand_right.urdf
LOWER = np.array([0.0, -0.698, 0.0, -0.174, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
UPPER = np.array([1.832, 1.570, 1.570, 0.174, 1.919, 1.919, 1.919, 1.919, 1.919, 1.919, 1.919, 1.919])
VELOCITY = np.array([8.63, 8.63, 14.38, 14.38, 8.63, 14.38, 8.63, 14.38, 8.63, 14.38, 8.63, 14.38])


@dataclass
class WeartSample:
    closure: float = 0.0
    adduction: float = 1.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def parse_weart_string(text: str) -> WeartSample:
    closure_match = CLOSURE_RE.search(text)
    adduction_match = ADDUCTION_RE.search(text)
    closure = float(closure_match.group(1)) if closure_match else 0.0
    adduction = float(adduction_match.group(1)) if adduction_match else 1.0
    return WeartSample(
        closure=clamp(closure, 0.0, 1.0),
        adduction=clamp(adduction, 0.0, 1.0),
    )


def shaped_closure(closure: float, exponent: float) -> float:
    return clamp(float(closure), 0.0, 1.0) ** max(float(exponent), 1e-3)


class XHandDirectDriver:
    def __init__(
        self,
        *,
        enabled: bool,
        hand_id: int,
        protocol: str,
        ethercat_ifname: str,
        serial_port: str,
        baud_rate: int,
        kp: int,
        ki: int,
        kd: int,
        tor_max: int,
        control_mode: int,
    ):
        self.enabled = bool(enabled)
        self.hand_id = int(hand_id)
        self.protocol = str(protocol)
        self.ethercat_ifname = str(ethercat_ifname)
        self.serial_port = str(serial_port)
        self.baud_rate = int(baud_rate)
        self.kp = int(kp)
        self.ki = int(ki)
        self.kd = int(kd)
        self.tor_max = int(tor_max)
        self.control_mode = int(control_mode)
        self.dev = None

    def connect(self) -> None:
        if not self.enabled:
            print("[XHAND] hardware output DISABLED (dry-run).", flush=True)
            return
        if not XHAND_CONTROLLER_AVAILABLE:
            raise RuntimeError("Python package 'xhand_controller' is not available.")

        self.dev = xhand_control.XHandControl()
        if self.protocol == "EtherCAT":
            ports = list(self.dev.enumerate_devices("EtherCAT"))
            print(f"[XHAND] EtherCAT ports: {ports}", flush=True)
            ifname = self.ethercat_ifname or (ports[0] if ports else "")
            if not ifname:
                raise RuntimeError("No EtherCAT interface found for XHAND.")
            reply = self.dev.open_ethercat(ifname)
        elif self.protocol == "RS485":
            reply = self.dev.open_serial(self.serial_port, self.baud_rate)
        else:
            raise ValueError(f"Unsupported XHAND protocol: {self.protocol}")

        if reply.error_code != 0:
            raise RuntimeError(f"Failed to open XHAND: {reply.error_message}")

        print(f"[XHAND] connected via {self.protocol}.", flush=True)

    def disconnect(self) -> None:
        if self.dev is not None:
            self.dev.close_device()
            self.dev = None
            print("[XHAND] disconnected.", flush=True)

    def command(self, q: np.ndarray) -> None:
        if not self.enabled:
            return
        if self.dev is None:
            raise RuntimeError("XHAND output enabled but device is not connected.")

        cmd = xhand_control.HandCommand_t()
        for i, value in enumerate(np.asarray(q, dtype=float).reshape(12)):
            finger = cmd.finger_command[i]
            finger.id = i
            finger.kp = self.kp
            finger.ki = self.ki
            finger.kd = self.kd
            finger.position = float(value)
            finger.tor_max = self.tor_max
            finger.mode = self.control_mode

        reply = self.dev.send_command(self.hand_id, cmd)
        if reply.error_code != 0:
            raise RuntimeError(f"Failed to send XHAND command: {reply.error_message}")

    def read_tactile_forces(self, force_update: bool = False) -> dict[str, np.ndarray]:
        if not self.enabled:
            return {}
        if self.dev is None:
            raise RuntimeError("XHAND is not connected.")

        reply, state = self.dev.read_state(self.hand_id, bool(force_update))
        if reply.error_code != 0:
            raise RuntimeError(f"Failed to read XHAND state: {reply.error_message}")

        # Robotera SDK exposes sensor_data[0..4] in fingertip order:
        # thumb, index, middle, ring, little. This is the same convention used
        # by the LeRobot XHand wrapper.
        forces = {}
        for sensor_index, finger_name in enumerate(("thumb", "index", "middle")):
            force = state.sensor_data[sensor_index].calc_force
            forces[finger_name] = np.array(
                [float(force.fx), float(force.fy), float(force.fz)],
                dtype=float,
            )
        return forces


class WeartHapticFeedback:
    def __init__(
        self,
        *,
        enabled: bool,
        ip: str,
        port: int,
        hand_side: str,
        tactile_full_scale_n: float,
        tactile_deadband_n: float,
        smoothing_alpha: float,
        min_force: float,
        log_period_s: float,
    ):
        self.enabled = bool(enabled)
        self.ip = str(ip)
        self.port = int(port)
        self.hand_side = str(hand_side)
        self.tactile_full_scale_n = float(tactile_full_scale_n)
        self.tactile_deadband_n = float(tactile_deadband_n)
        self.smoothing_alpha = float(smoothing_alpha)
        self.min_force = float(min_force)
        self.log_period_s = float(log_period_s)
        self.client = None
        self.haptics = {}
        self.effects = {}
        self.effect_active = {}
        self.smoothed = {name: 0.0 for name in ("thumb", "index", "middle")}
        self.last_log = 0.0
        self.lock = threading.RLock()

    def connect(self) -> None:
        if not self.enabled:
            return
        if not WEART_HAPTICS_AVAILABLE:
            raise RuntimeError("Python package 'weartsdk' is not available.")
        if self.tactile_full_scale_n <= self.tactile_deadband_n:
            raise ValueError("tactile_full_scale_n must be greater than tactile_deadband_n")

        self.client = WeArtClient(self.ip, self.port)
        self.client.Run()
        if not self.client.IsConnected():
            raise RuntimeError(f"Could not connect to WEART middleware at {self.ip}:{self.port}")
        self.client.Start()

        temperature = WeArtTemperature()
        temperature.active = False
        texture = WeArtTexture(
            active=False,
            texture_type=getattr(WeArtCommon.TextureType, "AluminiumFineMeshSlow"),
            velocity=0.0,
            volume=0.0,
        )

        points = {
            "thumb": WeArtCommon.ActuationPoint.Thumb,
            "index": WeArtCommon.ActuationPoint.Index,
            "middle": WeArtCommon.ActuationPoint.Middle,
        }
        hand_side_flag = getattr(WeArtCommon.HandSide, self.hand_side)
        for finger_name, point in points.items():
            force = WeArtForce(active=True, force=0.0)
            haptic = WeArtHapticObject(self.client)
            haptic.handSideFlag = hand_side_flag
            haptic.actuationPointFlag = point
            self.haptics[finger_name] = haptic
            self.effects[finger_name] = TouchEffect(temperature, force, texture)
            self.effect_active[finger_name] = False

        print(f"[WEART] haptic feedback connected at {self.ip}:{self.port}.", flush=True)

    def disconnect(self) -> None:
        if not self.enabled:
            return
        for finger_name in list(self.effect_active):
            self.stop_effect(finger_name)
        if self.client is not None:
            try:
                self.client.Stop()
            except Exception:
                pass
            try:
                self.client.Close()
            except Exception:
                pass
            self.client = None
            print("[WEART] haptic feedback disconnected.", flush=True)

    def normalize_force(self, force_vector: np.ndarray) -> float:
        magnitude = float(np.linalg.norm(force_vector))
        value = (magnitude - self.tactile_deadband_n) / max(
            self.tactile_full_scale_n - self.tactile_deadband_n,
            1.0e-9,
        )
        return float(np.clip(value, 0.0, 1.0))

    def update(self, tactile_forces: dict[str, np.ndarray]) -> dict[str, float]:
        if not self.enabled:
            return {}

        sent = {}
        for finger_name, force_vector in tactile_forces.items():
            raw = self.normalize_force(force_vector)
            previous = self.smoothed.get(finger_name, 0.0)
            value = self.smoothing_alpha * raw + (1.0 - self.smoothing_alpha) * previous
            value = float(np.clip(value, 0.0, 1.0))
            self.smoothed[finger_name] = value
            sent[finger_name] = value
            if value <= self.min_force:
                self.stop_effect(finger_name)
            else:
                self.send_force(finger_name, value)

        now = time.monotonic()
        if self.log_period_s > 0.0 and now - self.last_log >= self.log_period_s:
            self.last_log = now
            print(
                "[WEART] haptic force "
                + " | ".join(f"{name}={value:.2f}" for name, value in sent.items()),
                flush=True,
            )
        return sent

    def send_force(self, finger_name: str, force_value: float) -> None:
        if self.client is None or not self.client.IsConnected():
            return
        haptic = self.haptics.get(finger_name)
        effect = self.effects.get(finger_name)
        if haptic is None or effect is None:
            return
        with self.lock:
            try:
                temperature = WeArtTemperature()
                temperature.active = False
                force = WeArtForce(active=True, force=float(np.clip(force_value, 0.0, 1.0)))
                texture = WeArtTexture(
                    active=False,
                    texture_type=getattr(WeArtCommon.TextureType, "AluminiumFineMeshSlow"),
                    velocity=0.0,
                    volume=0.0,
                )
                effect.Set(temperature, force, texture)
                if not self.effect_active.get(finger_name, False):
                    haptic.AddEffect(effect)
                    self.effect_active[finger_name] = True
                haptic.UpdateEffects()
                haptic.SendMessage(WeArtMessages.SetForceMessage([force_value, 0.0, 0.0]))
            except Exception as exc:
                print(f"[WEART] failed to send haptic force for {finger_name}: {exc}", flush=True)

    def stop_effect(self, finger_name: str) -> None:
        if not self.effect_active.get(finger_name, False):
            return
        haptic = self.haptics.get(finger_name)
        effect = self.effects.get(finger_name)
        if haptic is None or effect is None:
            return
        with self.lock:
            try:
                haptic.RemoveEffect(effect)
                haptic.SendMessage(WeArtMessages.StopForceMessage())
            except Exception as exc:
                print(f"[WEART] failed to stop haptic force for {finger_name}: {exc}", flush=True)
            finally:
                self.effect_active[finger_name] = False


class WeartXHandDirectRetargetingNode(Node):
    def __init__(self):
        super().__init__("weart_xhand_direct_retargeter")

        self.declare_parameter("control_rate_hz", 30.0)
        self.declare_parameter("enable_hardware", False)
        self.declare_parameter("hand_id", 0)
        self.declare_parameter("protocol", "EtherCAT")
        self.declare_parameter("ethercat_ifname", "")
        self.declare_parameter("serial_port", "/dev/ttyUSB0")
        self.declare_parameter("baud_rate", 3000000)
        self.declare_parameter("kp", 50)
        self.declare_parameter("ki", 0)
        self.declare_parameter("kd", 0)
        self.declare_parameter("tor_max", 300)
        self.declare_parameter("control_mode", 3)
        self.declare_parameter("closure_scale", 0.9)
        self.declare_parameter("closure_exponent", 1.35)
        self.declare_parameter("low_pass_alpha", 0.25)
        self.declare_parameter("max_speed_rad_s", 1.5)
        self.declare_parameter("mirror_middle_to_ring_little", True)
        self.declare_parameter("max_sample_age_s", 0.30)
        self.declare_parameter("publish_xhand_tactile", False)
        self.declare_parameter("enable_weart_haptics", False)
        self.declare_parameter("weart_ip", "127.0.0.1")
        self.declare_parameter("weart_port", 13031)
        self.declare_parameter("weart_hand_side", "Right")
        self.declare_parameter("tactile_rate_hz", 20.0)
        self.declare_parameter("tactile_full_scale_n", 20.0)
        self.declare_parameter("tactile_deadband_n", 0.2)
        self.declare_parameter("haptic_smoothing_alpha", 0.25)
        self.declare_parameter("haptic_min_force", 0.02)
        self.declare_parameter("haptic_log_period_s", 1.0)

        self.closure_scale = float(self.get_parameter("closure_scale").value)
        self.closure_exponent = float(self.get_parameter("closure_exponent").value)
        self.low_pass_alpha = float(self.get_parameter("low_pass_alpha").value)
        self.max_speed_rad_s = float(self.get_parameter("max_speed_rad_s").value)
        self.mirror_middle = bool(self.get_parameter("mirror_middle_to_ring_little").value)
        self.max_sample_age_s = float(self.get_parameter("max_sample_age_s").value)
        self.publish_xhand_tactile = bool(self.get_parameter("publish_xhand_tactile").value)

        self.samples: Dict[str, WeartSample] = {}
        self.last_rx_time: Dict[str, float] = {}
        self.seen_first_sample = set()
        self.current_q = np.zeros(12, dtype=float)
        self.last_control_time_s = self.now_s()

        self.driver = XHandDirectDriver(
            enabled=bool(self.get_parameter("enable_hardware").value),
            hand_id=int(self.get_parameter("hand_id").value),
            protocol=str(self.get_parameter("protocol").value),
            ethercat_ifname=str(self.get_parameter("ethercat_ifname").value),
            serial_port=str(self.get_parameter("serial_port").value),
            baud_rate=int(self.get_parameter("baud_rate").value),
            kp=int(self.get_parameter("kp").value),
            ki=int(self.get_parameter("ki").value),
            kd=int(self.get_parameter("kd").value),
            tor_max=int(self.get_parameter("tor_max").value),
            control_mode=int(self.get_parameter("control_mode").value),
        )
        self.driver.connect()

        self.haptics = WeartHapticFeedback(
            enabled=bool(self.get_parameter("enable_weart_haptics").value),
            ip=str(self.get_parameter("weart_ip").value),
            port=int(self.get_parameter("weart_port").value),
            hand_side=str(self.get_parameter("weart_hand_side").value),
            tactile_full_scale_n=float(self.get_parameter("tactile_full_scale_n").value),
            tactile_deadband_n=float(self.get_parameter("tactile_deadband_n").value),
            smoothing_alpha=float(self.get_parameter("haptic_smoothing_alpha").value),
            min_force=float(self.get_parameter("haptic_min_force").value),
            log_period_s=float(self.get_parameter("haptic_log_period_s").value),
        )
        self.haptics.connect()

        for finger in ("thumb", "index", "middle"):
            self.create_subscription(
                String,
                f"/weart/{finger}/raw",
                lambda msg, finger_name=finger: self.weart_callback(finger_name, msg),
                20,
            )

        self.target_pub = self.create_publisher(Float64MultiArray, "/xhand/target_joint_positions", 20)
        self.haptic_pub = self.create_publisher(Float64MultiArray, "/xhand/tactile_normalized", 20)
        self.tactile_raw_pub = self.create_publisher(Float64MultiArray, "/xhand/tactile_force_norms", 20)
        self.diag_counter = 0
        self.wait_diag_counter = 0

        rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(rate_hz, 1.0), self.control_step)
        tactile_rate_hz = float(self.get_parameter("tactile_rate_hz").value)
        self.tactile_timer = None
        if self.publish_xhand_tactile or self.haptics.enabled:
            self.tactile_timer = self.create_timer(
                1.0 / max(tactile_rate_hz, 1.0),
                self.tactile_step,
            )

        self.get_logger().info("WEART -> XHAND direct retargeter started.")
        if self.driver.enabled:
            self.get_logger().warning("XHAND HARDWARE OUTPUT ENABLED.")
        else:
            self.get_logger().warning("Dry-run mode: publishing /xhand/target_joint_positions only.")
        if self.haptics.enabled:
            self.get_logger().warning("WEART HAPTIC FEEDBACK ENABLED.")
        elif self.publish_xhand_tactile:
            self.get_logger().info("Publishing XHAND tactile values on /xhand/tactile_normalized.")

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def weart_callback(self, finger_name: str, msg: String) -> None:
        self.samples[finger_name] = parse_weart_string(msg.data)
        self.last_rx_time[finger_name] = self.now_s()
        if finger_name not in self.seen_first_sample:
            self.seen_first_sample.add(finger_name)
            sample = self.samples[finger_name]
            self.get_logger().info(
                f"First WEART sample for {finger_name}: "
                f"closure={sample.closure:.3f}, adduction={sample.adduction:.3f}"
            )

    def sample_is_fresh(self, finger_name: str) -> bool:
        return finger_name in self.last_rx_time and (self.now_s() - self.last_rx_time[finger_name]) <= self.max_sample_age_s

    def target_from_samples(self) -> np.ndarray:
        q = np.zeros(12, dtype=float)

        thumb = self.samples.get("thumb", WeartSample())
        index = self.samples.get("index", WeartSample())
        middle = self.samples.get("middle", WeartSample())

        thumb_c = self.closure_scale * shaped_closure(thumb.closure, self.closure_exponent)
        index_c = self.closure_scale * shaped_closure(index.closure, self.closure_exponent)
        middle_c = self.closure_scale * shaped_closure(middle.closure, self.closure_exponent)

        q[0] = thumb_c * UPPER[0]
        q[1] = thumb_c * (1.0 - thumb.adduction) * 0.85
        q[2] = thumb_c * UPPER[2]

        q[3] = 0.0
        q[4] = index_c * UPPER[4]
        q[5] = index_c * UPPER[5]

        q[6] = middle_c * UPPER[6]
        q[7] = middle_c * UPPER[7]

        if self.mirror_middle:
            q[8] = middle_c * UPPER[8]
            q[9] = middle_c * UPPER[9]
            q[10] = middle_c * UPPER[10]
            q[11] = middle_c * UPPER[11]

        return np.clip(q, LOWER, UPPER)

    def control_step(self) -> None:
        if not all(name in self.samples for name in ("thumb", "index", "middle")):
            self.wait_diag_counter += 1
            if self.wait_diag_counter % 60 == 0:
                missing = [
                    name
                    for name in ("thumb", "index", "middle")
                    if name not in self.samples
                ]
                self.get_logger().warning(
                    "Waiting for WEART samples on: " + ", ".join(missing)
                )
            return
        if not all(self.sample_is_fresh(name) for name in ("thumb", "index", "middle")):
            self.get_logger().warning("WEART data stale; not sending XHAND target.", throttle_duration_sec=2.0)
            return

        now_s = self.now_s()
        dt = max(now_s - self.last_control_time_s, 1e-3)
        self.last_control_time_s = now_s

        target = self.target_from_samples()
        low_pass = self.current_q + self.low_pass_alpha * (target - self.current_q)
        max_step = np.minimum(VELOCITY, self.max_speed_rad_s) * dt
        self.current_q = np.clip(self.current_q + np.clip(low_pass - self.current_q, -max_step, max_step), LOWER, UPPER)

        msg = Float64MultiArray()
        msg.data = [float(v) for v in self.current_q]
        self.target_pub.publish(msg)

        try:
            self.driver.command(self.current_q)
        except Exception as exc:
            self.get_logger().error(f"XHAND hardware command failed: {exc}", throttle_duration_sec=1.0)

        self.diag_counter += 1
        if self.diag_counter % 30 == 0:
            self.get_logger().info(
                " | ".join(
                    f"{name}: c={self.samples[name].closure:.2f}"
                    for name in ("thumb", "index", "middle")
                )
                + " | q="
                + np.array2string(self.current_q, precision=2, suppress_small=True)
            )

    def tactile_step(self) -> None:
        try:
            tactile_forces = self.driver.read_tactile_forces(force_update=False)
        except Exception as exc:
            self.get_logger().warning(
                f"XHAND tactile read failed: {exc}",
                throttle_duration_sec=1.0,
            )
            return

        normalized = {
            name: self.haptics.normalize_force(force_vector)
            for name, force_vector in tactile_forces.items()
        }
        raw_msg = Float64MultiArray()
        raw_msg.data = [
            float(np.linalg.norm(tactile_forces.get(name, np.zeros(3))))
            for name in ("thumb", "index", "middle")
        ]
        self.tactile_raw_pub.publish(raw_msg)

        if self.haptics.enabled:
            normalized = self.haptics.update(tactile_forces)

        msg = Float64MultiArray()
        msg.data = [
            float(normalized.get("thumb", 0.0)),
            float(normalized.get("index", 0.0)),
            float(normalized.get("middle", 0.0)),
        ]
        self.haptic_pub.publish(msg)

    def destroy_node(self):
        try:
            self.haptics.disconnect()
            self.driver.disconnect()
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = WeartXHandDirectRetargetingNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
