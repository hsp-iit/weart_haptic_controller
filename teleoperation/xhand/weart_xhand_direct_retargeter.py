#!/usr/bin/env python3
"""
WEART retargeting + Robotera XHAND EtherCAT bridge.

It can operate in three modes:
  - subscribe to the existing WEART raw String topics;
  - map normalized WEART closure/adduction to the 12 XHAND joints;
  - standalone (default): map WEART and command XHAND directly;
  - lerobot_control_mode: publish the WEART proposal, but command hardware
    only after PandaEE.send_action() returns it on the LeRobot command topic;
  - replay_mode: ignore WEART and execute replay/policy targets.
"""

from __future__ import annotations

import atexit
import re
import select
import sys
import termios
import threading
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

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
UPPER = np.array([1.570, 1.570, 1.570, 0.174, 1.919, 1.919, 1.919, 1.919, 1.919, 1.919, 1.919, 1.919])
VELOCITY = np.array([8.63, 8.63, 14.38, 14.38, 8.63, 14.38, 8.63, 14.38, 8.63, 14.38, 8.63, 14.38])
TACTILE_SENSOR_IDS = tuple(range(0x11, 0x16))


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


def normalize_tactile_force(
    force_vector: np.ndarray,
    *,
    full_scale_n: float,
    deadband_n: float,
) -> float:
    magnitude = float(np.linalg.norm(force_vector))
    value = (magnitude - deadband_n) / max(full_scale_n - deadband_n, 1.0e-9)
    return float(np.clip(value, 0.0, 1.0))


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
        self.log_version_info()

    def log_version_info(self) -> None:
        sdk_version = self.dev.get_sdk_version()
        print(f"[XHAND] software SDK version: {sdk_version}", flush=True)

        version_reply, hw_version = self.dev.read_version(self.hand_id, 0)
        if version_reply.error_code == 0:
            print(f"[XHAND] hardware version (joint 0): {hw_version}", flush=True)
        else:
            print(f"[XHAND] failed to read hardware version: {version_reply.error_message}", flush=True)

        info_reply, info = self.dev.read_device_info(self.hand_id)
        if info_reply.error_code == 0:
            print(
                f"[XHAND] device info: serial_number={info.serial_number[0:16]!r} "
                f"hand_id={info.hand_id} ev_hand={info.ev_hand}",
                flush=True,
            )
        else:
            print(f"[XHAND] failed to read device info: {info_reply.error_message}", flush=True)

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

    def reset_tactile_sensors_sdk(self) -> None:
        if not self.enabled:
            return
        if self.dev is None:
            raise RuntimeError("XHAND is not connected.")

        for sensor_id in TACTILE_SENSOR_IDS:
            reply = self.dev.reset_sensor(self.hand_id, sensor_id)
            if reply.error_code != 0:
                raise RuntimeError(
                    f"Failed to reset XHAND tactile sensor 0x{sensor_id:02x}: "
                    f"{reply.error_message}"
                )
            print(
                f"[XHAND] SDK reset_sensor OK: sensor_id=0x{sensor_id:02x}",
                flush=True,
            )

    def read_measured_state(
        self, force_update: bool = False
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Return 12 joints, resultant fingertip forces, and every tactile taxel.

        ``calc_force`` is the SDK's force resultant for one fingertip.  In
        contrast, ``raw_force`` contains the 120 individual force vectors of
        that fingertip, ordered exactly as supplied by the Robotera SDK.
        """
        if not self.enabled:
            return np.zeros(12, dtype=float), {}, {}
        if self.dev is None:
            raise RuntimeError("XHAND is not connected.")

        reply, state = self.dev.read_state(self.hand_id, bool(force_update))
        if reply.error_code != 0:
            raise RuntimeError(f"Failed to read XHAND state: {reply.error_message}")

        joints = np.array([float(finger.position) for finger in state.finger_state], dtype=float)
        if joints.size < 12:
            raise RuntimeError(f"XHAND returned {joints.size} joints; expected 12.")

        # Robotera SDK exposes sensor_data[0..4] in fingertip order.
        forces: dict[str, np.ndarray] = {}
        raw_forces: dict[str, np.ndarray] = {}
        for sensor_index, finger_name in enumerate(("thumb", "index", "middle", "ring", "little")):
            sensor = state.sensor_data[sensor_index]
            force = sensor.calc_force
            forces[finger_name] = np.array(
                [float(force.fx), float(force.fy), float(force.fz)],
                dtype=float,
            )
            raw_forces[finger_name] = np.array(
                [[float(taxel.fx), float(taxel.fy), float(taxel.fz)] for taxel in sensor.raw_force],
                dtype=float,
            )
        return joints[:12], forces, raw_forces

class WeartXHandDirectRetargetingNode(Node):
    def __init__(self):
        super().__init__("weart_xhand_direct_retargeter")

        self.declare_parameter("control_rate_hz", 30.0)
        self.declare_parameter("enable_hardware", True)
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
        self.declare_parameter("publish_xhand_tactile", True)
        self.declare_parameter("tactile_rate_hz", 20.0)
        self.declare_parameter("tactile_full_scale_n", 8.0)
        self.declare_parameter("tactile_deadband_n", 0.0)
        self.declare_parameter("enable_voice_tactile_reset", True)
        self.declare_parameter("voice_tactile_reset_topic", "/speech/recognized")
        self.declare_parameter("voice_tactile_reset_phrase", "azzera")
        # In this mode WEART only generates proposed joint targets. LeRobot
        # must echo them through replay_joint_topic before EtherCAT is driven.
        self.declare_parameter("lerobot_control_mode", False)
        self.declare_parameter("replay_mode", False)
        self.declare_parameter("replay_joint_topic", "/xhand/replay_joint_positions")

        self.closure_scale = float(self.get_parameter("closure_scale").value)
        self.closure_exponent = float(self.get_parameter("closure_exponent").value)
        self.low_pass_alpha = float(self.get_parameter("low_pass_alpha").value)
        self.max_speed_rad_s = float(self.get_parameter("max_speed_rad_s").value)
        self.mirror_middle = bool(self.get_parameter("mirror_middle_to_ring_little").value)
        self.max_sample_age_s = float(self.get_parameter("max_sample_age_s").value)
        self.publish_xhand_tactile = bool(self.get_parameter("publish_xhand_tactile").value)
        self.tactile_full_scale_n = float(self.get_parameter("tactile_full_scale_n").value)
        self.tactile_deadband_n = float(self.get_parameter("tactile_deadband_n").value)
        self.enable_voice_tactile_reset = bool(self.get_parameter("enable_voice_tactile_reset").value)
        self.voice_tactile_reset_phrase = str(
            self.get_parameter("voice_tactile_reset_phrase").value
        ).strip().lower()
        self.lerobot_control_mode = bool(self.get_parameter("lerobot_control_mode").value)
        self.replay_mode = bool(self.get_parameter("replay_mode").value)
        self.replay_joint_topic = str(self.get_parameter("replay_joint_topic").value)
        if self.lerobot_control_mode and self.replay_mode:
            raise ValueError("lerobot_control_mode and replay_mode are mutually exclusive")

        self.samples: Dict[str, WeartSample] = {}
        self.last_rx_time: Dict[str, float] = {}
        self.seen_first_sample = set()
        self.current_q = np.zeros(12, dtype=float)
        self.weart_target_q = np.zeros(12, dtype=float)
        self.replay_target: np.ndarray | None = None
        self.last_control_time_s = self.now_s()
        self.last_weart_target_time_s = self.last_control_time_s

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
        if self.driver.enabled:
            measured_q, _, _ = self.driver.read_measured_state(force_update=False)
            self.current_q = np.clip(measured_q, LOWER, UPPER)
            self.weart_target_q = self.current_q.copy()

        for finger in ("thumb", "index", "middle"):
            self.create_subscription(
                String,
                f"/weart/{finger}/raw",
                lambda msg, finger_name=finger: self.weart_callback(finger_name, msg),
                20,
            )
        if self.replay_mode or self.lerobot_control_mode:
            self.create_subscription(
                Float64MultiArray,
                self.replay_joint_topic,
                self.replay_joint_callback,
                20,
            )
        if self.enable_voice_tactile_reset:
            voice_topic = str(self.get_parameter("voice_tactile_reset_topic").value)
            self.create_subscription(String, voice_topic, self.voice_reset_callback, 20)
            self.get_logger().info(
                f"Voice tactile reset enabled: topic={voice_topic} "
                f"phrase='{self.voice_tactile_reset_phrase}'"
            )

        self.target_pub = self.create_publisher(Float64MultiArray, "/xhand/target_joint_positions", 20)
        self.weart_target_pub = self.create_publisher(
            Float64MultiArray, "/xhand/weart_joint_positions", 20
        )
        # Measured state for recorder consumers. This node is the only EtherCAT
        # owner, so downstream processes must subscribe instead of opening XHand.
        self.joint_state_pub = self.create_publisher(Float64MultiArray, "/xhand/joint_positions", 20)
        self.tactile_force_pub = self.create_publisher(Float64MultiArray, "/xhand/tactile_forces", 20)
        self.tactile_raw_vectors_pub = self.create_publisher(
            Float64MultiArray, "/xhand/tactile_raw_forces", 20
        )
        self.tactile_norm_pub = self.create_publisher(Float64MultiArray, "/xhand/tactile_normalized", 20)
        self.tactile_raw_pub = self.create_publisher(Float64MultiArray, "/xhand/tactile_force_norms", 20)
        self.diag_counter = 0
        self.wait_diag_counter = 0

        rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(rate_hz, 1.0), self.control_step)
        tactile_rate_hz = float(self.get_parameter("tactile_rate_hz").value)
        self.tactile_timer = None
        if self.publish_xhand_tactile:
            self.tactile_timer = self.create_timer(
                1.0 / max(tactile_rate_hz, 1.0),
                self.tactile_step,
            )

        self.reset_tactile_requested = threading.Event()
        self.reset_tactile_cooldown_s = 0.5
        self.last_reset_tactile_time = -float("inf")
        self.keyboard_stop = threading.Event()
        self.keyboard_thread = None
        self.keyboard_fd = None
        self.keyboard_original_settings = None
        if self.driver.enabled:
            self.start_keyboard_listener()

        self.get_logger().info("WEART -> XHAND direct retargeter started.")
        if self.driver.enabled:
            self.get_logger().warning("XHAND HARDWARE OUTPUT ENABLED.")
        else:
            self.get_logger().warning("Dry-run mode: publishing /xhand/target_joint_positions only.")
        if self.publish_xhand_tactile:
            self.get_logger().info("Publishing XHAND tactile values on /xhand/tactile_normalized.")
        if self.replay_mode:
            self.get_logger().warning(
                f"XHAND REPLAY MODE: ignoring WEART commands; listening on {self.replay_joint_topic}."
            )
        if self.lerobot_control_mode:
            self.get_logger().warning(
                "XHAND LEROBOT CONTROL MODE: WEART publishes proposed targets on "
                f"/xhand/weart_joint_positions; hardware moves only from {self.replay_joint_topic}."
            )

    def start_keyboard_listener(self) -> None:
        if not sys.stdin.isatty():
            self.get_logger().info(
                "stdin is not a TTY; Shift+X tactile SDK reset shortcut is disabled."
            )
            return

        self.keyboard_fd = sys.stdin.fileno()
        self.keyboard_original_settings = termios.tcgetattr(self.keyboard_fd)
        tty.setcbreak(self.keyboard_fd)
        atexit.register(self.restore_keyboard_terminal)
        self.keyboard_thread = threading.Thread(
            target=self.keyboard_listener_loop,
            daemon=True,
        )
        self.keyboard_thread.start()
        self.get_logger().info("Press Shift+X to reset XHAND tactile sensors via SDK.")

    def keyboard_listener_loop(self) -> None:
        try:
            while not self.keyboard_stop.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue
                char = sys.stdin.read(1)
                if char == "X":
                    self.reset_tactile_requested.set()
        except Exception as exc:
            self.get_logger().warning(f"Keyboard listener stopped: {exc}")

    def restore_keyboard_terminal(self) -> None:
        if self.keyboard_fd is None or self.keyboard_original_settings is None:
            return
        try:
            termios.tcsetattr(
                self.keyboard_fd,
                termios.TCSADRAIN,
                self.keyboard_original_settings,
            )
        except Exception as exc:
            self.get_logger().warning(f"Failed to restore terminal settings: {exc}")
        finally:
            self.keyboard_fd = None
            self.keyboard_original_settings = None

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

    def voice_reset_callback(self, msg: String) -> None:
        text = msg.data.strip().lower()
        if self.voice_tactile_reset_phrase and self.voice_tactile_reset_phrase in text:
            self.reset_tactile_requested.set()
            self.get_logger().info(f"Voice tactile reset requested: {msg.data!r}")

    def replay_joint_callback(self, msg: Float64MultiArray) -> None:
        values = np.asarray(msg.data, dtype=float)
        if values.size != 12 or not np.all(np.isfinite(values)):
            self.get_logger().warning(
                f"Ignoring invalid XHAND replay target with {values.size} values.",
                throttle_duration_sec=1.0,
            )
            return
        self.replay_target = np.clip(values, LOWER, UPPER)

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

        # right_hand_thumb_bend_joint (q[0]) is the thumb's ab/adduction axis
        # on the real hand (0-105 deg), not flexion despite its URDF name.
        # Drive it directly from the WEART adduction reading across its full
        # range so the joint is actually exploited, independent of closure.
        q[0] = (1.0 - thumb.adduction) * UPPER[0]
        q[1] = thumb_c * UPPER[1]
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

    def weart_samples_ready(self) -> bool:
        required = ("thumb", "index", "middle")
        if not all(name in self.samples for name in required):
            self.wait_diag_counter += 1
            if self.wait_diag_counter % 60 == 0:
                missing = [name for name in required if name not in self.samples]
                self.get_logger().warning("Waiting for WEART samples on: " + ", ".join(missing))
            return False
        if not all(self.sample_is_fresh(name) for name in required):
            self.get_logger().warning(
                "WEART data stale; holding XHAND target.", throttle_duration_sec=2.0
            )
            return False
        return True

    def publish_weart_target(self) -> None:
        """Publish a safe WEART-derived proposal without touching hardware."""
        if self.weart_samples_ready():
            now_s = self.now_s()
            dt = max(now_s - self.last_weart_target_time_s, 1e-3)
            self.last_weart_target_time_s = now_s
            target = self.target_from_samples()
            low_pass = self.weart_target_q + self.low_pass_alpha * (
                target - self.weart_target_q
            )
            max_step = np.minimum(VELOCITY, self.max_speed_rad_s) * dt
            self.weart_target_q = np.clip(
                self.weart_target_q
                + np.clip(low_pass - self.weart_target_q, -max_step, max_step),
                LOWER,
                UPPER,
            )

        msg = Float64MultiArray()
        msg.data = [float(value) for value in self.weart_target_q]
        self.weart_target_pub.publish(msg)

    def execute_external_target(self) -> None:
        """Apply only the command received from PandaEE.send_action()."""
        if self.replay_target is None:
            return
        now_s = self.now_s()
        dt = max(now_s - self.last_control_time_s, 1e-3)
        self.last_control_time_s = now_s
        low_pass = self.current_q + self.low_pass_alpha * (self.replay_target - self.current_q)
        max_step = np.minimum(VELOCITY, self.max_speed_rad_s) * dt
        self.current_q = np.clip(
            self.current_q + np.clip(low_pass - self.current_q, -max_step, max_step),
            LOWER,
            UPPER,
        )
        msg = Float64MultiArray()
        msg.data = [float(value) for value in self.current_q]
        self.target_pub.publish(msg)
        try:
            self.driver.command(self.current_q)
        except Exception as exc:
            self.get_logger().error(
                f"XHAND external command failed: {exc}", throttle_duration_sec=1.0
            )

    def control_step(self) -> None:
        if self.reset_tactile_requested.is_set():
            self.reset_tactile_requested.clear()
            now_s = self.now_s()
            if now_s - self.last_reset_tactile_time >= self.reset_tactile_cooldown_s:
                self.last_reset_tactile_time = now_s
                try:
                    self.driver.reset_tactile_sensors_sdk()
                    self.get_logger().info("XHAND tactile sensors reset via SDK (Shift+X).")
                except Exception as exc:
                    self.get_logger().error(f"Failed to reset XHAND tactile sensors via SDK: {exc}")

        if self.lerobot_control_mode:
            self.publish_weart_target()
            self.execute_external_target()
            return

        if self.replay_mode:
            self.execute_external_target()
            return

        if not self.weart_samples_ready():
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
        self.weart_target_pub.publish(msg)

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
            measured_joints, tactile_forces, tactile_raw_forces = self.driver.read_measured_state(
                force_update=False
            )
        except Exception as exc:
            self.get_logger().warning(
                f"XHAND tactile read failed: {exc}",
                throttle_duration_sec=1.0,
            )
            return

        joint_msg = Float64MultiArray()
        joint_msg.data = [float(value) for value in measured_joints]
        self.joint_state_pub.publish(joint_msg)

        tactile_msg = Float64MultiArray()
        tactile_msg.data = [
            float(value)
            for finger_name in ("thumb", "index", "middle", "ring", "little")
            for value in tactile_forces.get(finger_name, np.zeros(3))
        ]
        self.tactile_force_pub.publish(tactile_msg)

        # Five fingertips × 120 taxels × (fx, fy, fz).  Ordering is
        # thumb/index/middle/ring/little, then the SDK's taxel order, then xyz.
        raw_vectors_msg = Float64MultiArray()
        raw_vectors_msg.data = [
            float(component)
            for finger_name in ("thumb", "index", "middle", "ring", "little")
            for taxel in tactile_raw_forces.get(finger_name, np.zeros((120, 3)))
            for component in taxel
        ]
        self.tactile_raw_vectors_pub.publish(raw_vectors_msg)

        normalized = {
            name: normalize_tactile_force(
                force_vector,
                full_scale_n=self.tactile_full_scale_n,
                deadband_n=self.tactile_deadband_n,
            )
            for name, force_vector in tactile_forces.items()
        }
        raw_msg = Float64MultiArray()
        raw_msg.data = [
            float(np.linalg.norm(tactile_forces.get(name, np.zeros(3))))
            for name in ("thumb", "index", "middle")
        ]
        self.tactile_raw_pub.publish(raw_msg)

        msg = Float64MultiArray()
        msg.data = [
            float(normalized.get("thumb", 0.0)),
            float(normalized.get("index", 0.0)),
            float(normalized.get("middle", 0.0)),
        ]
        self.tactile_norm_pub.publish(msg)

    def destroy_node(self):
        self.keyboard_stop.set()
        if self.keyboard_thread is not None and self.keyboard_thread.is_alive():
            self.keyboard_thread.join(timeout=0.5)
        self.restore_keyboard_terminal()
        try:
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
