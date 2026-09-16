#!/usr/bin/env python3
"""
Forward normalized XHAND tactile values to WEART haptic thimbles.

Run this on the PC connected to the WEART Middleware. The XHAND PC should
publish /xhand/tactile_normalized as std_msgs/Float64MultiArray in this order:
thumb, index, middle.
"""

from __future__ import annotations

import threading
import time

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

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


FINGERS = ("thumb", "index", "middle")
POINTS = {
    "thumb": WeArtCommon.ActuationPoint.Thumb,
    "index": WeArtCommon.ActuationPoint.Index,
    "middle": WeArtCommon.ActuationPoint.Middle,
}


class XHandTactileToWeartHaptics(Node):
    def __init__(self):
        super().__init__("xhand_tactile_to_weart_haptics")
        self.declare_parameter("input_topic", "/xhand/tactile_normalized")
        self.declare_parameter("weart_ip", WeArtCommon.DEFAULT_IP_ADDRESS)
        self.declare_parameter("weart_port", WeArtCommon.DEFAULT_TCP_PORT)
        self.declare_parameter("hand_side", "Right")
        self.declare_parameter("gain", 1.0)
        self.declare_parameter("deadband", 0.02)
        self.declare_parameter("smoothing_alpha", 0.35)
        self.declare_parameter("input_timeout_s", 0.5)
        self.declare_parameter("log_period_s", 1.0)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.weart_ip = str(self.get_parameter("weart_ip").value)
        self.weart_port = int(self.get_parameter("weart_port").value)
        self.hand_side = str(self.get_parameter("hand_side").value)
        self.gain = float(self.get_parameter("gain").value)
        self.deadband = float(self.get_parameter("deadband").value)
        self.smoothing_alpha = float(self.get_parameter("smoothing_alpha").value)
        self.input_timeout_s = float(self.get_parameter("input_timeout_s").value)
        self.log_period_s = float(self.get_parameter("log_period_s").value)

        self.lock = threading.RLock()
        self.client = None
        self.haptics = {}
        self.effects = {}
        self.effect_active = {name: False for name in FINGERS}
        self.smoothed = {name: 0.0 for name in FINGERS}
        self.last_input_time = 0.0
        self.last_log_time = 0.0

        self._connect_weart()
        self.create_subscription(Float64MultiArray, self.input_topic, self._on_tactile, 20)
        self.watchdog = self.create_timer(0.1, self._watchdog)
        self.get_logger().info(f"Listening to {self.input_topic}; forwarding to WEART {self.hand_side}.")

    def _connect_weart(self) -> None:
        self.client = WeArtClient(self.weart_ip, self.weart_port)
        self.client.Run()
        if not self.client.IsConnected():
            raise RuntimeError(f"Could not connect to WEART Middleware at {self.weart_ip}:{self.weart_port}")
        self.client.Start()

        hand_side_flag = getattr(WeArtCommon.HandSide, self.hand_side)
        for finger_name in FINGERS:
            temperature = WeArtTemperature()
            temperature.active = False
            force = WeArtForce(active=True, force=0.0)
            texture = WeArtTexture(
                active=False,
                texture_type=getattr(WeArtCommon.TextureType, "AluminiumFineMeshSlow"),
                velocity=0.0,
                volume=0.0,
            )
            haptic = WeArtHapticObject(self.client)
            haptic.handSideFlag = hand_side_flag
            haptic.actuationPointFlag = POINTS[finger_name]
            self.haptics[finger_name] = haptic
            self.effects[finger_name] = TouchEffect(temperature, force, texture)

        self.get_logger().info(f"Connected to WEART Middleware at {self.weart_ip}:{self.weart_port}.")

    def _on_tactile(self, msg: Float64MultiArray) -> None:
        self.last_input_time = time.monotonic()
        values = list(msg.data)
        if len(values) < len(FINGERS):
            self.get_logger().warning(
                f"Expected at least {len(FINGERS)} tactile values, got {len(values)}",
                throttle_duration_sec=1.0,
            )
            return

        sent = {}
        for finger_name, raw_value in zip(FINGERS, values):
            target = float(np.clip(float(raw_value) * self.gain, 0.0, 1.0))
            previous = self.smoothed[finger_name]
            value = self.smoothing_alpha * target + (1.0 - self.smoothing_alpha) * previous
            value = float(np.clip(value, 0.0, 1.0))
            self.smoothed[finger_name] = value
            sent[finger_name] = value
            if value <= self.deadband:
                self._stop_effect(finger_name)
            else:
                self._send_force(finger_name, value)

        now = time.monotonic()
        if self.log_period_s > 0.0 and now - self.last_log_time >= self.log_period_s:
            self.last_log_time = now
            self.get_logger().info(
                "WEART force: " + " | ".join(f"{name}={value:.2f}" for name, value in sent.items())
            )

    def _send_force(self, finger_name: str, force_value: float) -> None:
        if self.client is None or not self.client.IsConnected():
            return
        haptic = self.haptics[finger_name]
        effect = self.effects[finger_name]
        with self.lock:
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
            if not self.effect_active[finger_name]:
                haptic.AddEffect(effect)
                self.effect_active[finger_name] = True
            haptic.UpdateEffects()
            haptic.SendMessage(WeArtMessages.SetForceMessage([force_value, 0.0, 0.0]))

    def _stop_effect(self, finger_name: str) -> None:
        if not self.effect_active.get(finger_name, False):
            return
        with self.lock:
            try:
                self.haptics[finger_name].RemoveEffect(self.effects[finger_name])
                self.haptics[finger_name].SendMessage(WeArtMessages.StopForceMessage())
            finally:
                self.effect_active[finger_name] = False

    def _watchdog(self) -> None:
        if self.last_input_time <= 0.0:
            return
        if time.monotonic() - self.last_input_time > self.input_timeout_s:
            for finger_name in FINGERS:
                self.smoothed[finger_name] = 0.0
                self._stop_effect(finger_name)

    def shutdown(self) -> None:
        for finger_name in FINGERS:
            self._stop_effect(finger_name)
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


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = XHandTactileToWeartHaptics()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
