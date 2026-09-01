#!/usr/bin/env python3
"""Dummy WEART bridge that forwards force from a simulated tactile sensor."""

from __future__ import annotations

import threading
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from weartsdk import (
    MiddlewareStatusListener,
    TouchEffect,
    WeArtClient,
    WeArtCommon,
    WeArtForce,
    WeArtHapticObject,
    WeArtMessages,
    WeArtTemperature,
    WeArtTexture,
)


class BridgeNodeDummy(Node):
    def __init__(self) -> None:
        super().__init__("bridge_node_dummy")

        self._declare_parameters()
        self._load_parameters()

        self._weart_lock = threading.RLock()
        self._client: Optional[WeArtClient] = None
        self._middleware_listener: Optional[MiddlewareStatusListener] = None
        self._haptic: Optional[WeArtHapticObject] = None
        self._effect: Optional[TouchEffect] = None
        self._effect_active = False
        self._weart_started = False
        self._received_first_force = False

        if not self.dry_run:
            self._initialize_weart()

        self._subscription = self.create_subscription(
            Vector3,
            self.force_topic,
            self._on_force_vector,
            10,
        )
        self._watchdog_timer = self.create_timer(0.1, self._watchdog)
        self._last_update_monotonic = 0.0

        self.get_logger().info(
            f"Listening to {self.force_topic}; WEART target={self.hand_side}/{','.join(self.actuation_points)}"
        )

    def _declare_parameters(self) -> None:
        parameters = [
            ("force_topic", "/gelsight_processed/force"),
            ("weart_ip", "127.0.0.1"),
            ("weart_port", 13031),
            ("auto_start_weart", True),
            ("dry_run", False),
            ("hand_side", "Right"),
            ("actuation_points", ["Thumb"]),
            ("max_update_rate_hz", 5.0),
        ]
        self.declare_parameters(namespace="", parameters=parameters)

    def _load_parameters(self) -> None:
        value = lambda name: self.get_parameter(name).value

        self.force_topic = str(value("force_topic"))
        self.weart_ip = str(value("weart_ip"))
        self.weart_port = int(value("weart_port"))
        self.auto_start_weart = bool(value("auto_start_weart"))
        self.dry_run = bool(value("dry_run"))
        self.hand_side = str(value("hand_side"))
        self.actuation_points = [str(item) for item in value("actuation_points")]
        self.max_update_rate_hz = float(value("max_update_rate_hz"))

        if self.max_update_rate_hz <= 0.0:
            raise ValueError("max_update_rate_hz must be positive")

    def _initialize_weart(self) -> None:
        self._client = WeArtClient(self.weart_ip, self.weart_port)
        try:
            self._client.Run()
        except SystemExit as exc:
            raise RuntimeError(
                f"Could not connect to WEART middleware at {self.weart_ip}:{self.weart_port}"
            ) from exc
        if not self._client.IsConnected():
            raise RuntimeError(
                f"Could not connect to WEART middleware at {self.weart_ip}:{self.weart_port}"
            )

        self._middleware_listener = MiddlewareStatusListener()
        self._middleware_listener.AddStatusCallback(self._on_middleware_status)
        self._client.AddMessageListener(self._middleware_listener)

        temperature = WeArtTemperature()
        temperature.active = False

        force = WeArtForce(active=True, force=0.0)
        texture = WeArtTexture(
            active=False,
            texture_type=getattr(
                WeArtCommon.TextureType, "AluminiumFineMeshSlow"
            ),
            velocity=0.0,
            volume=0.0,
        )

        self._effect = TouchEffect(temperature, force, texture)
        self._haptic = WeArtHapticObject(self._client)
        self._haptic.handSideFlag = self._parse_hand_side(self.hand_side)
        self._haptic.actuationPointFlag = self._parse_actuation_points(
            self.actuation_points
        )

        if self.auto_start_weart:
            try:
                self._client.Start()
                self.get_logger().info("WEART haptic StartFromClient sent")
            except Exception as exc:
                self.get_logger().warning(f"Cannot start WEART session: {exc}")

        self.get_logger().info(
            f"WEART middleware client running at {self.weart_ip}:{self.weart_port}"
        )

    @staticmethod
    def _parse_hand_side(name: str):
        try:
            return getattr(WeArtCommon.HandSide, name)
        except AttributeError as exc:
            allowed = ["Left", "Right"]
            raise ValueError(f"hand_side must be one of {allowed}") from exc

    @staticmethod
    def _parse_actuation_points(names: list[str]):
        flag = WeArtCommon.ActuationPoint(0)
        for name in names:
            try:
                flag |= getattr(WeArtCommon.ActuationPoint, name)
            except AttributeError as exc:
                allowed = [
                    "Thumb",
                    "Index",
                    "Middle",
                    "Annular",
                    "Pinky",
                    "Palm",
                ]
                raise ValueError(
                    f"Unknown actuation point {name!r}; choose from {allowed}"
                ) from exc
        if int(flag) == 0:
            raise ValueError("actuation_points cannot be empty")
        return flag

    def _on_middleware_status(self, status) -> None:
        if getattr(status, "status", None) == WeArtCommon.MiddlewareStatus.RUNNING:
            if not self._weart_started:
                self._weart_started = True
                self.get_logger().info("WEART middleware RUNNING; ready to send forces")

    def _on_force_vector(self, msg: Vector3) -> None:
        now = time.monotonic()
        self._last_update_monotonic = now

        if not self._received_first_force:
            self._received_first_force = True
            self.get_logger().info(
                f"Received first force message from {self.force_topic}"
            )

        z_value = float(msg.z)
        force_value = z_value / -19.0
        force_value = max(0.0, min(1.0, force_value))

        self.get_logger().info(
            f"Received force vector z={z_value:.3f}; mapped force={force_value:.3f}"
        )

        if force_value <= 1.0e-3:
            self.get_logger().debug("Force value below threshold; stopping effect")
            self._stop_effect()
            return

        self._send_force(force_value)

    def _send_force(self, force_value: float) -> None:
        if self.dry_run or not self._effect or not self._haptic:
            return
        if not self._client or not self._client.IsConnected():
            return
        if not self._weart_started:
            self.get_logger().debug("WEART not started yet, skipping force send")
            return

        self.get_logger().debug(
            f"WEART send: force={force_value:.3f} active={self._effect_active}"
        )

        with self._weart_lock:
            try:
                temperature = WeArtTemperature()
                temperature.active = False
                force = WeArtForce(active=True, force=force_value)

                texture = WeArtTexture(
                    active=False,
                    texture_type=getattr(
                        WeArtCommon.TextureType, "AluminiumFineMeshSlow"
                    ),
                    velocity=0.0,
                    volume=0.0,
                )

                self._effect.Set(temperature, force, texture)
                texture.textureVelocity = 0.0

                if not self._effect_active:
                    self._haptic.AddEffect(self._effect)
                    self._effect_active = True
                self._haptic.UpdateEffects()

                # Send a direct force message as a fallback to ensure the command
                # reaches the middleware even if the effect update path is unreliable.
                direct_msg = WeArtMessages.SetForceMessage([force_value, 0.0, 0.0])
                self._haptic.SendMessage(direct_msg)
                self.get_logger().debug(
                    f"WEART direct force message sent: {force_value:.3f}"
                )
            except Exception as exc:
                self.get_logger().error(f"Failed to send WEART force: {exc}")

    def _stop_effect(self) -> None:
        if self.dry_run or not self._effect_active or self._haptic is None or self._effect is None:
            return

        with self._weart_lock:
            try:
                self._haptic.RemoveEffect(self._effect)
                stop_msg = WeArtMessages.StopForceMessage()
                self._haptic.SendMessage(stop_msg)
            except Exception as exc:
                self.get_logger().warning(f"Failed to stop WEART effect: {exc}")
            finally:
                self._effect_active = False

    def _watchdog(self) -> None:
        if self._last_update_monotonic <= 0.0:
            return
        if time.monotonic() - self._last_update_monotonic > 1.0:
            self._stop_effect()

    def shutdown(self) -> None:
        self._stop_effect()
        if self.dry_run or self._client is None:
            return
        with self._weart_lock:
            try:
                self._client.Stop()
            except Exception as exc:
                self.get_logger().warning(f"WEART Stop() failed: {exc}")
            try:
                self._client.Close()
            except Exception as exc:
                self.get_logger().warning(f"WEART Close() failed: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[BridgeNodeDummy] = None
    try:
        node = BridgeNodeDummy()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
