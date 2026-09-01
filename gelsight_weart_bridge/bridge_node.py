#!/usr/bin/env python3
"""ROS 2 bridge from AI4CE GelSight point clouds to a WEART haptic device."""

from __future__ import annotations

import math
import threading
import time
from typing import Optional

import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from weartsdk import (
    MiddlewareStatusListener,
    TouchEffect,
    WeArtClient,
    WeArtCommon,
    WeArtForce,
    WeArtHapticObject,
    WeArtTemperature,
    WeArtTexture,
)


class GelSightWeartBridge(Node):
    """Convert GelSight contact geometry into WEART force and texture commands."""

    def __init__(self) -> None:
        super().__init__("gelsight_weart_bridge")

        self._declare_parameters()
        self._load_parameters()

        self._weart_lock = threading.RLock()
        self._client: Optional[WeArtClient] = None
        self._middleware_listener: Optional[MiddlewareStatusListener] = None
        self._haptic: Optional[WeArtHapticObject] = None
        self._effect: Optional[TouchEffect] = None
        self._effect_active = False
        self._weart_started = False
        self._start_error_reported = False

        self._last_input_monotonic = 0.0
        self._last_send_monotonic = 0.0
        self._last_debug_monotonic = 0.0
        self._smoothed_force = 0.0
        self._previous_centroid: Optional[np.ndarray] = None
        self._previous_stamp: Optional[float] = None

        if not self.dry_run:
            self._initialize_weart()

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._patch_sub = message_filters.Subscriber(
            self, PointCloud2, self.patch_topic, qos_profile=qos
        )
        self._mask_sub = message_filters.Subscriber(
            self, PointCloud2, self.mask_topic, qos_profile=qos
        )
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self._patch_sub, self._mask_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop_sec,
        )
        self._synchronizer.registerCallback(self._on_tactile_pair)

        self._watchdog_timer = self.create_timer(0.05, self._watchdog)
        self._start_timer = None
        if not self.dry_run and self.auto_start_weart:
            self._start_timer = self.create_timer(1.0, self._try_start_weart)
            self._try_start_weart()

        self.get_logger().info(
            f"Listening to {self.patch_topic} and {self.mask_topic}; "
            f"WEART target={self.hand_side}/{','.join(self.actuation_points)}"
        )

    def _declare_parameters(self) -> None:
        params = [
            ("patch_topic", "/gelsight_capture/patch"),
            ("mask_topic", "/gelsight_capture/mask"),
            ("sync_queue_size", 10),
            ("sync_slop_sec", 0.08),
            ("weart_ip", "127.0.0.1"),
            ("weart_port", 13031),
            ("auto_start_weart", True),
            ("dry_run", False),
            ("hand_side", "Right"),
            ("actuation_points", ["Index"]),
            ("contact_mask_threshold", 0.5),
            ("min_contact_fraction", 0.002),
            ("depth_deadband", 0.01),
            ("depth_full_scale", 0.50),
            ("area_full_scale", 0.15),
            ("depth_weight", 0.80),
            ("force_gain", 1.0),
            ("force_smoothing_alpha", 0.25),
            ("enable_texture", True),
            ("texture_type", "AluminiumFineMeshSlow"),
            ("roughness_deadband", 0.002),
            ("roughness_full_scale", 0.08),
            ("texture_gain", 1.0),
            ("min_texture_volume", 3.0),
            ("minimum_texture_velocity", 0.02),
            ("slip_speed_full_scale_px_s", 200.0),
            ("max_update_rate_hz", 20.0),
            ("input_timeout_sec", 0.30),
            ("debug_log_period_sec", 1.0),
        ]
        self.declare_parameters(namespace="", parameters=params)

    def _load_parameters(self) -> None:
        value = lambda name: self.get_parameter(name).value

        self.patch_topic = str(value("patch_topic"))
        self.mask_topic = str(value("mask_topic"))
        self.sync_queue_size = int(value("sync_queue_size"))
        self.sync_slop_sec = float(value("sync_slop_sec"))
        self.weart_ip = str(value("weart_ip"))
        self.weart_port = int(value("weart_port"))
        self.auto_start_weart = bool(value("auto_start_weart"))
        self.dry_run = bool(value("dry_run"))
        self.hand_side = str(value("hand_side"))
        self.actuation_points = [str(item) for item in value("actuation_points")]
        self.contact_mask_threshold = float(value("contact_mask_threshold"))
        self.min_contact_fraction = float(value("min_contact_fraction"))
        self.depth_deadband = float(value("depth_deadband"))
        self.depth_full_scale = float(value("depth_full_scale"))
        self.area_full_scale = float(value("area_full_scale"))
        self.depth_weight = float(value("depth_weight"))
        self.force_gain = float(value("force_gain"))
        self.force_smoothing_alpha = float(value("force_smoothing_alpha"))
        self.enable_texture = bool(value("enable_texture"))
        self.texture_type_name = str(value("texture_type"))
        self.roughness_deadband = float(value("roughness_deadband"))
        self.roughness_full_scale = float(value("roughness_full_scale"))
        self.texture_gain = float(value("texture_gain"))
        self.min_texture_volume = float(value("min_texture_volume"))
        self.minimum_texture_velocity = float(value("minimum_texture_velocity"))
        self.slip_speed_full_scale_px_s = float(value("slip_speed_full_scale_px_s"))
        self.max_update_rate_hz = float(value("max_update_rate_hz"))
        self.input_timeout_sec = float(value("input_timeout_sec"))
        self.debug_log_period_sec = float(value("debug_log_period_sec"))

        if not 0.0 < self.force_smoothing_alpha <= 1.0:
            raise ValueError("force_smoothing_alpha must be in (0, 1]")
        if not 0.0 <= self.depth_weight <= 1.0:
            raise ValueError("depth_weight must be in [0, 1]")
        for name, number in (
            ("depth_full_scale", self.depth_full_scale),
            ("area_full_scale", self.area_full_scale),
            ("roughness_full_scale", self.roughness_full_scale),
            ("slip_speed_full_scale_px_s", self.slip_speed_full_scale_px_s),
            ("max_update_rate_hz", self.max_update_rate_hz),
            ("input_timeout_sec", self.input_timeout_sec),
        ):
            if number <= 0.0:
                raise ValueError(f"{name} must be positive")

    def _initialize_weart(self) -> None:
        self._client = WeArtClient(self.weart_ip, self.weart_port)
        try:
            self._client.Run()
        except SystemExit as exc:
            # The current SDK calls sys.exit() when the middleware socket cannot
            # be opened. Convert that into an ordinary node startup failure.
            raise RuntimeError(
                f"Could not connect to WEART middleware at "
                f"{self.weart_ip}:{self.weart_port}"
            ) from exc
        if not self._client.IsConnected():
            raise RuntimeError(
                f"Could not connect to WEART middleware at "
                f"{self.weart_ip}:{self.weart_port}"
            )

        self._middleware_listener = MiddlewareStatusListener()
        self._client.AddMessageListener(self._middleware_listener)

        temperature = WeArtTemperature()
        temperature.active = False

        force = WeArtForce(active=True, force=0.0)
        texture = WeArtTexture(
            active=False,
            texture_type=self._parse_texture_type(self.texture_type_name),
            velocity=0.0,
            volume=0.0,
        )

        self._effect = TouchEffect(temperature, force, texture)
        self._haptic = WeArtHapticObject(self._client)
        self._haptic.handSideFlag = self._parse_hand_side(self.hand_side)
        self._haptic.actuationPointFlag = self._parse_actuation_points(
            self.actuation_points
        )

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

    @staticmethod
    def _parse_texture_type(name: str):
        try:
            return getattr(WeArtCommon.TextureType, name)
        except AttributeError as exc:
            raise ValueError(f"Unknown WEART texture_type: {name!r}") from exc

    def _try_start_weart(self) -> None:
        if self.dry_run or self._weart_started or self._client is None:
            return

        try:
            if not self._client.IsConnected():
                return

            if self._middleware_listener is not None:
                status = self._middleware_listener.LastStatus()
                connected_devices = getattr(status, "connectedDevices", [])
                if not connected_devices:
                    return

            self._client.Start()
            self._weart_started = True
            self._start_error_reported = False
            self.get_logger().info("WEART haptic session started")
            if self._start_timer is not None:
                self._start_timer.cancel()
        except Exception as exc:  # SDK/network failures should not kill ROS.
            if not self._start_error_reported:
                self.get_logger().warning(f"Cannot start WEART session yet: {exc}")
                self._start_error_reported = True

    def _on_tactile_pair(self, patch_msg: PointCloud2, mask_msg: PointCloud2) -> None:
        now_monotonic = time.monotonic()
        self._last_input_monotonic = now_monotonic

        min_period = 1.0 / self.max_update_rate_hz
        if now_monotonic - self._last_send_monotonic < min_period:
            return
        self._last_send_monotonic = now_monotonic

        try:
            patch_xyz = self._cloud_to_xyz(patch_msg)
            mask_xyz = self._cloud_to_xyz(mask_msg)
        except Exception as exc:
            self.get_logger().error(f"Could not decode GelSight point clouds: {exc}")
            return

        if patch_xyz.shape != mask_xyz.shape or patch_xyz.shape[0] == 0:
            self.get_logger().warning(
                "GelSight patch and mask point clouds have different/empty shapes"
            )
            self._stop_effect()
            return

        finite = np.isfinite(patch_xyz).all(axis=1) & np.isfinite(mask_xyz[:, 2])
        if not np.any(finite):
            self._stop_effect()
            return

        patch_xyz = patch_xyz[finite]
        mask_values = mask_xyz[finite, 2]
        contact = mask_values > self.contact_mask_threshold
        contact_fraction = float(np.mean(contact))

        if contact_fraction < self.min_contact_fraction or not np.any(contact):
            self._previous_centroid = None
            self._previous_stamp = None
            self._smoothed_force = 0.0
            self._stop_effect()
            return

        z = patch_xyz[:, 2]
        self.get_logger().info(
            f"z: min={z.min():.3f}, "
            f"max={z.max():.3f}, "
            f"mean={z.mean():.3f}, "
            f"std={z.std():.3f}"
        )
        non_contact = ~contact
        if np.count_nonzero(non_contact) >= max(20, int(0.01 * z.size)):
            baseline = float(np.median(z[non_contact]))
        else:
            baseline = float(np.percentile(z, 90.0))

        contact_z = z[contact]
        contact_level = float(np.percentile(contact_z, 20.0))
        indentation = max(0.0, baseline - contact_level)

        contact_median = float(np.median(contact_z))
        roughness = float(
            1.4826 * np.median(np.abs(contact_z - contact_median))
        )

        stamp = self._stamp_seconds(patch_msg)
        centroid = np.mean(patch_xyz[contact, :2], axis=0)
        slip_speed = self._compute_slip_speed(centroid, stamp)

        depth_normalized = self._normalize(
            indentation, self.depth_deadband, self.depth_full_scale
        )
        if depth_normalized <= 0.0:
            self._previous_centroid = None
            self._previous_stamp = None
            self._smoothed_force = 0.0
            self._stop_effect()
            return

        area_normalized = float(
            np.clip(contact_fraction / self.area_full_scale, 0.0, 1.0)
        )
        raw_force = self.force_gain * (
            self.depth_weight * depth_normalized
            + (1.0 - self.depth_weight) * area_normalized
        )
        raw_force = float(np.clip(raw_force, 0.0, 1.0))
        self._smoothed_force = (
            self.force_smoothing_alpha * raw_force
            + (1.0 - self.force_smoothing_alpha) * self._smoothed_force
        )

        roughness_normalized = self._normalize(
            roughness, self.roughness_deadband, self.roughness_full_scale
        )
        texture_volume = float(
            np.clip(100.0 * self.texture_gain * roughness_normalized, 0.0, 100.0)
        )
        texture_velocity = float(
            np.clip(
                0.5 * slip_speed / self.slip_speed_full_scale_px_s,
                0.0,
                0.5,
            )
        )
        if texture_volume >= self.min_texture_volume:
            texture_velocity = max(
                self.minimum_texture_velocity, texture_velocity
            )

        self.get_logger().info(
            f"""
            baseline={baseline:.3f}
            contact={contact_level:.3f}
            indent={indentation:.3f}
            depth_norm={depth_normalized:.3f}
            area={contact_fraction:.3f}
            area_norm={area_normalized:.3f}
            raw_force={raw_force:.3f}
            smooth={self._smoothed_force:.3f}
            """
        )

        self._send_haptics(
            force_value=self._smoothed_force,
            texture_volume=texture_volume,
            texture_velocity=texture_velocity,
        )
        self._debug_log(
            indentation=indentation,
            contact_fraction=contact_fraction,
            roughness=roughness,
            slip_speed=slip_speed,
            force_value=self._smoothed_force,
            texture_volume=texture_volume,
        )

    @staticmethod
    def _cloud_to_xyz(msg: PointCloud2) -> np.ndarray:
        points = point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=False
        )

        if isinstance(points, np.ndarray):
            if points.dtype.names:
                return np.column_stack(
                    (
                        np.asarray(points["x"]).reshape(-1),
                        np.asarray(points["y"]).reshape(-1),
                        np.asarray(points["z"]).reshape(-1),
                    )
                ).astype(np.float32, copy=False)
            array = np.asarray(points, dtype=np.float32)
        else:
            array = np.asarray(list(points), dtype=np.float32)

        if array.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        return array.reshape(-1, 3)

    @staticmethod
    def _stamp_seconds(msg: PointCloud2) -> float:
        seconds = float(msg.header.stamp.sec) + 1.0e-9 * float(
            msg.header.stamp.nanosec
        )
        return seconds if seconds > 0.0 else time.monotonic()

    def _compute_slip_speed(self, centroid: np.ndarray, stamp: float) -> float:
        speed = 0.0
        if self._previous_centroid is not None and self._previous_stamp is not None:
            dt = stamp - self._previous_stamp
            if 1.0e-4 < dt < 1.0:
                speed = float(
                    np.linalg.norm(centroid - self._previous_centroid) / dt
                )
                if not math.isfinite(speed):
                    speed = 0.0
        self._previous_centroid = centroid
        self._previous_stamp = stamp
        return speed

    @staticmethod
    def _normalize(value: float, deadband: float, full_scale: float) -> float:
        denominator = max(full_scale - deadband, 1.0e-9)
        return float(np.clip((value - deadband) / denominator, 0.0, 1.0))

    def _send_haptics(
        self, force_value: float, texture_volume: float, texture_velocity: float
    ) -> None:
        if self.dry_run:
            return
        if not self._weart_started:
            return
        if self._haptic is None or self._effect is None:
            return

        with self._weart_lock:
            try:
                # Use fresh value objects. WeArtHapticObject caches the previous
                # objects by reference, so mutating the old objects in place would
                # make UpdateEffects() believe that nothing changed.
                temperature = WeArtTemperature()
                temperature.active = False
                force = WeArtForce(
                    active=True, force=float(np.clip(force_value, 0.0, 1.0))
                )

                texture_active = (
                    self.enable_texture
                    and texture_volume >= self.min_texture_volume
                )
                requested_velocity = (
                    float(np.clip(texture_velocity, 0.0, 0.5))
                    if texture_active
                    else 0.0
                )
                texture = WeArtTexture(
                    active=texture_active,
                    texture_type=self._parse_texture_type(self.texture_type_name),
                    velocity=requested_velocity,
                    volume=(
                        float(np.clip(texture_volume, 0.0, 100.0))
                        if texture_active
                        else 0.0
                    ),
                )

                self._effect.Set(temperature, force, texture)
                # In SDK 2.0.3, TouchEffect.Set() sets a changed texture's
                # velocity to 0.5. Restore the velocity computed by this bridge.
                texture.textureVelocity = requested_velocity

                if not self._effect_active:
                    self._haptic.AddEffect(self._effect)
                    self._effect_active = True
                else:
                    self._haptic.UpdateEffects()
            except Exception as exc:
                self.get_logger().error(f"Failed to send WEART haptics: {exc}")

    def _stop_effect(self) -> None:
        if self.dry_run:
            return
        if not self._effect_active or self._haptic is None or self._effect is None:
            return

        with self._weart_lock:
            try:
                self._haptic.RemoveEffect(self._effect)
            except Exception as exc:
                self.get_logger().warning(f"Failed to stop WEART effect: {exc}")
            finally:
                self._effect_active = False

    def _watchdog(self) -> None:
        if self._last_input_monotonic <= 0.0:
            return
        if time.monotonic() - self._last_input_monotonic > self.input_timeout_sec:
            self._smoothed_force = 0.0
            self._previous_centroid = None
            self._previous_stamp = None
            self._stop_effect()

    def _debug_log(
        self,
        *,
        indentation: float,
        contact_fraction: float,
        roughness: float,
        slip_speed: float,
        force_value: float,
        texture_volume: float,
    ) -> None:
        if self.debug_log_period_sec <= 0.0:
            return
        now = time.monotonic()
        if now - self._last_debug_monotonic < self.debug_log_period_sec:
            return
        self._last_debug_monotonic = now
        self.get_logger().info(
            "GelSight -> WEART: "
            f"depth={indentation:.5f}, area={contact_fraction:.4f}, "
            f"roughness={roughness:.5f}, slip={slip_speed:.1f}px/s, "
            f"force={force_value:.3f}, texture_volume={texture_volume:.1f}"
        )

    def shutdown(self) -> None:
        self._stop_effect()
        if self.dry_run or self._client is None:
            return
        with self._weart_lock:
            try:
                if self._weart_started:
                    self._client.Stop()
            except Exception as exc:
                self.get_logger().warning(f"WEART Stop() failed: {exc}")
            try:
                self._client.Close()
            except Exception as exc:
                self.get_logger().warning(f"WEART Close() failed: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[GelSightWeartBridge] = None
    try:
        node = GelSightWeartBridge()
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
