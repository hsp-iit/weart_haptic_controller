#!/usr/bin/env python3
"""Publish WEART finger data and forward tactile forces through one client.

This node is the sole owner of the WEART middleware connection. Two things
are independently configurable through the YAML config file:

  tactile_sensor: "gelsight" | "xhand"
      Where estimated per-finger contact force comes from, to be mapped onto
      the WEART haptic thimbles (force feedback into the glove):
        - "gelsight": per-finger geometry_msgs/Vector3 topics carrying a raw
          force vector (as produced by a GelSight force estimator).
        - "xhand": a single std_msgs/Float64MultiArray topic
          (default /xhand/tactile_normalized, published by
          teleoperation/weart_xhand_direct_retargeter.py) already normalized
          to [0, 1], ordered [thumb, index, middle]. This folds in the logic
          from teleoperation/xhand_tactile_to_weart_haptics.py so that only
          one process ever connects to the WEART middleware.

  hand_control: "leap_hand" | "xhand"
      Which downstream hand-control node consumes the /weart/{thumb,index,
      middle}/raw closure/adduction data this node always publishes (either
      weart_leap_retargeter_adaptive_lat.py or
      weart_xhand_direct_retargeter.py). Both retargeters read the exact same
      raw topics, so this does not change what gets published here; it is
      recorded and logged for clarity/validation, and to make the intended
      pairing with tactile_sensor explicit in one place.
"""

import logging
import threading
import time

import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from weartsdk import (
    DeviceStatusListener,
    MiddlewareStatusListener,
    TouchEffect,
    WeArtClient,
    WeArtCommon,
    WeArtForce,
    WeArtHapticObject,
    WeArtMessages,
    WeArtTemperature,
    WeArtTexture,
    WeArtThimbleTrackingObject,
    WeArtTrackingCalibration,
    WeArtTrackingRawData,
)


HAND = WeArtCommon.HandSide.Right
POLL_SLEEP_S = 0.001
DIAGNOSTIC_PERIOD_S = 1.0
GELSIGHT_FORCE_TIMEOUT_S = 1.0

TACTILE_SENSORS = ("gelsight", "xhand")
HAND_CONTROLS = ("leap_hand", "xhand")

FINGERS = {
    "thumb": {
        "point": WeArtCommon.ActuationPoint.Thumb,
        "topic": "/weart/thumb/raw",
        "has_abduction": True,
    },
    "index": {
        "point": WeArtCommon.ActuationPoint.Index,
        "topic": "/weart/index/raw",
        "has_abduction": False,
    },
    "middle": {
        "point": WeArtCommon.ActuationPoint.Middle,
        "topic": "/weart/middle/raw",
        "has_abduction": False,
    },
}


class CombinedController(Node):
    """The sole owner of the WEART middleware connection and lifecycle."""

    def __init__(self):
        super().__init__("weart_combined_controller")
        self.declare_parameters(
            namespace="",
            parameters=[
                ("tactile_sensor", "gelsight"),
                ("hand_control", "leap_hand"),
                ("thumb_force_topic", "/gelsight/thumb/force"),
                ("index_force_topic", "/gelsight/index/force"),
                ("middle_force_topic", "/gelsight/middle/force"),
                ("force_full_scale_z", 2),
                ("xhand_tactile_topic", "/xhand/tactile_normalized"),
                ("xhand_gain", 1.0),
                ("xhand_deadband", 0.02),
                ("xhand_smoothing_alpha", 0.35),
                ("xhand_input_timeout_s", 0.5),
                ("weart_ip", WeArtCommon.DEFAULT_IP_ADDRESS),
                ("weart_port", WeArtCommon.DEFAULT_TCP_PORT),
                ("hand_side", "Right"),
                ("force_log_period_s", 1.0),
            ],
        )

        self.tactile_sensor = str(self.get_parameter("tactile_sensor").value).strip().lower()
        if self.tactile_sensor not in TACTILE_SENSORS:
            raise ValueError(
                f"tactile_sensor must be one of {TACTILE_SENSORS}, got {self.tactile_sensor!r}"
            )

        self.hand_control = str(self.get_parameter("hand_control").value).strip().lower()
        if self.hand_control not in HAND_CONTROLS:
            raise ValueError(
                f"hand_control must be one of {HAND_CONTROLS}, got {self.hand_control!r}"
            )

        self.weart_ip = str(self.get_parameter("weart_ip").value)
        self.weart_port = int(self.get_parameter("weart_port").value)
        self.hand_side = str(self.get_parameter("hand_side").value)
        self.force_log_period_s = float(self.get_parameter("force_log_period_s").value)
        if self.force_log_period_s < 0.0:
            raise ValueError("force_log_period_s cannot be negative")

        self.finger_publishers = {
            name: self.create_publisher(String, config["topic"], 20)
            for name, config in FINGERS.items()
        }

        # State shared by both tactile-sensor backends, always keyed by all
        # three fingers so the watchdog and haptic effects behave the same
        # regardless of which backend is active.
        self._weart_lock = threading.RLock()
        self._client = None
        self._effects = {}
        self._haptics = {}
        self._effect_active = {finger_name: False for finger_name in FINGERS}
        self._last_force_update = {finger_name: 0.0 for finger_name in FINGERS}
        self._last_force_log = {finger_name: 0.0 for finger_name in FINGERS}
        self._raw_started = False

        self._tactile_subscriptions = []
        if self.tactile_sensor == "gelsight":
            self._force_timeout_s = GELSIGHT_FORCE_TIMEOUT_S
            self.force_topics = {
                "thumb": str(self.get_parameter("thumb_force_topic").value),
                "index": str(self.get_parameter("index_force_topic").value),
                "middle": str(self.get_parameter("middle_force_topic").value),
            }
            self.force_full_scale_z = float(self.get_parameter("force_full_scale_z").value)
            if self.force_full_scale_z <= 0.0:
                raise ValueError("force_full_scale_z must be positive")
            self._tactile_subscriptions = [
                self.create_subscription(
                    Vector3,
                    topic,
                    lambda message, finger_name=finger_name: self._on_gelsight_force(
                        finger_name, message
                    ),
                    10,
                )
                for finger_name, topic in self.force_topics.items()
            ]
        else:  # xhand
            self.xhand_tactile_topic = str(self.get_parameter("xhand_tactile_topic").value)
            self.xhand_gain = float(self.get_parameter("xhand_gain").value)
            self.xhand_deadband = float(self.get_parameter("xhand_deadband").value)
            self.xhand_smoothing_alpha = float(self.get_parameter("xhand_smoothing_alpha").value)
            self._force_timeout_s = float(self.get_parameter("xhand_input_timeout_s").value)
            self._xhand_smoothed = {finger_name: 0.0 for finger_name in FINGERS}
            self._xhand_last_log = 0.0
            self._tactile_subscriptions = [
                self.create_subscription(
                    Float64MultiArray,
                    self.xhand_tactile_topic,
                    self._on_xhand_tactile,
                    20,
                )
            ]

        self._force_watchdog = self.create_timer(0.1, self._watchdog)

        self._middleware_listener = MiddlewareStatusListener()
        self._device_listener = DeviceStatusListener()
        self._calibration = WeArtTrackingCalibration()
        self._tracking_objects = {}
        self._raw_objects = {}

    def start(self):
        """Connect, calibrate and enable raw data exactly once for this process.

        The finger-tracking half of this method (everything through the raw-data
        readiness check) mirrors teleoperation/weart_all_fingers_publisher.py so
        the two behave identically; only the haptic-force setup is specific to
        this node.
        """
        print(f"Tactile sensor source: {self.tactile_sensor}")
        print(f"Hand control target: {self.hand_control}")

        self._client = WeArtClient(
            self.weart_ip, self.weart_port, log_level=logging.INFO
        )
        self._middleware_listener.AddStatusCallback(self._on_middleware_status)
        self._device_listener.AddStatusCallback(self._on_device_status)
        self._client.AddMessageListener(self._middleware_listener)
        self._client.AddMessageListener(self._device_listener)
        self._client.AddMessageListener(self._calibration)

        for finger_name, config in FINGERS.items():
            tracking = WeArtThimbleTrackingObject(HAND, config["point"])
            raw = WeArtTrackingRawData(HAND, config["point"])
            self._tracking_objects[finger_name] = tracking
            self._raw_objects[finger_name] = raw
            self._client.AddThimbleTracking(tracking)
            self._client.AddMessageListener(raw)

        self._configure_haptics()

        print("Connessione al WEART Middleware...")
        self._client.Run()

        if not self._client.IsConnected():
            raise RuntimeError("Connessione TCP al WEART Middleware non riuscita.")

        print("TCP collegato correttamente.")
        print("Attendo 3 secondi per lo stato del Middleware...")

        deadline = time.time() + 3.0
        while time.time() < deadline:
            status = self._middleware_listener.LastStatus()
            if status.timestamp != 0:
                break
            time.sleep(0.1)

        status = self._middleware_listener.LastStatus()
        if status.timestamp == 0:
            print(
                "\n[ATTENZIONE] Nessun MiddlewareStatusUpdate ricevuto."
                "\nIl socket è comunque collegato: provo ad avviare il device."
            )
        else:
            print(
                f"\nUltimo stato Middleware: "
                f"{status.status}, devices={len(status.connectedDevices)}"
            )

        print("\nInvio client.Start()...")
        self._client.Start()
        time.sleep(2.0)

        status = self._middleware_listener.LastStatus()
        print(
            f"Stato dopo Start: "
            f"{status.status}, code={status.statusCode}, "
            f"error='{status.errorDesc}'"
        )

        input(
            "\nIndossa il guanto, tieni la mano ferma nella posizione "
            "di calibrazione e premi INVIO..."
        )

        print("Avvio calibrazione WEART...")
        self._client.StartCalibration()

        calib_deadline = time.time() + 20.0
        while not self._calibration.getResult():
            if time.time() > calib_deadline:
                raise RuntimeError(
                    "Timeout calibrazione WEART: nessun risultato dopo 20 secondi."
                )
            time.sleep(0.2)

        self._client.StopCalibration()
        print("Calibrazione WEART completata.")

        print("\nAvvio dati RAW di pollice, indice e medio...")
        self._client.StartRawData()
        self._raw_started = True

        waiting = set(FINGERS.keys())
        raw_deadline = time.time() + 10.0

        while waiting:
            for finger_name in list(waiting):
                sample = self._raw_objects[finger_name].GetLastSample()
                if sample.timestamp != 0:
                    print(f"Primo campione RAW ricevuto: {finger_name}")
                    waiting.remove(finger_name)

            if time.time() > raw_deadline:
                raise RuntimeError(
                    "Timeout: nessun campione RAW per: " + ", ".join(sorted(waiting))
                )

            time.sleep(0.01)

        print("\nPubblicazione ROS 2 attiva:")
        for finger_name, config in FINGERS.items():
            print(f"  {finger_name:>6}: {config['topic']}")

        print(
            "\nOgni dito viene pubblicato solo quando cambia "
            "il relativo timestamp RAW."
        )
        print("Premi CTRL+C per terminare.\n")

        if self.tactile_sensor == "gelsight":
            tactile_source_desc = ", ".join(self.force_topics.values())
        else:
            tactile_source_desc = self.xhand_tactile_topic

        self.get_logger().info(
            "Ready: publishing /weart/{thumb,index,middle}/raw; "
            f"hand_control={self.hand_control}; tactile_sensor={self.tactile_sensor}; "
            f"haptic input: {tactile_source_desc}"
        )

    def _configure_haptics(self):
        temperature = WeArtTemperature()
        temperature.active = False
        force = WeArtForce(active=True, force=0.0)
        texture = WeArtTexture(
            active=False,
            texture_type=getattr(WeArtCommon.TextureType, "AluminiumFineMeshSlow"),
            velocity=0.0,
            volume=0.0,
        )
        for finger_name in FINGERS:
            haptic = WeArtHapticObject(self._client)
            haptic.handSideFlag = self._parse_hand_side(self.hand_side)
            haptic.actuationPointFlag = FINGERS[finger_name]["point"]
            self._haptics[finger_name] = haptic
            self._effects[finger_name] = TouchEffect(temperature, force, texture)

    def publish_raw_data(self):
        """Match teleoperation/weart_all_fingers_publisher.py's publish loop exactly."""
        last_timestamp = {finger_name: None for finger_name in FINGERS}
        samples_since_diag = {finger_name: 0 for finger_name in FINGERS}
        latest_closure = {finger_name: 0.0 for finger_name in FINGERS}
        latest_tof = {finger_name: None for finger_name in FINGERS}
        diag_start = time.perf_counter()

        while rclpy.ok():
            for finger_name, config in FINGERS.items():
                sample = self._raw_objects[finger_name].GetLastSample()

                if sample.timestamp == 0:
                    continue

                if sample.timestamp == last_timestamp[finger_name]:
                    continue

                last_timestamp[finger_name] = sample.timestamp

                tracking = self._tracking_objects[finger_name]

                closure = float(tracking.GetClosure())
                opening = 1.0 - closure

                if config["has_abduction"]:
                    abduction = float(tracking.GetAbduction())
                    adduction = 1.0 - abduction
                else:
                    abduction = 0.0
                    adduction = 1.0

                data = sample.data
                acc = data.accelerometer
                gyro = data.gyroscope
                tof = data.timeOfFlight

                line = (
                    f"ts={sample.timestamp} | "
                    f"closure={closure:.6f} | "
                    f"opening={opening:.6f} | "
                    f"abduction={abduction:.6f} | "
                    f"adduction={adduction:.6f} | "
                    f"ACC[g]=({acc.x:.6f}, {acc.y:.6f}, {acc.z:.6f}) | "
                    f"GYRO[deg/s]=({gyro.x:.6f}, {gyro.y:.6f}, {gyro.z:.6f}) | "
                    f"ToF[mm]={tof.distance}"
                )

                message = String()
                message.data = line
                self.finger_publishers[finger_name].publish(message)

                samples_since_diag[finger_name] += 1
                latest_closure[finger_name] = closure
                latest_tof[finger_name] = tof.distance

            rclpy.spin_once(self, timeout_sec=0.0)

            now = time.perf_counter()
            elapsed = now - diag_start

            if elapsed >= DIAGNOSTIC_PERIOD_S:
                parts = []
                for finger_name in FINGERS:
                    hz = samples_since_diag[finger_name] / elapsed
                    parts.append(
                        f"{finger_name}: {hz:.1f} Hz "
                        f"(closure={latest_closure[finger_name]:.3f}, "
                        f"ToF={latest_tof[finger_name]})"
                    )
                    samples_since_diag[finger_name] = 0

                self.get_logger().info(" | ".join(parts))
                diag_start = now

            time.sleep(POLL_SLEEP_S)

    def _on_gelsight_force(self, finger_name, message):
        now = time.monotonic()
        self._last_force_update[finger_name] = now
        # GelSight publishes compression as a negative z force. Values around
        # -0.25 are the observed full-contact range for these topics.
        force_value = max(
            0.0, min(1.0, -float(message.z) / self.force_full_scale_z)
        )
        if now - self._last_force_log[finger_name] >= self.force_log_period_s:
            self.get_logger().info(
                f"{finger_name} force: topic={self.force_topics[finger_name]}, "
                f"received z={message.z:.6f}, WEART command={force_value:.3f}"
            )
            self._last_force_log[finger_name] = now
        if force_value <= 1.0e-3:
            self._stop_effect(finger_name)
        else:
            self._send_force(finger_name, force_value)

    def _on_xhand_tactile(self, message):
        """Forward normalized XHAND tactile values to WEART haptic thimbles.

        Folded in from teleoperation/xhand_tactile_to_weart_haptics.py: the
        publisher (teleoperation/weart_xhand_direct_retargeter.py) already
        normalizes each value to [0, 1] and orders them [thumb, index,
        middle], so this only applies gain/smoothing/deadband before driving
        the same haptic effects the gelsight backend uses.
        """
        values = list(message.data)
        if len(values) < len(FINGERS):
            self.get_logger().warning(
                f"Expected at least {len(FINGERS)} values on {self.xhand_tactile_topic} "
                f"(thumb, index, middle), got {len(values)}",
                throttle_duration_sec=1.0,
            )
            return

        now = time.monotonic()
        sent = {}
        for finger_name, raw_value in zip(FINGERS, values):
            self._last_force_update[finger_name] = now
            target = max(0.0, min(1.0, float(raw_value) * self.xhand_gain))
            previous = self._xhand_smoothed[finger_name]
            value = (
                self.xhand_smoothing_alpha * target
                + (1.0 - self.xhand_smoothing_alpha) * previous
            )
            value = max(0.0, min(1.0, value))
            self._xhand_smoothed[finger_name] = value
            sent[finger_name] = value
            if value <= self.xhand_deadband:
                self._stop_effect(finger_name)
            else:
                self._send_force(finger_name, value)

        if self.force_log_period_s > 0.0 and now - self._xhand_last_log >= self.force_log_period_s:
            self._xhand_last_log = now
            self.get_logger().info(
                f"XHAND tactile ({self.xhand_tactile_topic}) -> WEART force: "
                + " | ".join(f"{name}={value:.2f}" for name, value in sent.items())
            )

    def _send_force(self, finger_name, force_value):
        haptic = self._haptics.get(finger_name)
        effect = self._effects.get(finger_name)
        if not self._client or not self._client.IsConnected() or not effect or not haptic:
            return
        with self._weart_lock:
            try:
                temperature = WeArtTemperature()
                temperature.active = False
                force = WeArtForce(active=True, force=force_value)
                texture = WeArtTexture(
                    active=False,
                    texture_type=getattr(WeArtCommon.TextureType, "AluminiumFineMeshSlow"),
                    velocity=0.0,
                    volume=0.0,
                )
                effect.Set(temperature, force, texture)
                if not self._effect_active[finger_name]:
                    haptic.AddEffect(effect)
                    self._effect_active[finger_name] = True
                haptic.UpdateEffects()
                haptic.SendMessage(
                    WeArtMessages.SetForceMessage([force_value, 0.0, 0.0])
                )
            except Exception as exc:
                self.get_logger().error(f"Failed to send WEART force: {exc}")

    def _stop_effect(self, finger_name):
        haptic = self._haptics.get(finger_name)
        effect = self._effects.get(finger_name)
        if not self._effect_active[finger_name] or not haptic or not effect:
            return
        with self._weart_lock:
            try:
                haptic.RemoveEffect(effect)
                haptic.SendMessage(WeArtMessages.StopForceMessage())
            except Exception as exc:
                self.get_logger().warning(f"Failed to stop WEART effect: {exc}")
            finally:
                self._effect_active[finger_name] = False

    def _watchdog(self):
        for finger_name, last_update in self._last_force_update.items():
            if last_update and time.monotonic() - last_update > self._force_timeout_s:
                self._stop_effect(finger_name)

    def _on_middleware_status(self, status):
        self.get_logger().info(
            f"WEART middleware status={status.status}, devices={len(status.connectedDevices)}"
        )

    def _on_device_status(self, status):
        self.get_logger().info(f"WEART device status: {len(status.devices)} device(s)")

    @staticmethod
    def _parse_hand_side(name):
        try:
            return getattr(WeArtCommon.HandSide, name)
        except AttributeError as exc:
            raise ValueError("hand_side must be Left or Right") from exc

    def shutdown(self):
        self._force_watchdog.cancel()
        for finger_name in FINGERS:
            self._stop_effect(finger_name)
        if not self._client:
            return
        with self._weart_lock:
            if self._raw_started:
                try:
                    self._client.StopRawData()
                except Exception:
                    pass
            try:
                self._client.Stop()
            except Exception:
                pass
            try:
                self._client.Close()
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CombinedController()
        node.start()
        node.publish_raw_data()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().error(str(exc))
        else:
            print(f"Combined controller failed: {exc}")
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
