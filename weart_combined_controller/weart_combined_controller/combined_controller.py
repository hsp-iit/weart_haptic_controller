#!/usr/bin/env python3
"""Publish WEART finger data and forward GelSight forces through one client."""

import logging
import threading
import time

import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from std_msgs.msg import String
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
                ("thumb_force_topic", "/gelsight/thumb/force"),
                ("index_force_topic", "/gelsight/index/force"),
                ("weart_ip", WeArtCommon.DEFAULT_IP_ADDRESS),
                ("weart_port", WeArtCommon.DEFAULT_TCP_PORT),
                ("hand_side", "Right"),
                ("force_full_scale_z", 2),
                ("force_log_period_s", 1.0),
            ],
        )
        self.force_topics = {
            "thumb": str(self.get_parameter("thumb_force_topic").value),
            "index": str(self.get_parameter("index_force_topic").value),
        }
        self.weart_ip = str(self.get_parameter("weart_ip").value)
        self.weart_port = int(self.get_parameter("weart_port").value)
        self.hand_side = str(self.get_parameter("hand_side").value)
        self.force_full_scale_z = float(
            self.get_parameter("force_full_scale_z").value
        )
        if self.force_full_scale_z <= 0.0:
            raise ValueError("force_full_scale_z must be positive")
        self.force_log_period_s = float(
            self.get_parameter("force_log_period_s").value
        )
        if self.force_log_period_s < 0.0:
            raise ValueError("force_log_period_s cannot be negative")

        self.finger_publishers = {
            name: self.create_publisher(String, config["topic"], 20)
            for name, config in FINGERS.items()
        }
        self._force_subscriptions = [
            self.create_subscription(
                Vector3,
                topic,
                lambda message, finger_name=finger_name: self._on_force_vector(
                    finger_name, message
                ),
                10,
            )
            for finger_name, topic in self.force_topics.items()
        ]
        self._force_watchdog = self.create_timer(0.1, self._watchdog)

        self._weart_lock = threading.RLock()
        self._client = None
        self._effects = {}
        self._haptics = {}
        self._effect_active = {finger_name: False for finger_name in self.force_topics}
        self._last_force_update = {finger_name: 0.0 for finger_name in self.force_topics}
        self._last_force_log = {finger_name: 0.0 for finger_name in self.force_topics}
        self._raw_started = False

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
        self.get_logger().info(
            "Ready: publishing /weart/{thumb,index,middle}/raw; haptic inputs: %s"
            % ", ".join(self.force_topics.values())
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
        for finger_name in self.force_topics:
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

    def _on_force_vector(self, finger_name, message):
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
            if last_update and time.monotonic() - last_update > 1.0:
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
        for finger_name in self.force_topics:
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
