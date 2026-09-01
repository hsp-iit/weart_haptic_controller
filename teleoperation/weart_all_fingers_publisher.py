#!/usr/bin/env python3

import logging
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from weartsdk import *


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


class WeartMultiFingerPublisher(Node):
    def __init__(self):
        super().__init__("weart_multi_finger_publisher")
        self.finger_publishers = {}

        for finger_name, cfg in FINGERS.items():
            self.finger_publishers[finger_name] = self.create_publisher(
                String, cfg["topic"], 20
            )
            self.get_logger().info(
                f"{finger_name:>6} -> {cfg['topic']}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = WeartMultiFingerPublisher()

    client = WeArtClient(
        WeArtCommon.DEFAULT_IP_ADDRESS,
        WeArtCommon.DEFAULT_TCP_PORT,
        log_level=logging.INFO,
    )

    mw_listener = MiddlewareStatusListener()
    dev_listener = DeviceStatusListener()
    calibration = WeArtTrackingCalibration()

    client.AddMessageListener(mw_listener)
    client.AddMessageListener(dev_listener)
    client.AddMessageListener(calibration)

    tracking_objects = {}
    raw_objects = {}

    for finger_name, cfg in FINGERS.items():
        tracking = WeArtThimbleTrackingObject(HAND, cfg["point"])
        raw = WeArtTrackingRawData(HAND, cfg["point"])

        tracking_objects[finger_name] = tracking
        raw_objects[finger_name] = raw

        client.AddThimbleTracking(tracking)
        client.AddMessageListener(raw)

    raw_started = False

    def on_middleware_status(status):
        print(
            f"\n[MW STATUS] status={status.status}, "
            f"code={status.statusCode}, "
            f"version={status.version}, "
            f"devices={len(status.connectedDevices)}"
        )
        for d in status.connectedDevices:
            print(f"  -> MAC={d.macAddress}, hand={d.handSide}")

    def on_device_status(status):
        print(f"\n[DEVICE STATUS] devices={len(status.devices)}")
        for d in status.devices:
            print(
                f"  -> MAC={d.macAddress}, "
                f"hand={d.handSide}, "
                f"battery={d.batteryLevel}%"
            )
            for t in d.thimbles:
                print(
                    f"     thimble={t.id}, "
                    f"connected={t.connected}, "
                    f"code={t.statusCode}, "
                    f"error='{t.errorDesc}'"
                )

    mw_listener.AddStatusCallback(on_middleware_status)
    dev_listener.AddStatusCallback(on_device_status)

    try:
        print("Connessione al WEART Middleware...")
        client.Run()

        if not client.IsConnected():
            raise RuntimeError(
                "Connessione TCP al WEART Middleware non riuscita."
            )

        print("TCP collegato correttamente.")
        print("Attendo 3 secondi per lo stato del Middleware...")

        deadline = time.time() + 3.0
        while time.time() < deadline:
            status = mw_listener.LastStatus()
            if status.timestamp != 0:
                break
            time.sleep(0.1)

        status = mw_listener.LastStatus()

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
        client.Start()
        time.sleep(2.0)

        status = mw_listener.LastStatus()
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
        client.StartCalibration()

        calib_deadline = time.time() + 20.0
        while not calibration.getResult():
            if time.time() > calib_deadline:
                raise RuntimeError(
                    "Timeout calibrazione WEART: nessun risultato dopo 20 secondi."
                )
            time.sleep(0.2)

        client.StopCalibration()
        print("Calibrazione WEART completata.")

        print("\nAvvio dati RAW di pollice, indice e medio...")
        client.StartRawData()
        raw_started = True

        waiting = set(FINGERS.keys())
        raw_deadline = time.time() + 10.0

        while waiting:
            for finger_name in list(waiting):
                sample = raw_objects[finger_name].GetLastSample()
                if sample.timestamp != 0:
                    print(f"Primo campione RAW ricevuto: {finger_name}")
                    waiting.remove(finger_name)

            if time.time() > raw_deadline:
                raise RuntimeError(
                    "Timeout: nessun campione RAW per: "
                    + ", ".join(sorted(waiting))
                )

            time.sleep(0.01)

        print("\nPubblicazione ROS 2 attiva:")
        for finger_name, cfg in FINGERS.items():
            print(f"  {finger_name:>6}: {cfg['topic']}")

        print(
            "\nOgni dito viene pubblicato solo quando cambia "
            "il relativo timestamp RAW."
        )
        print("Premi CTRL+C per terminare.\n")

        last_timestamp = {finger_name: None for finger_name in FINGERS}
        samples_since_diag = {finger_name: 0 for finger_name in FINGERS}
        latest_closure = {finger_name: 0.0 for finger_name in FINGERS}
        latest_tof = {finger_name: None for finger_name in FINGERS}
        diag_start = time.perf_counter()

        while rclpy.ok():
            for finger_name, cfg in FINGERS.items():
                sample = raw_objects[finger_name].GetLastSample()

                if sample.timestamp == 0:
                    continue

                if sample.timestamp == last_timestamp[finger_name]:
                    continue

                last_timestamp[finger_name] = sample.timestamp

                tracking = tracking_objects[finger_name]

                closure = float(tracking.GetClosure())
                opening = 1.0 - closure

                if cfg["has_abduction"]:
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

                msg = String()
                msg.data = line
                node.finger_publishers[finger_name].publish(msg)

                samples_since_diag[finger_name] += 1
                latest_closure[finger_name] = closure
                latest_tof[finger_name] = tof.distance

            rclpy.spin_once(node, timeout_sec=0.0)

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

                node.get_logger().info(" | ".join(parts))
                diag_start = now

            time.sleep(POLL_SLEEP_S)

    except KeyboardInterrupt:
        print("\nInterruzione richiesta dall'utente.")

    except Exception as exc:
        node.get_logger().error(str(exc))

    finally:
        if raw_started:
            try:
                client.StopRawData()
            except Exception:
                pass

        try:
            client.Stop()
        except Exception:
            pass

        try:
            client.Close()
        except Exception:
            pass

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        print("Connessione WEART chiusa.")
        print("Nodo ROS 2 terminato.")


if __name__ == "__main__":
    main()
