#!/usr/bin/env python3

import logging
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from weartsdk import *


# ============================================================
# CONFIGURAZIONE
# ============================================================

HAND = WeArtCommon.HandSide.Right
POINT = WeArtCommon.ActuationPoint.Index
TOPIC_NAME = "/weart/index/raw"

# Polling molto rapido del buffer "ultimo campione" WEART.
# Non impone la frequenza di pubblicazione.
POLL_SLEEP_S = 0.001

# Diagnostica della frequenza osservata/pubblicata.
DIAGNOSTIC_PERIOD_S = 1.0


class WeartIndexPublisher(Node):
    def __init__(self):
        super().__init__("weart_index_publisher")
        self.publisher_ = self.create_publisher(String, TOPIC_NAME, 20)
        self.get_logger().info(f"Publisher ROS 2 creato su: {TOPIC_NAME}")


def main(args=None):
    rclpy.init(args=args)
    node = WeartIndexPublisher()

    client = WeArtClient(
        WeArtCommon.DEFAULT_IP_ADDRESS,
        WeArtCommon.DEFAULT_TCP_PORT,
        log_level=logging.INFO,
    )

    mw_listener = MiddlewareStatusListener()
    dev_listener = DeviceStatusListener()
    calibration = WeArtTrackingCalibration()

    index_tracking = WeArtThimbleTrackingObject(HAND, POINT)
    index_raw = WeArtTrackingRawData(HAND, POINT)

    client.AddMessageListener(mw_listener)
    client.AddMessageListener(dev_listener)
    client.AddMessageListener(calibration)
    client.AddThimbleTracking(index_tracking)
    client.AddMessageListener(index_raw)

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
            raise RuntimeError("Connessione TCP al WEART Middleware non riuscita.")

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
                f"{status.status}, "
                f"devices={len(status.connectedDevices)}"
            )

        print("\nInvio client.Start()...")
        client.Start()
        time.sleep(2.0)

        status = mw_listener.LastStatus()
        print(
            f"Stato dopo Start: "
            f"{status.status}, "
            f"code={status.statusCode}, "
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

        print("\nAvvio dati RAW dell'indice...")
        client.StartRawData()
        raw_started = True

        raw_deadline = time.time() + 10.0
        while index_raw.GetLastSample().timestamp == 0:
            if time.time() > raw_deadline:
                raise RuntimeError(
                    "Nessun campione RAW ricevuto dall'indice entro 10 secondi."
                )
            time.sleep(0.01)

        print("Primo campione RAW dell'indice ricevuto.")
        print(f"Pubblicazione ROS 2 su {TOPIC_NAME}")
        print("Pubblico solo quando cambia il timestamp RAW WEART.")
        print("Premi CTRL+C per terminare.\n")

        last_timestamp = None

        diag_start = time.perf_counter()
        samples_since_diag = 0
        last_diag_timestamp = None

        while rclpy.ok():
            sample = index_raw.GetLastSample()

            if sample.timestamp == 0:
                rclpy.spin_once(node, timeout_sec=0.0)
                time.sleep(POLL_SLEEP_S)
                continue

            # Nessun nuovo campione: non ripubblicare lo stesso dato.
            if sample.timestamp == last_timestamp:
                rclpy.spin_once(node, timeout_sec=0.0)
                time.sleep(POLL_SLEEP_S)
                continue

            # Nuovo campione RAW.
            last_timestamp = sample.timestamp

            closure = float(index_tracking.GetClosure())
            opening = 1.0 - closure

            # Placeholder richiesti dal parser del consumer.
            abduction = 0.0
            adduction = 1.0

            data = sample.data
            acc = data.accelerometer
            gyro = data.gyroscope
            tof = data.timeOfFlight

            # Manteniamo più precisione rispetto alla versione precedente.
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
            node.publisher_.publish(msg)

            samples_since_diag += 1
            rclpy.spin_once(node, timeout_sec=0.0)

            now = time.perf_counter()
            elapsed = now - diag_start

            if elapsed >= DIAGNOSTIC_PERIOD_S:
                rate_hz = samples_since_diag / elapsed

                ts_info = ""
                if last_diag_timestamp is not None:
                    ts_info = (
                        f" | WEART Δts="
                        f"{sample.timestamp - last_diag_timestamp}"
                    )

                node.get_logger().info(
                    f"RAW observed/published: {rate_hz:.1f} Hz | "
                    f"closure={closure:.4f} | "
                    f"ToF={tof.distance}"
                    f"{ts_info}"
                )

                diag_start = now
                samples_since_diag = 0
                last_diag_timestamp = sample.timestamp

            # Serve solo a non saturare una CPU.
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
