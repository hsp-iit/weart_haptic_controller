#!/usr/bin/env python3
"""Dummy tactile sensor that publishes sinusoidal force vectors."""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension


class TactileSensorDummy(Node):
    def __init__(self) -> None:
        super().__init__("tactile_sensor_dummy")

        self._declare_parameters()
        self._load_parameters()

        self._publisher = self.create_publisher(
            Float32MultiArray,
            self.force_topic,
            10,
        )
        self._start_time = time.monotonic()
        self._timer = self.create_timer(
            1.0 / self.publish_rate_hz, self._publish_force_vector
        )

        self.get_logger().info(
            f"Publishing dummy force vectors on {self.force_topic} at {self.publish_rate_hz} Hz"
        )

    def _declare_parameters(self) -> None:
        parameters = [
            ("force_topic", "/dummy_tactile/force_vector"),
            ("vector_size", 1),
            ("frequency_hz", 0.5),
            ("publish_rate_hz", 5.0),
        ]
        self.declare_parameters(namespace="", parameters=parameters)

    def _load_parameters(self) -> None:
        value = lambda name: self.get_parameter(name).value

        self.force_topic = str(value("force_topic"))
        self.vector_size = int(value("vector_size"))
        self.frequency_hz = float(value("frequency_hz"))
        self.publish_rate_hz = float(value("publish_rate_hz"))

        if self.vector_size <= 0:
            raise ValueError("vector_size must be positive")
        if self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive")
        if self.publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be positive")

    def _publish_force_vector(self) -> None:
        t = time.monotonic() - self._start_time
        phase_offsets = np.linspace(
            0.0,
            2.0 * math.pi * 0.75,
            self.vector_size,
            endpoint=False,
        )
        wave = np.sin(2.0 * math.pi * self.frequency_hz * t + phase_offsets)
        base = 0.5 + 0.4 * math.sin(0.25 * 2.0 * math.pi * self.frequency_hz * t)
        values = np.clip(base + 0.3 * wave, 0.0, 1.0).astype(np.float32)

        msg = Float32MultiArray(data=values.tolist())
        msg.layout.dim.append(
            MultiArrayDimension(
                label="force_vector",
                size=self.vector_size,
                stride=self.vector_size,
            )
        )

        self._publisher.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[TactileSensorDummy] = None
    try:
        node = TactileSensorDummy()
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
