#!/usr/bin/env python3
"""
ROS 2 minimal LEAP test: move ONLY joint q0.

All other joints are read from the physical hand at startup and then kept fixed.

Example:
python3 test_leap_q0_only_ros.py \
  --ros-args \
  -p urdf_path:=/home/panda-admin/users/sberti/weart_haptic_controller/teleoperation/leap_hand_right.urdf \
  -p leap_port:=/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT94W0JX-if00-port0
"""

import math
import time
import numpy as np

import rclpy
from rclpy.node import Node

import pinocchio as pin

from weart_leap_index_ik import (
    DEFAULT_KP,
    DEFAULT_KI,
    DEFAULT_KD,
    DEFAULT_LEAP_PORT,
    _load_leap_cpp,
    autodetect_leap_port,
    leap_sim_to_motor,
)


class Q0OnlyTestNode(Node):
    def __init__(self):
        super().__init__("leap_q0_only_test")

        # Same ROS-style parameters as your other LEAP script.
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("leap_port", DEFAULT_LEAP_PORT)
        self.declare_parameter("kp", DEFAULT_KP)
        self.declare_parameter("ki", DEFAULT_KI)
        self.declare_parameter("kd", DEFAULT_KD)

        # q0 trajectory.
        self.declare_parameter(
            "q0_values",
            [0.0, 0.20, 0.40, 0.60, 0.40, 0.20, 0.0],
        )
        self.declare_parameter("move_time_s", 1.0)
        self.declare_parameter("hold_time_s", 0.5)
        self.declare_parameter("control_hz", 50.0)

        urdf_path = str(self.get_parameter("urdf_path").value)
        if not urdf_path:
            raise RuntimeError(
                "Set -p urdf_path:=/path/to/leap_hand_right.urdf"
            )

        self.port_arg = str(self.get_parameter("leap_port").value)
        self.kp = float(self.get_parameter("kp").value)
        self.ki = float(self.get_parameter("ki").value)
        self.kd = float(self.get_parameter("kd").value)

        self.q0_values = [
            float(v) for v in self.get_parameter("q0_values").value
        ]
        self.move_time_s = float(self.get_parameter("move_time_s").value)
        self.hold_time_s = float(self.get_parameter("hold_time_s").value)
        self.control_hz = float(self.get_parameter("control_hz").value)

        # Use URDF only to get the true q0 hard limits.
        self.model = pin.buildModelFromUrdf(urdf_path)

        joint_id = self.model.getJointId("0")
        if joint_id == 0:
            raise RuntimeError("Joint '0' not found in URDF.")

        joint = self.model.joints[joint_id]
        self.q0_idx_pin = joint.idx_q

        lower = np.asarray(
            self.model.lowerPositionLimit, dtype=float
        )
        upper = np.asarray(
            self.model.upperPositionLimit, dtype=float
        )

        self.q0_lower = float(lower[self.q0_idx_pin])
        self.q0_upper = float(upper[self.q0_idx_pin])

        self.get_logger().info(
            f"URDF q0 limits: [{self.q0_lower:.3f}, {self.q0_upper:.3f}] rad"
        )

        # Connect to LEAP.
        self.port = autodetect_leap_port(self.port_arg)

        leap_cpp = _load_leap_cpp()
        self.ctrl = leap_cpp.LeapController(self.port)
        self.ctrl.connect()
        self.ctrl.setGains(
            int(self.kp),
            int(self.ki),
            int(self.kd),
        )

        self.get_logger().info(f"Connected to LEAP on {self.port}")

        # Capture CURRENT physical pose.
        motor_q = np.asarray(
            self.ctrl.read_pos(),
            dtype=float,
        ).reshape(-1)

        if motor_q.size != 16:
            raise RuntimeError(
                f"Expected 16 joints from LEAP, got {motor_q.size}"
            )

        # Same convention used in your existing driver.
        self.q_fixed = motor_q - math.pi

        self.get_logger().info(
            f"Captured physical pose. Initial q0={self.q_fixed[0]:.3f} rad"
        )
        self.get_logger().info(
            "ONLY q0 will move. q1..q15 remain frozen."
        )

    def smooth_move_q0(self, q0_start: float, q0_target: float):
        q0_target = float(
            np.clip(q0_target, self.q0_lower, self.q0_upper)
        )

        dt = 1.0 / self.control_hz
        steps = max(
            1,
            int(self.move_time_s * self.control_hz),
        )

        for k in range(1, steps + 1):
            s = k / steps

            # Smoothstep interpolation.
            alpha = s * s * (3.0 - 2.0 * s)

            q_cmd = self.q_fixed.copy()

            # ----------------------------------------------------------
            # THE ONLY JOINT MODIFIED
            # ----------------------------------------------------------
            q_cmd[0] = (
                q0_start
                + alpha * (q0_target - q0_start)
            )

            self.ctrl.set_leap(
                leap_sim_to_motor(q_cmd)
            )

            time.sleep(dt)

        self.q_fixed[0] = q0_target
        return q0_target

    def run_test(self):
        current_q0 = float(self.q_fixed[0])

        self.get_logger().info(
            f"q0 test sequence: {self.q0_values}"
        )

        for requested_q0 in self.q0_values:
            if not rclpy.ok():
                break

            target_q0 = float(
                np.clip(
                    requested_q0,
                    self.q0_lower,
                    self.q0_upper,
                )
            )

            self.get_logger().info(
                f"q0: {current_q0:+.3f} -> {target_q0:+.3f} rad"
            )

            current_q0 = self.smooth_move_q0(
                current_q0,
                target_q0,
            )

            time.sleep(self.hold_time_s)

        self.get_logger().info("q0 test completed.")

    def close(self):
        if getattr(self, "ctrl", None) is not None:
            self.ctrl.disconnect()
            self.ctrl = None
            self.get_logger().info("LEAP disconnected.")


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = Q0OnlyTestNode()
        node.run_test()

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.close()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
