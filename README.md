# WEART haptic controller

This repository holds `weart_combined_controller`, a ROS 2 Python node that is the
**sole owner** of the connection to the WEART middleware: it connects, runs the
interactive glove calibration, publishes raw per-finger tracking data, and drives
haptic force feedback from incoming force topics.

Only one process should ever open a WEART middleware session at a time — running
more than one WEART client concurrently (e.g. an old bridge node alongside this
one) leaves the middleware in a state where this node's calibration step will
hang or fail.

## Requirements

- ROS 2 (matching your `weartsdk`/middleware setup)
- WEART middleware and device connected
- Python package `weartsdk`

## Build

```bash
cd ~/ros2_ws/src
# Copy the weart_combined_controller directory here.
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select weart_combined_controller
source install/setup.bash
```

## Run

```bash
ros2 launch weart_combined_controller weart_combined_controller.launch.py
```

On startup the node connects to the middleware, then prompts:

```
Wear the glove, keep your hand still in the calibration position, then press Enter...
```

After calibration it starts publishing raw tracking data on
`/weart/{thumb,index,middle}/raw` and begins listening for per-finger contact
force on the topics configured in
`weart_combined_controller/config/weart_combined_controller.yaml`
(`geometry_msgs/Vector3`, `z` = compression force).

## Teleoperation scripts

The `teleoperation/` directory holds standalone experimentation scripts used
during development (synergy mappings, retargeters, viewers). They are not part
of the `weart_combined_controller` package build.
