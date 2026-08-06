# GelSight → WEART ROS 2 bridge

This ROS 2 Python node synchronizes the AI4CE GelSight `PointCloud2` topics:

- `/gelsight_capture/patch`: reconstructed surface height in the `z` field
- `/gelsight_capture/mask`: contact mask in the `z` field (`0` or `1`)

It estimates:

- indentation depth and contact area → WEART force `[0, 1]`
- contact-height roughness → WEART texture volume `[0, 100]`
- contact-centroid motion → WEART texture velocity `[0, 0.5]`

The WEART texture itself is selected from the SDK's built-in `TextureType` enum; GelSight modulates its intensity and playback velocity.

## Requirements

- Ubuntu 22.04 / ROS 2 Humble (matching the AI4CE sensor repository)
- AI4CE GelSight driver running
- WEART middleware and device connected
- Python package `weartsdk`

Install the SDK into the same Python environment used by ROS 2:

```bash
python3 -m pip install --user weartsdk
```

## Build

```bash
cd ~/ros2_ws/src
# Copy this gelsight_weart_bridge directory here.
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select gelsight_weart_bridge
source install/setup.bash
```

## Run

Start the GelSight driver first, then:

```bash
ros2 launch gelsight_weart_bridge bridge.launch.py
```

To test the tactile feature extraction without commanding hardware:

```bash
ros2 run gelsight_weart_bridge bridge_node --ros-args \
  --params-file "$(ros2 pkg prefix gelsight_weart_bridge)/share/gelsight_weart_bridge/config/bridge.yaml" \
  -p dry_run:=true
```

Or run the executable directly with overrides:

```bash
ros2 run gelsight_weart_bridge bridge_node --ros-args \
  -p hand_side:=Right \
  -p actuation_points:="[Index]" \
  -p depth_full_scale:=0.5
```

## Calibration

The reconstructed GelSight height scale depends on the sensor calibration and reconstruction configuration, so tune these while watching the once-per-second diagnostic log:

1. `depth_deadband`: value that rejects no-contact noise.
2. `depth_full_scale`: indentation that should command full WEART force.
3. `area_full_scale`: image fraction that should count as full contact area.
4. `roughness_deadband` and `roughness_full_scale`: texture intensity range.
5. `slip_speed_full_scale_px_s`: centroid speed that maps to texture velocity `0.5`.

Begin with low `force_gain`, verify contact/no-contact behavior, and increase gradually.

## Safety behavior

- The effect is removed when the contact mask falls below `min_contact_fraction`.
- A watchdog removes the effect if synchronized tactile input is stale for `input_timeout_sec`.
- Force, texture volume, and texture velocity are clamped to WEART's documented ranges.
