#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Overlay LEAP target and actual URDF poses in Rerun.

Default inputs match ``weart_three_fingers_synergy.py``:

    /leap/target_joint_positions  complete commanded 16-joint LEAPsim vector
    /leap/actual_joint_positions  complete measured 16-joint LEAPsim vector

The viewer never opens the LEAP serial port. Actual feedback is published by
the controller that already owns the hardware connection. Per-finger target
topics remain available as an optional fallback.
"""

from __future__ import annotations

import math
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

DEFAULT_URDF_PATH = str(Path(__file__).with_name("model.urdf"))
DEFAULT_INDEX_TARGET_TOPIC = ""
DEFAULT_MIDDLE_TARGET_TOPIC = ""
DEFAULT_THUMB_TARGET_TOPIC = ""
DEFAULT_FULL_TARGET_TOPIC = "/leap/target_joint_positions"
DEFAULT_ACTUAL_TOPIC = "/leap/actual_joint_positions"
DEFAULT_ACTUAL_JOINT_STATE_TOPIC = ""

WORLD_FRAME_ID = "tf#/leap_overlay/world"
HAND_COLORS = {
    "target": [255, 95, 55, 125],
    "actual": [45, 185, 255, 180],
}

TARGET_INDICES = {
    "index": (1, 2, 3),
    "middle": (5, 6, 7),
    "thumb": (12, 13, 14, 15),
}


@dataclass(frozen=True)
class LeapUrdfInfo:
    path: Path
    root_link_name: str
    motor_count: int
    joint_name_by_index: tuple[str, ...]
    child_link_by_index: tuple[str, ...]
    index_by_joint_name: dict[str, int]
    index_by_child_link: dict[str, int]


@dataclass
class PreparedUrdf:
    path: Path
    temporary_directory: Any | None = None

    def close(self) -> None:
        if self.temporary_directory is not None:
            self.temporary_directory.cleanup()
            self.temporary_directory = None


def load_leap_urdf_info(urdf_path: str) -> LeapUrdfInfo:
    path = Path(urdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LEAP URDF not found: {path}")

    root = ET.parse(path).getroot()
    link_names = {
        str(link.get("name")) for link in root.findall("link") if link.get("name")
    }
    child_links = {
        str(child.get("link"))
        for joint in root.findall("joint")
        if (child := joint.find("child")) is not None and child.get("link")
    }
    root_links = sorted(link_names.difference(child_links))
    if len(root_links) != 1:
        raise ValueError(f"Expected one URDF root link, found {root_links}")

    numerical: dict[int, ET.Element] = {}
    for joint in root.findall("joint"):
        name = str(joint.get("name", ""))
        if name.isdecimal():
            numerical[int(name)] = joint

    if not numerical:
        raise ValueError(f"No numerical joints found in {path}")
    indices = list(range(max(numerical) + 1))
    if sorted(numerical) != indices:
        raise ValueError(
            "LEAP numerical joint names must be contiguous from zero; found "
            f"{sorted(numerical)}"
        )

    joint_names: list[str] = []
    child_link_names: list[str] = []
    for index in indices:
        joint = numerical[index]
        child = joint.find("child")
        if child is None or child.get("link") is None:
            raise ValueError(f"Joint {index} has no child link")
        joint_names.append(str(joint.get("name")))
        child_link_names.append(str(child.get("link")))

    return LeapUrdfInfo(
        path=path,
        root_link_name=root_links[0],
        motor_count=len(indices),
        joint_name_by_index=tuple(joint_names),
        child_link_by_index=tuple(child_link_names),
        index_by_joint_name={name: i for i, name in enumerate(joint_names)},
        index_by_child_link={name: i for i, name in enumerate(child_link_names)},
    )


def _automatic_mesh_directory() -> Path | None:
    project_parent = Path(__file__).resolve().parents[2]
    candidate = (
        project_parent
        / "lerobot"
        / "src"
        / "lerobot"
        / "robots"
        / "custom_manipulator"
        / "grippers"
        / "urdfs"
        / "meshes"
    )
    return candidate if candidate.is_dir() else None


def prepare_urdf_for_rerun(
    urdf_path: Path,
    mesh_directory: str,
) -> PreparedUrdf:
    """Resolve meshes and remove OBJ materials so overlay tints stay visible."""
    tree = ET.parse(urdf_path)
    mesh_elements = tree.getroot().findall(".//mesh")
    unresolved = []

    explicit_mesh_directory = Path(mesh_directory).expanduser().resolve()
    selected_mesh_directory = (
        explicit_mesh_directory
        if mesh_directory.strip()
        else _automatic_mesh_directory()
    )

    resolved_meshes: list[tuple[ET.Element, Path]] = []
    for mesh in mesh_elements:
        filename = str(mesh.get("filename", ""))
        source = Path(filename)
        if source.is_absolute() and source.is_file():
            resolved = source.resolve()
        elif (urdf_path.parent / source).is_file():
            resolved = (urdf_path.parent / source).resolve()
        else:
            parts = (
                source.parts[1:] if source.parts[:1] == ("meshes",) else source.parts
            )
            replacement = (
                selected_mesh_directory.joinpath(*parts)
                if selected_mesh_directory is not None
                else None
            )
            if replacement is None or not replacement.is_file():
                unresolved.append(filename)
                continue
            resolved = replacement.resolve()
        resolved_meshes.append((mesh, resolved))

    if unresolved:
        raise FileNotFoundError(
            "URDF mesh files not found: "
            + ", ".join(sorted(set(unresolved)))
            + ". Pass --ros-args -p mesh_directory:=/path/to/meshes"
        )
    temporary_directory = tempfile.TemporaryDirectory(prefix="leap-rerun-urdf-")
    temporary_root = Path(temporary_directory.name)
    neutral_mesh_directory = temporary_root / "meshes"
    neutral_mesh_directory.mkdir()
    neutral_mesh_by_source: dict[Path, Path] = {}

    for mesh, source in resolved_meshes:
        prepared_mesh = source
        if source.suffix.lower() == ".obj":
            prepared_mesh = neutral_mesh_by_source.get(source)
            if prepared_mesh is None:
                prepared_mesh = (
                    neutral_mesh_directory
                    / f"{len(neutral_mesh_by_source):02d}_{source.name}"
                )
                lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
                prepared_mesh.write_text(
                    "".join(
                        line
                        for line in lines
                        if not line.lstrip().startswith(("mtllib ", "usemtl "))
                    ),
                    encoding="utf-8",
                )
                neutral_mesh_by_source[source] = prepared_mesh
        mesh.set("filename", str(prepared_mesh))

    prepared_path = Path(temporary_directory.name) / "leap_hand_visual.urdf"
    tree.write(prepared_path, encoding="utf-8", xml_declaration=True)
    return PreparedUrdf(
        path=prepared_path,
        temporary_directory=temporary_directory,
    )


class RerunLeapPosePair:
    def __init__(
        self,
        *,
        urdf_path: Path,
        root_link_name: str,
        mesh_directory: str,
        spawn_viewer: bool,
    ) -> None:
        import rerun as rr
        import rerun.blueprint as rrb
        from rerun.urdf import UrdfTree

        self.rr = rr
        self.prepared_urdf = prepare_urdf_for_rerun(urdf_path, mesh_directory)

        if not rr.is_enabled():
            rr.init("weart_leap_target_actual", spawn=spawn_viewer)
        self.recording = rr.get_global_data_recording()
        self.start_time = time.perf_counter()

        self.trees = {
            "target": UrdfTree.from_file_path(
                self.prepared_urdf.path,
                entity_path_prefix="leap_target",
                frame_prefix="tf#/leap_target/",
            ),
            "actual": UrdfTree.from_file_path(
                self.prepared_urdf.path,
                entity_path_prefix="leap_actual",
                frame_prefix="tf#/leap_actual/",
            ),
        }

        rr.set_time("stable_time", duration=0.0, recording=self.recording)
        for kind, tree in self.trees.items():
            tree.log_urdf_to_recording(recording=self.recording)
            self._tint_tree(tree, HAND_COLORS[kind])

        rr.log(
            "/leap_overlay",
            rr.CoordinateFrame(WORLD_FRAME_ID),
            rr.ViewCoordinates.UBR,
            static=True,
            recording=self.recording,
        )

        for kind in self.trees:
            prefix = f"leap_{kind}"
            rr.log(
                f"/{prefix}/{root_link_name}",
                rr.Transform3D(
                    translation=[0.0, 0.0, 0.0],
                    parent_frame=WORLD_FRAME_ID,
                    child_frame=f"tf#/{prefix}/{root_link_name}",
                ),
                rr.ViewCoordinates.UBR,
                static=True,
                recording=self.recording,
            )

        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Vertical(
                    rrb.Spatial3DView(
                        name="LEAP OVERLAY: TARGET (orange) / ACTUAL (blue)",
                        origin="/",
                        contents=[
                            "/leap_target/**",
                            "/leap_actual/**",
                            "/leap_overlay/**",
                        ],
                        spatial_information=rrb.SpatialInformation(
                            target_frame=WORLD_FRAME_ID,
                            show_axes=True,
                            show_bounding_box=True,
                        ),
                    ),
                    rrb.TimeSeriesView(
                        name="JOINT ERROR: actual - target [rad]",
                        origin="/comparison/error",
                        contents=["/comparison/error/**"],
                    ),
                    row_shares=[3.0, 1.0],
                ),
                collapse_panels=True,
            ),
            recording=self.recording,
        )

    def _tint_tree(self, tree: Any, rgba: list[int]) -> None:
        link_names = {tree.root_link().name}
        for joint in tree.joints():
            link_names.add(joint.parent_link)
            link_names.add(joint.child_link)
        for link_name in link_names:
            for entity_path in tree.get_visual_geometry_paths(link_name):
                self.rr.log(
                    entity_path,
                    self.rr.Asset3D.from_fields(albedo_factor=rgba),
                    static=True,
                    recording=self.recording,
                )

    def close(self) -> None:
        self.recording.flush(timeout_sec=2.0)
        self.prepared_urdf.close()

    def _set_time(self) -> None:
        self.rr.set_time(
            "stable_time",
            duration=time.perf_counter() - self.start_time,
            recording=self.recording,
        )

    def log_pose(
        self,
        kind: str,
        joint_values_by_child_link: dict[str, float],
    ) -> None:
        if kind not in self.trees:
            raise ValueError(f"Unknown pose kind: {kind}")
        self._set_time()
        tree = self.trees[kind]
        for index, joint in enumerate(tree.joints()):
            value = joint_values_by_child_link.get(joint.child_link)
            if value is None:
                continue
            self.rr.log(
                f"/{'leap_target' if kind == 'target' else 'leap_actual'}/transforms",
                joint.compute_transform(float(value)),
                recording=self.recording,
            )
            self.rr.log(
                f"/comparison/{kind}/joint_{index}",
                self.rr.Scalars([float(value)]),
                recording=self.recording,
            )

    def log_error(self, error: np.ndarray) -> None:
        self._set_time()
        for index, value in enumerate(np.asarray(error, dtype=float)):
            self.rr.log(
                f"/comparison/error/joint_{index}",
                self.rr.Scalars([float(value)]),
                recording=self.recording,
            )


class WeartLeapRerunViewerNode(Node):
    def __init__(self) -> None:
        super().__init__("weart_leap_rerun_viewer")

        self.declare_parameter("urdf_path", DEFAULT_URDF_PATH)
        self.declare_parameter("mesh_directory", "")
        self.declare_parameter("spawn_viewer", True)
        self.declare_parameter("index_target_topic", DEFAULT_INDEX_TARGET_TOPIC)
        self.declare_parameter("middle_target_topic", DEFAULT_MIDDLE_TARGET_TOPIC)
        self.declare_parameter("thumb_target_topic", DEFAULT_THUMB_TARGET_TOPIC)
        self.declare_parameter("full_target_topic", DEFAULT_FULL_TARGET_TOPIC)
        self.declare_parameter("actual_topic", DEFAULT_ACTUAL_TOPIC)
        self.declare_parameter(
            "actual_joint_state_topic", DEFAULT_ACTUAL_JOINT_STATE_TOPIC
        )

        self.urdf_info = load_leap_urdf_info(str(self.get_parameter("urdf_path").value))
        self.viewer = RerunLeapPosePair(
            urdf_path=self.urdf_info.path,
            root_link_name=self.urdf_info.root_link_name,
            mesh_directory=str(self.get_parameter("mesh_directory").value),
            spawn_viewer=bool(self.get_parameter("spawn_viewer").value),
        )

        self.target_q = np.zeros(self.urdf_info.motor_count, dtype=float)
        self.actual_q = np.zeros(self.urdf_info.motor_count, dtype=float)
        self.actual_received = False
        self.subscriptions_by_source = []

        for finger_name in TARGET_INDICES:
            topic = str(self.get_parameter(f"{finger_name}_target_topic").value)
            if not topic:
                continue
            subscription = self.create_subscription(
                Float64MultiArray,
                topic,
                lambda msg, name=finger_name: self._on_finger_target(name, msg),
                20,
            )
            self.subscriptions_by_source.append(subscription)
            self.get_logger().info(f"{finger_name} target: {topic}")

        full_target_topic = str(self.get_parameter("full_target_topic").value)
        if full_target_topic:
            self.subscriptions_by_source.append(
                self.create_subscription(
                    Float64MultiArray,
                    full_target_topic,
                    self._on_full_target,
                    20,
                )
            )
            self.get_logger().info(f"Full target: {full_target_topic}")

        actual_topic = str(self.get_parameter("actual_topic").value)
        if actual_topic:
            self.subscriptions_by_source.append(
                self.create_subscription(
                    Float64MultiArray,
                    actual_topic,
                    self._on_actual,
                    20,
                )
            )
            self.get_logger().info(f"Actual: {actual_topic}")

        joint_state_topic = str(self.get_parameter("actual_joint_state_topic").value)
        if joint_state_topic:
            self.subscriptions_by_source.append(
                self.create_subscription(
                    JointState,
                    joint_state_topic,
                    self._on_actual_joint_state,
                    20,
                )
            )
            self.get_logger().info(f"Actual JointState: {joint_state_topic}")

        self.viewer.log_pose("target", self._joint_values(self.target_q))
        self.viewer.log_pose("actual", self._joint_values(self.actual_q))
        self.get_logger().info(f"Rerun URDF: {self.urdf_info.path}")

    def destroy_node(self):
        try:
            self.viewer.close()
        finally:
            return super().destroy_node()

    def _validate_full_q(self, values: Any, source: str) -> np.ndarray:
        q = np.asarray(values, dtype=float).reshape(-1)
        if q.size != self.urdf_info.motor_count:
            raise ValueError(
                f"{source}: expected {self.urdf_info.motor_count} joints, "
                f"got {q.size}"
            )
        if not np.all(np.isfinite(q)):
            raise ValueError(f"{source}: non-finite joint values")
        return q

    def _joint_values(self, q: np.ndarray) -> dict[str, float]:
        return {
            child_link: float(q[index])
            for index, child_link in enumerate(self.urdf_info.child_link_by_index)
        }

    def _log_target(self) -> None:
        self.viewer.log_pose("target", self._joint_values(self.target_q))
        self._log_error_if_ready()

    def _log_actual(self) -> None:
        self.actual_received = True
        self.viewer.log_pose("actual", self._joint_values(self.actual_q))
        self._log_error_if_ready()

    def _log_error_if_ready(self) -> None:
        if self.actual_received:
            self.viewer.log_error(self.actual_q - self.target_q)

    def _on_finger_target(
        self,
        finger_name: str,
        msg: Float64MultiArray,
    ) -> None:
        try:
            values = np.asarray(msg.data, dtype=float).reshape(-1)
            indices = TARGET_INDICES[finger_name]
            if finger_name == "thumb" and values.size == 3:
                indices = indices[1:]
            if values.size != len(indices):
                raise ValueError(f"expected {len(indices)} values, got {values.size}")
            if not np.all(np.isfinite(values)):
                raise ValueError("non-finite target values")
            self.target_q[list(indices)] = values
            self._log_target()
        except Exception as exc:
            self.get_logger().warning(f"Invalid {finger_name} target message: {exc}")

    def _on_full_target(self, msg: Float64MultiArray) -> None:
        try:
            self.target_q = self._validate_full_q(msg.data, "full target")
            self._log_target()
        except Exception as exc:
            self.get_logger().warning(f"Invalid full target message: {exc}")

    def _on_actual(self, msg: Float64MultiArray) -> None:
        try:
            self.actual_q = self._validate_full_q(msg.data, "actual")
            self._log_actual()
        except Exception as exc:
            self.get_logger().warning(f"Invalid actual message: {exc}")

    def _on_actual_joint_state(self, msg: JointState) -> None:
        try:
            if len(msg.name) != len(msg.position):
                raise ValueError("JointState name/position lengths differ")
            updated = self.actual_q.copy()
            found = 0
            for name, value in zip(msg.name, msg.position, strict=True):
                index = self.urdf_info.index_by_joint_name.get(str(name))
                if index is None:
                    index = self.urdf_info.index_by_child_link.get(str(name))
                if index is None:
                    continue
                if not math.isfinite(value):
                    raise ValueError(f"non-finite value for joint {name}")
                updated[index] = float(value)
                found += 1
            if found == 0:
                raise ValueError("no LEAP numerical joint names recognized")
            self.actual_q = updated
            self._log_actual()
        except Exception as exc:
            self.get_logger().warning(f"Invalid actual JointState: {exc}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = WeartLeapRerunViewerNode()
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
