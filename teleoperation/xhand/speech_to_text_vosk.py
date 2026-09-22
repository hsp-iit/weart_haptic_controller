#!/usr/bin/env python3
"""Offline keyword speech-to-text node for tactile reset commands.

It listens to the default ALSA microphone with arecord, recognizes speech with
Vosk, and publishes std_msgs/String on /speech/recognized when the keyword is
heard. The retargeter can subscribe to that topic and trigger SDK tactile reset.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except ImportError:
    rclpy = None
    Node = object
    String = None


DEFAULT_MODEL_URL = "https://alphacephei.com/kaldi/models/vosk-model-small-it-0.22.zip"
DEFAULT_MODEL_DIR = Path("/home/panda-admin/users/sberti/weart_haptic_controller/models/vosk-model-small-it-0.22")


def ensure_vosk_imported():
    try:
        from vosk import KaldiRecognizer, Model, SetLogLevel
    except ImportError as exc:
        raise SystemExit(
            "Python package 'vosk' is missing. Install it first, for example:\n"
            "  pixi add --pypi vosk\n"
            "or:\n"
            "  python3 -m pip install vosk"
        ) from exc
    return KaldiRecognizer, Model, SetLogLevel


def download_model(model_dir: Path, url: str) -> None:
    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"Model already exists: {model_dir}")
        return

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path = model_dir.parent / Path(url).name
    print(f"Downloading Vosk model:\n  {url}\n  -> {zip_path}")
    urllib.request.urlretrieve(url, zip_path)
    print(f"Extracting -> {model_dir.parent}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(model_dir.parent)
    print(f"Ready: {model_dir}")


class VoskKeywordPublisher(Node):
    def __init__(
        self,
        *,
        model_path: Path,
        topic: str,
        keyword: str,
        audio_device: str,
        sample_rate: int,
        cooldown_s: float,
        grammar_only: bool,
    ):
        if rclpy is None or String is None:
            raise RuntimeError(
                "ROS 2 Python package 'rclpy' is not available. Source ROS first, e.g.:\n"
                "  source /opt/ros/jazzy/setup.bash"
            )
        super().__init__("speech_to_text_vosk")
        KaldiRecognizer, Model, SetLogLevel = ensure_vosk_imported()
        SetLogLevel(-1)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Vosk model not found: {model_path}\n"
                "Run this first:\n"
                f"  python3 {Path(__file__).as_posix()} --download-model"
            )

        self.keyword = keyword.strip().lower()
        self.cooldown_s = float(cooldown_s)
        self.last_publish_time = 0.0
        self.publisher = self.create_publisher(String, topic, 10)

        grammar = json.dumps([self.keyword, "[unk]"]) if grammar_only else None
        model = Model(str(model_path))
        self.recognizer = (
            KaldiRecognizer(model, sample_rate, grammar)
            if grammar is not None
            else KaldiRecognizer(model, sample_rate)
        )

        arecord = shutil.which("arecord")
        if arecord is None:
            raise RuntimeError("arecord not found. Install alsa-utils.")

        self.audio_proc = subprocess.Popen(
            [
                arecord,
                "-q",
                "-D",
                audio_device,
                "-f",
                "S16_LE",
                "-r",
                str(sample_rate),
                "-c",
                "1",
                "-t",
                "raw",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.get_logger().info(
            f"Listening for '{self.keyword}' on ALSA device '{audio_device}', publishing {topic}"
        )
        self.timer = self.create_timer(0.02, self.audio_step)

    def audio_step(self) -> None:
        if self.audio_proc.stdout is None:
            return
        chunk = self.audio_proc.stdout.read(4000)
        if not chunk:
            return

        text = ""
        if self.recognizer.AcceptWaveform(chunk):
            text = json.loads(self.recognizer.Result()).get("text", "")
        else:
            text = json.loads(self.recognizer.PartialResult()).get("partial", "")

        if self.keyword in text.lower():
            now = time.monotonic()
            if now - self.last_publish_time >= self.cooldown_s:
                self.last_publish_time = now
                msg = String()
                msg.data = self.keyword
                self.publisher.publish(msg)
                self.get_logger().info(f"Published voice command: {msg.data!r}")

    def destroy_node(self):
        if hasattr(self, "audio_proc") and self.audio_proc is not None:
            self.audio_proc.terminate()
            try:
                self.audio_proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.audio_proc.kill()
        return super().destroy_node()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--model-url", default=DEFAULT_MODEL_URL)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--topic", default="/speech/recognized")
    parser.add_argument("--keyword", default="azzera")
    parser.add_argument("--audio-device", default="default")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--cooldown-s", type=float, default=1.5)
    parser.add_argument("--full-vocabulary", action="store_true")
    args, ros_args = parser.parse_known_args()
    args.ros_args = [sys.argv[0], *ros_args]
    return args


def main() -> None:
    args = parse_args()
    if args.download_model:
        download_model(args.model_path, args.model_url)
        return

    if rclpy is None:
        raise SystemExit(
            "ROS 2 Python package 'rclpy' is not available. Source ROS first, e.g.:\n"
            "  source /opt/ros/jazzy/setup.bash\n"
            "  source /usr/local/robot/weart_haptic_controller/install/setup.bash"
        )

    rclpy.init(args=args.ros_args)
    node = None
    try:
        node = VoskKeywordPublisher(
            model_path=args.model_path,
            topic=args.topic,
            keyword=args.keyword,
            audio_device=args.audio_device,
            sample_rate=args.sample_rate,
            cooldown_s=args.cooldown_s,
            grammar_only=not args.full_vocabulary,
        )
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
