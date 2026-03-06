from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np


@dataclass
class CameraConfig:
    id: str
    rtsp_url: str
    name: str
    enabled: bool = True


@dataclass
class Frame:
    image: np.ndarray
    camera_id: str
    timestamp: float
    frame_number: int


class CameraStream:
    def __init__(self, config: CameraConfig, buffer_size: int = 30):
        self.config = config
        self.frame_buffer: queue.Queue[Frame] = queue.Queue(maxsize=buffer_size)
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_count = 0

    def start(self) -> bool:
        self.cap = cv2.VideoCapture(self.config.rtsp_url)
        if not self.cap.isOpened():
            return False
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def _capture_loop(self) -> None:
        while self.running and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.2)
                continue
            self.frame_count += 1
            item = Frame(frame, self.config.id, time.time(), self.frame_count)
            if self.frame_buffer.full():
                try:
                    self.frame_buffer.get_nowait()
                except queue.Empty:
                    pass
            self.frame_buffer.put_nowait(item)

    def get_latest_frame(self) -> Optional[Frame]:
        latest = None
        while not self.frame_buffer.empty():
            latest = self.frame_buffer.get_nowait()
        return latest

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()


class SurveillanceAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cameras: Dict[str, CameraStream] = {}
        self._init_cameras()

    def _init_cameras(self) -> None:
        for cam in self.config.get("cameras", []):
            if cam.get("enabled", True):
                cfg = CameraConfig(cam["id"], cam["rtsp_url"], cam.get("name", cam["id"]))
                self.cameras[cfg.id] = CameraStream(cfg)

    def start(self) -> bool:
        return all(stream.start() for stream in self.cameras.values())

    def stop(self) -> None:
        for stream in self.cameras.values():
            stream.stop()

    def get_frames(self) -> Dict[str, Frame]:
        output: Dict[str, Frame] = {}
        for cam_id, stream in self.cameras.items():
            frame = stream.get_latest_frame()
            if frame is not None:
                output[cam_id] = frame
        return output


class DemoSurveillanceAgent(SurveillanceAgent):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.frame_count = 0

    def start(self) -> bool:
        source = self.config.get("demo", {}).get("video_source", 0)
        self.cap = cv2.VideoCapture(source)
        self.running = self.cap.isOpened()
        return self.running

    def stop(self) -> None:
        self.running = False
        if self.cap:
            self.cap.release()

    def get_frames(self) -> Dict[str, Frame]:
        if not self.running or self.cap is None:
            return {}
        ok, img = self.cap.read()
        if not ok:
            return {}
        self.frame_count += 1
        return {"demo_cam": Frame(img, "demo_cam", time.time(), self.frame_count)}
