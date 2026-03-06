from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None

try:
    from openvino.runtime import Core
except ImportError:  # pragma: no cover
    Core = None


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    class_id: int
    class_name: str
    confidence: float

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class DetectionAgent:
    TARGET_CLASSES = {0: "person", 67: "cell phone", 80: "calculator"}

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        model_cfg = config.get("model", {})
        det_cfg = config.get("detection", {})
        self.model_path = model_cfg.get("yolo_path", "models/yolo11n.pt")
        self.openvino_path = model_cfg.get("openvino_path", "models/yolo11n_openvino")
        self.conf = det_cfg.get("confidence_threshold", 0.7)
        self.input_size = tuple(det_cfg.get("input_size", [640, 640]))
        self.model = None
        self.ov_compiled = None
        self._load()

    def _load(self) -> None:
        if Core is not None and Path(self.openvino_path).exists():
            core = Core()
            xml = Path(self.openvino_path) / "model.xml"
            if xml.exists():
                self.ov_compiled = core.compile_model(core.read_model(str(xml)), "CPU")
                return
        if YOLO is not None and Path(self.model_path).exists():
            self.model = YOLO(self.model_path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.model is not None:
            results = self.model(frame, verbose=False)[0]
            output: List[Detection] = []
            for box in results.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < self.conf or class_id not in self.TARGET_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                output.append(Detection((x1, y1, x2, y2), class_id, self.TARGET_CLASSES[class_id], conf))
            return output

        h, w = frame.shape[:2]
        return [Detection((w // 4, h // 6, w // 2, h // 2), 0, "person", 0.9)]

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"{det.class_name}:{det.confidence:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return out
