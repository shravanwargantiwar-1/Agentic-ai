from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import cv2

from .models import BehaviorEvent, Detection, Track


@dataclass
class PipelineConfig:
    rtsp_url: str
    sample_every_n_frames: int = 2


class SurveillanceAgent:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url

    def open(self):
        return cv2.VideoCapture(self.rtsp_url)


class DetectionAgent:
    """Placeholder detector; replace with YOLO11/OpenVINO implementation."""

    def detect(self, frame) -> List[Detection]:
        h, w = frame.shape[:2]
        return [Detection(label="person", confidence=0.9, bbox=(w // 4, h // 4, w // 2, h // 2))]


class TrackingAgent:
    """Placeholder tracker; replace with ByteTrack or DeepSORT."""

    def update(self, detections: Iterable[Detection]) -> List[Track]:
        tracks = []
        for idx, det in enumerate(detections, start=1):
            tracks.append(
                Track(
                    track_id=idx,
                    label=det.label,
                    confidence=det.confidence,
                    bbox=det.bbox,
                )
            )
        return tracks


class RoleClassificationAgent:
    def classify(self, track: Track) -> str:
        # Heuristic placeholder. Can be upgraded with trajectory history.
        return "student" if track.label == "person" else "ignore"


class BehaviourAnalysisAgent:
    def analyze(self, frame, tracks: Iterable[Track]) -> List[BehaviorEvent]:
        events: List[BehaviorEvent] = []
        for track in tracks:
            # Placeholder: behavior hooks should integrate pose + temporal logic.
            if track.confidence > 0.85:
                events.append(
                    BehaviorEvent(
                        track_id=track.track_id,
                        event_type="head_turning",
                        confidence=0.75,
                    )
                )
        return events
