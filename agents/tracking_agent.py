from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detection_agent import Detection


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    class_id: int
    class_name: str
    confidence: float
    hits: int = 1
    time_since_update: int = 0
    state: str = "tentative"
    history: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def is_confirmed(self) -> bool:
        return self.state == "confirmed"

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua else 0.0


class TrackingAgent:
    def __init__(self, config):
        tr = config.get("tracking", {})
        self.max_age = tr.get("max_age", 30)
        self.min_hits = tr.get("min_hits", 3)
        self.iou_threshold = tr.get("iou_threshold", 0.3)
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def update(self, detections: List[Detection]) -> List[Track]:
        persons = [d for d in detections if d.class_id == 0]
        assigned = set()
        for track in self.tracks.values():
            best_idx, best_iou = -1, 0.0
            for i, det in enumerate(persons):
                if i in assigned:
                    continue
                score = iou(track.bbox, det.bbox)
                if score > best_iou:
                    best_iou, best_idx = score, i
            if best_idx >= 0 and best_iou >= self.iou_threshold:
                det = persons[best_idx]
                assigned.add(best_idx)
                track.bbox = det.bbox
                track.confidence = det.confidence
                track.hits += 1
                track.time_since_update = 0
                track.history.append(track.center)
                if track.hits >= self.min_hits:
                    track.state = "confirmed"
            else:
                track.time_since_update += 1

        for i, det in enumerate(persons):
            if i in assigned:
                continue
            self.tracks[self.next_id] = Track(self.next_id, det.bbox, det.class_id, det.class_name, det.confidence)
            self.tracks[self.next_id].history.append(self.tracks[self.next_id].center)
            self.next_id += 1

        stale = [tid for tid, t in self.tracks.items() if t.time_since_update > self.max_age]
        for tid in stale:
            del self.tracks[tid]

        return [t for t in self.tracks.values() if t.is_confirmed]

    def get_track(self, track_id: int) -> Optional[Track]:
        return self.tracks.get(track_id)

    def draw_tracks(self, frame, tracks=None):
        tracks = tracks or [t for t in self.tracks.values() if t.is_confirmed]
        out = frame.copy()
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(out, f"ID:{t.track_id}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            if len(t.history) > 1:
                cv2.polylines(out, [np.array(t.history, dtype=np.int32)], False, (255, 0, 255), 2)
        return out

    def reset(self):
        self.tracks = {}
        self.next_id = 1
