from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import statistics



class BehaviorType(Enum):
    HEAD_TURNING = "head_turning"
    PHONE_USAGE = "phone_usage"
    TALKING = "talking"
    LOOKING_AROUND = "looking_around"


@dataclass
class BehaviorEvent:
    behavior_type: BehaviorType
    confidence: float
    timestamp: float
    details: dict = field(default_factory=dict)


@dataclass
class BehaviorAnalysis:
    track_id: int
    current_behaviors: List[BehaviorEvent]
    suspicion_score: float


class BehaviorAnalysisAgent:
    def __init__(self, config):
        b = config.get("behavior", {})
        self.head_turn_threshold = b.get("head_turn_angle_threshold", 30)
        self.talking_frames = b.get("talking_detection_frames", 15)
        self.head_angle_history = defaultdict(list)

    def analyze(self, frame, tracks: List[object]) -> Dict[int, BehaviorAnalysis]:
        output = {}
        for t in tracks:
            events: List[BehaviorEvent] = []
            angle = self._synthetic_head_angle(t)
            self.head_angle_history[t.track_id].append(angle)
            if abs(angle) > self.head_turn_threshold:
                events.append(BehaviorEvent(BehaviorType.HEAD_TURNING, 0.75, time.time(), {"angle": angle}))
            recent = self.head_angle_history[t.track_id][-self.talking_frames:]
            if len(recent) >= self.talking_frames:
                var = statistics.pvariance(recent) if len(recent) > 1 else 0.0
                if 5 < var < 50:
                    events.append(BehaviorEvent(BehaviorType.TALKING, 0.6, time.time()))
            suspicion = min(1.0, sum(e.confidence for e in events) / 2)
            output[t.track_id] = BehaviorAnalysis(t.track_id, events, suspicion)
        return output

    def _synthetic_head_angle(self, track: Track) -> float:
        if len(track.history) < 2:
            return 0.0
        x1, _ = track.history[-2]
        x2, _ = track.history[-1]
        return float(max(-45, min(45, x2 - x1)))

    def reset(self):
        self.head_angle_history.clear()
