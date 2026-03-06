from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .behavior_analysis_agent import BehaviorAnalysis, BehaviorType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .detection_agent import Detection


@dataclass
class RiskEvent:
    event_type: str
    score: float
    confidence: float
    timestamp: float
    details: dict = field(default_factory=dict)


@dataclass
class RiskScore:
    track_id: int
    total_score: float
    breakdown: Dict[str, float]
    events: List[RiskEvent]
    is_alert_triggered: bool


class RiskScoringAgent:
    def __init__(self, config):
        risk = config.get("risk", {})
        self.threshold = risk.get("threshold", 70)
        self.decay_rate = risk.get("decay_rate", 5)
        self.persistence_time = risk.get("persistence_time", 2)
        self.scores_cfg = risk.get("scores", {"mobile_detected": 60, "head_turning": 20, "talking_gesture": 15})
        self.min_confidence = 0.7
        self.event_history = defaultdict(list)
        self.pending_events = defaultdict(dict)
        self.scores: Dict[int, RiskScore] = {}

    def calculate_scores(self, detections: list, behavior_analyses: Dict[int, BehaviorAnalysis], track_to_detections: dict) -> Dict[int, RiskScore]:
        now = time.time()
        current_scores: Dict[int, RiskScore] = {}
        current_track_ids = set(behavior_analyses.keys())

        # Clear stale pending/event history for tracks no longer visible in this frame.
        stale_tracks = set(self.event_history.keys()) - current_track_ids
        for track_id in stale_tracks:
            self.event_history.pop(track_id, None)
            self.pending_events.pop(track_id, None)

        for tid, analysis in behavior_analyses.items():
            self._apply_decay(tid, now)
            for det in track_to_detections.get(tid, []):
                if det.class_name == "cell phone":
                    self._add_event(tid, "mobile_detected", det.confidence, now)
                elif det.class_name == "calculator":
                    self._add_event(tid, "calculator_detected", det.confidence, now)
            for b in analysis.current_behaviors:
                et = self._map_behavior(b.behavior_type)
                if et:
                    self._add_event(tid, et, b.confidence, now)
            current_scores[tid] = self._build_score(tid)

        self.scores = current_scores
        return current_scores

    def _map_behavior(self, behavior: BehaviorType) -> Optional[str]:
        mapping = {BehaviorType.HEAD_TURNING: "head_turning", BehaviorType.TALKING: "talking_gesture", BehaviorType.PHONE_USAGE: "mobile_detected"}
        return mapping.get(behavior)

    def _add_event(self, track_id: int, event_type: str, confidence: float, ts: float):
        if confidence < self.min_confidence:
            return
        if self.persistence_time <= 0:
            self.event_history[track_id].append(RiskEvent(event_type, float(self.scores_cfg.get(event_type, 10)), confidence, ts))
            return
        pending = self.pending_events[track_id]
        if event_type not in pending:
            pending[event_type] = ts
            return
        if ts - pending[event_type] < self.persistence_time:
            return
        self.event_history[track_id].append(RiskEvent(event_type, float(self.scores_cfg.get(event_type, 10)), confidence, ts))
        del pending[event_type]

    def _apply_decay(self, track_id: int, now: float):
        ev = self.event_history.get(track_id, [])
        for e in ev:
            e.score = max(0.0, e.score - ((now - e.timestamp) * self.decay_rate / 10))
        self.event_history[track_id] = [e for e in ev if now - e.timestamp < 60 and e.score > 0]

    def _build_score(self, track_id: int) -> RiskScore:
        breakdown = defaultdict(float)
        for e in self.event_history.get(track_id, []):
            breakdown[e.event_type] += e.score
        total = min(100.0, sum(breakdown.values()))
        return RiskScore(track_id, total, dict(breakdown), list(self.event_history.get(track_id, [])), total >= self.threshold)

    def associate_detections_to_tracks(self, detections: list, tracks: list, max_distance: float = 100) -> dict:
        out = defaultdict(list)
        objects = [d for d in detections if d.class_id != 0]
        for obj in objects:
            ox, oy = obj.center
            best_id, best_dist = None, float("inf")
            for t in tracks:
                tx, ty = t.center
                dist = ((ox - tx) ** 2 + (oy - ty) ** 2) ** 0.5
                if dist < best_dist and dist < max_distance:
                    best_dist = dist
                    best_id = t.track_id
            if best_id is not None:
                out[best_id].append(obj)
        return out

    def get_statistics(self):
        vals = [s.total_score for s in self.scores.values()]
        return {"total_tracks": len(self.scores), "high_risk": sum(1 for v in vals if v >= self.threshold), "average_score": round(sum(vals) / len(vals), 2) if vals else 0}

    def reset(self):
        self.event_history.clear()
        self.pending_events.clear()
        self.scores.clear()
