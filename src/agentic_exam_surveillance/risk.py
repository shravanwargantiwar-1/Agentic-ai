from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Deque, Dict, Iterable, List, Optional

from .models import Alert, BehaviorEvent


EVENT_WEIGHTS = {
    "mobile_detected": 60,
    "head_turning": 20,
    "talking_gesture": 15,
    "paper_passing": 30,
}


class RiskScoringEngine:
    def __init__(self, min_confidence: float = 0.7, persistence_seconds: int = 2):
        self.min_confidence = min_confidence
        self.persistence_window = timedelta(seconds=persistence_seconds)
        self._events: Dict[int, Deque[BehaviorEvent]] = defaultdict(deque)

    def add_events(self, events: Iterable[BehaviorEvent]) -> None:
        for event in events:
            if event.confidence < self.min_confidence:
                continue
            queue = self._events[event.track_id]
            queue.append(event)
            self._expire_old(queue, event.timestamp)

    def _expire_old(self, queue: Deque[BehaviorEvent], now: datetime) -> None:
        while queue and now - queue[0].timestamp > self.persistence_window:
            queue.popleft()

    def calculate(self, track_id: int) -> float:
        queue = self._events.get(track_id)
        if not queue:
            return 0.0
        self._expire_old(queue, datetime.utcnow())
        return float(sum(EVENT_WEIGHTS.get(e.event_type, 0) for e in queue))

    def event_types(self, track_id: int) -> List[str]:
        return [e.event_type for e in self._events.get(track_id, [])]


class DecisionAgent:
    def __init__(self, threshold: float = 70.0):
        self.threshold = threshold

    def decide(self, track_id: int, risk_score: float, event_types: List[str]) -> Optional[Alert]:
        if risk_score < self.threshold:
            return None
        return Alert(track_id=track_id, risk_score=risk_score, events=event_types)
