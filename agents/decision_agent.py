from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List

from .risk_scoring_agent import RiskScore


class AlertLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Decision:
    track_id: int
    risk_score: float
    level: AlertLevel
    should_alert: bool
    message: str
    timestamp: float = field(default_factory=time.time)


class DecisionAgent:
    def __init__(self, config):
        self.threshold = config.get("risk", {}).get("threshold", 70)
        self.callbacks: List[Callable[[Decision], None]] = []
        self.decisions: Dict[int, Decision] = {}

    def register_callback(self, cb: Callable[[Decision], None]) -> None:
        self.callbacks.append(cb)

    def decide(self, risk_scores: Dict[int, RiskScore]) -> Dict[int, Decision]:
        out = {}
        for tid, risk in risk_scores.items():
            level = self._level(risk.total_score)
            should_alert = risk.total_score >= self.threshold
            decision = Decision(
                track_id=tid,
                risk_score=risk.total_score,
                level=level,
                should_alert=should_alert,
                message=f"Track {tid} risk={risk.total_score:.1f} level={level.value}",
            )
            out[tid] = decision
            self.decisions[tid] = decision
            if should_alert:
                for cb in self.callbacks:
                    cb(decision)
        return out

    def _level(self, score: float) -> AlertLevel:
        if score >= 90:
            return AlertLevel.CRITICAL
        if score >= 70:
            return AlertLevel.HIGH
        if score >= 40:
            return AlertLevel.MEDIUM
        return AlertLevel.LOW

    def get_statistics(self):
        active = sum(1 for d in self.decisions.values() if d.should_alert)
        return {"active_alerts": active, "total_decisions": len(self.decisions)}

    def reset(self):
        self.decisions.clear()
