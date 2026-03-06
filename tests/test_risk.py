from datetime import datetime, timedelta

from agentic_exam_surveillance.models import BehaviorEvent
from agentic_exam_surveillance.risk import DecisionAgent, RiskScoringEngine


def test_risk_accumulates_high_confidence_events():
    now = datetime.utcnow()
    engine = RiskScoringEngine(min_confidence=0.7, persistence_seconds=3)
    engine.add_events(
        [
            BehaviorEvent(track_id=1, event_type="mobile_detected", confidence=0.95, timestamp=now),
            BehaviorEvent(track_id=1, event_type="head_turning", confidence=0.8, timestamp=now),
        ]
    )
    assert engine.calculate(1) == 80.0


def test_risk_ignores_low_confidence_events():
    engine = RiskScoringEngine(min_confidence=0.7, persistence_seconds=3)
    engine.add_events([BehaviorEvent(track_id=1, event_type="mobile_detected", confidence=0.5)])
    assert engine.calculate(1) == 0.0


def test_risk_expires_old_events():
    now = datetime.utcnow()
    old = now - timedelta(seconds=10)
    engine = RiskScoringEngine(min_confidence=0.7, persistence_seconds=2)
    engine.add_events(
        [
            BehaviorEvent(track_id=2, event_type="mobile_detected", confidence=0.9, timestamp=old),
            BehaviorEvent(track_id=2, event_type="head_turning", confidence=0.9, timestamp=now),
        ]
    )
    assert engine.calculate(2) == 20.0


def test_decision_agent_threshold():
    decider = DecisionAgent(threshold=70)
    assert decider.decide(track_id=1, risk_score=60, event_types=["head_turning"]) is None
    assert decider.decide(track_id=1, risk_score=75, event_types=["mobile_detected"]) is not None
