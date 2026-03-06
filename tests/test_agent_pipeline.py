from agents.behavior_analysis_agent import BehaviorAnalysis, BehaviorEvent, BehaviorType
from agents.decision_agent import AlertLevel, DecisionAgent
from agents.risk_scoring_agent import RiskScoringAgent


def test_decision_agent_threshold_levels():
    agent = DecisionAgent({"risk": {"threshold": 70}})

    class S:  # simple score object
        def __init__(self, total):
            self.total_score = total

    results = agent.decide({1: S(25), 2: S(75), 3: S(95)})
    assert results[1].level == AlertLevel.LOW
    assert results[2].level == AlertLevel.HIGH
    assert results[3].level == AlertLevel.CRITICAL
    assert results[2].should_alert is True


def test_risk_scoring_persistence_filtering():
    cfg = {"risk": {"threshold": 70, "persistence_time": 2, "scores": {"head_turning": 20}}}
    agent = RiskScoringAgent(cfg)

    # simulate two calls before persistence window: no confirmed event
    agent._add_event(1, "head_turning", 0.9, 1.0)
    agent._add_event(1, "head_turning", 0.9, 2.0)
    assert len(agent.event_history[1]) == 0

    # third call after persistence: event should be committed
    agent._add_event(1, "head_turning", 0.9, 3.2)
    assert len(agent.event_history[1]) == 1


def test_risk_scoring_from_behavior_analysis():
    cfg = {"risk": {"threshold": 10, "persistence_time": 0, "scores": {"talking_gesture": 15}}}
    agent = RiskScoringAgent(cfg)
    analysis = {
        5: BehaviorAnalysis(
            track_id=5,
            current_behaviors=[BehaviorEvent(behavior_type=BehaviorType.TALKING, confidence=0.9, timestamp=0.0)],
            suspicion_score=0.8,
        )
    }
    scores = agent.calculate_scores([], analysis, {})
    assert scores[5].total_score >= 10
    assert scores[5].is_alert_triggered is True


def test_risk_scores_only_for_current_tracks():
    cfg = {"risk": {"threshold": 70, "persistence_time": 0, "scores": {"talking_gesture": 15}}}
    agent = RiskScoringAgent(cfg)

    analysis1 = {
        1: BehaviorAnalysis(
            track_id=1,
            current_behaviors=[BehaviorEvent(behavior_type=BehaviorType.TALKING, confidence=0.95, timestamp=0.0)],
            suspicion_score=0.8,
        )
    }
    first = agent.calculate_scores([], analysis1, {})
    assert 1 in first

    # Track 1 disappears; output should not keep stale risk entries.
    second = agent.calculate_scores([], {}, {})
    assert 1 not in second
    assert second == {}

