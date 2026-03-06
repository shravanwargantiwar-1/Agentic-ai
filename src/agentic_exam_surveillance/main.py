from __future__ import annotations

import argparse

from .agents import (
    BehaviourAnalysisAgent,
    DetectionAgent,
    RoleClassificationAgent,
    SurveillanceAgent,
    TrackingAgent,
)
from .evidence import EvidenceManager
from .risk import DecisionAgent, RiskScoringEngine


def run(rtsp_url: str, max_frames: int = 200) -> None:
    surveillance = SurveillanceAgent(rtsp_url)
    detector = DetectionAgent()
    tracker = TrackingAgent()
    role_classifier = RoleClassificationAgent()
    behavior = BehaviourAnalysisAgent()
    risk_engine = RiskScoringEngine(min_confidence=0.7, persistence_seconds=3)
    decision = DecisionAgent(threshold=70)
    evidence = EvidenceManager(output_dir="evidence")

    cap = surveillance.open()
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open RTSP stream: {rtsp_url}")

    frame_idx = 0
    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        student_tracks = [t for t in tracks if role_classifier.classify(t) == "student"]

        events = behavior.analyze(frame, student_tracks)
        risk_engine.add_events(events)

        for track in student_tracks:
            score = risk_engine.calculate(track.track_id)
            alert = decision.decide(track.track_id, score, risk_engine.event_types(track.track_id))
            if alert:
                path = evidence.save(alert, frame)
                print(
                    f"[ALERT] track={alert.track_id} risk={alert.risk_score} events={alert.events} evidence={path}"
                )

    cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agentic AI exam surveillance pipeline")
    parser.add_argument("--rtsp-url", required=True, help="RTSP source URL")
    parser.add_argument("--max-frames", default=200, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(rtsp_url=args.rtsp_url, max_frames=args.max_frames)
