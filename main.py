from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import cv2
import yaml

from agents.behavior_analysis_agent import BehaviorAnalysisAgent
from agents.decision_agent import DecisionAgent
from agents.detection_agent import DetectionAgent
from agents.risk_scoring_agent import RiskScoringAgent
from agents.role_classification_agent import RoleClassificationAgent
from agents.surveillance_agent import DemoSurveillanceAgent, SurveillanceAgent
from agents.tracking_agent import TrackingAgent
from alerts.evidence_capture import EvidenceCapture
from alerts.hardware_alert import HardwareAlert
from utils.alert_utils import alert_message


class ExamSurveillanceSystem:
    def __init__(self, config_path: str = "config/config.yaml", demo_mode: bool = False):
        self.config = self._load_config(config_path)
        self.demo_mode = demo_mode
        self.running = False
        self.frame_count = 0
        self.start_time = 0.0

        if demo_mode:
            self.config["demo"] = {"video_source": 0}
            self.surveillance_agent = DemoSurveillanceAgent(self.config)
        else:
            self.surveillance_agent = SurveillanceAgent(self.config)

        self.detection_agent = DetectionAgent(self.config)
        self.evidence = EvidenceCapture(self.config.get("alerts", {}).get("evidence_dir", "evidence"))
        self.hardware = HardwareAlert(self.config.get("alerts", {}).get("hardware_enabled", False))

        # Per-camera stateful agents to isolate streams.
        self.tracking_agents: dict[str, TrackingAgent] = {}
        self.role_agents: dict[str, RoleClassificationAgent] = {}
        self.behavior_agents: dict[str, BehaviorAnalysisAgent] = {}
        self.risk_agents: dict[str, RiskScoringAgent] = {}
        self.decision_agents: dict[str, DecisionAgent] = {}

    def _load_config(self, path: str):
        p = Path(path)
        if p.exists():
            return yaml.safe_load(p.read_text())
        return {}

    def _on_alert(self, decision):
        print(alert_message(decision.track_id, decision.risk_score))
        self.hardware.trigger(decision.message)

    def _camera_agents(self, camera_id: str):
        if camera_id not in self.tracking_agents:
            self.tracking_agents[camera_id] = TrackingAgent(self.config)
            self.role_agents[camera_id] = RoleClassificationAgent(self.config)
            self.behavior_agents[camera_id] = BehaviorAnalysisAgent(self.config)
            self.risk_agents[camera_id] = RiskScoringAgent(self.config)
            decision_agent = DecisionAgent(self.config)
            decision_agent.register_callback(self._on_alert)
            self.decision_agents[camera_id] = decision_agent

        return (
            self.tracking_agents[camera_id],
            self.role_agents[camera_id],
            self.behavior_agents[camera_id],
            self.risk_agents[camera_id],
            self.decision_agents[camera_id],
        )

    def start(self):
        self.running = self.surveillance_agent.start()
        self.start_time = time.time()
        return self.running

    def stop(self):
        self.running = False
        self.surveillance_agent.stop()

    def process_frame(self):
        frames = self.surveillance_agent.get_frames()
        if not frames:
            return {}

        out = {}
        for cam_id, frame_data in frames.items():
            frame = frame_data.image
            tracker, role_agent, behavior_agent, risk_agent, decision_agent = self._camera_agents(cam_id)

            detections = self.detection_agent.detect(frame)
            tracks = tracker.update(detections)
            role_agent.classify(tracks)
            student_tracks = [t for t in tracks if role_agent.is_student(t.track_id)]
            behavior = behavior_agent.analyze(frame, student_tracks)
            associations = risk_agent.associate_detections_to_tracks(detections, student_tracks)
            scores = risk_agent.calculate_scores(detections, behavior, associations)
            decisions = decision_agent.decide(scores)

            for tid, dec in decisions.items():
                if dec.should_alert:
                    events = [e.event_type for e in scores[tid].events]
                    self.evidence.save_screenshot(frame, tid, dec.risk_score, events)

            out[cam_id] = {
                "frame": frame,
                "detections": detections,
                "tracks": tracks,
                "decisions": decisions,
                "risk_scores": scores,
            }
        return out

    def run(self, display=True):
        if not self.start():
            print("Failed to start surveillance agent")
            return
        try:
            while self.running:
                results = self.process_frame()
                if display:
                    for cam_id, data in results.items():
                        tracker = self.tracking_agents.get(cam_id)
                        img = self.detection_agent.draw_detections(data["frame"], data["detections"])
                        if tracker is not None:
                            img = tracker.draw_tracks(img, data["tracks"])
                        cv2.imshow(f"Surveillance - {cam_id}", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                time.sleep(1 / max(1, self.config.get("performance", {}).get("max_fps", 18)))
        finally:
            self.stop()
            if display:
                cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Agentic AI Smart Exam Surveillance System")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    system = ExamSurveillanceSystem(config_path=args.config, demo_mode=args.demo)

    def handler(_sig, _frame):
        system.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    system.run(display=not args.no_display)


if __name__ == "__main__":
    main()
