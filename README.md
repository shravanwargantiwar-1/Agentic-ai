# Agentic AI Smart Exam Surveillance & Alert System

This project monitors exam halls from CCTV RTSP streams and raises malpractice alerts using a multi-agent AI pipeline.

## Architecture

CCTV (RTSP)
↓
SurveillanceAgent
↓
DetectionAgent
↓
TrackingAgent
↓
RoleClassificationAgent
↓
BehaviourAnalysisAgent
↓
RiskScoringEngine
↓
DecisionAgent
↓
AlertManager

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --config config/config.yaml --demo