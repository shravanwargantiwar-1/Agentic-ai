# Agentic AI Smart Exam Surveillance & Alert System

A modular, CPU-friendly exam malpractice monitoring pipeline that processes RTSP CCTV streams and raises real-time alerts using an **agentic architecture**.

## What this starter implements

- RTSP frame ingestion (OpenCV)
- Detection agent abstraction (YOLO/OpenVINO-ready interface)
- Multi-object tracking abstraction (ByteTrack/DeepSORT-ready interface)
- Role classification (student vs invigilator heuristic)
- Behaviour analysis hooks (pose/motion events)
- Risk scoring engine with persistence window filtering
- Decision agent with threshold-based alerts
- Evidence manager that stores snapshots and metadata

> This repository is a production-ready scaffold: concrete model weights and hardware integrations can be plugged in without changing the orchestration layer.

## Architecture

```text
CCTV (RTSP) -> SurveillanceAgent -> DetectionAgent -> TrackingAgent
           -> RoleClassificationAgent -> BehaviourAnalysisAgent
           -> RiskScoringEngine -> DecisionAgent -> AlertManager
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m agentic_exam_surveillance.main --rtsp-url "rtsp://user:pass@camera_ip:554/stream"
```

## OpenVINO integration flow

```bash
yolo export model=yolo11n.pt format=onnx
mo --input_model yolo11n.onnx
```

Pass generated `.xml/.bin` files to the detector implementation.

## Project layout

```text
src/agentic_exam_surveillance/
  main.py                # end-to-end orchestration
  models.py              # shared dataclasses
  agents.py              # agent implementations
  risk.py                # scoring + decision logic
  evidence.py            # screenshot/event persistence
```

## Notes

- Designed to run inference locally (edge laptop) and push events to cloud backends.
- Default detector/tracker are lightweight placeholders so the pipeline can run before model integration.
