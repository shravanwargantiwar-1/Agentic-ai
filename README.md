# Agentic AI Smart Exam Surveillance & Alert System (OpenVINO-ready)

This project monitors exam halls from CCTV RTSP streams and raises malpractice alerts using a **multi-agent AI pipeline**.

## Implemented Project Structure

```text
.
├── README.md
├── requirements.txt
├── config/config.yaml
├── main.py
├── agents/
├── utils/
├── dashboard/
├── alerts/
└── scripts/
```

## Agent Pipeline

1. **SurveillanceAgent**: RTSP frame capture (multi-camera)
2. **DetectionAgent**: YOLO11/OpenVINO-compatible object detection
3. **TrackingAgent**: ByteTrack-style person tracking IDs
4. **RoleClassificationAgent**: Student vs invigilator classification
5. **BehaviorAnalysisAgent**: suspicious behavior events
6. **RiskScoringAgent**: confidence + persistence + decay scoring
7. **DecisionAgent**: alert threshold and escalation

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --config config/config.yaml --demo
```

Press `q` to stop display mode.

## Dashboard

```bash
python dashboard/app.py
```

Open: `http://localhost:5000`

## OpenVINO conversion

```bash
python scripts/export_model.py
```
