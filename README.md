# Agentic AI Smart Exam Surveillance & Alert System (OpenVINO-ready)

This project monitors exam halls from CCTV RTSP streams and raises malpractice alerts using a multi-agent AI pipeline.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── config/config.yaml
├── main.py
├── agents/
├── alerts/
├── dashboard/
├── scripts/
├── utils/
├── src/agentic_exam_surveillance/
└── tests/
```

## Runtime Pipeline

1. SurveillanceAgent (RTSP / demo capture)
2. DetectionAgent (YOLO / OpenVINO compatible)
3. TrackingAgent (camera-isolated tracking state)
4. RoleClassificationAgent
5. BehaviorAnalysisAgent
6. RiskScoringAgent (current-track-only score output)
7. DecisionAgent + EvidenceCapture + optional HardwareAlert

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --config config/config.yaml --demo
```

Press `q` in display mode to exit.

## Dashboard

```bash
python dashboard/app.py
```

Open `http://localhost:5000`.

## Model Conversion Notes

```bash
python scripts/export_model.py
python scripts/download_models.py
```
