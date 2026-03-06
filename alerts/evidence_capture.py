from __future__ import annotations

import json
from pathlib import Path

import cv2


class EvidenceCapture:
    def __init__(self, output_dir: str = "evidence"):
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)

    def save_screenshot(self, frame, track_id: int, risk: float, events: list[str]):
        name = f"track_{track_id}_{int(__import__('time').time())}"
        img = self.output / f"{name}.jpg"
        cv2.imwrite(str(img), frame)
        meta = self.output / f"{name}.json"
        meta.write_text(json.dumps({"track_id": track_id, "risk": risk, "events": events, "image": str(img)}, indent=2))
        return img
