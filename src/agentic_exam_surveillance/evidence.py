from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2

from .models import Alert


class EvidenceManager:
    def __init__(self, output_dir: str = "evidence"):
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)

    def save(self, alert: Alert, frame) -> Optional[Path]:
        ts = alert.timestamp.strftime("%Y%m%d_%H%M%S")
        prefix = f"track_{alert.track_id}_{ts}"

        image_path = self.output / f"{prefix}.jpg"
        ok = cv2.imwrite(str(image_path), frame)
        if not ok:
            return None

        meta_path = self.output / f"{prefix}.json"
        meta_path.write_text(
            json.dumps(
                {
                    "track_id": alert.track_id,
                    "risk_score": alert.risk_score,
                    "events": alert.events,
                    "timestamp": alert.timestamp.isoformat(),
                    "metadata": alert.metadata,
                    "screenshot": str(image_path),
                },
                indent=2,
            )
        )
        return image_path
