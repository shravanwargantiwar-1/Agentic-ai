from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import statistics



class PersonRole(Enum):
    STUDENT = "student"
    INVIGILATOR = "invigilator"


@dataclass
class RoleClassification:
    track_id: int
    role: PersonRole
    confidence: float
    mobility_score: float


class RoleClassificationAgent:
    def __init__(self, config):
        rc = config.get("role_classification", {})
        self.mobility_threshold = rc.get("mobility_threshold", 50)
        self.invigilator_min_mobility = rc.get("invigilator_min_mobility", 100)
        self.classifications: Dict[int, RoleClassification] = {}
        self.role_history = defaultdict(list)

    def classify(self, tracks: List[object]) -> Dict[int, RoleClassification]:
        out = {}
        for t in tracks:
            mobility = self._mobility(t)
            if mobility >= self.invigilator_min_mobility:
                role, conf = PersonRole.INVIGILATOR, 0.85
            elif mobility < self.mobility_threshold:
                role, conf = PersonRole.STUDENT, 0.85
            else:
                role, conf = PersonRole.STUDENT, 0.6
            c = RoleClassification(t.track_id, role, conf, mobility)
            out[t.track_id] = c
            self.classifications[t.track_id] = c
            self.role_history[t.track_id].append(role)
        return out

    def _mobility(self, track) -> float:
        if len(track.history) < 2:
            return 0.0
        distances = []
        for i in range(1, len(track.history)):
            x1, y1 = track.history[i - 1]
            x2, y2 = track.history[i]
            distances.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        return float(statistics.mean(distances))

    def is_student(self, track_id: int) -> bool:
        entry = self.classifications.get(track_id)
        return bool(entry and entry.role == PersonRole.STUDENT)

    def get_statistics(self):
        students = sum(1 for c in self.classifications.values() if c.role == PersonRole.STUDENT)
        invigilators = sum(1 for c in self.classifications.values() if c.role == PersonRole.INVIGILATOR)
        return {"students": students, "invigilators": invigilators, "total_classified": len(self.classifications)}

    def reset(self):
        self.classifications.clear()
        self.role_history.clear()
