from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple


BBox = Tuple[int, int, int, int]


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: BBox


@dataclass
class Track:
    track_id: int
    label: str
    confidence: float
    bbox: BBox


@dataclass
class BehaviorEvent:
    track_id: int
    event_type: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StudentState:
    track_id: int
    role: str = "student"
    risk_score: float = 0.0
    events: List[BehaviorEvent] = field(default_factory=list)


@dataclass
class Alert:
    track_id: int
    risk_score: float
    events: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, str] = field(default_factory=dict)
