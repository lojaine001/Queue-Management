from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class Person:
    track_id: int
    entry_time: datetime
    gender: str = "unknown"
    age: Optional[float] = None
    has_bag: bool = False
    has_caddy: bool = False
    is_group: bool = False
    group_id: Optional[str] = None
    
    picks_bakery: bool = False
    is_express: bool = False        # grab-and-go, 1–3 items, short shop time

    # Simulation state
    entrance_eta: Optional[datetime] = None          # when they leave the entrance zone
    entrance_done_time: Optional[datetime] = None    # left entrance zone
    entrance_dwell_sec: float = 0.0                  # seconds spent at entrance
    shopping_end_eta: Optional[datetime] = None
    shopping_done_time: Optional[datetime] = None
    service_start_time: Optional[datetime] = None
    service_end_time: Optional[datetime] = None
    lane_id: Optional[int] = None
    
    @property
    def dwell_seconds(self) -> float:
        if self.service_end_time:
            return (self.service_end_time - self.entry_time).total_seconds()
        return 0.0

@dataclass
class Lane:
    lane_id: int
    camera_id: str
    active: bool = True
    current_person: Optional[Person] = None
    queue: list[Person] = field(default_factory=list)
    service_end_eta: Optional[datetime] = None

    def is_free(self) -> bool:
        return self.current_person is None
