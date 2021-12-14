from __future__ import annotations
from dataclasses import dataclass

@dataclass
class F1RaceCourse:
    name: str
    length: int
    num_bends: int

    @classmethod
    def from_id(cls, id: int) -> F1RaceCourse:
        ...