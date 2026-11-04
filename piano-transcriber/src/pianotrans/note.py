"""Shared data types passed between pipeline stages."""
from dataclasses import dataclass


@dataclass
class Note:
    onset: float        # start, in seconds (or beats after quantisation)
    offset: float       # end, in seconds (or beats)
    pitch: int          # MIDI note number (60 = middle C)
    velocity: int = 80  # 0-127, optional in v1
