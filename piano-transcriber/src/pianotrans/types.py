"""Shared data types passed between pipeline stages."""
from dataclasses import dataclass


@dataclass
class Note:
    onset: float        # start time in seconds (or beats after quantisation)
    offset: float       # end time in seconds (or beats)
    pitch: int          # MIDI note number (60 = middle C)
    velocity: int = 80  # loudness 0-127 (optional for v1)
