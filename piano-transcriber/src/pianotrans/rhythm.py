"""Stage 6: tempo estimation and quantisation (seconds -> beats)."""
import numpy as np

from .config import SR, HOP
from .note import Note


def estimate_tempo(y: np.ndarray, sr: int = SR, hop: int = HOP) -> tuple[float, float]:
    """Return (bpm, first_beat_time_seconds).

    librosa: tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    Watch for octave errors in tempo (half or double the true BPM).
    Later: write your own from inter-onset intervals and compare.
    """
    raise NotImplementedError


def quantise(notes: list[Note], bpm: float, t0: float = 0.0,
             grid: float = 0.25) -> list[Note]:
    """Return new Notes with onset/offset in BEATS, snapped to `grid`
    (0.25 = sixteenth note in 4/4).

    beats = (seconds - t0) * bpm / 60, then round(beats / grid) * grid.
    Enforce a minimum duration of one grid step so nothing collapses to zero.
    """
    raise NotImplementedError
