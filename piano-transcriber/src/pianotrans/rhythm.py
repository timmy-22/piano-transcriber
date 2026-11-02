"""Stage 6: tempo estimation and quantisation to a beat grid."""
from .types import Note


def estimate_tempo(notes: list[Note]) -> float:
    """Return tempo in BPM from inter-onset intervals (histogram or autocorrelation)."""
    raise NotImplementedError


def quantise(notes: list[Note], bpm: float, grid: float = 0.25) -> list[Note]:
    """Convert seconds to beats and snap onsets/durations to `grid` (0.25 = 16th note)."""
    raise NotImplementedError
