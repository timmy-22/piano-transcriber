"""Stage 5: combine onsets + pitch track into discrete notes."""
import numpy as np
from .types import Note


def segment_notes(onset_times: np.ndarray, f0: np.ndarray, voiced: np.ndarray,
                  sr: int, hop_length: int = 256, tuning_cents: float = 0.0,
                  min_duration: float = 0.05) -> list[Note]:
    """Each note runs from an onset to the next onset or until unvoiced.

    Pitch = median of frame pitches (after tuning correction), rounded to MIDI.
    Drop notes shorter than min_duration.
    """
    raise NotImplementedError
