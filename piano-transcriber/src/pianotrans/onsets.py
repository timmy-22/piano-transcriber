"""Stage 4: onset detection (works unchanged for polyphonic audio)."""
import numpy as np
import librosa

from .config import SR, HOP


def onset_envelope(y: np.ndarray, sr: int = SR, hop: int = HOP) -> np.ndarray:
    """Onset strength per frame: spectral flux on a log-mel spectrogram.

    High where energy suddenly INCREASES across many frequencies (note attacks).
    Same frame grid as the pitch track (same sr and hop).
    """
    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)


def detect_onsets(env: np.ndarray, sr: int = SR, hop: int = HOP,
                  delta: float = 0.1, wait: int = 3) -> np.ndarray:
    """Return onset FRAME indices.

    delta: how far above the local average a peak must be.
           Lower = more sensitive (catches quiet notes, more false onsets).
    wait:  minimum frames between onsets (3 frames = ~35 ms).
    """
    return librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=hop,
        units="frames", delta=delta, wait=wait,
    )