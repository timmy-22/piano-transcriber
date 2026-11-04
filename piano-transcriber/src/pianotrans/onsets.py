"""Stage 4: onset detection (works unchanged for polyphonic audio)."""
import numpy as np

from .config import SR, HOP


def onset_envelope(y: np.ndarray, sr: int = SR, hop: int = HOP) -> np.ndarray:
    """Onset strength per frame (spectral flux).

    librosa: librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    Plot it under a spectrogram in a notebook: peaks should sit on note attacks.
    """
    raise NotImplementedError


def detect_onsets(env: np.ndarray, sr: int = SR, hop: int = HOP) -> np.ndarray:
    """Return onset FRAME indices (ints).

    librosa: librosa.onset.onset_detect(onset_envelope=env, sr=sr,
                                        hop_length=hop, units="frames")
    Tunable: `delta` (threshold) and `wait` (min frames between onsets).
    Convert to seconds when needed: librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    """
    raise NotImplementedError
