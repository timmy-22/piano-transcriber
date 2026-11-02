"""Stage 2: frame-by-frame fundamental frequency (f0) tracking.

Write `yin` yourself first (de Cheveigné & Kawahara 2002), then compare
against `librosa.pyin` in a notebook and save the comparison plot to figures/.
"""
import numpy as np


def yin(y: np.ndarray, sr: int, fmin: float = 27.5, fmax: float = 4186.0,
        frame_length: int = 2048, hop_length: int = 256,
        threshold: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Return (f0, voiced) per frame. f0 in Hz (nan where unvoiced), voiced is bool.

    Steps: difference function -> cumulative mean normalised difference ->
    absolute threshold -> parabolic interpolation.
    """
    raise NotImplementedError


def hz_to_midi(f: np.ndarray) -> np.ndarray:
    """Fractional MIDI number: m = 69 + 12*log2(f/440)."""
    return 69 + 12 * np.log2(np.asarray(f, dtype=float) / 440.0)
