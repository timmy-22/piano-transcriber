"""Stage 2: pitch tracking -> activation matrix.

THE KEY INTERFACE: everything downstream consumes an activation matrix of shape
(n_frames, 88), where A[i, k] = 1 if piano key k is sounding in frame i.

v1 (monophonic): pYIN gives one f0 per frame -> at most one active key per row.
v2 (polyphonic): swap in NMF / Basic Pitch, which fills several keys per row.
Nothing after this module needs to change.
"""
import numpy as np

from .config import SR, HOP, N_KEYS, FMIN_NOTE, FMAX_NOTE, midi_to_key


def track_f0(y: np.ndarray, sr: int = SR, hop: int = HOP
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (f0, voiced_flag, voiced_prob), one value per frame.

    f0 in Hz, NaN where unvoiced.
    librosa: librosa.pyin(y, fmin=librosa.note_to_hz(FMIN_NOTE),
                          fmax=librosa.note_to_hz(FMAX_NOTE),
                          sr=sr, hop_length=hop)
    Keep hop = HOP so these frames line up with the onset frames.
    """
    raise NotImplementedError


def f0_to_activation(f0: np.ndarray, voiced: np.ndarray,
                     tuning_cents: float = 0.0) -> np.ndarray:
    """Return activation matrix A of shape (len(f0), N_KEYS), dtype bool.

    For each voiced frame:
      1. m = librosa.hz_to_midi(f0)            (fractional MIDI number)
      2. subtract the tuning offset: m - tuning_cents / 100
      3. round to the nearest integer key
      4. set A[frame, midi_to_key(m)] = True if it lies inside 0..87
    Unvoiced frames stay all-False.

    Why subtract tuning BEFORE rounding: a piano 45 cents sharp would otherwise
    have half its notes rounded up to the wrong key.
    """
    raise NotImplementedError
