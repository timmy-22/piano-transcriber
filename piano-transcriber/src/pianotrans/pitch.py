"""Stage 2: pitch tracking -> activation matrix.

THE KEY INTERFACE: everything downstream consumes an activation matrix of shape
(n_frames, 88), where A[i, k] = 1 if piano key k is sounding in frame i.

v1 (monophonic): pYIN gives one f0 per frame -> at most one active key per row.
v2 (polyphonic): swap in NMF / Basic Pitch, which fills several keys per row.
Nothing after this module needs to change.
"""
import numpy as np
import librosa

from .config import SR, HOP, N_KEYS, MIDI_MIN, FMIN_NOTE, FMAX_NOTE, midi_to_key


def track_f0(y: np.ndarray, sr: int = SR, hop: int = HOP
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (f0, voiced, voiced_prob), one value per frame.

    f0:          Hz, NaN where unvoiced
    voiced:      bool, True where pYIN thinks a note is sounding
    voiced_prob: 0-1, how confident it is
    Frame i is centred at time i * hop / sr.
    """
    f0, voiced, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz(FMIN_NOTE),   # lowest pitch to search for (C2, 65 Hz)
        fmax=librosa.note_to_hz(FMAX_NOTE),   # highest (C7, 2093 Hz)
        sr=sr,
        hop_length=hop,                       # same HOP as onsets, so frames line up
    )
    return f0, voiced, voiced_prob


def f0_to_activation(f0: np.ndarray, voiced: np.ndarray,
                     tuning_cents: float = 0.0) -> np.ndarray:
    """turns pYIN's “one frequency per frame” into the shared format: a grid of frames x 88 keys, 
    True where a key is sounding. It's a piano roll as a matrix

    Return activation matrix A, shape (n_frames, N_KEYS), dtype bool.

    A[i, k] is True if piano key k is sounding in frame i.
    v1: at most one True per row (one f0 per frame).
    """
    n_frames = len(f0)
    A = np.zeros((n_frames, N_KEYS), dtype=bool)

    # Only frames with a pitch: voiced, and f0 is a real number (not NaN)
    frames = np.where(voiced & ~np.isnan(f0))[0]

    # Hz -> fractional MIDI number, e.g. 446 Hz -> 69.24
    midi = librosa.hz_to_midi(f0[frames])

    # Undo the global tuning BEFORE rounding, then snap to the nearest key
    midi = np.round(midi - tuning_cents / 100).astype(int)

    # MIDI number -> column index (A0 = MIDI 21 = column 0)
    keys = midi - MIDI_MIN

    # Ignore anything outside the 88 keys
    ok = (keys >= 0) & (keys < N_KEYS)
    A[frames[ok], keys[ok]] = True
    return A
