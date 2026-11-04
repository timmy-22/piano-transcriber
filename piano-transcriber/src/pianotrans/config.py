"""Shared settings. Every stage uses the same SR and HOP so frame i means
the same moment in time everywhere (pitch track, onsets, activation matrix)."""

SR = 22050          # sample rate (Hz)
HOP = 256           # hop length (samples) -> 11.6 ms per frame
MIDI_MIN = 21       # A0, lowest piano key
MIDI_MAX = 108      # C8, highest piano key
N_KEYS = MIDI_MAX - MIDI_MIN + 1   # 88

# v1 pitch search range. Narrower than the full piano = fewer octave errors.
# Matches the test melodies (MIDI 48-84). Widen later if needed.
FMIN_NOTE = "C2"
FMAX_NOTE = "C7"


def key_to_midi(k: int) -> int:
    """Activation-matrix column -> MIDI note number."""
    return k + MIDI_MIN


def midi_to_key(m: int) -> int:
    """MIDI note number -> activation-matrix column."""
    return m - MIDI_MIN
