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

# --- Segmentation (tuned on data/tune, reported on data/test) ---
RELEASE_SLOPE_DB = 3.0   # note ends where loudness falls > this many dB per 4 frames...
RELEASE_RATIO = 1.5      # ...and > this many times faster than just before
REQUIRE_ONSET = True     # drop candidate notes with no onset near their start
ONSET_BEFORE_TOL = 3     # frames an onset may sit BEFORE a pitch run starts (~35 ms)
ONSET_AFTER_TOL = 2      # frames an onset may sit AFTER a pitch run starts (~23 ms)

# --- Rhythm ---
TEMPO_MIN = 60.0         # BPM search range for estimate_tempo
TEMPO_MAX = 180.0
GRID = 0.25              # quantisation grid in beats (0.25 = sixteenth note)
LEGATO_GAP = 0.5         # silence (beats) before the next note that counts as a real rest
