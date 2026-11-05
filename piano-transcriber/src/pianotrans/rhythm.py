"""Stage 6: tempo estimation and quantisation (seconds -> beats)."""
import numpy as np

from .config import TEMPO_MIN, TEMPO_MAX, GRID, LEGATO_GAP
from .note import Note

# Note lengths (in beats) that appear in ordinary notation:
# sixteenth, eighth, dotted eighth, quarter, dotted quarter, half, dotted half, whole
COMMON_DURATIONS = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])


def _ioi_error(ioi: np.ndarray, bpm: float, subdivisions: int) -> float:
    """Mean squared distance of each IOI from a whole number of units."""
    unit = 60.0 / bpm / subdivisions                # seconds per eighth note
    x = ioi / unit
    return float(np.mean((x - np.round(x)) ** 2))


def estimate_tempo(onset_times: np.ndarray, bpm_min: float = TEMPO_MIN,
                   bpm_max: float = TEMPO_MAX, subdivisions: int = 2,
                   step: float = 0.25, octave_tol: float = 0.01) -> float:
    """Return tempo in BPM: the beat length that best explains the gaps
    between note starts as whole multiples of 1/`subdivisions` of a beat.

    For each candidate tempo, the "unit" is one eighth note (subdivisions=2).
    Every inter-onset interval (IOI) divided by the unit should be close to a
    whole number. Score = mean squared distance from whole numbers; lowest wins.

    Octave ambiguity: if 90 BPM fits, 180 BPM fits equally well (every IOI is
    still a whole number of the smaller unit), and with a tempo that falls
    between search steps either can win by a hair. So after the search, if
    half (or a third) of the winner fits within `octave_tol` of it, take the
    slower tempo. Rule: the SLOWEST tempo that explains everything.
    """
    ioi = np.diff(np.sort(np.asarray(onset_times, dtype=float)))
    ioi = ioi[ioi > 0.03]                           # ignore near-simultaneous onsets
    if len(ioi) == 0:
        return 120.0

    candidates = np.arange(bpm_min, bpm_max + step, step)
    errors = np.array([_ioi_error(ioi, b, subdivisions) for b in candidates])
    best = float(candidates[np.argmin(errors)])
    best_err = float(errors.min())

    for divisor in (2, 3):
        slower = best / divisor
        if slower < bpm_min:
            continue
        # refine around best/divisor: the true slow tempo may sit between steps
        local = np.arange(slower - 2 * step, slower + 2 * step + 1e-9, step / 4)
        local_err = np.array([_ioi_error(ioi, b, subdivisions) for b in local])
        if local_err.min() <= best_err + octave_tol:
            return float(local[np.argmin(local_err)])
    return best


def quantise(notes: list[Note], bpm: float, t0: float | None = None,
             grid: float = GRID, legato_gap: float = LEGATO_GAP) -> list[Note]:
    """Return new Notes with onset/offset in BEATS, snapped to `grid`
    (0.25 = sixteenth note when the beat is a quarter note).

    t0: time (s) of beat 0. Default: the first note's onset, i.e. assume the
        piece starts on a beat (no pickup bar).

    Durations: a note lasts until the next note starts (legato) unless the
    silence before the next note is at least `legato_gap` beats. Only then
    (and for the last note) is the note's own measured length used, snapped to
    the nearest COMMON note value rather than any sixteenth multiple: players
    release slightly early, so 1.8 beats should read as a half note (2), not a
    double-dotted quarter (1.75).
    Quantising every end independently to the grid got only 58% of lengths
    right on the synthetic test set; legato + common values gets nearly all.
    """
    if not notes:
        return []
    notes = sorted(notes, key=lambda n: (n.onset, n.pitch))
    if t0 is None:
        t0 = notes[0].onset

    def to_beats(t):
        return (t - t0) * bpm / 60.0

    def snap(b):
        return round(b / grid) * grid

    on = [snap(to_beats(n.onset)) for n in notes]
    out = []
    for i, n in enumerate(notes):
        raw_end = to_beats(n.offset)
        nxt = next((on[j] for j in range(i + 1, len(notes)) if on[j] > on[i]), None)
        if nxt is not None and nxt - raw_end < legato_gap:
            end = nxt                           # legato: hold until the next note
        else:
            # real gap (or last note): use own length, snapped to a common value
            length = raw_end - on[i]
            if length > COMMON_DURATIONS[-1]:
                length = snap(length)                       # longer than a whole: plain grid
            else:
                length = COMMON_DURATIONS[np.argmin(np.abs(COMMON_DURATIONS - length))]
            end = on[i] + length
            if nxt is not None:
                end = min(end, nxt)             # never overlap the next note
        end = max(end, on[i] + grid)            # at least one grid step long
        out.append(Note(on[i], end, n.pitch, n.velocity))
    return out
