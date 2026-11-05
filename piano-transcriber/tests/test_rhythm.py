import numpy as np
import pytest

from pianotrans.note import Note
from pianotrans.rhythm import estimate_tempo, quantise


def onsets_from_beats(beats, bpm, t0=0.5, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = t0 + np.array(beats) * 60 / bpm
    return t + rng.normal(0, jitter, len(t))


RHYTHM = [0, 1, 1.5, 2, 3, 3.5, 4, 6, 7, 7.5, 8]      # quarters, eighths, a half note


@pytest.mark.parametrize("bpm", [72, 100, 133])
def test_tempo_exact(bpm):
    assert estimate_tempo(onsets_from_beats(RHYTHM, bpm)) == pytest.approx(bpm, rel=0.02)


def test_tempo_with_human_timing():
    """+/-10 ms of random timing error (a tidy human) should not change the answer."""
    est = estimate_tempo(onsets_from_beats(RHYTHM, 96, jitter=0.01))
    assert est == pytest.approx(96, rel=0.03)


def test_quantise_onsets_and_legato_durations():
    bpm = 120                                   # 1 beat = 0.5 s
    # quarter, quarter, half; each released slightly early (90% of its slot)
    notes = [Note(1.0, 1.45, 60), Note(1.5, 1.95, 62), Note(2.0, 2.9, 64)]
    q = quantise(notes, bpm)
    assert [n.onset for n in q] == [0, 1, 2]
    assert [n.offset - n.onset for n in q] == [1, 1, 2]


def test_real_rest_is_kept():
    """A gap of a whole beat before the next note is a rest, not legato."""
    bpm = 120
    notes = [Note(0.0, 0.5, 60), Note(1.0, 1.5, 62)]   # quarter, quarter rest, quarter
    q = quantise(notes, bpm)
    assert q[0].offset == 1.0 and q[1].onset == 2.0


def test_tempo_between_search_steps_prefers_slower():
    """True tempo 89.93 BPM (not on the 0.25 search grid), perfect onsets.
    Double tempo must not win just because it lands closer to a grid step."""
    beats = [0, 2, 6, 10, 11, 14, 16, 19, 21, 23, 25, 27, 31, 34, 38, 42]
    est = estimate_tempo(onsets_from_beats(beats, 89.93))
    assert est == pytest.approx(89.93, rel=0.02)
