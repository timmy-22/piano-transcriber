"""Segmentation is pure logic: test it with hand-made activation matrices."""
import numpy as np

from pianotrans.config import N_KEYS, midi_to_key, SR, HOP
from pianotrans.segment import active_runs, split_at_onsets, activation_to_notes

FRAME = HOP / SR


def test_active_runs():
    col = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], bool)
    assert active_runs(col) == [(1, 3), (4, 5), (7, 10)]


def test_split_repeated_note():
    """Key held frames 0-40, but an onset at 20: two notes, not one."""
    assert split_at_onsets((0, 40), np.array([0, 20])) == [(0, 20), (20, 40)]


def test_two_notes_from_matrix():
    A = np.zeros((100, N_KEYS), bool)
    A[5:40, midi_to_key(60)] = True
    A[45:90, midi_to_key(64)] = True
    notes = activation_to_notes(A, onset_frames=np.array([4, 45]))
    assert [n.pitch for n in notes] == [60, 64]
    assert abs(notes[0].onset - 4 * FRAME) < 2 * FRAME   # snapped back to onset


def test_chord_polyphony_ready():
    """Two keys at once (v2 case) should give two notes with the same onset."""
    A = np.zeros((50, N_KEYS), bool)
    A[10:40, midi_to_key(60)] = True
    A[10:40, midi_to_key(67)] = True
    notes = activation_to_notes(A, onset_frames=np.array([10]))
    assert sorted(n.pitch for n in notes) == [60, 67]
    assert abs(notes[0].onset - notes[1].onset) < 1e-9


def test_drops_blips():
    A = np.zeros((50, N_KEYS), bool)
    A[10:12, midi_to_key(60)] = True    # ~23 ms: noise, not a note
    assert activation_to_notes(A, onset_frames=np.array([10])) == []


def test_fragment_without_onset_dropped():
    """A run with no attack anywhere near its start (e.g. a reverb-tail
    fragment) is not a played note."""
    A = np.zeros((100, N_KEYS), bool)
    A[5:40, midi_to_key(60)] = True       # real note, onset at 5
    A[60:80, midi_to_key(61)] = True      # fragment, no onset near 60
    notes = activation_to_notes(A, onset_frames=np.array([5]))
    assert [n.pitch for n in notes] == [60]


def test_onset_just_after_pitch_start():
    """pYIN can report pitch a frame or two BEFORE the detected onset.
    The note must be kept and start at the onset."""
    A = np.zeros((60, N_KEYS), bool)
    A[10:50, midi_to_key(60)] = True
    notes = activation_to_notes(A, onset_frames=np.array([12]))
    assert len(notes) == 1
    assert abs(notes[0].onset - 12 * FRAME) < 1e-9
