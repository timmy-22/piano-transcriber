"""Stage 5: activation matrix + onsets -> discrete notes.

Pure NumPy logic, no audio, so it's easy to unit-test with hand-made matrices.
Works per key, so it already handles polyphony (several keys active at once).
"""
import numpy as np
import librosa

from .config import (SR, HOP, key_to_midi, RELEASE_SLOPE_DB, RELEASE_RATIO,
                     REQUIRE_ONSET, ONSET_BEFORE_TOL, ONSET_AFTER_TOL)
from .note import Note


def active_runs(column: np.ndarray) -> list[tuple[int, int]]:
    """Runs of consecutive True frames: [(start, end), ...], end exclusive.
    e.g. [F, T, T, F, T] -> [(1, 3), (4, 5)]
    """
    # Pad with False at both ends so runs touching the edges are still detected
    padded = np.concatenate([[False], column.astype(bool), [False]])
    # Difference is +1 where a run starts, -1 where it ends
    change = np.diff(padded.astype(int))
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def split_at_onsets(run: tuple[int, int], onset_frames: np.ndarray,
                    tol: int = 2) -> list[tuple[int, int]]:
    """Split one run wherever an onset falls inside it (repeated notes).
    Onsets within `tol` frames of the run's start are its own attack: ignore.
    """
    start, end = run
    cuts = [int(o) for o in onset_frames if start + tol < o < end]
    bounds = [start] + cuts + [end]
    return list(zip(bounds[:-1], bounds[1:]))


def find_release(start: int, end: int, rms_db: np.ndarray,
                 slope_db: float = RELEASE_SLOPE_DB, ratio: float = RELEASE_RATIO,
                 window: int = 4, skip: int = 9) -> int | None:
    """Return the frame where the key was released, or None.

    Release = loudness suddenly falls much FASTER than the note was already
    fading. Comparing to the note's own recent fade means fast-decaying high
    notes don't trigger it, but a damper cutting in does, at any pitch.

    slope_db: minimum fall over the next `window` frames (dB)
    ratio:    how many times faster than the previous `window` frames
    skip:     frames to ignore after the start (the attack decays fast)
    """
    for f in range(start + max(skip, window), end - window):
        recent = rms_db[f - window] - rms_db[f]      # fade just before f
        now = rms_db[f] - rms_db[f + window]         # fade just after f
        if now > slope_db and now > ratio * max(recent, 0.5):
            return f
    return None


def activation_to_notes(A: np.ndarray, onset_frames: np.ndarray,
                        sr: int = SR, hop: int = HOP,
                        min_duration: float = 0.05,
                        rms_db: np.ndarray | None = None,
                        slope_db: float = RELEASE_SLOPE_DB,
                        ratio: float = RELEASE_RATIO,
                        require_onset: bool = REQUIRE_ONSET,
                        before_tol: int = ONSET_BEFORE_TOL,
                        after_tol: int = ONSET_AFTER_TOL) -> list[Note]:
    """Activation matrix -> list of Notes in SECONDS, sorted by onset.

    Each candidate note (a run of active frames for one key) is matched to the
    nearest onset within `before_tol` frames before its start or `after_tol`
    frames after it, and its start is moved to that onset.
    The window is two-sided because pYIN's long analysis window "sees" a note
    slightly early, while detected onsets arrive slightly late.

    require_onset: drop candidates with no matching onset. Real notes start
                   with an attack; reverb-tail fragments and junk pitches don't.
    rms_db:        frame loudness in dB, same frame grid (pipeline.loudness_db).
                   Enables release detection for note endings.
    """
    onset_frames = np.asarray(onset_frames)
    notes = []

    for k in range(A.shape[1]):                      # each piano key
        for run in active_runs(A[:, k]):              # each stretch it's active
            for start, end in split_at_onsets(run, onset_frames):

                # Match this candidate to the nearest onset around its start
                near = onset_frames[(onset_frames >= start - before_tol) &
                                    (onset_frames <= start + after_tol)]
                if len(near):
                    start = int(near[np.argmin(np.abs(near - start))])
                elif require_onset:
                    continue                          # no attack: not a real note
                if start >= end:
                    continue

                # End at the key release rather than the end of the reverb tail
                if rms_db is not None:
                    release = find_release(start, end, rms_db,
                                           slope_db=slope_db, ratio=ratio)
                    if release is not None:
                        end = release

                t_on, t_off = librosa.frames_to_time([start, end], sr=sr, hop_length=hop)
                if t_off - t_on >= min_duration:     # drop blips
                    notes.append(Note(float(t_on), float(t_off), key_to_midi(k)))

    return sorted(notes, key=lambda n: (n.onset, n.pitch))
