"""Stage 5: activation matrix + onsets -> discrete notes.

Pure NumPy logic, no audio, so it's easy to unit-test with hand-made matrices.
Written per key, so it already handles polyphony (several keys active at once).
"""
import numpy as np

from .config import SR, HOP, key_to_midi
from .note import Note


def active_runs(column: np.ndarray) -> list[tuple[int, int]]:
    """Runs of consecutive True frames in one key's column: [(start, end), ...],
    end exclusive. e.g. [F, T, T, F, T] -> [(1, 3), (4, 5)]

    Hint: np.diff on the column padded with False at both ends gives +1 at
    run starts and -1 at run ends.
    """
    raise NotImplementedError


def split_at_onsets(run: tuple[int, int], onset_frames: np.ndarray,
                    tol: int = 2) -> list[tuple[int, int]]:
    """Split one run wherever an onset falls strictly inside it.

    This is how REPEATED notes are separated: the same key stays active,
    so only the onset tells you a new note started. Ignore onsets within
    `tol` frames of the run's start (that's the run's own attack).
    """
    raise NotImplementedError


def activation_to_notes(A: np.ndarray, onset_frames: np.ndarray,
                        sr: int = SR, hop: int = HOP,
                        min_duration: float = 0.05,
                        snap_tol: int = 3) -> list[Note]:
    """Convert activation matrix to a list of Notes in SECONDS, sorted by onset.

    For each key column:
      1. active_runs -> split_at_onsets
      2. snap each note's start back to an onset within `snap_tol` frames
         (pYIN often reports pitch a few frames after the attack)
      3. frames -> seconds with librosa.frames_to_time
      4. drop notes shorter than min_duration
    """
    raise NotImplementedError
