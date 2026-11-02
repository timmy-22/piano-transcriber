"""Stage 3: estimate global tuning offset in cents."""
import numpy as np


def estimate_tuning(midi_frac: np.ndarray) -> float:
    """Return tuning offset in cents in (-50, 50].

    Use a CIRCULAR mean of the fractional part: +49 and -49 cents are neighbours.
    Hint: map cents to angles (2*pi*cents/100), average as unit vectors.
    """
    raise NotImplementedError
