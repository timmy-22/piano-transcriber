"""Stage 3: global tuning estimate."""
import numpy as np


def estimate_tuning(f0: np.ndarray, voiced: np.ndarray) -> float:
    """Return the global tuning offset in cents, in [-50, 50).

    Steps:
      1. m = librosa.hz_to_midi(f0[voiced])
      2. cents = 100 * (m - np.round(m))           # each frame's offset
      3. CIRCULAR mean: +49 and -49 cents are 2 cents apart, not 98.
         angle = 2*pi*cents/100 -> mean of cos and sin -> atan2 -> back to cents.
    Return 0.0 if there are no voiced frames.

    Cross-check against librosa.pitch_tuning (returns a fraction of a semitone).
    """
    raise NotImplementedError
