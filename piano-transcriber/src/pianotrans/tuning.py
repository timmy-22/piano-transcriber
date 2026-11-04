"""Stage 3: global tuning estimate."""
import numpy as np
import librosa

def estimate_tuning(f0: np.ndarray, voiced: np.ndarray) -> float:
    """Return the global tuning offset in cents, in [-50, 50).
    Positive = piano is sharp of A440.

    Note: pYIN reports f0 on a 10-cent grid by default, so this estimate
    is good to about +/-5 cents. That's ample for rounding to the right key.
    """
    f = f0[voiced & ~np.isnan(f0)]
    if len(f) == 0:
        return 0.0

    # Each frame's offset from its nearest key, in cents (-50 to +50)
    midi = librosa.hz_to_midi(f)
    cents = 100 * (midi - np.round(midi))

    # Circular mean: treat 100 cents as one full turn of a circle
    angles = 2 * np.pi * cents / 100
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))

    return float(mean_angle * 100 / (2 * np.pi))
