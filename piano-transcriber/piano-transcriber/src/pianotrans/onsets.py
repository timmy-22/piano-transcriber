"""Stage 4: note onset detection."""
import numpy as np


def spectral_flux(y: np.ndarray, sr: int, n_fft: int = 2048,
                  hop_length: int = 256) -> np.ndarray:
    """Onset strength per frame: sum of POSITIVE magnitude increases between frames."""
    raise NotImplementedError


def pick_onsets(flux: np.ndarray, sr: int, hop_length: int = 256) -> np.ndarray:
    """Return onset times in seconds using an adaptive (e.g. moving median) threshold."""
    raise NotImplementedError
