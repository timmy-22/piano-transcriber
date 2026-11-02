"""Stage 1: load audio as mono, resampled and normalised."""
import numpy as np

SR = 22050


def load_audio(path: str, sr: int = SR) -> tuple[np.ndarray, int]:
    """Return (signal, sample_rate). Mono, resampled to `sr`, peak-normalised to 1.

    Hint: librosa.load does mono + resampling; normalise with np.max(np.abs(y)).
    """
    raise NotImplementedError
