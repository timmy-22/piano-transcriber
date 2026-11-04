"""Stage 1: load audio."""
import numpy as np
import soundfile as sf
import librosa

from .config import SR


def load_audio(path: str, sr: int = SR) -> tuple[np.ndarray, int]:
    """Return (y, sr): mono, resampled to `sr`, peak-normalised to 1.0.

    y[i] is the waveform value at time i / sr seconds.
    """
    # Read the file exactly as stored.
    # y has shape (n_samples,) for mono or (n_samples, n_channels) for stereo.
    y, file_sr = sf.read(path, dtype="float32")

    # 1. Stereo -> mono: average the channels (one column per channel).
    if y.ndim == 2:
        y = y.mean(axis=1)

    # 2. Resample to the project's sample rate (e.g. 44100 -> 22050 Hz).
    #    librosa low-pass filters first, so nothing above the new Nyquist aliases.
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)

    # 3. Peak-normalise so the loudest sample is exactly +/-1.0.
    #    Skip if the file is silent, to avoid dividing by zero.
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    return y, sr