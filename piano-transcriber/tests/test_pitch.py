"""Run with: pytest   (all fail with NotImplementedError until you fill the stubs)"""
import numpy as np
import pytest

from pianotrans.config import SR, N_KEYS, midi_to_key
from pianotrans.pitch import track_f0, f0_to_activation


def tone(f, dur=0.6, harmonics=(1.0, 0.5, 0.3)):
    t = np.arange(int(SR * dur)) / SR
    return sum(a * np.sin(2 * np.pi * f * (k + 1) * t) for k, a in enumerate(harmonics))


@pytest.mark.parametrize("f", [110.0, 261.63, 440.0, 1046.5])
def test_track_f0_harmonic_tone(f):
    f0, voiced, _ = track_f0(tone(f))
    est = np.nanmedian(f0[voiced])
    assert abs(1200 * np.log2(est / f)) < 15


def test_activation_shape_and_one_hot():
    f0 = np.array([440.0, np.nan, 261.63])
    voiced = np.array([True, False, True])
    A = f0_to_activation(f0, voiced)
    assert A.shape == (3, N_KEYS)
    assert A[0, midi_to_key(69)] and A[2, midi_to_key(60)]
    assert A.sum(axis=1).tolist() == [1, 0, 1]


def test_activation_uses_tuning():
    """A4 played 45 cents sharp must still map to A4 once tuning is applied,
    not get rounded up to A#4."""
    f_sharp = 440.0 * 2 ** (45 / 1200)
    A = f0_to_activation(np.array([f_sharp]), np.array([True]), tuning_cents=45)
    assert A[0, midi_to_key(69)]
