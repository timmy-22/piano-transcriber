"""Sanity tests on synthetic signals. Run with: pytest
These fail until you implement the stubs; making them pass is milestone 1."""
import numpy as np
import pytest

from pianotrans.pitch import yin, hz_to_midi

SR = 22050


def tone(f, dur=0.5, harmonics=(1.0, 0.5, 0.3)):
    t = np.arange(int(SR * dur)) / SR
    return sum(a * np.sin(2 * np.pi * f * (k + 1) * t) for k, a in enumerate(harmonics))


def test_hz_to_midi():
    assert hz_to_midi(440.0) == pytest.approx(69)
    assert hz_to_midi(261.6256) == pytest.approx(60, abs=1e-3)


@pytest.mark.parametrize("f", [110.0, 261.63, 440.0, 1046.5])
def test_yin_on_harmonic_tone(f):
    f0, voiced = yin(tone(f), SR)
    est = np.nanmedian(f0[voiced])
    cents_error = 1200 * np.log2(est / f)
    assert abs(cents_error) < 10, f"{est:.2f} Hz vs {f} Hz ({cents_error:.1f} cents)"


def test_yin_missing_fundamental():
    """Harmonics 2-4 only: YIN should still find the fundamental, not an octave up."""
    f0, voiced = yin(tone(220.0, harmonics=(0.0, 1.0, 0.8, 0.6)), SR)
    assert abs(1200 * np.log2(np.nanmedian(f0[voiced]) / 220.0)) < 10
