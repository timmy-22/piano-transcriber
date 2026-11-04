import numpy as np
import pytest

from pianotrans.tuning import estimate_tuning


def detuned(midi, cents):
    return 440.0 * 2 ** ((np.asarray(midi) - 69 + cents / 100) / 12)


def test_in_tune():
    f = detuned([60, 62, 64, 67], 0)
    assert estimate_tuning(f, np.ones(4, bool)) == pytest.approx(0, abs=1)


def test_sharp():
    f = detuned([60, 62, 64, 67], 30)
    assert estimate_tuning(f, np.ones(4, bool)) == pytest.approx(30, abs=1)


def test_wraparound():
    """Half the frames read +48, half -48 (they're really all ~50 cents off).
    A naive mean gives 0, which is wrong. Circular mean gives about +/-50."""
    m = np.array([60.48, 62.48, 63.52, 65.52])     # +48, +48, -48, -48 cents
    f = 440.0 * 2 ** ((m - 69) / 12)
    assert abs(estimate_tuning(f, np.ones(4, bool))) > 45
