"""Plot helpers for notebooks and README figures. Save figures to figures/."""


def plot_overview(y, sr, A, onset_frames, ref_notes=None, hop=256, ax=None):
    """One figure, shared time axis:
      - spectrogram (librosa.display.specshow, CQT or log-STFT)
      - activation matrix overlaid as a piano roll
      - onsets as vertical lines
      - optional ground-truth notes as outlined boxes
    This is your main debugging view: you can SEE where errors come from.
    """
    raise NotImplementedError
