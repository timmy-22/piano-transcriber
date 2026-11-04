"""Stage 7: quantised notes -> sheet music (MusicXML) via music21."""
from .note import Note


def estimate_key(notes: list[Note]):
    """Return a music21.key.Key, e.g. <music21.key.Key of G major>.

    Simplest: build a music21 stream of the pitches and call stream.analyze('key')
    (Krumhansl-Schmuckler under the hood). Later: write your own from a
    duration-weighted pitch-class histogram and compare.
    """
    raise NotImplementedError


def build_score(notes: list[Note], bpm: float, key, time_signature: str = "4/4"):
    """Return a music21.stream.Score.

    Steps:
      - Part with TimeSignature, the Key, and MetronomeMark(number=bpm)
      - insert each note at its beat offset: part.insert(n.onset, m21note)
        with quarterLength = n.offset - n.onset
      - spell notes using the key (key.Key has helpers; see pitch.spellingIsInferred)
      - part.makeRests(fillGaps=True), then part.makeMeasures() and
        makeTies() so notes crossing bar lines get tied
    """
    raise NotImplementedError


def write_musicxml(score, out_path: str) -> None:
    """score.write('musicxml', fp=out_path). Open the result in MuseScore."""
    raise NotImplementedError
