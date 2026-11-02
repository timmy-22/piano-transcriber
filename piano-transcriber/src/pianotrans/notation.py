"""Stage 7: notes in beats -> MusicXML via music21."""
from .types import Note


def estimate_key(notes: list[Note]) -> str:
    """Return a key name like 'G major' (Krumhansl-Schmuckler on a pitch-class histogram)."""
    raise NotImplementedError


def to_musicxml(notes: list[Note], bpm: float, key: str, out_path: str,
                time_signature: str = "4/4") -> None:
    """Build a music21 Stream (key, time sig, tempo, notes, rests) and write MusicXML."""
    raise NotImplementedError
