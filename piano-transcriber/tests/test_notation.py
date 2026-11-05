from music21 import key as m21key, converter

from pianotrans.note import Note
from pianotrans.notation import estimate_key, spell, build_score, write_musicxml


def test_spelling_follows_key():
    assert spell(66, m21key.Key("G")).nameWithOctave == "F#4"
    assert spell(66, m21key.Key("D-")).nameWithOctave == "G-4"
    assert spell(70, m21key.Key("F")).nameWithOctave == "B-4"


def test_spelling_keeps_midi_number():
    for k in ["C", "G", "F", "D-", "B", "E-"]:
        for m in range(48, 85):
            assert spell(m, m21key.Key(k)).midi == m


def test_key_of_g_major_melody():
    pitches = [67, 69, 71, 72, 74, 76, 78, 79, 74, 71, 67]       # G major scale + arpeggio
    notes = [Note(i, i + 1, p) for i, p in enumerate(pitches)]
    k = estimate_key(notes)
    assert k.tonic.name == "G" and k.mode == "major"


def test_bar_crossing_note_is_tied(tmp_path):
    """A half note starting on beat 4 of a 4/4 bar must be split across the
    bar line and tied, and the round trip through MusicXML must keep it."""
    notes = [Note(0, 3, 60), Note(3, 5, 62), Note(5, 8, 64)]
    score = build_score(notes, 100, m21key.Key("C"))
    path = write_musicxml(score, str(tmp_path / "t.musicxml"))
    back = converter.parse(path).flatten().notes
    assert [n.pitch.midi for n in back] == [60, 62, 62, 64]
    assert back[1].tie is not None and back[1].tie.type == "start"
    assert sum(n.quarterLength for n in back) == 8
