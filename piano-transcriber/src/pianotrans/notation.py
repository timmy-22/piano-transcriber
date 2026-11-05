"""Stage 7: quantised notes (in beats) -> sheet music (MusicXML) via music21."""
from music21 import stream, note, meter, metadata, key as m21key, tempo, pitch as m21pitch

from .note import Note


MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)


def estimate_key(notes: list[Note]) -> m21key.Key:
    """Return the key to write the score in, e.g. <music21.key.Key of G major>.

    1. Key SIGNATURE: of the 13 signatures from 6 flats to 6 sharps, pick the
       one whose major scale covers the most note-time (pitch classes weighted
       by note length). Ties go to fewer sharps/flats. This directly minimises
       accidentals, which is what matters on the page.
    2. Major or relative minor (same signature): ask Krumhansl-Schmuckler
       (music21's analyze('key')); if it names the relative minor, use that.

    On the synthetic test set, step 1 chose a signature needing no accidentals
    on 30/30 melodies; Krumhansl-Schmuckler alone managed 18/30, because
    random-walk melodies don't emphasise the tonic the way real music does.
    """
    weight = [0.0] * 12
    for n in notes:
        weight[n.pitch % 12] += max(n.offset - n.onset, 0.25)
    total = sum(weight) or 1.0

    best_major, best_score = m21key.Key("C"), None
    for sharps in range(-6, 7):                           # Gb major .. F# major
        k = m21key.KeySignature(sharps).asKey("major")
        cover = sum(weight[(k.tonic.pitchClass + i) % 12] for i in MAJOR_STEPS) / total
        score = (round(cover, 6), -abs(sharps))           # coverage first, then simplicity
        if best_score is None or score > best_score:
            best_major, best_score = k, score

    s = stream.Stream()
    for n in notes:
        m = note.Note(n.pitch)
        m.quarterLength = max(n.offset - n.onset, 0.25)
        s.append(m)
    ks = s.analyze("key")
    if ks.mode == "minor" and ks.relative.tonic.pitchClass == best_major.tonic.pitchClass:
        return m21key.Key(best_major.relative.tonic.name, "minor")
    return best_major


def spell(midi: int, k: m21key.Key) -> m21pitch.Pitch:
    """Spell a MIDI note number using the key, e.g. 66 -> F#4 in G major,
    Gb4 in Db major. MIDI can't tell these apart; the key decides.

    In-scale notes take the scale's spelling. Out-of-scale notes use sharps in
    sharp keys (and C major), flats in flat keys.
    """
    scale_names = {p.pitchClass: p.name for p in k.getPitches()}
    p = m21pitch.Pitch(midi=midi)
    name = scale_names.get(p.pitchClass)
    if name is None:
        use_flats = k.sharps < 0
        names = (["C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B"] if use_flats
                 else ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
        name = names[p.pitchClass]
    spelled = m21pitch.Pitch(name)
    # set the octave so the spelled pitch has the same MIDI number (B#3 = C4)
    spelled.octave = 4
    spelled.octave += (midi - spelled.midi) // 12
    return spelled


def build_score(notes: list[Note], bpm: float, k: m21key.Key,
                time_signature: str = "4/4", title: str = "Transcription") -> stream.Score:
    """Notes with onset/offset in BEATS -> a music21 Score.

    makeNotation() does the engraving bookkeeping: fills gaps with rests,
    splits the music into bars, ties notes that cross a bar line, and beams
    eighths and sixteenths.
    """
    part = stream.Part()
    part.insert(0, meter.TimeSignature(time_signature))
    part.insert(0, k)
    part.insert(0, tempo.MetronomeMark(number=round(bpm)))
    for n in notes:
        m = note.Note(spell(n.pitch, k))
        m.quarterLength = n.offset - n.onset
        part.insert(n.onset, m)

    score = stream.Score()
    score.metadata = metadata.Metadata(title=title)
    score.insert(0, part)
    return score.makeNotation()


def write_musicxml(score: stream.Score, out_path: str) -> str:
    """Write MusicXML. Open the file in MuseScore to see the sheet music."""
    return str(score.write("musicxml", fp=out_path))
