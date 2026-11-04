"""Full pipeline.

    python -m pianotrans.pipeline input.wav -o out.musicxml
"""
import argparse

from .note import Note
from . import load, pitch, tuning, onsets, segment, rhythm, notation


def transcribe(path: str) -> list[Note]:
    """Audio file -> notes in SECONDS. Used by eval/evaluate.py, so this is
    what gets scored. Stages 1-5 only."""
    y, sr = load.load_audio(path)
    f0, voiced, _ = pitch.track_f0(y, sr)
    cents = tuning.estimate_tuning(f0, voiced)
    A = pitch.f0_to_activation(f0, voiced, cents)
    onset_frames = onsets.detect_onsets(onsets.onset_envelope(y, sr), sr)
    return segment.activation_to_notes(A, onset_frames, sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("-o", "--out", default="output.musicxml")
    ap.add_argument("--time-signature", default="4/4")
    args = ap.parse_args()

    notes = transcribe(args.audio)
    y, sr = load.load_audio(args.audio)
    bpm, t0 = rhythm.estimate_tempo(y, sr)
    beats = rhythm.quantise(notes, bpm, t0)
    key = notation.estimate_key(beats)
    score = notation.build_score(beats, bpm, key, args.time_signature)
    notation.write_musicxml(score, args.out)
    print(f"{len(notes)} notes | {bpm:.0f} BPM | {key} -> {args.out}")


if __name__ == "__main__":
    main()
