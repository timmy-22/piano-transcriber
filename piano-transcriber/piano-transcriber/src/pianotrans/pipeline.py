"""Run the full pipeline: python -m pianotrans.pipeline input.wav -o out.musicxml"""
import argparse

from .types import Note
from . import load, pitch, tuning, onsets, segment, rhythm, notation


def transcribe(path: str) -> list[Note]:
    """Audio file -> notes in SECONDS (used by the evaluator)."""
    y, sr = load.load_audio(path)
    f0, voiced = pitch.yin(y, sr)
    cents = tuning.estimate_tuning(pitch.hz_to_midi(f0[voiced]))
    flux = onsets.spectral_flux(y, sr)
    onset_times = onsets.pick_onsets(flux, sr)
    return segment.segment_notes(onset_times, f0, voiced, sr, tuning_cents=cents)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("-o", "--out", default="output.musicxml")
    ap.add_argument("--time-signature", default="4/4")
    args = ap.parse_args()

    notes = transcribe(args.audio)
    bpm = rhythm.estimate_tempo(notes)
    beats = rhythm.quantise(notes, bpm)
    key = notation.estimate_key(notes)
    notation.to_musicxml(beats, bpm, key, args.out, args.time_signature)
    print(f"{len(notes)} notes, {bpm:.0f} BPM, {key} -> {args.out}")


if __name__ == "__main__":
    main()
