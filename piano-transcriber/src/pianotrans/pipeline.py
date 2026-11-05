"""Full pipeline.

    python -m pianotrans.pipeline input.wav -o out.musicxml
"""
import argparse
from pathlib import Path

import numpy as np
import librosa

from .config import HOP
from .note import Note
from . import load, pitch, tuning, onsets, segment, rhythm, notation


def loudness_db(y: np.ndarray, hop: int = HOP) -> np.ndarray:
    """Frame loudness in dB (RMS), on the same frame grid as everything else.
    0 dB = the loudest frame in the file."""
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    return librosa.amplitude_to_db(rms, ref=np.max)


def transcribe(path: str) -> list[Note]:
    """Audio file -> notes in SECONDS. Used by eval/evaluate.py, so this is
    what gets scored. Stages 1-5 only."""
    y, sr = load.load_audio(path)
    f0, voiced, _ = pitch.track_f0(y, sr)
    cents = tuning.estimate_tuning(f0, voiced)
    A = pitch.f0_to_activation(f0, voiced, cents)
    onset_frames = onsets.detect_onsets(onsets.onset_envelope(y, sr), sr)
    rms_db = loudness_db(y)
    return segment.activation_to_notes(A, onset_frames, sr, rms_db=rms_db)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("-o", "--out", default="output.musicxml")
    ap.add_argument("--time-signature", default="4/4")
    args = ap.parse_args()

    notes = transcribe(args.audio)
    bpm = rhythm.estimate_tempo([n.onset for n in notes])
    beats = rhythm.quantise(notes, bpm)
    key = notation.estimate_key(beats)
    score = notation.build_score(beats, bpm, key, args.time_signature,
                                 title=Path(args.audio).stem)
    notation.write_musicxml(score, args.out)
    print(f"{len(notes)} notes | {bpm:.0f} BPM | {key} -> {args.out}")


if __name__ == "__main__":
    main()
