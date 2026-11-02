"""Generate a synthetic test set: random monophonic melodies -> MIDI -> audio.

Ground truth is exact because we wrote the MIDI ourselves.

Usage:
    python eval/make_test_data.py --soundfont path/to/piano.sf2 --n 20
    python eval/make_test_data.py --soundfont piano.sf2 --n 20 --detune 30   # 30 cents sharp

Needs FluidSynth installed (e.g. `brew install fluidsynth` / `apt install fluidsynth`)
and pyfluidsynth (`pip install pyfluidsynth`).
"""
import argparse
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

SR = 22050
OUT = Path(__file__).resolve().parent.parent / "data" / "test"


def random_melody(rng, n_notes=16, bpm=None, low=48, high=84):
    """Random walk melody in a major scale, durations on a 16th-note grid."""
    bpm = bpm or rng.uniform(70, 140)
    beat = 60.0 / bpm
    tonic = rng.integers(low, low + 12)
    scale = tonic + np.array([0, 2, 4, 5, 7, 9, 11])
    degrees = np.concatenate([scale - 12, scale, scale + 12])
    degrees = degrees[(degrees >= low) & (degrees <= high)]

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    piano = pretty_midi.Instrument(program=0)
    idx, t = len(degrees) // 2, 0.5
    for _ in range(n_notes):
        idx = int(np.clip(idx + rng.integers(-2, 3), 0, len(degrees) - 1))
        dur = rng.choice([0.5, 1.0, 1.0, 1.5, 2.0]) * beat
        piano.notes.append(pretty_midi.Note(int(rng.integers(60, 110)),
                                            int(degrees[idx]), t, t + dur * 0.9))
        t += dur
    pm.instruments.append(piano)
    return pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soundfont", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--detune", type=float, default=0.0, help="global detune in cents")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.n):
        pm = random_melody(rng)
        name = f"melody_{i:03d}"
        pm.write(str(OUT / f"{name}.mid"))
        audio = pm.fluidsynth(fs=SR, sf2_path=args.soundfont)
        if args.detune:
            import librosa
            audio = librosa.effects.pitch_shift(audio, sr=SR, n_steps=args.detune / 100)
        sf.write(OUT / f"{name}.wav", audio / (np.max(np.abs(audio)) + 1e-9), SR)
    print(f"Wrote {args.n} melodies to {OUT}")


if __name__ == "__main__":
    main()
