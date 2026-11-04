# Dev Log

Three lines per session, max. Add entries with `python log.py` or by hand.

---

## 2026-11-02
- Did: set up repo structure, evaluation harness and module stubs
- Learned: planned v1 scope (monophonic, steady tempo, MusicXML output)
- Next: fundamentals (sampling, DFT, windowing), write YIN from scratch on synthetic sines

## 2026-11-02
- Did: set up venv, FluidSynth and soundfont; generated synthetic test set
- Learned: FluidSynth runs as a library from Python, the exe is only for live MIDI
- Next: fundamentals: sampling, DFT, windowing; start YIN

## 2026-11-04
- Did: cd D:\FMP
- Learned: python -m venv .venv
- Next: .venv\Scripts\Activate.ps1

## 2026-11-04
- Did: onsets: librosa spectral flux + peak picking; swept delta on test set
- Learned: F=1.0 for delta 0.05-0.1, drops to 0.63 at 0.5 (quiet notes missed); chose 0.1 for margin
- Next: segmentation

## 2026-11-04
- Did: first end-to-end evaluation
- Learned: onset F1 0.976, full F1 0.914; offsets are the main weakness
- Next: error analysis on melody_002 and 004

## 2026-11-04
- Did: fixed slope_db default override; release detection now active in evaluate.py
- Learned: held-out F1 0.901 -> 0.961 with release detection; remaining errors are mostly extra notes (P 0.942 < R 0.981)
- Next: attack pitch glitches / extra notes
