"""Evaluate the pipeline on the test set and LOG the result automatically.

Every run appends one row to results/results.csv (date, commit, your note,
scores) and redraws results/progress.png. Commit both: that's your progress record.

Usage:
    python eval/evaluate.py --note "added parabolic interpolation to YIN"
    python eval/evaluate.py --no-log      # quick check without logging
"""
import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data" / "test"
RESULTS = ROOT / "results"
CSV = RESULTS / "results.csv"
PLOT = RESULTS / "progress.png"
FIELDS = ["timestamp", "commit", "note", "n_files",
          "precision", "recall", "f1", "f1_no_offset"]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return out + ("+dirty" if dirty else "")
    except Exception:
        return "no-git"


def notes_to_arrays(notes):
    intervals = np.array([[n.onset, n.offset] for n in notes]).reshape(-1, 2)
    pitches_hz = np.array([440.0 * 2 ** ((n.pitch - 69) / 12) for n in notes])
    return intervals, pitches_hz


def load_reference(mid_path):
    import pretty_midi
    from pianotrans.types import Note
    pm = pretty_midi.PrettyMIDI(str(mid_path))
    return [Note(n.start, n.end, n.pitch, n.velocity)
            for inst in pm.instruments for n in inst.notes]


def evaluate():
    import mir_eval
    from pianotrans.pipeline import transcribe

    wavs = sorted(DATA.glob("*.wav"))
    if not wavs:
        sys.exit(f"No test audio in {DATA}. Run eval/make_test_data.py first.")

    scores = {"precision": [], "recall": [], "f1": [], "f1_no_offset": []}
    for wav in wavs:
        ref_i, ref_p = notes_to_arrays(load_reference(wav.with_suffix(".mid")))
        est = transcribe(str(wav))
        est_i, est_p = notes_to_arrays(est)
        p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_i, ref_p, est_i, est_p)
        _, _, f_on, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_i, ref_p, est_i, est_p, offset_ratio=None)
        for k, v in zip(scores, (p, r, f, f_on)):
            scores[k].append(v)
        print(f"  {wav.name}: F1={f:.3f}  (onset-only F1={f_on:.3f})")
    return len(wavs), {k: float(np.mean(v)) for k, v in scores.items()}


def append_csv(row):
    RESULTS.mkdir(exist_ok=True)
    new = not CSV.exists()
    with CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def plot_progress():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    x = range(1, len(rows) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, [float(r["f1_no_offset"]) for r in rows], "o-", label="F1 (onset + pitch)")
    ax.plot(x, [float(r["f1"]) for r in rows], "s--", label="F1 (onset + pitch + offset)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["commit"] for r in rows], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Note-level F1")
    ax.set_xlabel("Evaluation run (commit)")
    ax.set_title("Transcription accuracy over development")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOT, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", default="", help="what changed since the last run")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    n, s = evaluate()
    print(f"\nMean over {n} files: P={s['precision']:.3f} R={s['recall']:.3f} "
          f"F1={s['f1']:.3f} onset-only F1={s['f1_no_offset']:.3f}")
    if args.no_log:
        return
    append_csv({"timestamp": datetime.now().isoformat(timespec="minutes"),
                "commit": git_commit(), "note": args.note, "n_files": n,
                **{k: f"{v:.4f}" for k, v in s.items()}})
    plot_progress()
    print(f"Logged to {CSV.relative_to(ROOT)} and redrew {PLOT.relative_to(ROOT)}. Commit both.")


if __name__ == "__main__":
    main()
