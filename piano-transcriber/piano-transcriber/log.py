"""Append a dev log entry in one command.

Usage:
    python log.py "what I did" "what I learned" "what's next"
    python log.py            # prompts for each line
"""
import sys
from datetime import date
from pathlib import Path

LOG = Path(__file__).parent / "DEVLOG.md"


def main():
    if len(sys.argv) == 4:
        did, learned, nxt = sys.argv[1:4]
    else:
        did = input("Did: ")
        learned = input("Learned: ")
        nxt = input("Next: ")

    entry = f"\n## {date.today().isoformat()}\n- Did: {did}\n- Learned: {learned}\n- Next: {nxt}\n"
    with LOG.open("a", encoding="utf-8") as f:
        f.write(entry)
    print(f"Logged to {LOG.name}. Now commit and push.")


if __name__ == "__main__":
    main()
