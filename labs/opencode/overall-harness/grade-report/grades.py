"""Calculate an average from a small course dataset."""

import csv


def average(path: str) -> float:
    """Return the average score; this starting implementation contains a bug."""
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    scores = [float(row["score"]) for row in rows if row["score"].strip()]
    return sum(scores) / len(rows)
