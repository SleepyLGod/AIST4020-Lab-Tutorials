"""Prepare a reproducible local movie-review teaching bundle; no model calls."""

import argparse
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import zipfile

QUOTAS = {
    "taken_3": {"POSITIVE": 4, "NEGATIVE": 12},
    "ant_man_and_the_wasp_quantumania": {"POSITIVE": 8, "NEGATIVE": 8},
}
SPLITS = ("demo", "optimization", "test")
FIELDS = ("id", "reviewId", "reviewText")


def encode_json(value: object) -> bytes:
    """Encode JSON consistently for provenance and repeatable bundles."""
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def sample_key(row: dict[str, str]) -> str:
    """Order records independently of CSV row order or model outputs."""
    return hashlib.sha256(f"4020|{row['id']}|{row['reviewId']}".encode()).hexdigest()


def build_payloads(reviews_path: Path, movies_path: Path) -> dict[str, bytes]:
    """Validate source rows and return separate inputs, labels, and manifest."""
    with reviews_path.open(encoding="utf-8-sig", newline="") as handle:
        raw_reviews = list(csv.DictReader(handle))
    with movies_path.open(encoding="utf-8-sig", newline="") as handle:
        raw_movies = list(csv.DictReader(handle))
    unique: dict[tuple[str, str], dict[str, str]] = {}
    original_counts = {movie: 0 for movie in QUOTAS}
    for row in raw_reviews:
        if row["id"] not in QUOTAS:
            continue
        movie = row["id"]
        original_counts[movie] += 1
        if not row["reviewId"].strip() or not row["reviewText"].strip():
            raise ValueError("Empty review ID or text in selected movies")
        if row["scoreSentiment"] not in QUOTAS[movie]:
            raise ValueError(f"Unexpected reference label: {row['scoreSentiment']}")
        key = (movie, row["reviewId"])
        if key in unique:
            if any(unique[key][f] != row[f] for f in ("reviewText", "scoreSentiment")):
                raise ValueError(f"Conflicting duplicate review: {key}")
        else:
            unique[key] = row
    titles = {}
    for movie in QUOTAS:
        values = {row["title"] for row in raw_movies if row["id"] == movie}
        if len(values) != 1:
            raise ValueError(f"Missing or conflicting movie title: {movie}")
        titles[movie] = values.pop()

    splits: dict[str, list[dict[str, str]]] = {name: [] for name in SPLITS}
    for movie, quotas in QUOTAS.items():
        for label, count in quotas.items():
            candidates = sorted(
                (r for r in unique.values() if r["id"] == movie and r["scoreSentiment"] == label),
                key=sample_key,
            )
            if len(candidates) < len(SPLITS) * count:
                raise ValueError(f"Not enough unique reviews for {movie}, {label}")
            for index, name in enumerate(SPLITS):
                splits[name].extend(candidates[index * count:(index + 1) * count])
    payloads, selection = {}, {}
    seen_ids: set[tuple[str, str]] = set()
    seen_text: dict[str, str] = {}
    for name, selected in splits.items():
        selected.sort(key=sample_key)
        inputs, labels = [], []
        for row in selected:
            key = (row["id"], row["reviewId"])
            text = html.unescape(row["reviewText"]).strip()
            if not text:
                raise ValueError(f"Empty decoded text: {key}")
            if key in seen_ids:
                raise ValueError(f"Review appears in multiple splits: {key}")
            if text in seen_text and seen_text[text] != name:
                raise ValueError("The same text appears in different splits")
            seen_ids.add(key)
            seen_text[text] = name
            inputs.append({"id": row["id"], "reviewId": row["reviewId"], "reviewText": text})
            labels.append({"id": row["id"], "reviewId": row["reviewId"],
                           "scoreSentiment": row["scoreSentiment"]})
        payloads[f"{name}.json"] = encode_json(inputs)
        payloads[f"{name}_labels.json"] = encode_json(labels)
        selection[name] = [[r["id"], r["reviewId"]] for r in selected]
    manifest = {
        "format_version": 1,
        "distribution": "Teaching subset distributed under the upstream dataset's stated CC0-1.0 terms; see DATA_SOURCES.md.",
        "license": {
            "upstream_declared_license": "CC0-1.0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "verified_on": "2026-10-05",
            "verification": "Kaggle public dataset API lists CC0: Public Domain for andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews.",
            "scope": "Upstream publisher's declaration; third-party review rights have not been independently audited.",
        },
        "origin": {
            "dataset": "SemBench movie scenario, sf_2000, adapted for teaching",
            "upstream": "https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews",
            "scenario": "files/movie/data/sf_2000",
            "source_sha256": {"Reviews.csv": hashlib.sha256(reviews_path.read_bytes()).hexdigest(),
                              "Movies.csv": hashlib.sha256(movies_path.read_bytes()).hexdigest()},
        },
        "preparation": {
            "order": "SHA-256(4020|movie_id|review_id), ascending within each movie/label",
            "split_order": list(SPLITS), "quotas_per_split": QUOTAS,
            "text_transform": "html.unescape followed by strip; no model rewriting",
            "input_fields": list(FIELDS),
            "deduplication": "(id, reviewId); conflicting raw text or label is an error",
        },
        "movies": titles,
        "source_rows": original_counts,
        "unique_source_rows": {m: sum(k[0] == m for k in unique) for m in QUOTAS},
        "selection": selection,
        "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
    }
    payloads["manifest.json"] = encode_json(manifest)
    return payloads


def write_bundle(payloads: dict[str, bytes], output_dir: Path) -> Path:
    """Write data and a portable ZIP, refusing to overwrite different files."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, contents in sorted(payloads.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, contents)
    archive_path = output_dir / "movie_lab_data.zip"
    targets = {output_dir / "data" / name: data for name, data in payloads.items()}
    targets[archive_path] = buffer.getvalue()
    for path, data in targets.items():
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(f"Different existing content, not overwritten: {path}")
    for path, data in targets.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    return archive_path


def main() -> None:
    """Build the local bundle from explicitly supplied SemBench CSV files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--movies", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    archive = write_bundle(build_payloads(args.reviews, args.movies), args.output_dir)
    print(f"Prepared three disjoint 32-review splits. Local bundle: {archive}")
    print("No model calls. Upstream declares CC0-1.0; retain DATA_SOURCES.md with distributed teaching data.")


if __name__ == "__main__":
    main()
