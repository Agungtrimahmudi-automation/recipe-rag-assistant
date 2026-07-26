#!/usr/bin/env python3
"""Sample a bulk recipe dataset (from Kaggle CSVs) into data/recipes.jsonl.

Source: Kaggle "canggih/indonesian-food-recipes" (CC0-1.0), 8 category CSVs
(ayam, ikan, kambing, sapi, tahu, telur, tempe, udang) scraped from Cookpad
Indonesia. Downloaded with the Kaggle CLI into .tmp/kaggle/ (disposable —
re-download any time with `kaggle datasets download -d
canggih/indonesian-food-recipes -p .tmp/kaggle --unzip`).

Takes the top N recipes per category by "Loves" (community likes) so the
sample favors recipes that real users validated, rather than a random draw.

Usage:
    python tools/prepare_dataset.py [--per-category 125] [--kaggle-dir .tmp/kaggle]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from _common import PROJECT_ROOT

SOURCE_CITATION = (
    "Kaggle dataset canggih/indonesian-food-recipes (CC0-1.0), "
    "scraped from Cookpad Indonesia"
)


def parse_loves(value):
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return 0


def clean_list(raw):
    return [item.strip() for item in raw.split("--") if item.strip()]


def load_category(csv_path, category):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            title = (row.get("Title") or "").strip()
            ingredients = clean_list(row.get("Ingredients") or "")
            steps = clean_list(row.get("Steps") or "")
            if not title or not ingredients or not steps:
                continue
            url = (row.get("URL") or "").strip()
            source_url = f"https://cookpad.com{url}" if url.startswith("/") else url
            rows.append(
                {
                    "title": title,
                    "category": category,
                    "ingredients": ingredients,
                    "steps": steps,
                    "loves": parse_loves(row.get("Loves")),
                    "source_url": source_url,
                    "source_citation": SOURCE_CITATION,
                }
            )
    rows.sort(key=lambda r: r["loves"], reverse=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Sample the Kaggle recipe dataset into JSONL")
    parser.add_argument("--per-category", type=int, default=125)
    parser.add_argument("--kaggle-dir", default=".tmp/kaggle")
    parser.add_argument("--out", default="data/recipes.jsonl")
    args = parser.parse_args()

    kaggle_dir = PROJECT_ROOT / args.kaggle_dir
    csv_files = sorted(kaggle_dir.glob("dataset-*.csv"))
    if not csv_files:
        print(f"No dataset-*.csv files found in {kaggle_dir}", file=sys.stderr)
        sys.exit(1)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for csv_path in csv_files:
            category = csv_path.stem.replace("dataset-", "")
            rows = load_category(csv_path, category)
            sample = rows[: args.per_category]
            for i, row in enumerate(sample, start=1):
                row["id"] = f"{category}-{i:04d}"
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += len(sample)
            print(f"{category}: {len(sample)} recipes (of {len(rows)} usable)")

    print(f"Wrote {total} recipes -> {out_path}")


if __name__ == "__main__":
    main()
