#!/usr/bin/env python3
"""Build the recipe embeddings index.

Reads every .md file in the recipes folder, embeds it with the Gemini
embedding API, and writes a cached index (vectors + source text) to disk.
Re-run this after adding, editing, or removing recipe files.

Usage:
    python tools/build_index.py [--config config/rag_config.json]
"""
import argparse
import json
import sys
from pathlib import Path

from _common import PROJECT_ROOT, embed_text, load_config


def build_index(config):
    recipes_dir = PROJECT_ROOT / config["recipes_dir"]
    index_path = PROJECT_ROOT / config["index_path"]

    recipe_files = sorted(recipes_dir.glob("*.md"))
    if not recipe_files:
        print(f"No .md recipe files found in {recipes_dir}", file=sys.stderr)
        sys.exit(1)

    entries = []
    for path in recipe_files:
        text = path.read_text(encoding="utf-8").strip()
        title_line = text.splitlines()[0].lstrip("#").strip() if text else path.stem
        print(f"Embedding: {path.name}")
        vector = embed_text(
            text,
            model=config["embedding_model"],
            output_dim=config["embedding_output_dim"],
            task_type="RETRIEVAL_DOCUMENT",
        )
        entries.append(
            {
                "source_file": path.name,
                "title": title_line,
                "text": text,
                "embedding": vector,
            }
        )

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "embedding_model": config["embedding_model"],
                "embedding_output_dim": config["embedding_output_dim"],
                "entries": entries,
            },
            f,
            ensure_ascii=False,
        )

    print(f"Indexed {len(entries)} recipe(s) -> {index_path}")


def main():
    parser = argparse.ArgumentParser(description="Build the recipe embeddings index")
    parser.add_argument("--config", default=None, help="Path to rag_config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    build_index(config)


if __name__ == "__main__":
    main()
