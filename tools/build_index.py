#!/usr/bin/env python3
"""Build the recipe embeddings index from data/recipes.jsonl.

Reads every recipe row, embeds it in batches with the Gemini embedding API,
and writes a cached index (vectors + source text) to disk. Resumable: if
interrupted (rate limit, network error), re-run the same command and it
picks up from the last checkpoint instead of re-embedding everything (and
re-spending quota) from scratch.

Usage:
    python tools/build_index.py [--config config/rag_config.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

from _common import PROJECT_ROOT, batch_embed_texts, load_config


def recipe_text(entry):
    ingredients = "\n".join(f"- {i}" for i in entry["ingredients"])
    steps = "\n".join(f"{n}. {s}" for n, s in enumerate(entry["steps"], start=1))
    return f"# {entry['title']}\n\n## Bahan\n{ingredients}\n\n## Langkah\n{steps}"


def load_recipes(recipes_path):
    recipes = []
    with open(recipes_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recipes.append(json.loads(line))
    return recipes


def load_existing_index(index_path):
    if not index_path.exists():
        return {}
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {entry["id"]: entry for entry in data["entries"]}


def save_index(index_path, config, entries_by_id, recipe_order):
    ordered = [entries_by_id[rid] for rid in recipe_order if rid in entries_by_id]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "embedding_model": config["embedding_model"],
                "embedding_output_dim": config["embedding_output_dim"],
                "entries": ordered,
            },
            f,
            ensure_ascii=False,
        )


def build_index(config, batch_size, pace_seconds):
    recipes_path = PROJECT_ROOT / config["recipes_path"]
    index_path = PROJECT_ROOT / config["index_path"]

    recipes = load_recipes(recipes_path)
    if not recipes:
        print(f"No recipes found in {recipes_path}", file=sys.stderr)
        sys.exit(1)
    recipe_order = [r["id"] for r in recipes]
    recipes_by_id = {r["id"]: r for r in recipes}

    existing = load_existing_index(index_path)

    # Backfill fields added after entries were already cached (e.g. `category`)
    # without re-embedding — join on id against the current recipes file.
    backfilled = False
    for rid, entry in existing.items():
        recipe = recipes_by_id.get(rid)
        if recipe and entry.get("category") != recipe.get("category", ""):
            entry["category"] = recipe.get("category", "")
            backfilled = True

    todo = [r for r in recipes if r["id"] not in existing]

    if not todo:
        if backfilled:
            save_index(index_path, config, existing, recipe_order)
            print(f"Backfilled category on {len(existing)} cached entries (no re-embedding).")
        else:
            print(f"Index already up to date ({len(existing)} recipes). Nothing to do.")
        return

    print(f"{len(existing)} already indexed, embedding {len(todo)} more...")

    checkpoint_every = max(batch_size, 100)
    num_chunks = (len(todo) + checkpoint_every - 1) // checkpoint_every
    for chunk_idx, start in enumerate(range(0, len(todo), checkpoint_every)):
        # Pace before every chunk, including the first: resuming soon after a
        # prior run's failed/rate-limited attempt can still be inside the
        # same quota window, so there's no safe chunk to skip the wait for.
        print(f"  pacing {pace_seconds}s to stay under the free-tier per-minute quota...")
        time.sleep(pace_seconds)

        chunk = todo[start : start + checkpoint_every]
        texts = [recipe_text(r) for r in chunk]
        vectors = batch_embed_texts(
            texts,
            model=config["embedding_model"],
            output_dim=config["embedding_output_dim"],
            task_type="RETRIEVAL_DOCUMENT",
            batch_size=batch_size,
        )
        for r, text, vector in zip(chunk, texts, vectors):
            existing[r["id"]] = {
                "id": r["id"],
                "title": r["title"],
                "text": text,
                "category": r.get("category", ""),
                "source_url": r.get("source_url", ""),
                "source_citation": r.get("source_citation", ""),
                "embedding": vector,
            }
        save_index(index_path, config, existing, recipe_order)
        print(f"  checkpointed {min(start + checkpoint_every, len(todo))}/{len(todo)} (chunk {chunk_idx + 1}/{num_chunks})")

    print(f"Indexed {len(existing)} recipe(s) total -> {index_path}")


def main():
    parser = argparse.ArgumentParser(description="Build the recipe embeddings index")
    parser.add_argument("--config", default=None, help="Path to rag_config.json")
    parser.add_argument("--batch-size", type=int, default=90, help="Recipes per API call")
    parser.add_argument(
        "--pace-seconds",
        type=int,
        default=65,
        help="Seconds to wait between batches (free tier: 100 embed requests/minute)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    build_index(config, args.batch_size, args.pace_seconds)


if __name__ == "__main__":
    main()
