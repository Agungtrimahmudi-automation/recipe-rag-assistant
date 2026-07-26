#!/usr/bin/env python3
"""Ask a question against the recipe RAG index.

Embeds the question, retrieves the top-k most similar recipes from the
cached index (cosine similarity, no vector DB needed at this scale), then
asks Gemini to answer grounded only in those recipes.

Usage:
    python tools/ask.py "resep apa yang bisa dibuat dari ayam dan kecap?"
"""
import argparse
import json
import sys

import numpy as np

from _common import PROJECT_ROOT, embed_text, generate_text, load_config

# Recipe titles/answers can contain emoji; Windows consoles default to cp1252,
# which can't encode them and crashes print() mid-answer.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_index(config):
    index_path = PROJECT_ROOT / config["index_path"]
    if not index_path.exists():
        print(
            f"Index not found at {index_path}. Run tools/build_index.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_category(question, categories):
    """Cheap keyword match against the recipe collection's own category field
    (e.g. "tempe", "ayam") — lets a broad ingredient question pull from the
    full ~125-recipe category instead of whatever 3 rank highest by pure
    cosine similarity across all 1000. Ambiguous (0 or 2+ hits) means no
    filter, so retrieval falls back to plain semantic search."""
    q = question.lower()
    hits = {c for c in categories if c in q}
    return next(iter(hits)) if len(hits) == 1 else None


def retrieve(question_vector, index, top_k, category=None):
    entries = index["entries"]
    if category:
        narrowed = [e for e in entries if e.get("category") == category]
        if narrowed:
            entries = narrowed
    scored = [
        (cosine_similarity(question_vector, entry["embedding"]), entry)
        for entry in entries
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:top_k]


def build_prompt(system_prompt, question, retrieved, history=None):
    context_blocks = "\n\n".join(
        f"### {entry['title']} (sumber: {entry.get('source_url', 'tidak diketahui')})\n{entry['text']}"
        for _, entry in retrieved
    )
    history_block = ""
    if history:
        turns = "\n".join(
            f"User: {turn['question']}\nAsisten: {turn['answer']}" for turn in history
        )
        history_block = f"Riwayat percakapan sebelumnya:\n{turns}\n\n"
    return (
        f"{system_prompt}\n\n"
        f"{history_block}"
        f"Konteks resep yang tersedia:\n{context_blocks}\n\n"
        f"Pertanyaan: {question}\n\nJawaban:"
    )


def main():
    parser = argparse.ArgumentParser(description="Ask the recipe RAG assistant a question")
    parser.add_argument("question", help="Question in Indonesian")
    parser.add_argument("--config", default=None, help="Path to rag_config.json")
    parser.add_argument("--top-k", type=int, default=None, help="Override top_k from config")
    parser.add_argument(
        "--show-sources", action="store_true", help="Print retrieved recipe titles and scores"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    category = detect_category(args.question, config.get("categories", []))
    if args.top_k:
        top_k = args.top_k
    elif category:
        top_k = config.get("top_k_category", config["top_k"])
    else:
        top_k = config["top_k"]

    index = load_index(config)
    question_vector = embed_text(
        args.question,
        model=config["embedding_model"],
        output_dim=config["embedding_output_dim"],
        task_type="RETRIEVAL_QUERY",
    )

    retrieved = retrieve(question_vector, index, top_k, category=category)

    if args.show_sources:
        print("Sumber yang diambil:", file=sys.stderr)
        for score, entry in retrieved:
            print(f"  - {entry['title']} (skor={score:.3f})", file=sys.stderr)

    prompt = build_prompt(config["system_prompt"], args.question, retrieved)
    answer = generate_text(prompt, model=config["generation_model"])
    print(answer.strip())


if __name__ == "__main__":
    main()
