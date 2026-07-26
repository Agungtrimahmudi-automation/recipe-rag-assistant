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


def retrieve(question_vector, index, top_k):
    scored = [
        (cosine_similarity(question_vector, entry["embedding"]), entry)
        for entry in index["entries"]
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:top_k]


def build_prompt(system_prompt, question, retrieved):
    context_blocks = "\n\n".join(
        f"### {entry['title']} (sumber: {entry['source_file']})\n{entry['text']}"
        for _, entry in retrieved
    )
    return (
        f"{system_prompt}\n\n"
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
    top_k = args.top_k or config["top_k"]

    index = load_index(config)
    question_vector = embed_text(
        args.question,
        model=config["embedding_model"],
        output_dim=config["embedding_output_dim"],
        task_type="RETRIEVAL_QUERY",
    )

    retrieved = retrieve(question_vector, index, top_k)

    if args.show_sources:
        print("Sumber yang diambil:", file=sys.stderr)
        for score, entry in retrieved:
            print(f"  - {entry['title']} (skor={score:.3f})", file=sys.stderr)

    prompt = build_prompt(config["system_prompt"], args.question, retrieved)
    answer = generate_text(prompt, model=config["generation_model"])
    print(answer.strip())


if __name__ == "__main__":
    main()
