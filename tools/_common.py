"""Shared helpers for the recipe RAG tools: env loading, config loading, Gemini API calls."""
import json
import os
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def _server_retry_delay(resp):
    """Read the server's own RetryInfo.retryDelay (e.g. "7s") if present."""
    try:
        details = resp.json().get("error", {}).get("details", [])
        for d in details:
            if d.get("@type", "").endswith("RetryInfo"):
                raw = d.get("retryDelay", "")
                if raw.endswith("s"):
                    return float(raw[:-1])
    except (ValueError, requests.exceptions.JSONDecodeError):
        pass
    return None


def _post_with_retry(url, payload, timeout=60, max_retries=8):
    delay = 5
    for attempt in range(max_retries):
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code == 429 and attempt < max_retries - 1:
            wait = _server_retry_delay(resp) or delay
            wait += 3  # small buffer on top of the server's own hint
            print(f"Rate limited, retrying in {wait:.0f}s (attempt {attempt + 1}/{max_retries})...", file=sys.stderr)
            time.sleep(wait)
            delay = min(delay * 2, 90)
            continue
        return resp
    return resp


def _load_env_file(env_path):
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def load_env():
    """Load .env: parent folder first (this author's own multi-project layout),
    falling back to a project-local .env (for anyone else cloning this repo)."""
    _load_env_file(PROJECT_ROOT.parent / ".env")
    _load_env_file(PROJECT_ROOT / ".env")


def load_config(config_path=None):
    path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "rag_config.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_api_key():
    """Prefer a RAG-specific key (separate Google project/quota from this
    author's other projects) and fall back to the shared GEMINI_API_KEY."""
    load_env()
    key = os.environ.get("GEMINI_API_KEY_RAG") or os.environ.get("GEMINI_API_KEY")
    if not key:
        print(
            "GEMINI_API_KEY_RAG (or GEMINI_API_KEY) not found. Set one in a "
            ".env file (see .env.example) with your own Gemini API key.",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def embed_text(text, model, output_dim, task_type):
    key = get_api_key()
    url = f"{GEMINI_API_BASE}/{model}:embedContent?key={key}"
    payload = {
        "content": {"parts": [{"text": text}]},
        "taskType": task_type,
        "outputDimensionality": output_dim,
    }
    resp = _post_with_retry(url, payload, timeout=30)
    if not resp.ok:
        print(f"Gemini embedContent failed: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    return resp.json()["embedding"]["values"]


def batch_embed_texts(texts, model, output_dim, task_type, batch_size=100):
    """Embed many texts with batchEmbedContents (one HTTP call per batch_size
    texts) instead of one call per text — far fewer round trips at scale."""
    key = get_api_key()
    url = f"{GEMINI_API_BASE}/{model}:batchEmbedContents?key={key}"
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        payload = {
            "requests": [
                {
                    "model": f"models/{model}",
                    "content": {"parts": [{"text": t}]},
                    "taskType": task_type,
                    "outputDimensionality": output_dim,
                }
                for t in chunk
            ]
        }
        resp = _post_with_retry(url, payload, timeout=120)
        if not resp.ok:
            print(f"Gemini batchEmbedContents failed: {resp.status_code} {resp.text}", file=sys.stderr)
            sys.exit(1)
        all_vectors.extend(e["values"] for e in resp.json()["embeddings"])
    return all_vectors


def generate_text(prompt, model):
    key = get_api_key()
    url = f"{GEMINI_API_BASE}/{model}:generateContent?key={key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = _post_with_retry(url, payload, timeout=60)
    if not resp.ok:
        print(f"Gemini generateContent failed: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        print(f"Gemini returned no candidates: {data}", file=sys.stderr)
        sys.exit(1)
    parts = candidates[0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts)
