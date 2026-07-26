"""Shared helpers for the recipe RAG tools: env loading, config loading, Gemini API calls."""
import json
import os
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


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
    load_env()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print(
            "GEMINI_API_KEY not found. Set it in a .env file "
            "(see .env.example) with your own Gemini API key.",
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
    resp = requests.post(url, json=payload, timeout=30)
    if not resp.ok:
        print(f"Gemini embedContent failed: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    return resp.json()["embedding"]["values"]


def generate_text(prompt, model):
    key = get_api_key()
    url = f"{GEMINI_API_BASE}/{model}:generateContent?key={key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = requests.post(url, json=payload, timeout=60)
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
