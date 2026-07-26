#!/usr/bin/env python3
"""HTTP API wrapping the recipe RAG pipeline, for the Telegram bot.

n8n's Telegram Trigger calls this over HTTP instead of running Python
directly, so the heavy retrieval/generation logic stays in Python while n8n
handles the Telegram integration and hosting/orchestration.

Run with:
    uvicorn tools.api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health          -> {"status": "ok", "recipes_indexed": N}
    POST /ask {"question": "..."} -> {"answer": "...", "sources": [...], "reply_text": "..."}
"""
import html
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import clean_answer_text, embed_text, generate_text, load_config  # noqa: E402
from ask import build_prompt, detect_category, load_index, retrieve  # noqa: E402
from conversation_store import get_recent_turns, save_turn  # noqa: E402

app = FastAPI(title="Recipe RAG API")
_config = load_config()
_index = load_index(_config)


class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    top_k: Optional[int] = None


class Source(BaseModel):
    title: str
    source_url: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    reply_text: str


def build_reply_text(answer, sources):
    """Telegram-ready HTML: the answer plus a numbered "Sumber" footer where
    each recipe title is the clickable link text (not a raw URL). Built here
    instead of as an n8n expression — n8n's expression sandbox choked on the
    regex literals this needed ("invalid syntax"), and this is easier to
    test and version than JS embedded in a workflow node anyway."""
    text = html.escape(answer, quote=False)
    if sources:
        lines = [
            f'{i + 1}. <a href="{s.source_url}">{html.escape(s.title, quote=False)}</a>'
            for i, s in enumerate(sources)
        ]
        text += "\n\n\U0001f4ce Sumber:\n" + "\n".join(lines)
    return text


@app.get("/health")
def health():
    return {"status": "ok", "recipes_indexed": len(_index["entries"])}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    history = get_recent_turns(req.session_id, _config.get("history_turns", 0))

    category = detect_category(question, _config.get("categories", []))
    if req.top_k:
        top_k = req.top_k
    elif category:
        top_k = _config.get("top_k_category", _config["top_k"])
    else:
        top_k = _config["top_k"]

    # Fold the previous turn into the retrieval query so a follow-up like
    # "boleh yang pedas?" still retrieves against what was actually discussed,
    # not just those three words on their own.
    retrieval_query = question
    if history:
        retrieval_query = f"{history[-1]['question']} {question}"

    question_vector = embed_text(
        retrieval_query,
        model=_config["embedding_model"],
        output_dim=_config["embedding_output_dim"],
        task_type="RETRIEVAL_QUERY",
    )
    retrieved = retrieve(question_vector, _index, top_k, category=category)
    prompt = build_prompt(_config["system_prompt"], question, retrieved, history=history)
    answer = generate_text(prompt, model=_config["generation_model"])
    answer = clean_answer_text(answer.strip())

    save_turn(req.session_id, question, answer)

    sources = [
        Source(title=entry["title"], source_url=entry.get("source_url", ""), score=round(score, 3))
        for score, entry in retrieved
    ]
    reply_text = build_reply_text(answer, sources)
    return AskResponse(answer=answer, sources=sources, reply_text=reply_text)
