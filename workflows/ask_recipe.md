# Workflow: Tanya Resep (Recipe RAG)

## Objective
Answer a home-cooking question by retrieving the most relevant recipe(s) from
a personal recipe collection and generating a grounded answer — never
inventing a recipe that isn't in the collection.

## Required inputs
- A question in Indonesian, e.g. "resep apa yang bisa dibuat dari ayam dan kecap?"
- `GEMINI_API_KEY_RAG` (or `GEMINI_API_KEY`) present in a `.env` file
  (project-local, or a shared parent `.env` one directory above the project —
  see `tools/_common.py`)
- An up-to-date index at `data/index/embeddings.json`

## Tool sequence
1. **When `data/recipes.jsonl` changes** (recipes added, edited, or removed):
   run `tools/build_index.py`. It only embeds rows whose `id` isn't already
   cached — existing entries are skipped (no repeat API cost), and if a
   cached recipe's `category` field changed, that gets backfilled onto the
   existing entry for free (no re-embedding). Full re-embed only happens for
   genuinely new ids.
2. **To answer a question:** run `tools/ask.py "<question>"`.
   - Add `--show-sources` to see which recipes were retrieved and their
     similarity scores (useful for debugging bad answers).
   - Add `--top-k N` to override how many recipes are retrieved.
   - The Telegram bot talks to `tools/api.py` (`POST /ask`) instead of the
     CLI directly; the CLI is for manual testing/debugging.

## Category-aware retrieval
The dataset (`data/recipes.jsonl`, ~1000 Cookpad recipes) has a `category`
field (`ayam`, `ikan`, `kambing`, `sapi`, `tahu`, `telur`, `tempe`, `udang`),
~125 recipes each. `detect_category()` in `tools/ask.py` does a plain keyword
match of the question against `config.categories`; if exactly one category
matches, retrieval narrows to that ~125-recipe subset before ranking by
cosine similarity, and `top_k_category` (broader than the default `top_k`) is
used instead. This exists because plain semantic search across all 1000
recipes for a broad question like "punya tempe di rumah" would silently
surface only 3 of ~150 matching recipes with no signal that more exist.
Ambiguous questions (0 or 2+ category keywords) fall back to plain semantic
search over the whole index.

## Conversation history (Telegram bot only)
`tools/api.py` accepts an optional `session_id` (the Telegram chat id) per
`POST /ask` request. When present, the last `history_turns` question/answer
pairs are read from `data/state/conversations.db` (SQLite, keyed by
session_id), folded into the prompt so the model can resolve follow-ups like
"yang kedua aja deh", and the previous question is appended to the retrieval
query text so follow-ups still retrieve against what was actually discussed.
Every answered turn is saved back to the same store. The CLI (`ask.py`) has
no session concept — each invocation is a fresh, single-turn call.

`data/state/` is bind-mounted as a Docker volume (see `docker-compose.yml`)
so history survives a container redeploy; it's gitignored since it holds
real user conversation content, not source data.

## Expected output
A short Indonesian answer citing the recipe title(s) it used as its source.
If more than one recipe in the retrieved context is genuinely relevant, the
system prompt requires the model to list the candidate titles and ask the
user to pick, rather than blending steps from different recipes or hiding
the fact that other matches exist. If nothing in the collection is relevant,
it says so explicitly ("resepnya tidak ada di koleksi").

## Edge cases
- **Index missing:** `ask.py` exits with a clear error telling you to run
  `build_index.py` first, instead of failing silently.
- **No relevant recipe:** the model is instructed (via `system_prompt` in the
  config) to say so plainly rather than hallucinate a recipe. Retrieval always
  returns the top-k nearest vectors even when none are truly relevant — the
  similarity score (visible with `--show-sources`) is the signal that the
  match is weak, not an error.
- **Recipes added but index not rebuilt:** the new recipe simply won't be
  retrievable yet. There is no auto-rebuild — this is a manual step by design,
  since re-indexing spends API quota.

## How to add your own recipes
Append a row to `data/recipes.jsonl` (same shape as existing rows: `id`,
`title`, `category`, `ingredients`, `steps`, `source_url`,
`source_citation`), then re-run `tools/build_index.py`. No code changes
needed.

## Tuning knobs (in `config/rag_config.json`)
- `top_k` / `top_k_category`: how many recipes to retrieve when no
  category / exactly one category is detected in the question.
- `categories`: the keyword list `detect_category()` matches against —
  must match the `category` values actually present in `data/recipes.jsonl`.
- `history_turns`: how many past Q/A pairs from `data/state/conversations.db`
  get folded into the prompt for a given `session_id`.
- `embedding_model` / `generation_model`: swap Gemini model names here.
- `system_prompt`: controls tone, the "don't invent recipes" rule, and the
  "list titles before giving full instructions" behavior.

## Known limitation: not a real vector database
Retrieval is brute-force cosine similarity over a flat JSON file
(`data/index/embeddings.json`), not a vector database (pgvector/Chroma/
FAISS/etc.) — no ANN index, no native metadata filtering. At 1000 recipes
this is still fast (well under 50ms per query in numpy) and the category
keyword filter above covers the metadata-filtering need cheaply. Recipes are
also intentionally *not* chunked: each is embedded whole (title + ingredients
+ steps) because a recipe is an atomic answer unit — splitting ingredients
from steps would produce fragments that can't stand on their own. A real
vector DB would only start to matter if the collection grows by an order of
magnitude or more.
