# Workflow: Tanya Resep (Recipe RAG)

## Objective
Answer a home-cooking question by retrieving the most relevant recipe(s) from
a personal recipe collection and generating a grounded answer — never
inventing a recipe that isn't in the collection.

## Required inputs
- A question in Indonesian, e.g. "resep apa yang bisa dibuat dari ayam dan kecap?"
- `GEMINI_API_KEY` present in a `.env` file (project-local, or a shared parent
  `.env` one directory above the project — see `tools/_common.py`)
- An up-to-date index at `data/index/embeddings.json`

## Tool sequence
1. **When recipes change** (added, edited, or removed under `data/recipes/`):
   run `tools/build_index.py` first. This re-embeds every `.md` file and
   overwrites the cached index. Skip this step if the index is already
   current — it costs one embedding API call per recipe file.
2. **To answer a question:** run `tools/ask.py "<question>"`.
   - Add `--show-sources` to see which recipes were retrieved and their
     similarity scores (useful for debugging bad answers).
   - Add `--top-k N` to change how many recipes are retrieved (default from
     `config/rag_config.json`).

## Expected output
A short Indonesian answer citing the recipe title it used as its source, or
an explicit "resepnya tidak ada di koleksi" if nothing in the collection is
actually relevant.

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
Drop a new `.md` file into `data/recipes/` (same shape as the existing ones:
`# Title` heading, `## Bahan`, `## Langkah`), then re-run
`tools/build_index.py`. No code changes needed.

## Tuning knobs (in `config/rag_config.json`)
- `top_k`: how many recipes to retrieve per question.
- `embedding_model` / `generation_model`: swap Gemini model names here.
- `system_prompt`: controls tone and the "don't invent recipes" rule.
