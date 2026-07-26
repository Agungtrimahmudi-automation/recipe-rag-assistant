# Recipe RAG — Personal Cooking Assistant

A small retrieval-augmented generation (RAG) pipeline that answers everyday
cooking questions using only a personal collection of recipes — no invented
recipes, no generic internet answers.

## Problem

"What can I cook with chicken and soy sauce?" is a question I ask myself
several times a week, and the answer I want is one of *my own* recipes, not
whatever a search engine surfaces. A plain LLM call can't see my recipe
collection, and it will happily invent a plausible-sounding recipe that isn't
actually one I use. I wanted a minimal RAG pipeline that grounds its answer
in a real, personal document set, and says "not in my collection" instead of
guessing.

## Solution

The pipeline has two stages, kept as separate, composable scripts rather than
one script that does everything:

1. **`tools/build_index.py`** — reads every recipe (`.md`) in `data/recipes/`,
   embeds each one with Gemini's `gemini-embedding-001` model, and caches the
   vectors to `data/index/embeddings.json`. Run this once, and again whenever
   a recipe is added, edited, or removed.
2. **`tools/ask.py`** — embeds the incoming question, ranks cached recipe
   vectors by cosine similarity (plain NumPy — no vector database needed at
   this scale), and passes only the top-k matching recipes to
   `gemini-flash-lite-latest` as context. The system prompt explicitly
   instructs the model to answer only from that context and admit when
   nothing matches, rather than hallucinate.

I chose full-recipe chunking (one recipe = one chunk) instead of splitting
documents into smaller passages, since each recipe is short enough to fit
whole and splitting would have separated ingredients from steps. I also
skipped a vector database — cosine similarity over a NumPy array is a single
function call at this scale (a handful to a few hundred recipes), and adding
FAISS/Chroma would have been complexity the problem doesn't need yet.

Everything tunable — which models, how many recipes to retrieve, the
grounding instructions — lives in `config/rag_config.json`, not hardcoded in
the scripts.

### Why Gemini instead of Claude

The original plan for this project used Claude Haiku for generation. I
dropped that: a Claude Pro subscription is billed completely separately from
the Anthropic API, so using Claude here would have meant setting up a new,
separate paid API account just for a demo. Gemini's free tier already
covered both embeddings and generation, so the whole pipeline runs on Gemini
end to end.

## Result

Example run, retrieving from a 12-recipe collection sourced from Kompas Food
(see `Sumber` in each recipe file):

```
$ python tools/ask.py "punya sisa nasi mau dijadiin apa ya yang gampang?" --show-sources
Sumber yang diambil:
  - Cireng Nasi (skor=0.781)
  - Nasi Goreng Sederhana (skor=0.727)
  - Nasi Telur Dadar Sambal Tomat (skor=0.702)
Berdasarkan resep yang tersedia di koleksi, sisa nasi dapat diolah menjadi
Cireng Nasi atau Nasi Goreng Sederhana.
```

And the grounding check — a recipe that isn't in the collection:

```
$ python tools/ask.py "resep rendang daging sapi bagaimana?" --show-sources
Sumber yang diambil:
  - Sayur Lodeh Tempe Goreng (skor=0.613)
  - Oseng Sosis (skor=0.610)
  - Oseng Tempe Kering (skor=0.607)
Maaf, resep rendang daging sapi tidak ada di koleksi.
```

Retrieval still returns its nearest neighbors (similarity scores around
0.6), but the model correctly refuses to answer from weak matches instead of
fabricating a rendang recipe.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in a Gemini API key from
[Google AI Studio](https://aistudio.google.com/apikey):

```
GEMINI_API_KEY=your-key-here
```

Then build the index and ask a question:

```bash
python tools/build_index.py
python tools/ask.py "resep apa yang bisa dibuat dari telur?"
```

## Project layout

```
config/rag_config.json   # models, top_k, grounding prompt — all tunable
data/recipes/*.md         # the source recipes (demo data — replace with your own)
data/index/               # cached embeddings, rebuilt by build_index.py
tools/build_index.py       # embed recipes -> cache
tools/ask.py                # retrieve + generate an answer
tools/_common.py            # shared env/config/API helpers
workflows/ask_recipe.md     # SOP for this pipeline
```

## Known limitations

- **Small dataset.** The 12 recipes here are sourced from a single Kompas
  Food roundup article (cited in each recipe's `Sumber` line) to prove the
  pipeline works end to end — add your own recipes to `data/recipes/` to make
  it genuinely useful day to day.
- **No relevance threshold.** Retrieval always returns its top-k nearest
  vectors, even when nothing is truly relevant; refusal relies entirely on
  the system prompt's instruction rather than a hard similarity cutoff.
- **No chunking for long documents.** Each recipe is embedded whole. A much
  longer document (e.g. a multi-page cookbook) would need real chunking,
  which this pipeline doesn't implement.
- **In Indonesian.** Recipes and answers are in Indonesian since that's the
  actual daily-use case; the system prompt and model would need adjusting
  for another language.

## License

MIT — see [LICENSE](LICENSE).
