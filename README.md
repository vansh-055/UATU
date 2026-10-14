# Uatu RAG

A RAG system over the Marvel Database wiki (222k+ articles, every reality) that answers
questions in the voice of Uatu, the Watcher -- with hybrid retrieval (vector + BM25 +
a relationship graph) and a web chat UI.

## Pipeline

Raw MediaWiki XML dump isn't included in this repo (multi-GB) -- point `scripts/extract.py`
at your own copy, or adjust `DUMP` in that file.

```
enmarveldatabase_pages_current.xml   (MediaWiki XML export, not in repo)
          |
          v  extract.py       streams XML, parses wikitext via mwparserfromhell
data/articles.jsonl            222,817 articles: {id, title, url, universe, type, categories, infobox, sections}
          |
          v  chunk.py          packs paragraphs into ~1000-char pieces, prefixes "Title -- Heading"
data/chunks.jsonl              501,931 chunks, all realities, junk page types filtered out
          |
          +--v embed.py        -> data/chroma/       dense vectors (all-MiniLM-L6-v2, cosine)
          +--v build_fts.py    -> data/fts.sqlite3   lexical index (SQLite FTS5, real BM25)
          +--v build_graph.py  -> data/graph.sqlite3 relationship graph from infobox fields (no LLM calls)
```

Run in order:

```bash
.venv/bin/python scripts/extract.py
.venv/bin/python scripts/chunk.py
.venv/bin/python scripts/embed.py      # slow -- supervised batches recommended, see --skip/--limit
.venv/bin/python scripts/build_fts.py
.venv/bin/python scripts/build_graph.py
```

Then either:

```bash
.venv/bin/python scripts/query.py "How did Asgard fall?"   # CLI
.venv/bin/python web/app.py                                 # web UI -> http://localhost:5000
```

## Retrieval design (`scripts/rag.py`)

`RagEngine.retrieve()` combines three channels, fused with Reciprocal Rank Fusion
(rank-position-based, since cosine similarity / BM25 scores / graph hops aren't on
comparable scales):

1. **Vector search** -- semantic, finds chunks that mean the same thing even with
   different wording.
2. **BM25 search** -- lexical, nails exact rare terms (e.g. character names) that
   embeddings can blur together.
3. **Graph expansion** -- the top fused results from (1)+(2) seed a lookup into the
   relationship graph (`Relatives`, `Affiliation`, `KilledBy`, etc., parsed straight
   from infobox fields extract.py already structured -- no LLM extraction needed).
   Their 1-hop neighbors are added as extra candidates. The `created_by` relation is
   excluded from expansion (Stan Lee/Jack Kirby have enormous fan-out and would pollute
   results with unrelated characters).

A **universe bias** (small RRF bonus for Earth-616 / untagged pages) is applied twice --
once before picking graph seeds, once on the final pool -- as a tiebreaker so an
ambiguous query like "Who is the Abomination?" doesn't scatter evenly across 100+
same-named alternate-reality variants. It's a nudge, not an override: a query that
names a specific other reality explicitly can still win on actual relevance.

## Generation

- Persona system prompt (Uatu's voice; instructed to narrate events as stories, not
  bullet lists) + Qwen2.5-1.5B-Instruct, bf16, GPU.
- `answer_stream()` uses `transformers.TextIteratorStreamer` on a background thread
  for token-by-token output (the web UI reads it as real SSE).
- Conversation memory: last 3 exchanges, kept server-side keyed by a cookie id --
  deliberately NOT Flask's signed session, since that cookie is written into response
  headers before a streaming generator body runs, so mutating `session[...]` inside
  one doesn't reliably persist.

## Known limitations (honest, not fixed yet)

- **No reranker.** The three-channel fusion above gets a good candidate pool, but a
  cross-encoder reranking pass (query + candidate text read jointly, not just rank-fused)
  would likely fix remaining precision misses -- e.g. "What happened to the father of
  Odin?" still doesn't surface the right page, because Bor Burison's chunk genuinely
  scores low on relevance for that phrasing even after the universe-bias fix.
- **Retrieval isn't memory-aware.** Conversation history is fed to the LLM for tone/
  continuity, but retrieval only ever searches the literal current question -- a
  follow-up like "what's the name of his hammer?" can retrieve the wrong entity
  entirely if "his" isn't resolved first. Needs a query-rewrite step before retrieval.
- **A small number of articles parsed badly.** ~0.3% of non-junk articles have an
  empty infobox (e.g. `Thor Odinson (Earth-616)` specifically) -- an extract.py edge
  case with some unusual wikitext structure, not yet root-caused.

## Project layout

```
scripts/
  extract.py      XML dump -> articles.jsonl
  chunk.py        articles.jsonl -> chunks.jsonl
  embed.py        chunks.jsonl -> Chroma vector DB
  build_fts.py    chunks.jsonl -> SQLite FTS5 (BM25)
  build_graph.py  articles.jsonl infobox fields -> relationship graph (SQLite)
  rag.py          shared retrieval + generation engine (RagEngine)
  query.py        CLI wrapper around RagEngine
web/
  app.py          Flask server, SSE streaming chat endpoint
  templates/index.html   chat UI
```
