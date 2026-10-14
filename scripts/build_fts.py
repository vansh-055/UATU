"""Build a BM25 full-text index over chunks.jsonl, using SQLite's FTS5.

Chroma's vector search finds chunks that mean roughly the same thing as the
query, even with different wording -- but a sentence embedding can blur
exact proper nouns (e.g. every alternate-reality "Abomination" chunk looks
nearly identical to a dense embedding, since they're all about a character
named Abomination). FTS5's bm25() ranking is the complementary tool:
lexical, exact-term matching, weighted by how rare/informative a term is
across the whole corpus -- it doesn't care about meaning, only about
"does this exact word appear, and how unusual is it."

rag.py combines both (hybrid search) via Reciprocal Rank Fusion, so neither
retriever's blind spot alone sinks a query.

This is a small, disk-backed SQLite file, not a big in-memory structure --
much lighter than loading something like rank_bm25 for 500k+ documents.

    .venv/bin/python scripts/build_fts.py
"""
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "chunks.jsonl"
OUT = ROOT / "data" / "fts.sqlite3"


def main():
    OUT.unlink(missing_ok=True)
    con = sqlite3.connect(OUT)
    # "id UNINDEXED" -- stored and retrievable, but not tokenized into the
    # search index itself (we never want to full-text-search on the id string).
    # "porter unicode61" -- stems words (Powers/Power/Powered all match) and
    # handles basic unicode the way a normal tokenizer would.
    con.execute(
        "CREATE VIRTUAL TABLE chunks USING fts5(id UNINDEXED, text, tokenize='porter unicode61')"
    )

    n = 0
    batch = []
    with IN.open() as f:
        for line in f:
            d = json.loads(line)
            batch.append((d["id"], d["text"]))
            if len(batch) >= 1000:
                con.executemany("INSERT INTO chunks (id, text) VALUES (?, ?)", batch)
                batch = []
                n += 1000
                if n % 50000 == 0:
                    print(f"{n} chunks indexed", flush=True)
    if batch:
        con.executemany("INSERT INTO chunks (id, text) VALUES (?, ?)", batch)
        n += len(batch)

    con.commit()
    con.close()
    print(f"done: {n} chunks -> {OUT}")


if __name__ == "__main__":
    main()
