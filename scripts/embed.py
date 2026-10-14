"""Embed chunks.jsonl and load them into a local Chroma vector DB.

Chroma is a file-based vector database -- no server to run, it just writes
to a folder on disk (data/chroma/). "Vector DB" here really just means:
store each chunk's embedding (a list of floats capturing its meaning) next
to its text and metadata, indexed so that "find the chunks whose embedding
is closest to this query's embedding" is fast even at ~750k rows.

Model: all-MiniLM-L6-v2 -- a small (22M param) sentence-transformer. It's
not the strongest embedding model available, but it's fast enough to run
this whole corpus on CPU in a reasonable time, which matters more for a
first working pipeline than squeezing out extra retrieval quality. Swapping
in a stronger model later (e.g. BAAI/bge-small-en-v1.5) only means re-running
this script -- nothing downstream changes.

Runs in bounded slices (--skip/--limit) so a long embed job can be done as
several short, checkpointed runs instead of one multi-hour unattended
process -- each slice exits cleanly (flushing Chroma's on-disk index
properly) before the next one starts, which is much safer on a machine
that's already shown it can crash mid-run.

    .venv/bin/python scripts/embed.py [--skip N] [--limit N] [--reset]
"""
import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "chunks.jsonl"
DB_DIR = ROOT / "data" / "chroma"
COLLECTION = "uatu_rag"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256


def iter_batches(path, skip, limit, batch_size):
    batch = []
    with path.open() as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            if limit and i >= skip + limit:
                break
            batch.append(json.loads(line))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=0, help="skip the first N chunks (resuming a slice)")
    ap.add_argument("--limit", type=int, default=0, help="only embed N chunks after --skip")
    ap.add_argument("--reset", action="store_true", help="drop and recreate the collection first")
    args = ap.parse_args()

    model = SentenceTransformer(MODEL_NAME)
    print(f"model loaded on device: {model.device}")

    client = chromadb.PersistentClient(path=str(DB_DIR))
    if args.reset:
        client.delete_collection(COLLECTION) if COLLECTION in {c.name for c in client.list_collections()} else None
    collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    n = args.skip
    for batch in iter_batches(IN, args.skip, args.limit, BATCH_SIZE):
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {"title": c["title"], "heading": c["heading"], "url": c["url"],
                 "universe": c["universe"], "type": c["type"]}
                for c in batch
            ],
        )
        n += len(batch)
        print(f"{n} chunks embedded", flush=True)

    print(f"done: {n} chunks -> {DB_DIR} (collection={COLLECTION})")


if __name__ == "__main__":
    main()
