"""Split articles.jsonl sections into embedding-sized chunks.

Each article's sections are already split by heading (done in extract.py).
This script just makes sure no single chunk is too big for an embedding
model's context window, while keeping short sections whole.

Chunks are character-based, not token-based -- simpler to reason about,
and close enough for a ~1000 char target (most English text is ~4 chars/token,
so 1000 chars is roughly 250 tokens).

    .venv/bin/python scripts/chunk.py [--limit N]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "articles.jsonl"
OUT = ROOT / "data" / "chunks.jsonl"

# Pages with no (Earth-XXXX) tag in the title -- universe is None -- are NOT
# alternate-reality noise, they're the opposite: events, locations, story
# arcs and teams that exist once and don't need a reality disambiguator
# (e.g. "Ragnarok (Event)", "Asgard (Realm)"). Dropping universe=None, as an
# earlier version of this script did, silently threw away exactly the pages
# that answer "what happened" story questions. The real noise at that
# universe=None bucket is individual comic-issue and bookkeeping pages
# (~69k of them) -- those are excluded by type instead.
JUNK_TYPES = {"comic", "volume", "creator", "handbook"}

MAX_CHARS = 1000
OVERLAP = 150  # chars repeated between consecutive hard-split chunks, so a
               # fact sitting right on a cut boundary isn't lost to either side


def hard_split(text, max_chars=MAX_CHARS, overlap=OVERLAP):
    """Slice text with no natural break points (one giant paragraph) into
    overlapping fixed-size windows."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def pack_paragraphs(text, max_chars=MAX_CHARS, overlap=OVERLAP):
    """Greedily pack whole paragraphs into chunks up to max_chars.
    A paragraph longer than max_chars on its own falls back to hard_split."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paras:
        if len(p) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(hard_split(p, max_chars, overlap))
            continue
        candidate = f"{current}\n\n{p}" if current else p
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks


def chunk_article(article):
    """Yield one dict per chunk. Each chunk's text is prefixed with the
    article title and section heading -- a chunk read in isolation (which is
    exactly how retrieval will see it) often has no idea who "he" or "the
    team" refers to otherwise.

    The infobox (Name/Aliases/Powers/Affiliation/...) is chunked the same way
    as a section, under the synthetic heading "Infobox" -- it's the densest
    fact source on the page and was being dropped entirely before."""
    sections = article["sections"]
    if article["infobox"]:
        sections = [{"heading": "Infobox", "text": article["infobox"]}] + sections
    for s_idx, section in enumerate(sections):
        pieces = pack_paragraphs(section["text"])
        for c_idx, piece in enumerate(pieces):
            yield {
                "id": f"{article['id']}:{s_idx}:{c_idx}",
                "text": f"{article['title']} — {section['heading']}\n{piece}",
                "title": article["title"],
                "heading": section["heading"],
                "url": article["url"],
                "universe": article["universe"] or "",
                "type": article["type"] or "",
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stop after N articles (testing)")
    ap.add_argument(
        "--universe", default="all",
        help="only keep this universe's articles (default 'all' = every reality). "
             "Uatu watches the whole multiverse, not just Earth-616, so the "
             "default is unrestricted; narrow it for a faster/smaller test build.",
    )
    args = ap.parse_args()

    n_articles = n_seen = n_chunks = 0
    with IN.open() as f, OUT.open("w") as out:
        for line in f:
            article = json.loads(line)
            n_seen += 1
            if article["type"] in JUNK_TYPES:
                continue
            if args.universe != "all" and article["universe"] != args.universe:
                continue
            for chunk in chunk_article(article):
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                n_chunks += 1
            n_articles += 1
            if args.limit and n_articles >= args.limit:
                break
    print(f"done: {n_seen} scanned, {n_articles} kept ({args.universe}) -> {n_chunks} chunks -> {OUT}")


if __name__ == "__main__":
    main()
