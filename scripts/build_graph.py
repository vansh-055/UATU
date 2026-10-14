"""Build a relationship graph from infobox fields already parsed by extract.py
-- no LLM calls. This is the cheap alternative to full GraphRAG: that approach
runs an LLM over every chunk to extract entities/relationships from prose;
we don't need to, because the Marvel wiki's own infobox already hands us
structured relationships for free (Relatives, Affiliation, KilledBy, ...).
Runs in seconds over the whole corpus, not hours.

Two tables:
  nodes(title, bare_name, universe, type, url)
  edges(source_title, relation, target_name, target_title, note)

target_title is a best-effort RESOLVED match (infobox values are free text
like "Thunderbolts", not "Thunderbolts (Earth-616)" -- we match on the bare
name and prefer a candidate in the SAME universe as the source article, since
a character's relatives/affiliations are almost always in their own reality).
It's NULL when nothing matched -- target_name (the raw text) is always kept
either way, so no information is silently dropped even when resolution fails.

    .venv/bin/python scripts/build_graph.py
"""
import json
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "data" / "articles.jsonl"
OUT = ROOT / "data" / "graph.sqlite3"

# Same junk-type filter as chunk.py -- graph nodes should correspond to
# articles that are actually retrievable, no point graphing comic-issue or
# bookkeeping pages that were never embedded.
JUNK_TYPES = {"comic", "volume", "creator", "handbook"}

# infobox key (lowercased) -> relation name. Picked from an actual frequency
# scan of the corpus's infobox fields, keeping the ones that name another
# entity rather than describe an attribute.
RELATION_FIELDS = {
    "relatives": "relative",
    "parents": "parent",
    "siblings": "sibling",
    "children": "child",
    "affiliation": "affiliated_with",
    "currentmembers": "has_member",
    "formermembers": "former_member",
    "leaders": "led_by",
    "enemies": "enemy_of",
    "allies": "ally_of",
    "killedby": "killed_by",
    "casualtyof": "casualty_of",
    "creators": "created_by",
}

TRAILING_PAREN_RE = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")
LABEL_ONLY_RE = re.compile(r"^[A-Za-z /]+:$")  # e.g. "Formerly:" with no name attached


def bare_name(title):
    """Strip a trailing "(Earth-616)" / "(Realm)" / "(City)" / etc. tag."""
    m = TRAILING_PAREN_RE.match(title)
    return m.group(1).strip() if m else title


def parse_segment(segment):
    """"Nadia Dornova Blonsky (wife, deceased)" -> ("Nadia Dornova Blonsky", "wife, deceased")
    "Thunderbolts" -> ("Thunderbolts", None)"""
    m = TRAILING_PAREN_RE.match(segment)
    return (m.group(1).strip(), m.group(2).strip()) if m else (segment.strip(), None)


def main():
    OUT.unlink(missing_ok=True)
    con = sqlite3.connect(OUT)
    con.execute("CREATE TABLE nodes (title TEXT PRIMARY KEY, bare_name TEXT, universe TEXT, type TEXT, url TEXT)")
    con.execute("""CREATE TABLE edges (
        id INTEGER PRIMARY KEY, source_title TEXT, relation TEXT,
        target_name TEXT, target_title TEXT, note TEXT)""")

    print("pass 1/2: loading nodes + building name index...")
    articles = []  # keep infoboxes in memory for pass 2, avoid re-reading 417MB twice
    name_index = {}  # bare_name.lower() -> [(title, universe), ...]
    with IN.open() as f:
        for line in f:
            d = json.loads(line)
            if d["type"] in JUNK_TYPES:
                continue
            articles.append(d)
            bn = bare_name(d["title"])
            name_index.setdefault(bn.lower(), []).append((d["title"], d["universe"]))

    con.executemany(
        "INSERT INTO nodes VALUES (?, ?, ?, ?, ?)",
        ((d["title"], bare_name(d["title"]), d["universe"], d["type"], d["url"]) for d in articles),
    )
    print(f"  {len(articles)} nodes")

    def resolve(name, source_universe):
        candidates = name_index.get(name.lower())
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0][0]
        # Prefer: same universe as the source > Earth-616 > untagged > first match.
        for title, universe in candidates:
            if universe == source_universe:
                return title
        for title, universe in candidates:
            if universe == "Earth-616":
                return title
        for title, universe in candidates:
            if universe is None:
                return title
        return candidates[0][0]

    print("pass 2/2: extracting edges from infobox fields...")
    edges = []
    for d in articles:
        for line in d["infobox"].split("\n"):
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            relation = RELATION_FIELDS.get(key.strip().lower())
            if not relation or not value.strip():
                continue
            for segment in value.split(";"):
                segment = segment.strip()
                if not segment or LABEL_ONLY_RE.match(segment):
                    continue
                name, note = parse_segment(segment)
                if not name:
                    continue
                target_title = resolve(name, d["universe"])
                edges.append((d["title"], relation, name, target_title, note))

    con.executemany(
        "INSERT INTO edges (source_title, relation, target_name, target_title, note) VALUES (?, ?, ?, ?, ?)",
        edges,
    )
    con.execute("CREATE INDEX idx_edges_source ON edges(source_title)")
    con.execute("CREATE INDEX idx_edges_target ON edges(target_title)")
    con.commit()
    con.close()

    resolved = sum(1 for e in edges if e[3] is not None)
    print(f"done: {len(edges)} edges ({resolved} resolved, {len(edges) - resolved} unresolved) -> {OUT}")


if __name__ == "__main__":
    main()
