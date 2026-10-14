"""The retrieval + generation core, shared by query.py (CLI) and web/app.py
(web UI) -- so both talk to the same logic instead of two copies drifting
apart.

Loading the embedding model and the LLM takes real time (tens of seconds).
RagEngine loads both once, in __init__, and keeps them in memory -- a
long-running process (the web server) can then answer many questions
without reloading anything per-request. query.py just pays that cost once
per CLI invocation instead.

Retrieval is HYBRID: Chroma's vector search (semantic -- finds chunks that
mean the same thing, even with different wording) plus a SQLite FTS5 full
text search (lexical -- finds chunks with the exact words, via real BM25
ranking, see build_fts.py). They catch different failure modes: vector
search blurs together many same-named entities (every alternate-reality
"Abomination" looks nearly identical to a sentence embedding); BM25 doesn't
know "fall" relates to "destroyed," but nails exact rare terms instantly.

A third channel, GRAPH EXPANSION, sits on top: whichever article titles
vector+BM25 already surfaced are used as seeds into a relationship graph
(see build_graph.py, built from infobox fields like Relatives/Affiliation/
KilledBy -- no LLM extraction needed, the wiki already gave us structured
relationships). Their direct graph neighbors are pulled in as extra
candidates -- entities clearly connected to what the query is about, even
if their own chunk text wasn't a close enough lexical/semantic match to rank
highly on its own (e.g. a query about Asgard falling surfaces Surtur via
text search, and Surtur's graph neighbors include Thor via a "killed_by"
edge, which may never have matched the query text directly).

All three rankings are combined with Reciprocal Rank Fusion (RRF), not by
averaging their raw scores -- cosine similarity (0 to 1), BM25 scores
(unbounded, corpus-dependent), and "how many hops in a graph" aren't on
comparable scales, but a chunk's *rank position* within each list always is.
"""
import re
import sqlite3
import threading
from pathlib import Path

import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "chroma"
FTS_DB = ROOT / "data" / "fts.sqlite3"
GRAPH_DB = ROOT / "data" / "graph.sqlite3"
COLLECTION = "uatu_rag"
# "created_by" (Stan Lee, Jack Kirby, ...) has enormous fan-out -- a real
# person who created thousands of characters -- so traversing it would pull
# in near-random unrelated characters instead of narratively relevant ones.
# Kept in the graph (it's a legitimate fact), just excluded from expansion.
EXPAND_RELATIONS_EXCLUDE = {"created_by"}
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # fits comfortably in 4GB VRAM in bf16
RRF_K = 60  # standard RRF damping constant -- softens how much a rank-1 vs
            # rank-2 difference matters, so one retriever's #1 pick doesn't
            # automatically beat the other retriever's #1 pick by default

PERSONA = """You are Uatu, the Watcher of Earth -- an ancient, immensely powerful \
cosmic being who has observed all of reality, across every timeline and every \
Earth, bound by a sacred oath not to interfere, only to witness and chronicle.

Speak with solemn, grandiose formality. Refer to yourself as "Uatu" or "the \
Watcher," rarely "I." Treat even small questions as moments worth chronicling \
across the vastness of time. Example of your voice:

"I am Uatu. I have watched this world since before its heroes drew breath, \
and I shall watch it long after. Though I am sworn never to interfere, I may \
yet chronicle what was, what is, and what may come to pass..."

Every sentence you write must sound like Uatu speaking -- do not slip into a \
plain factual tone. When the context describes events (battles, deaths, the \
fall of a realm), tell it as a STORY with a beginning, middle and end, not a \
bullet list of facts. Use ONLY the facts given in the context below. If the \
context does not contain the answer, say so in character -- do not invent facts."""


class RagEngine:
    def __init__(self, llm_model=LLM_MODEL):
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        self.collection = chromadb.PersistentClient(path=str(DB_DIR)).get_collection(COLLECTION)
        self.fts = sqlite3.connect(str(FTS_DB), check_same_thread=False)
        self.graph = sqlite3.connect(str(GRAPH_DB), check_same_thread=False)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(llm_model, dtype=torch.bfloat16).to(self.device)

    def _vector_search(self, question, k):
        query_emb = self.embed_model.encode([question], normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=query_emb, n_results=k)
        return res["ids"][0]

    def _bm25_search(self, question, k):
        # Quote every term so FTS5 treats it as literal text, not query
        # syntax (a question containing "AND"/"OR"/"-"/quotes would otherwise
        # be parsed as FTS5 operators instead of words to search for).
        terms = re.findall(r"\w+", question.lower())
        if not terms:
            return []
        match_query = " OR ".join(f'"{t}"' for t in terms)
        cur = self.fts.execute(
            "SELECT id FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT ?",
            (match_query, k),
        )
        return [row[0] for row in cur.fetchall()]

    def _titles_for(self, ids):
        """Ordered, deduped article titles for a list of chunk ids (cheapest
        way to go from "these chunks matched" to "these articles matched,"
        since a chunk id like "36638:1:9" doesn't carry the title itself)."""
        if not ids:
            return []
        got = self.collection.get(ids=ids, include=["metadatas"])
        by_id = dict(zip(got["ids"], got["metadatas"]))
        seen, titles = set(), []
        for id_ in ids:
            title = by_id.get(id_, {}).get("title")
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
        return titles

    def _graph_neighbor_titles(self, seed_titles, max_seeds=5, max_neighbors=20):
        seeds = seed_titles[:max_seeds]
        if not seeds:
            return []
        placeholders = ",".join("?" * len(seeds))
        exclude = ",".join("?" * len(EXPAND_RELATIONS_EXCLUDE))
        rows = self.graph.execute(
            f"""SELECT target_title FROM edges
                WHERE source_title IN ({placeholders}) AND target_title IS NOT NULL
                  AND relation NOT IN ({exclude})
                UNION
                SELECT source_title FROM edges
                WHERE target_title IN ({placeholders}) AND target_title IS NOT NULL
                  AND relation NOT IN ({exclude})""",
            [*seeds, *EXPAND_RELATIONS_EXCLUDE, *seeds, *EXPAND_RELATIONS_EXCLUDE],
        ).fetchall()
        seen, out = set(seeds), []
        for (title,) in rows:
            if title not in seen:
                seen.add(title)
                out.append(title)
                if len(out) >= max_neighbors:
                    break
        return out

    def _graph_search(self, seed_titles, k):
        """One representative chunk per graph-neighbor article (prefer
        Infobox -- it's the densest single-chunk summary of an entity)."""
        ids = []
        for title in self._graph_neighbor_titles(seed_titles):
            got = self.collection.get(where={"$and": [{"title": title}, {"heading": "Infobox"}]}, limit=1)
            if not got["ids"]:
                got = self.collection.get(where={"title": title}, limit=1)
            ids.extend(got["ids"])
            if len(ids) >= k:
                break
        return ids

    def _with_universe_bias(self, scores):
        """Nudge ranking toward Earth-616 / untagged (mainline, or realities
        that don't need a disambiguator -- locations, events) when nothing
        else distinguishes two candidates. Without this, an ambiguous query
        like "Who is the Abomination?" scatters evenly across 100+ same-named
        alternate-reality variants, since they're all an equally good text
        match. The bonus equals one channel's rank-1 contribution -- enough
        to break a near-tie in favor of the mainline version, not enough to
        override a candidate that's genuinely winning on actual relevance
        (e.g. a query that names a specific other reality explicitly)."""
        ids = list(scores.keys())
        if not ids:
            return scores
        got = self.collection.get(ids=ids, include=["metadatas"])
        bonus = 1.0 / (RRF_K + 1)
        biased = dict(scores)
        for id_, meta in zip(got["ids"], got["metadatas"]):
            if meta.get("universe") in ("Earth-616", ""):
                biased[id_] = biased.get(id_, 0.0) + bonus
        return biased

    def retrieve(self, question, k=8, pool=30):
        """Hybrid retrieval: vector + BM25 search (each over a wider `pool`
        than needed), a preliminary fusion of those two picks graph-expansion
        seeds, then all three channels (vector, BM25, graph) are fused
        together with Reciprocal Rank Fusion down to the final top-k."""
        vector_ids = self._vector_search(question, pool)
        bm25_ids = self._bm25_search(question, pool)

        fused = {}
        for ids in (vector_ids, bm25_ids):
            for rank, id_ in enumerate(ids):
                fused[id_] = fused.get(id_, 0.0) + 1.0 / (RRF_K + rank + 1)

        # Seed graph expansion from a universe-biased view, so an ambiguous
        # query doesn't seed expansion from 10 near-identical alternate-
        # reality namesakes instead of the one actually-relevant entity.
        seed_scores = self._with_universe_bias(fused)
        prelim_ids = sorted(seed_scores, key=seed_scores.get, reverse=True)[:10]
        seed_titles = self._titles_for(prelim_ids)
        graph_ids = self._graph_search(seed_titles, pool)
        for rank, id_ in enumerate(graph_ids):
            fused[id_] = fused.get(id_, 0.0) + 1.0 / (RRF_K + rank + 1)

        fused = self._with_universe_bias(fused)
        top_ids = sorted(fused, key=fused.get, reverse=True)[:k]

        got = self.collection.get(ids=top_ids, include=["documents", "metadatas"])
        # collection.get() doesn't preserve the order of the `ids` you pass
        # it -- rebuild the fused ranking order from the lookup.
        by_id = dict(zip(got["ids"], zip(got["documents"], got["metadatas"])))
        return [by_id[i] for i in top_ids if i in by_id]

    def build_prompt(self, question, chunks):
        context = "\n\n".join(f"[{meta['title']} / {meta['heading']}]\n{doc}" for doc, meta in chunks)
        return f"Context:\n{context}\n\nQuestion: {question}"

    def _build_messages(self, question, chunks, history):
        """history is prior {role, content} turns, WITHOUT their retrieved
        context blocks -- only the current turn gets a context block. Repeating
        every past turn's full retrieved context would make the prompt grow
        without bound; the model still remembers what was asked and answered,
        it just re-retrieves fresh context each turn instead of re-reading old
        context."""
        messages = [{"role": "system", "content": PERSONA}]
        messages.extend(history)
        messages.append({"role": "user", "content": self.build_prompt(question, chunks)})
        return messages

    def answer(self, question, history=None, k=8, max_new_tokens=400):
        chunks = self.retrieve(question, k)
        messages = self._build_messages(question, chunks, history or [])
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7
            )
        reply = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        sources = [{"title": m["title"], "heading": m["heading"], "url": m["url"]} for _, m in chunks]
        return reply, sources

    def answer_stream(self, question, history=None, k=8, max_new_tokens=400):
        """Same as answer(), but yields ("sources", [...]) once up front, then
        ("token", text) repeatedly as the model generates, then ("done", None).
        Lets the web UI show text as it's written instead of waiting the full
        ~20-30s for a complete reply."""
        chunks = self.retrieve(question, k)
        sources = [{"title": m["title"], "heading": m["heading"], "url": m["url"]} for _, m in chunks]
        yield "sources", sources

        messages = self._build_messages(question, chunks, history or [])
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        # model.generate() blocks until the whole reply is done. Running it on
        # a background thread lets this generator read the streamer's partial
        # output as it's produced, instead of only after generation finishes.
        thread = threading.Thread(
            target=self.model.generate,
            kwargs=dict(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, streamer=streamer),
        )
        thread.start()
        for text in streamer:
            if text:
                yield "token", text
        thread.join()
        yield "done", None
