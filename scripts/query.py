"""CLI: ask a question, answered in Uatu's voice using retrieved wiki chunks.

Thin wrapper around rag.RagEngine -- see rag.py for the actual retrieval +
generation logic (also used by web/app.py).

    .venv/bin/python scripts/query.py "How did Asgard fall?"
"""
import argparse

from rag import RagEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=8, help="number of chunks to retrieve")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    args = ap.parse_args()

    engine = RagEngine()
    reply, sources = engine.answer(args.question, k=args.k, max_new_tokens=args.max_new_tokens)

    print("--- retrieved ---")
    for s in sources:
        print(f"* {s['title']} / {s['heading']}")
    print()
    print("--- Uatu answers ---")
    print(reply)


if __name__ == "__main__":
    main()
