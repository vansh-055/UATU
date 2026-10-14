"""Web chat UI for the Uatu RAG pipeline.

One page, a streaming chat endpoint, a reset endpoint. RagEngine (the
embedding model + the LLM) is loaded ONCE when the server process starts --
not per-request. Model loading takes tens of seconds; a single answer
should take a few.

Conversation memory: each browser gets a random id cookie; the actual
history (list of {role, content} turns) lives in the CONVOS dict below, in
this process's memory -- not in the cookie itself, and not in Flask's
built-in `session`. That's a deliberate choice, not an oversight: Flask
writes the signed session cookie into the response HEADERS, which are sent
before a streaming response's generator body has produced anything --
mutating `session[...]` *inside* a streaming generator doesn't reliably
persist, because by the time the mutation happens the cookie header has
already gone out. A plain server-side dict, mutated directly, has no such
timing issue (and conversation history being lost on a server restart is a
fine tradeoff for a local single-user app).

    .venv/bin/python web/app.py
    -> open http://localhost:5000
"""
import json
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from flask import Flask, Response, jsonify, render_template, request

from rag import RagEngine

app = Flask(__name__)
print("Loading models (this happens once, at startup)...")
engine = RagEngine()
print("Ready.")

SID_COOKIE = "uatu_sid"
MAX_TURNS = 3  # keep the last 3 exchanges (6 messages) -- enough for natural
               # follow-ups, bounded enough that a small model's limited
               # context window isn't swamped by old conversation
CONVOS = {}    # sid -> [{"role": "user"/"assistant", "content": ...}, ...]


def get_sid():
    return request.cookies.get(SID_COOKIE) or secrets.token_hex(16)


def set_sid_cookie(resp, sid):
    resp.set_cookie(SID_COOKIE, sid, max_age=60 * 60 * 24 * 30, httponly=True, samesite="Lax")
    return resp


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/reset", methods=["POST"])
def reset():
    sid = get_sid()
    CONVOS.pop(sid, None)
    return set_sid_cookie(jsonify({"ok": True}), sid)


@app.route("/api/chat", methods=["POST"])
def chat():
    question = (request.json or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "empty question"}), 400

    sid = get_sid()
    history = CONVOS.get(sid, [])
    reply, sources = engine.answer(question, history=history)
    CONVOS[sid] = (history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": reply},
    ])[-MAX_TURNS * 2:]

    return set_sid_cookie(jsonify({"answer": reply, "sources": sources}), sid)


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    question = (request.json or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "empty question"}), 400

    sid = get_sid()
    history = CONVOS.get(sid, [])

    def generate():
        reply_parts = []
        for event, payload in engine.answer_stream(question, history=history):
            if event == "token":
                reply_parts.append(payload)
            yield f"event: {event}\ndata: {json.dumps(payload)}\n\n"
        CONVOS[sid] = (history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "".join(reply_parts)},
        ])[-MAX_TURNS * 2:]

    return set_sid_cookie(Response(generate(), mimetype="text/event-stream"), sid)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
