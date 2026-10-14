"""Stream the Marvel Database XML dump into a clean JSONL, one article per line.

Keeps main-namespace, non-redirect pages only. Each record has the infobox as
"Key: value" text, the body split into sections, the universe (e.g. Earth-616)
parsed from the title, and the page's categories.

    .venv/bin/python scripts/extract.py [--limit N]
"""
import argparse
import json
import multiprocessing as mp
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import mwparserfromhell as mwp

ROOT = Path(__file__).resolve().parent.parent
DUMP = ROOT / "enmarveldatabase_pages_current.xml"
OUT = ROOT / "data" / "articles.jsonl"

WIKI_URL = "https://marvel.fandom.com/wiki/"
UNIVERSE_RE = re.compile(r"\((Earth-[^()]+)\)\s*$")
HEADING_MARK = "\x00H\x00"
MIN_CHARS = 150

# Inline templates whose first positional argument is real text ({{m|Thunderbolts}}).
# Every other template (citations, {{r|..}} refs, navigation, flags) is dropped.
KEEP_FIRST_ARG = {
    "m", "a", "1st", "1st dies", "referenced", "invoked", "narration", "apn",
    "power", "ability", "wp", "g", "speaker", "cameo", "death", "vision",
}
DROP_TAGS = {"ref", "references", "gallery", "noinclude", "onlyinclude"}
# Infobox params that hold article prose; they become sections, not infobox facts.
# Any other param over PROSE_LEN chars is treated the same way.
PROSE_PARAMS = {
    "history", "powers", "abilities", "weaknesses", "equipment", "paraphernalia",
    "weapons", "transportation", "notes", "trivia", "synopsis", "overview",
}
PROSE_LEN = 1500
# Infobox params that hold media or bookkeeping, not facts.
SKIP_PARAM = re.compile(r"(^image|ref$|^gallery|^imagesize|^imagecaption|^type$)", re.I)


def _first_positional(tmpl):
    for p in tmpl.params:
        if not p.showkey:
            return str(p.value)
    return ""


def clean_code(code):
    """Flatten templates/links/tags in a parsed Wikicode to plain text (in place)."""
    for tmpl in reversed(code.filter_templates()):  # inner templates first
        try:
            name = str(tmpl.name).strip().lower()
            code.replace(tmpl, _first_positional(tmpl) if name in KEEP_FIRST_ARG else "")
        except ValueError:
            pass  # already detached by an outer replacement
    for tag in code.filter_tags():
        if str(tag.tag).strip().lower() in DROP_TAGS:
            try:
                code.remove(tag)
            except ValueError:
                pass
    categories = []
    for link in code.filter_wikilinks():
        title = str(link.title).strip()
        prefix = title.split(":", 1)[0].lower() if ":" in title else ""
        try:
            if prefix == "category":
                categories.append(title.split(":", 1)[1].split("|")[0].strip())
                code.remove(link)
            elif prefix in ("file", "image") or (2 <= len(prefix) <= 3 and prefix.isalpha()):
                code.remove(link)  # media and interlanguage links (es:, fr:, ...)
        except ValueError:
            pass
    return categories


def to_text(code):
    text = code.strip_code(normalize=True, collapse=True)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sections(code, base):
    """Turn cleaned Wikicode into [{heading, text}], marking sub-headings under `base`."""
    for h in code.filter_headings():
        heading = h.title.strip_code().strip()
        try:
            code.replace(h, f"\n\n{HEADING_MARK}{heading}\n")
        except ValueError:
            pass
    sections = []
    for i, chunk in enumerate(to_text(code).split(HEADING_MARK)):
        head, _, body = chunk.partition("\n") if i else ("", "", chunk)
        body, head = body.strip(), head.strip()
        if body:
            label = head if base == "Intro" else f"{base} / {head}" if head else base
            sections.append({"heading": label or "Intro", "text": body})
    return sections


def parse_page(args):
    page_id, title, wikitext = args
    try:
        code = mwp.parse(wikitext)

        infobox_lines, sections, categories, page_type = [], [], [], None
        for tmpl in code.filter_templates(recursive=False):
            name = str(tmpl.name).strip()
            if name.lower().startswith("marvel database:") and name.lower().endswith("template"):
                page_type = name.split(":", 1)[1][: -len(" template")].strip().lower()
                for p in tmpl.params:
                    key = str(p.name).strip()
                    if SKIP_PARAM.search(key):
                        continue
                    val_code = mwp.parse(str(p.value))
                    categories += clean_code(val_code)
                    if key.lower() in PROSE_PARAMS or len(str(val_code)) > PROSE_LEN:
                        sections += split_sections(val_code, key)
                        continue
                    val = re.sub(r"\s*\n\s*", "; ", to_text(val_code))
                    val = re.sub(r"(^|; )\*\s*", r"\1", val).strip("; ")
                    if val:
                        infobox_lines.append(f"{key}: {val}")
                code.remove(tmpl)
                break

        categories += clean_code(code)
        sections += split_sections(code, "Intro")

        infobox = "\n".join(infobox_lines)
        if len(infobox) + sum(len(s["text"]) for s in sections) < MIN_CHARS:
            return None
        m = UNIVERSE_RE.search(title)
        return {
            "id": page_id,
            "title": title,
            "url": WIKI_URL + quote(title.replace(" ", "_"), safe=":/()',!*"),
            "universe": m.group(1) if m else None,
            "type": page_type,
            "categories": categories,
            "infobox": infobox,
            "sections": sections,
        }
    except Exception as e:  # one bad page must not kill a 90k-page run
        return {"error": f"{title}: {type(e).__name__}: {e}"}


def iter_pages(limit):
    seen = 0
    for _, el in ET.iterparse(DUMP):
        if not el.tag.endswith("}page"):
            continue
        if el.find("{*}ns").text == "0" and el.find("{*}redirect") is None:
            text = el.find(".//{*}text").text or ""
            yield el.find("{*}id").text, el.find("{*}title").text, text
            seen += 1
            if limit and seen >= limit:
                return
        el.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stop after N articles (testing)")
    args = ap.parse_args()

    OUT.parent.mkdir(exist_ok=True)
    kept = skipped = errors = 0
    with mp.Pool() as pool, OUT.open("w") as out, (OUT.parent / "extract_errors.log").open("w") as elog:
        for rec in pool.imap(parse_page, iter_pages(args.limit), chunksize=32):
            if rec is None:
                skipped += 1
            elif "error" in rec:
                errors += 1
                elog.write(rec["error"] + "\n")
            else:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            if (kept + skipped + errors) % 5000 == 0:
                print(f"{kept} kept, {skipped} skipped, {errors} errors", file=sys.stderr, flush=True)
    print(f"done: {kept} kept, {skipped} too short, {errors} errors -> {OUT}")


if __name__ == "__main__":
    main()
