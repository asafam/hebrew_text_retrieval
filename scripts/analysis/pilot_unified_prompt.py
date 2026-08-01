#!/usr/bin/env python3
"""nfcorpus pilot: unified prompt (v20260801) vs split query/document (v20260531).

Two questions, both answered on the same rows so the comparison is paired:

  1. CONSISTENCY — does the same English string still translate differently
     depending on whether it is being handled as a query or as a document?
     v20260531 diverged on 38% of strings at temperature 0.
  2. QUALITY — does unifying the wording cost translation quality? Scored with
     the same judge and the same rubric the ladder's QA gate uses for nfcorpus
     (`translation_evaluation_nogold_technical_v20260531.yaml`,
     gemini-3.1-pro-preview), so the numbers are comparable to the gate.

Runs entirely on sampled rows in a scratch directory. Does NOT touch the run
directory, progress.json, or any accumulated CSV.

Usage:
    python scripts/analysis/pilot_unified_prompt.py --n 60
"""

import os
import re
import json
import argparse
import random
from concurrent.futures import ThreadPoolExecutor

import yaml

BASE = ("outputs/translation/runs/"
        "full_corpus_zeroshot_nocontext_gemini31flashlite_promptv20260531/corpus")
OLD = "prompts/translation/api/translation/translation_prompts_zeroshot_nocontext_v20260531.yaml"
NEW = "prompts/translation/api/translation/translation_prompts_zeroshot_nocontext_v20260801.yaml"
JUDGE_PROMPT = "prompts/translation/api/evaluation/translation_evaluation_nogold_technical_v20260531.yaml"
CFG = "config/translation/full_corpus.yaml"

SCORE_SCHEMA = {
    "type": "object",
    "properties": {"critique": {"type": "string"},
                   "score": {"type": "integer", "minimum": 0, "maximum": 5}},
    "required": ["critique", "score"],
}


def render(prompt_yaml, kind, hebrew_key, col, src, english_key="Text"):
    pr = prompt_yaml[kind]
    body = (pr["user_prompt_template"]
            .replace("{english_key}", english_key)
            .replace("{" + col + "}", src)
            .replace("{hebrew_key}", hebrew_key))
    return pr["system_prompt"], pr["user_prompt_prefix"] + "\n\n" + body


def make_client():
    from google import genai
    return genai.Client(vertexai=True, project="iucc-tsarfaty-lab-gcp-asaf", location="global")


def gen(client, model, sys_p, user_p, schema=None, temp=0.0):
    cfg = {"system_instruction": sys_p, "temperature": temp}
    if schema:
        cfg |= {"response_mime_type": "application/json", "response_json_schema": schema}
    r = client.models.generate_content(model=model, contents=user_p, config=cfg)
    return (r.text or "").strip()


def sample_nfcorpus(n):
    """Half short query-like titles, half document abstracts."""
    rng = random.Random(0)
    qs, ds = [], []
    with open(f"{BASE}/BeIR_nfcorpus/beir/queries.jsonl") as f:
        for line in f:
            t = (json.loads(line).get("text_en") or "").strip()
            if 10 < len(t) < 300:
                qs.append(t)
    with open(f"{BASE}/BeIR_nfcorpus/beir/corpus.jsonl") as f:
        for i, line in enumerate(f):
            if i > 3000:
                break
            r = json.loads(line)
            t = ((r.get("title_en") or "") + " " + (r.get("text_en") or "")).strip()
            if 200 < len(t) < 1200:
                ds.append(t)
    return rng.sample(qs, min(n // 2, len(qs))), rng.sample(ds, min(n // 2, len(ds)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--judge", default="gemini-3.1-pro-preview")
    ap.add_argument("--out", default="outputs/analysis/pilot_unified/results.json")
    a = ap.parse_args()

    old_p, new_p = yaml.safe_load(open(OLD)), yaml.safe_load(open(NEW))
    judge_p = yaml.safe_load(open(JUDGE_PROMPT))
    cfg = yaml.safe_load(open(CFG))
    model = cfg["queries"]["model"]
    client = make_client()

    queries, docs = sample_nfcorpus(a.n)
    print(f"nfcorpus pilot: {len(queries)} queries + {len(docs)} documents, model={model}\n")

    # ── 1. Consistency: same string as query vs as document, per prompt version ──
    def consistency(src, pv, hk_q, hk_d):
        q = gen(client, model, *render(pv, "query", hk_q, "text", src))
        d = gen(client, model, *render(pv, "document", hk_d, "segment_text", src))
        return q == d, q, d

    tok = lambda s: set(re.findall(r"[\w֐-׿]+", s))

    def one_consistency(src):
        out = {"src": src}
        for tag, pv, hkq, hkd in [("old", old_p, "Hebrew Query", "Hebrew Document"),
                                  ("new", new_p, "Hebrew", "Hebrew")]:
            try:
                same, q, d = consistency(src, pv, hkq, hkd)
                out[tag] = {"same": same,
                            "same_words": tok(q) == tok(d),
                            "as_query": q, "as_document": d}
            except Exception as e:
                out[tag] = {"error": str(e)[:120]}
        return out

    all_src = queries + docs
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        cons = list(ex.map(one_consistency, all_src))

    def rate(tag, field):
        ok = [c for c in cons if field in c.get(tag, {})]
        return (sum(1 for c in ok if c[tag][field]) / len(ok), len(ok)) if ok else (0, 0)

    old_same, n_old = rate("old", "same")
    new_same, n_new = rate("new", "same")
    old_words, _ = rate("old", "same_words")
    new_words, _ = rate("new", "same_words")

    print("1. CONSISTENCY — same source string as query vs as document")
    print(f"   {'':<26} {'v20260531 (split)':>18} {'v20260801 (unified)':>21}")
    print(f"   {'byte-identical output':<26} {100*old_same:>17.0f}% {100*new_same:>20.0f}%")
    print(f"   {'identical word set':<26} {100*old_words:>17.0f}% {100*new_words:>20.0f}%")

    # ── 2. Quality: judge both versions on the same rows ────────────────────────
    def judge(kind, src, he):
        jp = judge_p[kind]
        col = "text" if kind == "query" else "segment_text"
        user = jp["user_prompt_template"].replace("{" + col + "}", src).replace("{translation}", he)
        raw = gen(client, a.judge, jp["system_prompt"], user, schema=SCORE_SCHEMA)
        return json.loads(raw)["score"]

    def one_quality(item):
        kind, src = item
        col = "text" if kind == "query" else "segment_text"
        row = {"kind": kind, "src": src}
        for tag, pv, hk in [("old", old_p, "Hebrew Query" if kind == "query" else "Hebrew Document"),
                            ("new", new_p, "Hebrew")]:
            try:
                he = gen(client, model, *render(pv, kind, hk, col, src))
                row[tag] = {"he": he, "score": judge(kind, src, he)}
            except Exception as e:
                row[tag] = {"error": str(e)[:120]}
        return row

    items = [("query", q) for q in queries] + [("document", d) for d in docs]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        qual = list(ex.map(one_quality, items))

    def scores(tag, kind=None):
        return [r[tag]["score"] for r in qual
                if "score" in r.get(tag, {}) and (kind is None or r["kind"] == kind)]

    import statistics as st
    print("\n2. QUALITY — judge score 0-5 (same judge + rubric as the nfcorpus QA gate)")
    print(f"   {'':<14} {'n':>4} {'v20260531':>11} {'v20260801':>11} {'delta':>8}")
    for kind in (None, "query", "document"):
        o, n = scores("old", kind), scores("new", kind)
        if not o or not n:
            continue
        label = "ALL" if kind is None else kind
        print(f"   {label:<14} {len(o):>4} {st.mean(o):>11.3f} {st.mean(n):>11.3f} "
              f"{st.mean(n)-st.mean(o):>+8.3f}")
    o, n = scores("old"), scores("new")
    if o and n:
        paired = [(x, y) for r in qual
                  if "score" in r.get("old", {}) and "score" in r.get("new", {})
                  for x, y in [(r["old"]["score"], r["new"]["score"])]]
        worse = sum(1 for x, y in paired if y < x)
        better = sum(1 for x, y in paired if y > x)
        print(f"\n   paired: {better} improved, {worse} regressed, "
              f"{len(paired)-better-worse} unchanged")
        gate = cfg.get("qa", {}).get("min_score", 3.5)
        print(f"   QA gate (min_score={gate}): v20260801 mean {st.mean(n):.3f} "
              f"-> {'PASS' if st.mean(n) >= gate else 'FAIL'}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"consistency": cons, "quality": qual}, open(a.out, "w"),
              ensure_ascii=False, indent=1)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
