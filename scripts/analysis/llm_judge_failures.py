#!/usr/bin/env python3
"""LLM judge over Hebrew-only retrieval failures, with a blind control group.

Why a control group is mandatory
--------------------------------
The mechanical defect signals (length, script, digits) found nothing separating
Hebrew-only failures from successes, but they cannot see semantics — a *correct*
translation can still break retrieval by introducing ambiguity
(`suppositories -> נרות`, which also means "candles"). An LLM judge can see that.

But judging only the failures is uninterpretable: if the judge flags 30% of them
as problematic, that means nothing unless we know the rate among queries Hebrew
answered correctly. Translation noise is present everywhere; only an *elevated*
rate in the failure group implicates translation.

So each failure is paired with a randomly drawn concordant-success control from
the same dataset, the two are shuffled together, and the judge is never told
which group a row belongs to. The judge also cannot see the retrieval outcome.

What the judge rates (per query, on the query text and its gold document):
  translation_quality  faithful | minor_drift | ambiguous | wrong
  ambiguity_risk       whether a correct translation still introduces a word
                       sense that could pull retrieval off-target
  pair_relevance       whether the gold document plausibly answers the query at
                       all — separates genuine qrels noise from translation fault

Usage:
    python scripts/analysis/llm_judge_failures.py --limit 20      # smoke test
    python scripts/analysis/llm_judge_failures.py                 # full run
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

HE_ROOT = ("outputs/translation/runs/"
           "full_corpus_zeroshot_nocontext_gemini31flashlite_promptv20260531/corpus")
PER_QUERY = "outputs/analysis/per_query"
DATASETS = ["BeIR_arguana", "BeIR_fiqa", "BeIR_nfcorpus", "BeIR_scidocs", "BeIR_scifact"]

SYSTEM = """You are auditing Hebrew translations of an information-retrieval benchmark.

You are shown an English query and its Hebrew translation, plus one English document
and its Hebrew translation. The document is labelled relevant to the query in the
benchmark's answer key.

Judge ONLY the translation and the pair, using the English as ground truth. You are
NOT told whether any retrieval system succeeded or failed on this item, and you must
not speculate about that.

Return JSON with exactly these fields:

  "query_translation": one of
      "faithful"     - accurate and natural
      "minor_drift"  - slightly awkward or loose, meaning preserved
      "ambiguous"    - defensible translation, but a key term is polysemous in
                       Hebrew and a reader could take the wrong sense
                       (e.g. English "suppositories" -> Hebrew "נרות", which also
                       means "candles")
      "wrong"        - meaning changed, term mistranslated, or content dropped

  "doc_translation": same four values, for the document

  "key_term_lost": true if a term the query hinges on is missing, transliterated
                   unrecognisably, or rendered with the wrong domain sense in the
                   Hebrew query or document; else false

  "pair_relevance": one of
      "clear"     - the document plainly addresses the query
      "loose"     - same topic, but does not really answer it
      "unrelated" - no meaningful connection even in English
    Judge this from the ENGLISH texts, so it measures the answer key, not the translation.

  "retrieval_risk": one of "none" | "low" | "high"
      How likely the Hebrew wording alone would make a search engine miss this
      document for this query, ignoring model quality.

  "note": one short sentence citing the specific word or phrase, if any.

Output only the JSON object."""

USER_TMPL = """ENGLISH QUERY:
{qen}

HEBREW QUERY:
{qhe}

ENGLISH DOCUMENT:
{den}

HEBREW DOCUMENT:
{dhe}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "query_translation": {"type": "string",
                              "enum": ["faithful", "minor_drift", "ambiguous", "wrong"]},
        "doc_translation": {"type": "string",
                            "enum": ["faithful", "minor_drift", "ambiguous", "wrong"]},
        "key_term_lost": {"type": "boolean"},
        "pair_relevance": {"type": "string", "enum": ["clear", "loose", "unrelated"]},
        "retrieval_risk": {"type": "string", "enum": ["none", "low", "high"]},
        "note": {"type": "string"},
    },
    "required": ["query_translation", "doc_translation", "key_term_lost",
                 "pair_relevance", "retrieval_risk", "note"],
}


def load_dataset(ds):
    b = Path(HE_ROOT) / ds / "beir"
    q = {json.loads(l)["_id"]: json.loads(l) for l in open(b / "queries.jsonl")}
    c = {json.loads(l)["_id"]: json.loads(l) for l in open(b / "corpus.jsonl")}
    qrels = {}
    for f in sorted((b / "qrels").iterdir()):
        if f.name.startswith(("test", "dev", "validation")):
            for l in open(f):
                r = json.loads(l)
                if int(r["score"]) > 0:
                    qrels.setdefault(str(r["query-id"]), []).append(str(r["corpus-id"]))
            break
    return q, c, qrels


def build_items(limit_per_ds=None, seed=42):
    """Return shuffled [(group, dataset, qid, payload)] with matched controls."""
    rng = random.Random(seed)
    items = []
    for ds in DATASETS:
        pq = Path(PER_QUERY) / f"{ds}.jsonl"
        if not pq.exists():
            continue
        rows = [json.loads(l) for l in open(pq)]
        hit = lambda r, k: bool(r[k] and r[k] <= 10)
        loss = [r for r in rows if not hit(r, "he_rank") and hit(r, "en_rank")]
        ctrl_pool = [r for r in rows if hit(r, "he_rank") and hit(r, "en_rank")]
        if limit_per_ds:
            loss = loss[:limit_per_ds]
        # one control per failure, same dataset, no reuse
        rng.shuffle(ctrl_pool)
        ctrl = ctrl_pool[:len(loss)]

        q, c, qrels = load_dataset(ds)
        for group, group_rows in (("loss", loss), ("control", ctrl)):
            for r in group_rows:
                qid = r["qid"]
                gold = qrels.get(qid) or []
                if qid not in q or not gold or gold[0] not in c:
                    continue
                d = c[gold[0]]
                items.append((group, ds, qid, {
                    "qen": q[qid].get("text_en", ""), "qhe": q[qid].get("text", ""),
                    "den": ((d.get("title_en") or "") + " " + (d.get("text_en") or "")).strip()[:2500],
                    "dhe": ((d.get("title") or "") + " " + (d.get("text") or "")).strip()[:2500],
                }))
    rng.shuffle(items)   # judge sees groups interleaved
    return items


def judge_one(client, model, item, retries=4):
    group, ds, qid, p = item
    for a in range(retries):
        try:
            r = client.models.generate_content(
                model=model,
                contents=USER_TMPL.format(**p),
                config={"system_instruction": SYSTEM, "temperature": 0.0,
                        "response_mime_type": "application/json",
                        "response_json_schema": SCHEMA},
            )
            v = json.loads(r.text)
            v.update({"group": group, "dataset": ds, "qid": qid})
            return v
        except Exception as e:
            if a == retries - 1:
                return {"group": group, "dataset": ds, "qid": qid, "error": str(e)[:200]}
            time.sleep(2 ** a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--project", default="iucc-tsarfaty-lab-gcp-asaf")
    ap.add_argument("--location", default="global")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max failures per dataset (smoke test). Controls scale with it.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="outputs/analysis/judge/verdicts.jsonl")
    a = ap.parse_args()

    from google import genai
    client = genai.Client(vertexai=True, project=a.project, location=a.location)

    items = build_items(limit_per_ds=a.limit)
    n_loss = sum(1 for i in items if i[0] == "loss")
    print(f"Judging {len(items)} items ({n_loss} failures + {len(items)-n_loss} blind controls) "
          f"with {a.model}\n")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with open(a.out, "w") as f, ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(judge_one, client, a.model, it): it for it in items}
        for fut in as_completed(futs):
            v = fut.result()
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(items)}")
    print(f"\nWrote {done} verdicts -> {a.out}")


if __name__ == "__main__":
    main()
