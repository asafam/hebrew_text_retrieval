#!/usr/bin/env python3
"""Export the 141 queries no retrieval model could answer, for manual review.

These are row 3 of the attribution in docs/benchmark/why-not-translation.md: the
query succeeds on the English original, fails in Hebrew, and stays failed for
every model tried. No automated method separates the possible causes, so this
writes them out for a human to adjudicate.

Each row pairs the English and Hebrew of both the query and its gold document, so
the reviewer can see whether the Hebrew is a faithful rendering and whether the
gold document actually answers the query at all.

Two verdict columns are left blank to fill in:

  cause     bad_translation  - the Hebrew is wrong, or a key term is mistranslated
            bad_answer_key   - the "correct" document does not answer the query,
                               visible in the English alone
            term_mismatch    - both translations fine, but the query and document
                               render the same term differently
            hard_query       - everything is fine; the query is just difficult
            other

  notes     free text, ideally quoting the specific word or phrase

Usage:
    python scripts/analysis/export_unresolved_141.py
"""

import json
import argparse
from pathlib import Path

import pandas as pd

BASE = ("outputs/translation/runs/"
        "full_corpus_zeroshot_nocontext_gemini31flashlite_promptv20260531/corpus")
PER_QUERY = "outputs/analysis/per_query"
E = "outputs/eval/beir_zeroshot"
MODELS = ["neodictabert-dualenc-beir", "intfloat_multilingual-e5-large", "ndb-meanpool",
          "neodictabert-dualenc-beir-hn", "hmb-20250622-hn"]
DATASETS = ["BeIR_arguana", "BeIR_fiqa", "BeIR_nfcorpus", "BeIR_scidocs", "BeIR_scifact"]


def load_dataset(ds):
    b = Path(BASE) / ds / "beir"
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


def model_hits(ds):
    """Query ids each model retrieved correctly (hit@10) on the Hebrew text."""
    import sys, os, torch, importlib.util
    sys.path.insert(0, "src")
    spec = importlib.util.spec_from_file_location(
        "ev", "src/model/eval/eval_beir_retrieval_zeroshot.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    corpus, queries, qrels = ev.load_beir_local(f"{BASE}/{ds}/beir")
    doc_ids, qids = list(corpus.keys()), list(queries.keys())
    exs = len(set(qids) & set(doc_ids)) / len(qids) >= ev.SELF_OVERLAP_THRESHOLD
    per_model = {}
    for m in MODELS:
        d = f"{E}/{m}/{ds}"
        if not os.path.exists(f"{d}/query_embeddings.pt"):
            continue
        q = torch.load(f"{d}/query_embeddings.pt", weights_only=False)
        dd = torch.load(f"{d}/doc_embeddings.pt", weights_only=False)
        mp = json.load(open(f"{d}/results.json")).get("config", {}).get("model_path")
        if mp and os.path.isdir(mp) and ev._is_infonce(mp):
            q = torch.nn.functional.normalize(q, dim=-1)
            dd = torch.nn.functional.normalize(dd, dim=-1)
        _, idx = ev.faiss_search(q, dd, 101 if exs else 100)
        got = set()
        for i, qid in enumerate(qids):
            j = qrels.get(qid)
            if not j or not any(s > 0 for s in j.values()):
                continue
            rank = 0
            for k in idx[i]:
                did = doc_ids[k]
                if did == qid and exs and j.get(did, 0) <= 0:
                    continue
                rank += 1
                if rank > 10:
                    break
                if j.get(did, 0) > 0:
                    got.add(qid)
                    break
        per_model[m] = got
    return per_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs/analysis/manual_review")
    a = ap.parse_args()

    rows = []
    for ds in DATASETS:
        pq = Path(PER_QUERY) / f"{ds}.jsonl"
        if not pq.exists():
            continue
        # Hebrew failed, English succeeded
        suspects = {}
        for l in open(pq):
            r = json.loads(l)
            if not (r["he_rank"] and r["he_rank"] <= 10) and (r["en_rank"] and r["en_rank"] <= 10):
                suspects[r["qid"]] = r
        if not suspects:
            continue
        hits = model_hits(ds)
        rescued = set().union(*hits.values()) if hits else set()
        never = [qid for qid in suspects if qid not in rescued]

        q, c, qrels = load_dataset(ds)
        for qid in never:
            gold = [g for g in qrels.get(qid, []) if g in c]
            if qid not in q or not gold:
                continue
            d = c[gold[0]]
            rows.append({
                "dataset": ds.replace("BeIR_", ""),
                "query_id": qid,
                "query_en": q[qid].get("text_en", ""),
                "query_he": q[qid].get("text", ""),
                "doc_id": gold[0],
                "doc_title_en": d.get("title_en", ""),
                "doc_title_he": d.get("title", ""),
                "doc_text_en": (d.get("text_en") or "")[:4000],
                "doc_text_he": (d.get("text") or "")[:4000],
                "english_found_it_at_rank": suspects[qid]["en_rank"],
                "n_gold_docs": len(gold),
                "cause": "",     # <- fill in
                "notes": "",     # <- fill in
            })

    df = pd.DataFrame(rows).sort_values(["dataset", "query_id"]).reset_index(drop=True)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "unresolved_queries.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")   # BOM so Excel reads Hebrew

    xlsx_path = out / "unresolved_queries.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="unresolved")
            ws = w.sheets["unresolved"]
            widths = {"A": 11, "B": 16, "C": 55, "D": 55, "E": 16,
                      "F": 40, "G": 40, "H": 70, "I": 70, "J": 10, "K": 8, "L": 18, "M": 40}
            for col, wd in widths.items():
                ws.column_dimensions[col].width = wd
            ws.freeze_panes = "A2"
            from openpyxl.styles import Alignment
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
    except Exception as e:
        xlsx_path = None
        print(f"  (xlsx skipped: {e})")

    print(f"\n{len(df)} queries written")
    print(f"  {csv_path}")
    if xlsx_path:
        print(f"  {xlsx_path}")
    print("\nper dataset:")
    for ds, n in df["dataset"].value_counts().items():
        print(f"   {ds:<12} {n}")
    print("\nFill in the 'cause' column with one of:")
    print("   bad_translation | bad_answer_key | term_mismatch | hard_query | other")


if __name__ == "__main__":
    main()
