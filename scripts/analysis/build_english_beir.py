#!/usr/bin/env python3
"""Build English-source mirrors of the translated Hebrew BEIR exports.

Each exported record keeps its English source alongside the Hebrew
(`text`/`text_en`, `title`/`title_en`), so an English mirror can be produced by
swapping the fields. Ids and qrels are identical, which means the Hebrew and
English runs are the same retrieval problem in two languages — the only thing
that differs is the text the model sees.

Output: <out_root>/<dataset>/beir/{corpus.jsonl,queries.jsonl,qrels/...}
"""

import json
import shutil
import argparse
from pathlib import Path

DEFAULT_SRC = ("outputs/translation/runs/"
               "full_corpus_zeroshot_nocontext_gemini31flashlite_promptv20260531/corpus")


def build(src_root, out_root):
    src_root, out_root = Path(src_root), Path(out_root)
    for ds_dir in sorted(src_root.iterdir()):
        beir = ds_dir / "beir"
        if not (beir / "corpus.jsonl").exists():
            continue
        out = out_root / ds_dir.name / "beir"
        (out / "qrels").mkdir(parents=True, exist_ok=True)

        stats = {}
        for fname in ("corpus.jsonl", "queries.jsonl"):
            n = fallback = 0
            with open(beir / fname) as fin, open(out / fname, "w") as fout:
                for line in fin:
                    r = json.loads(line)
                    en_text = r.get("text_en")
                    # Fall back to the Hebrew if the English is missing, and count
                    # it — a silent fallback would quietly turn this into a
                    # Hebrew-vs-Hebrew comparison.
                    if en_text is None or not str(en_text).strip():
                        en_text = r.get("text", "")
                        fallback += 1
                    out_rec = {"_id": r["_id"], "text": en_text}
                    if "title" in r:
                        out_rec["title"] = r.get("title_en") or ""
                    fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    n += 1
            stats[fname] = (n, fallback)

        for q in (beir / "qrels").iterdir():
            shutil.copy2(q, out / "qrels" / q.name)

        c, qy = stats["corpus.jsonl"], stats["queries.jsonl"]
        print(f"{ds_dir.name:<16} corpus={c[0]:<7} queries={qy[0]:<6} "
              f"fallback_to_hebrew: corpus={c[1]} queries={qy[1]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", default=DEFAULT_SRC)
    ap.add_argument("--out_root", default="outputs/analysis/english_mirror")
    a = ap.parse_args()
    build(a.src_root, a.out_root)
    print(f"\nWrote English mirrors to {a.out_root}")
