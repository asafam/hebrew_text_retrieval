#!/usr/bin/env python3
"""Objective, model-free defect signals for each Hebrew translation.

None of these judge fluency — they detect mechanical failures that would plausibly
break retrieval:

  len_ratio        len(He)/len(En) in characters. The translation pipeline itself
                   treated <0.5 as a failure worth repairing, so a low ratio means
                   content was dropped or the output was truncated.
  latin_residue    share of alphabetic characters still in Latin script. High
                   values mean terms were left untranslated.
  hebrew_frac      share of alphabetic characters that are Hebrew. Near zero means
                   the record was effectively not translated at all.
  digit_jaccard    Jaccard overlap of the multisets of numbers in En vs He.
                   Numbers are the most retrieval-relevant tokens that a
                   translator can silently drop or garble.
  empty            Hebrew text is blank.

Output: outputs/analysis/defects/<dataset>_{queries,corpus}.jsonl
"""

import re
import json
import argparse
from pathlib import Path

HE_ROOT = ("outputs/translation/runs/"
           "full_corpus_zeroshot_nocontext_gemini31flashlite_promptv20260531/corpus")

HEB = re.compile(r"[֐-׿]")
LATIN = re.compile(r"[A-Za-z]")
NUM = re.compile(r"\d+(?:[.,]\d+)?")


def signals(he_text, en_text, he_title="", en_title=""):
    he = f"{he_title} {he_text}".strip()
    en = f"{en_title} {en_text}".strip()
    n_he, n_lat = len(HEB.findall(he)), len(LATIN.findall(he))
    alpha = n_he + n_lat

    en_nums, he_nums = NUM.findall(en), NUM.findall(he)
    if en_nums or he_nums:
        a, b = set(en_nums), set(he_nums)
        dj = len(a & b) / len(a | b) if (a | b) else 1.0
    else:
        dj = 1.0

    return {
        "len_ratio": (len(he) / len(en)) if en else 0.0,
        "latin_residue": (n_lat / alpha) if alpha else 0.0,
        "hebrew_frac": (n_he / alpha) if alpha else 0.0,
        "digit_jaccard": dj,
        "n_digits_en": len(en_nums),
        "empty": not he.strip(),
        "he_chars": len(he),
        "en_chars": len(en),
    }


def process(path, out_path):
    n = 0
    with open(path) as fin, open(out_path, "w") as fout:
        for line in fin:
            r = json.loads(line)
            s = signals(r.get("text", "") or "", r.get("text_en", "") or "",
                        r.get("title", "") or "", r.get("title_en", "") or "")
            s["_id"] = r["_id"]
            fout.write(json.dumps(s, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--he_root", default=HE_ROOT)
    ap.add_argument("--out_dir", default="outputs/analysis/defects")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for ds_dir in sorted(Path(a.he_root).iterdir()):
        beir = ds_dir / "beir"
        if not (beir / "corpus.jsonl").exists():
            continue
        nq = process(beir / "queries.jsonl", out / f"{ds_dir.name}_queries.jsonl")
        nc = process(beir / "corpus.jsonl", out / f"{ds_dir.name}_corpus.jsonl")
        print(f"{ds_dir.name:<16} queries={nq:<6} corpus={nc}")
    print(f"\nWrote defect signals to {out}")


if __name__ == "__main__":
    main()
