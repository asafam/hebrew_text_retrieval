"""
Title translation for BeIR ladder pipeline.

Translates the `title` column of a documents_accumulated.csv in-place,
writing results into a `title_translation` column. Deduplicates by title
text so each unique title is translated exactly once regardless of how many
rows share it.

Designed to run:
  (a) as a post-processing step on already-completed datasets, and
  (b) from run_beir_ladder_pipeline after ladder_all_done=True.

Auth: same as the rest of the sync pipeline — Vertex AI ADC, GEMINI_API_KEY
must be unset. translate.sh already handles this; the standalone entry-point
below also enforces it.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ── Core translation helper ───────────────────────────────────────────────────

def _load_prompts(prompt_file: str, prompt_type: str = "document") -> tuple[str, str]:
    """Return (system_prompt, user_prompt_prefix) from a prompt YAML."""
    with open(prompt_file, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    t = cfg.get(prompt_type, cfg.get("document", {}))
    return t.get("system_prompt", "").strip(), t.get("user_prompt_prefix", "").strip()


def _translate_one(title: str, model_name: str, system_prompt: str,
                   user_prompt_prefix: str, temperature: float) -> str:
    from translation.api.translate import translate as _translate
    user_prompt = (
        f"{user_prompt_prefix}\n\n"
        f"Text: {title}\n"
        f"Hebrew:"
    )
    result = _translate(system_prompt, user_prompt, model_name,
                        temperature=temperature, fail_on_error=False)
    return (result or {}).get("translation", "") or ""


# ── Main function ─────────────────────────────────────────────────────────────

def translate_titles(
    docs_csv: str,
    model_name: str,
    prompt_file: str,
    prompt_type: str = "document",
    temperature: float = 0.3,
    n_workers: int = 32,
    force: bool = False,
) -> dict:
    """Translate the `title` column of docs_csv, writing `title_translation`.

    Skips rows where title is empty/NaN. Deduplicates: each unique title text
    is translated once and the result propagated to all rows with that title.
    Idempotent: if `title_translation` already exists and force=False, only
    untranslated rows (NaN) are re-attempted.

    Returns stats dict: {total_rows, unique_titles, translated, skipped, failed}.
    """
    df = pd.read_csv(docs_csv, encoding="utf-8")

    if "title" not in df.columns:
        return {"total_rows": len(df), "unique_titles": 0, "translated": 0,
                "skipped": 0, "failed": 0}

    if "title_translation" not in df.columns:
        df["title_translation"] = pd.NA
        df["title_translation"] = df["title_translation"].astype(object)

    # Rows with a non-empty title that still need translation
    need_mask = (
        df["title"].notna() &
        (df["title"].astype(str).str.strip() != "") &
        (force | df["title_translation"].isna())
    )

    unique_titles = list(df.loc[need_mask, "title"].astype(str).str.strip().unique())
    skipped = int((~need_mask).sum())

    if not unique_titles:
        return {"total_rows": len(df), "unique_titles": 0,
                "translated": 0, "skipped": skipped, "failed": 0}

    system_prompt, user_prompt_prefix = _load_prompts(prompt_file, prompt_type)

    logger.info(f"Translating {len(unique_titles)} unique titles "
                f"({n_workers} workers, model={model_name})")

    def _job(title):
        try:
            return title, _translate_one(title, model_name, system_prompt,
                                         user_prompt_prefix, temperature)
        except Exception as e:
            logger.warning(f"Title translation failed: {e!r} — title={title[:60]!r}")
            return title, ""

    translation_map: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        for title, result in ex.map(_job, unique_titles):
            translation_map[title] = result

    translated = failed = 0
    for idx, row in df[need_mask].iterrows():
        t = str(row["title"]).strip()
        result = translation_map.get(t, "")
        if result:
            df.at[idx, "title_translation"] = result
            translated += 1
        else:
            failed += 1

    df.to_csv(docs_csv, index=False, encoding="utf-8")
    logger.info(f"Title translation complete: {translated} translated, "
                f"{failed} failed, {skipped} skipped (already done or empty)")
    return {
        "total_rows": len(df),
        "unique_titles": len(unique_titles),
        "translated": translated,
        "skipped": skipped,
        "failed": failed,
    }


# ── Standalone entry-point ────────────────────────────────────────────────────

def main():
    import argparse, sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    ap = argparse.ArgumentParser(
        description="Translate title column of documents_accumulated.csv for completed datasets."
    )
    ap.add_argument("--run-dir", required=True,
                    help="Corpus subdir of a ladder run "
                         "(e.g. outputs/translation/runs/<run_id>/corpus).")
    ap.add_argument("--dataset", default=None,
                    help="Single dataset slug filter, e.g. 'nfcorpus'.")
    ap.add_argument("--config", default="config/translation/full_corpus.yaml")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--force", action="store_true",
                    help="Re-translate even already-translated rows.")
    args = ap.parse_args()

    import json
    from translation.api.run_beir_translation_pipeline import load_config
    config = load_config(args.config)
    d_cfg = config.get("documents", {})
    model_name  = d_cfg.get("model", "gemini-3.1-flash-lite")
    prompt_file = d_cfg.get("prompt", {}).get("file", "")
    temperature = config.get("repair", {}).get("temperature", 0.3)

    corpus_dir = Path(args.run_dir)
    slugs = sorted(p.name for p in corpus_dir.iterdir()
                   if p.is_dir() and (p / "documents_accumulated.csv").exists())
    if args.dataset:
        slugs = [s for s in slugs if args.dataset.lower() in s.lower()]
    if not slugs:
        sys.exit("No datasets with documents_accumulated.csv found.")

    # Set up Vertex AI env vars from config (same as the ladder / pilot)
    gcs = config.get("gcs", {})
    project  = gcs.get("project", "")
    location = gcs.get("location", "global")
    if project:
        os.environ["GEMINI_PROJECT"]  = project
        os.environ["GEMINI_LOCATION"] = location
    os.environ.pop("GEMINI_API_KEY", None)   # force Vertex ADC

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    for slug in slugs:
        docs_csv = str(corpus_dir / slug / "documents_accumulated.csv")
        print(f"\n[{slug}] translating titles ...")
        stats = translate_titles(
            docs_csv=docs_csv,
            model_name=model_name,
            prompt_file=prompt_file,
            temperature=temperature,
            n_workers=args.workers,
            force=args.force,
        )
        print(f"  unique={stats['unique_titles']} translated={stats['translated']} "
              f"failed={stats['failed']} skipped={stats['skipped']}")


if __name__ == "__main__":
    main()
