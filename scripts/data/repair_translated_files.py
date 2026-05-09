"""
Repairs *_translated.csv files that lost rows due to a resume bug.

For each translated file, merges it against its source (queries.csv /
documents.csv) so that every source row is present. Rows that were
previously translated are preserved; rows that went missing get NaN
translation and will be retried on the next pipeline run.
"""
import glob
import os
import pandas as pd


CANDIDATES_BASE = "outputs/translation/BeIR/candidates"


def repair(dry_run: bool = False):
    translated_files = glob.glob(f"{CANDIDATES_BASE}/**/*_translated.csv", recursive=True)

    total_repaired = 0
    for trans_path in sorted(translated_files):
        # Derive the matching source file path
        # Structure: candidates/<dataset>/<model>/<prompt>/<type>_translated.csv
        parts = trans_path.split(os.sep)
        dataset_dir = os.path.join(*parts[:parts.index(os.path.basename(trans_path)) - 3 + 1])

        filename = os.path.basename(trans_path)           # e.g. queries_translated.csv
        src_name = filename.replace("_translated", "")    # e.g. queries.csv

        # Find dataset dir (3 levels up from the translated file)
        trans_dir = os.path.dirname(trans_path)
        dataset_dir = os.path.abspath(os.path.join(trans_dir, "../../.."))
        src_path = os.path.join(dataset_dir, src_name)

        if not os.path.exists(src_path):
            print(f"SKIP  {trans_path}  (no source at {src_path})")
            continue

        src_df = pd.read_csv(src_path, encoding="utf-8")
        trans_df = pd.read_csv(trans_path, encoding="utf-8")

        if len(trans_df) == len(src_df):
            print(f"OK    {trans_path}  ({len(trans_df)} rows)")
            continue

        print(f"FIX   {trans_path}  source={len(src_df)} translated={len(trans_df)}")

        id_cols = [c for c in ["_id", "segment_id"] if c in src_df.columns and c in trans_df.columns]
        extra_cols = [c for c in trans_df.columns if c not in src_df.columns]

        repaired = src_df.merge(trans_df[id_cols + extra_cols], on=id_cols, how="left")

        if not dry_run:
            repaired.to_csv(trans_path, index=False, encoding="utf-8")
            print(f"      → saved ({len(repaired)} rows)")

        total_repaired += 1

    print(f"\nDone. {total_repaired} file(s) repaired.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Report without modifying files")
    args = parser.parse_args()
    repair(dry_run=args.dry_run)
