"""Tests for the fixed-shard ladder translation pipeline.

All network calls are mocked, so these run locally with no API keys or GCP access.

The pipeline translates each shard through Vertex batch in three stages —
submit → poll → collect. The fakes below replace those three functions while
preserving their contracts, so everything downstream (repair, accumulate, QA
gating, progress persistence, resume) runs for real.

Scenarios covered:
  1. Happy path     — all shards pass QA → dataset marked ladder_all_done
  2. QA failure     — a shard fails the gate → ladder_stopped, later shards never submitted
  3. Resume         — completed shards are not re-submitted on a second run
  4. Kill guard     — starting without --resume when a run exists exits with an error
  5. Missing manifest — dataset skipped without crashing
  6. qa_scores.csv  — one row per shard with the expected columns
  7. Dataset filter — only the requested dataset runs
  8. Dedup cache    — a fully-cached shard skips batch submission entirely
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from translation.api.run_beir_ladder_pipeline import (
    _append_to_accumulated,
    _find_existing_run_dir,
    _load_or_init_progress,
    run_ladder,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

DATASET_NAME = "BeIR/nfcorpus"
DATASET_SLUG = "BeIR_nfcorpus"
OTHER_NAME   = "BeIR/scifact"
OTHER_SLUG   = "BeIR_scifact"
SHARD_ROWS   = 5
NUM_SHARDS   = 3
BUCKET       = "fake-bucket"
RUN_ID       = "test_ladder_run"

MOD = "translation.api.run_beir_ladder_pipeline"

MINIMAL_CONFIG = {
    "run_id": RUN_ID,
    "queries": {
        "model": "fake-model",
        "temperature": 0.0,
        "prompt": {"file": "fake/prompts.yaml", "type": "query", "text_col": "text",
                   "english_key": "Text", "hebrew_key": "Hebrew"},
    },
    "documents": {
        "model": "fake-model",
        "temperature": 0.0,
        "prompt": {"file": "fake/prompts.yaml", "type": "document", "text_col": "segment_text",
                   "english_key": "Text", "hebrew_key": "Hebrew"},
    },
    "datasets": {"names": [DATASET_NAME], "default_shard_size": SHARD_ROWS,
                 "shard_sizes": {DATASET_SLUG: SHARD_ROWS}},
    "execution": {"num_workers": 1, "sleep_time": 0, "force_translation": True},
    "ladder": {"cadence": "static", "cadence_start": 1},
    "batch": {"poll_interval_seconds": 0, "max_wait_hours": 1},
    # Repair needs a live client; the fake translations have no failures anyway.
    "repair": {"enabled": False},
    "dedup": {"enabled": False},
    "qa": {"enabled": True, "min_score": 3.5, "sample_size": 3, "sample_seed": 42,
           "judge_model": "fake-judge", "judge_location": None, "sleep_time": 0,
           "baseline_csv": ""},
    "gcs": {"project": "fake-project", "bucket": BUCKET, "location": "global"},
}


def _config(**overrides):
    import copy
    cfg = copy.deepcopy(MINIMAL_CONFIG)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def _make_shard_csv(path, n_rows=SHARD_ROWS, start=0, translated=False):
    """A candidate shard. `start` offsets the ids so shards don't collide —
    without that, duplicate-row checks can't tell a real double-append from the
    fixture reusing ids."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ids = range(start, start + n_rows)
    d = {"_id": [str(i) for i in ids],
         "text": [f"sentence {i}" for i in ids],
         "segment_text": [f"segment {i}" for i in ids],
         "dataset_name": DATASET_NAME}
    if translated:
        d["translation"] = [f"תרגום מלא מספר {i}" for i in ids]
    pd.DataFrame(d).to_csv(path, index=False, encoding="utf-8")


def _make_manifest(candidates_base, slug, num_shards=NUM_SHARDS):
    """Write shard CSVs + manifest under <candidates_base>/<slug>/."""
    slug_dir = os.path.join(candidates_base, slug)
    q, d = [], []
    for i in range(num_shards):
        qf, df = f"queries_shard_{i:03d}.csv", f"documents_shard_{i:03d}.csv"
        _make_shard_csv(os.path.join(slug_dir, qf), start=i * SHARD_ROWS)
        _make_shard_csv(os.path.join(slug_dir, df), start=i * SHARD_ROWS)
        q.append({"index": i, "file": qf, "rows": SHARD_ROWS})
        d.append({"index": i, "file": df, "rows": SHARD_ROWS})
    with open(os.path.join(slug_dir, "shard_manifest.json"), "w") as f:
        json.dump({"shard_size": SHARD_ROWS, "types": {"queries": q, "documents": d}}, f)


# ── Fake batch flow: submit → poll → collect ──────────────────────────────────
# Mirrors the real contracts. _submit_shard_job writes what the batch job would
# have produced; poll is a no-op; collect reports the paths back with token counts.

SUBMITTED = []   # (slug, shard_idx, text_type) for each job actually submitted


def _fake_submit(shard_csv, output_path, text_type, type_cfg, run_id, slug,
                 shard_idx, gemini_client, gcs_client, bucket):
    """Stand in for the batch job: copy the shard through, filling `translation`.

    Reads the real source rather than regenerating, so ids and row order survive
    exactly as they would in production."""
    SUBMITTED.append((slug, shard_idx, text_type))
    df = pd.read_csv(shard_csv)
    col = type_cfg["prompt"]["text_col"]
    df["translation"] = [f"תרגום מלא של {v}" for v in df[col]]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return {"job_name": f"fake-job-{slug}-{shard_idx}-{text_type}",
            "shard_csv": shard_csv, "output_path": output_path,
            "gcs_output_prefix": f"gs://{bucket}/{slug}/{shard_idx}/{text_type}/output"}


def _fake_poll(jobs, gemini_client, poll_interval, max_wait_seconds, **kwargs):
    return None


def _fake_collect(jobs, gcs_client, bucket, config=None):
    return {k: {"output_path": v["output_path"], "input_tokens": 100, "output_tokens": 100}
            for k, v in jobs.items()}


def _passing_qa(*a, **k):
    return {"passed": True, "score_mean": 4.5, "score_std": 0.3, "n": 3}


def _failing_qa(*a, **k):
    return {"passed": False, "score_mean": 2.0, "score_std": 0.5, "n": 3}


class _Harness:
    """Temp run dir with candidates in place, plus the patched batch flow."""

    def __init__(self, qa=_passing_qa, num_shards=NUM_SHARDS, slugs=(DATASET_SLUG,), **cfg):
        self.qa, self.num_shards, self.datasets, self.cfg_over = qa, num_shards, slugs, cfg

    def __enter__(self):
        SUBMITTED.clear()
        self._tmp = tempfile.TemporaryDirectory()
        tmp = self._tmp.name
        self.config = _config(**self.cfg_over)
        self.run_dir = os.path.join(tmp, "runs", f"20990101_000000_{RUN_ID}")
        candidates = os.path.join(self.run_dir, "candidates")
        for slug in self.datasets:
            _make_manifest(candidates, slug, self.num_shards)
        self.progress = _load_or_init_progress(self.run_dir, self.config, RUN_ID)
        self._patches = [
            patch(f"{MOD}._submit_shard_job", side_effect=_fake_submit),
            patch(f"{MOD}._poll_until_all_complete", side_effect=_fake_poll),
            patch(f"{MOD}._collect_shard_results", side_effect=_fake_collect),
            patch(f"{MOD}._ladder_qa", side_effect=self.qa),
            patch(f"{MOD}.render_plots"),
        ]
        for p in self._patches:
            p.start()
        return self

    def run(self, dataset_filter=None, max_cadence_steps=0):
        run_ladder(self.config, self.run_dir, self.progress, dataset_filter,
                   MagicMock(), MagicMock(), BUCKET,
                   max_cadence_steps=max_cadence_steps)

    def state(self, slug=DATASET_SLUG):
        with open(os.path.join(self.run_dir, "progress.json")) as f:
            return json.load(f)["datasets"][slug]

    def qa_scores(self):
        p = os.path.join(self.run_dir, "qa_scores.csv")
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    def accumulated(self, kind="queries", slug=DATASET_SLUG):
        p = os.path.join(self.run_dir, "corpus", slug, f"{kind}_accumulated.csv")
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    def __exit__(self, *a):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHappyPath:
    def test_all_shards_complete(self):
        with _Harness() as h:
            h.run()
            s = h.state()
            assert s["ladder_all_done"] is True
            assert s["ladder_stopped"] is False

    def test_every_shard_submitted_once(self):
        with _Harness() as h:
            h.run()
            assert len(SUBMITTED) == NUM_SHARDS * 2          # queries + documents
            assert len(set(SUBMITTED)) == len(SUBMITTED)     # no duplicates

    def test_qa_row_per_shard(self):
        with _Harness() as h:
            h.run()
            df = h.qa_scores()
            assert len(df) == NUM_SHARDS
            assert list(df["stage"]) == list(range(NUM_SHARDS))
            assert df["overall_passed"].all()

    def test_accumulated_grows_per_shard(self):
        with _Harness(num_shards=2) as h:
            h.run()
            assert len(h.accumulated("queries")) == 2 * SHARD_ROWS
            assert len(h.accumulated("documents")) == 2 * SHARD_ROWS


class TestQaGating:
    def test_stops_at_first_failing_shard(self):
        with _Harness(qa=_failing_qa) as h:
            h.run()
            s = h.state()
            assert s["ladder_stopped"] is True
            assert s["ladder_all_done"] is False

    def test_later_shards_never_submitted(self):
        with _Harness(qa=_failing_qa) as h:
            h.run()
            # only shard 0 should ever have been sent to the batch API
            assert {idx for _, idx, _ in SUBMITTED} == {0}

    def test_stop_reason_recorded(self):
        with _Harness(qa=_failing_qa) as h:
            h.run()
            assert h.state().get("ladder_stop_reason")

    def test_failing_score_logged(self):
        with _Harness(qa=_failing_qa) as h:
            h.run()
            df = h.qa_scores()
            assert len(df) == 1
            assert not df["overall_passed"].iloc[0]


class TestResume:
    def test_partial_run_resumes_without_resubmitting(self):
        """The real resume path: a dataset that stopped part-way must re-submit
        only the shards it never finished.

        Note this is NOT the same as re-running a finished dataset — that exits at
        the dataset level via ladder_all_done and never reaches the shard loop.
        Here the dataset is deliberately left incomplete.
        """
        with _Harness(num_shards=3) as h:
            h.run(max_cadence_steps=1)          # stop after the first cadence step
            done = sorted(SUBMITTED)
            assert {idx for _, idx, _ in done} == {0}, "only shard 0 should have run"
            assert not h.state().get("ladder_all_done")

            SUBMITTED.clear()
            h.run()                              # resume
            resumed = {idx for _, idx, _ in SUBMITTED}
            assert 0 not in resumed, "shard 0 was already appended — must not re-submit"
            assert resumed == {1, 2}
            assert h.state()["ladder_all_done"] is True

            # The data check, which is what a broken resume actually corrupts:
            # re-processing shard 0 would append its rows a second time.
            for kind in ("queries", "documents"):
                acc = h.accumulated(kind)
                assert len(acc) == 3 * SHARD_ROWS, (
                    f"{kind}: expected {3*SHARD_ROWS} rows, got {len(acc)} — "
                    "a shard was appended twice")
                assert acc["_id"].duplicated().sum() == 0, f"{kind}: duplicate ids"

    def test_crash_between_submit_and_append_does_not_duplicate(self):
        """The case the per-shard guards exist for.

        A normal resume never revisits a finished shard — the cadence cursor has
        moved past it. The guards matter when a shard was submitted and appended
        but the process died before the cursor advanced, so the ladder comes back
        to a shard whose rows are already in the accumulated file.
        """
        with _Harness(num_shards=2) as h:
            h.run(max_cadence_steps=1)
            before = {k: len(h.accumulated(k)) for k in ("queries", "documents")}
            assert before["queries"] == SHARD_ROWS

            # Rewind the cursor as a crash would, leaving shard 0's per-shard
            # records intact. The ladder must re-examine shard 0 and not re-append.
            entry = h.progress["datasets"][DATASET_SLUG]
            entry["ladder_current_stage"] = 0
            SUBMITTED.clear()
            h.run()

            for kind in ("queries", "documents"):
                acc = h.accumulated(kind)
                assert len(acc) == 2 * SHARD_ROWS, (
                    f"{kind}: {len(acc)} rows — shard 0 was appended twice")
                assert acc["_id"].duplicated().sum() == 0, f"{kind}: duplicate ids"

    def test_finished_dataset_short_circuits(self):
        with _Harness() as h:
            h.run()
            assert len(SUBMITTED) == NUM_SHARDS * 2
            SUBMITTED.clear()
            h.run()
            assert SUBMITTED == []

    def test_done_dataset_skipped(self):
        with _Harness() as h:
            h.progress["datasets"][DATASET_SLUG]["ladder_all_done"] = True
            h.run()
            assert SUBMITTED == []

    def test_stopped_dataset_skipped(self):
        with _Harness() as h:
            h.progress["datasets"][DATASET_SLUG]["ladder_stopped"] = True
            h.run()
            assert SUBMITTED == []


class TestDatasetFilter:
    def test_filter_runs_only_target(self):
        with _Harness(slugs=(DATASET_SLUG, OTHER_SLUG),
                      datasets={"names": [DATASET_NAME, OTHER_NAME],
                                "default_shard_size": SHARD_ROWS,
                                "shard_sizes": {DATASET_SLUG: SHARD_ROWS,
                                                OTHER_SLUG: SHARD_ROWS}}) as h:
            h.run(dataset_filter=OTHER_SLUG)
            assert {slug for slug, _, _ in SUBMITTED} == {OTHER_SLUG}
            assert h.state(OTHER_SLUG)["ladder_all_done"] is True
            # the non-target dataset must be untouched
            assert not h.state(DATASET_SLUG).get("ladder_all_done")


class TestMissingManifest:
    def test_skipped_without_crash(self):
        with _Harness() as h:
            os.remove(os.path.join(h.run_dir, "candidates", DATASET_SLUG,
                                   "shard_manifest.json"))
            h.run()          # must not raise
            assert SUBMITTED == []


class TestProgressPersistence:
    def test_progress_json_valid_after_run(self):
        with _Harness() as h:
            h.run()
            with open(os.path.join(h.run_dir, "progress.json")) as f:
                p = json.load(f)          # must parse
            assert p["run_id"] == RUN_ID
            assert DATASET_SLUG in p["datasets"]

    def test_token_counts_recorded(self):
        with _Harness(num_shards=1) as h:
            h.run()
            rec = h.state()["shards"]["0"]["queries"]
            assert rec["appended"] is True
            assert rec["input_tokens"] == 100


class TestDedupCache:
    """With dedup on, a shard already in the cache must skip batch submission."""

    def test_fully_cached_shard_skips_submission(self):
        from translation.api.ladder_dedup import SqliteTranslationCache
        with _Harness(num_shards=1, dedup={"enabled": True}) as h:
            # Pre-populate the cache with every source string in shard 0.
            cache_path = os.path.join(h.run_dir, "translation_cache.sqlite")
            os.makedirs(h.run_dir, exist_ok=True)
            cache = SqliteTranslationCache(cache_path)
            for kind, col in (("queries", "text"), ("documents", "segment_text")):
                src = os.path.join(h.run_dir, "candidates", DATASET_SLUG,
                                   f"{kind}_shard_000.csv")
                for t in pd.read_csv(src)[col]:
                    cache.store("fake-model", "fake/prompts.yaml", str(t), "", f"HE::{t}")
            cache.close()

            h.run()

            assert SUBMITTED == [], "a fully cached shard must not be submitted"
            assert h.state()["ladder_all_done"] is True
            acc = h.accumulated("queries")
            assert len(acc) == SHARD_ROWS
            assert all(str(v).startswith("HE::") for v in acc["translation"])


class TestAppendToAccumulated:
    def test_first_call_creates_file_with_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "s.csv"); acc = os.path.join(tmp, "a.csv")
            _make_shard_csv(src, translated=True)
            n = _append_to_accumulated(src, acc)
            assert n == SHARD_ROWS
            assert len(pd.read_csv(acc)) == SHARD_ROWS

    def test_append_without_duplicate_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "s.csv"); acc = os.path.join(tmp, "a.csv")
            _make_shard_csv(src, translated=True)
            _append_to_accumulated(src, acc)
            n = _append_to_accumulated(src, acc)
            assert n == 2 * SHARD_ROWS
            df = pd.read_csv(acc)
            assert len(df) == 2 * SHARD_ROWS
            assert "_id" not in list(df["_id"].astype(str))   # header not re-written as data


class TestKillGuard:
    def test_no_existing_run_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert _find_existing_run_dir(tmp, RUN_ID) is None

    def test_finds_existing_run(self):
        """A run dir only counts once it has a progress.json — a bare directory
        left behind by a crashed setup must not be mistaken for a resumable run."""
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, f"20990101_000000_{RUN_ID}")
            os.makedirs(d)
            assert _find_existing_run_dir(tmp, RUN_ID) is None
            with open(os.path.join(d, "progress.json"), "w") as f:
                json.dump({"run_id": RUN_ID, "datasets": {}}, f)
            assert _find_existing_run_dir(tmp, RUN_ID) == d

    def test_picks_latest_of_several_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            for stamp in ("20990101_000000", "20990301_000000", "20990201_000000"):
                d = os.path.join(tmp, f"{stamp}_{RUN_ID}")
                os.makedirs(d)
                with open(os.path.join(d, "progress.json"), "w") as f:
                    json.dump({"run_id": RUN_ID, "datasets": {}}, f)
            assert _find_existing_run_dir(tmp, RUN_ID).endswith(f"20990301_000000_{RUN_ID}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
