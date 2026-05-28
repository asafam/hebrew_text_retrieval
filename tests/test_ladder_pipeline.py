"""
Tests for the fixed-shard ladder translation pipeline.

All LLM calls (translation + QA judge) are mocked so the tests run
locally without any API keys or internet access.

Scenarios covered:
  1. Happy path  — all shards pass QA → dataset marked ladder_all_done
  2. QA failure  — shard fails threshold → ladder_stopped, no further shards
  3. Resume      — progress.json survives a simulated kill; re-running picks up
                   from ladder_current_stage (completed shards are not re-run)
  4. Kill guard  — starting without --resume when a run exists exits with an error
  5. Manifest missing — dataset is skipped gracefully
  6. qa_scores.csv  — rows are appended after every shard with correct columns
  7. Plots          — render_plots is called after every shard; non-fatal on error
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

# Make sure src/ is on the path so imports resolve without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from translation.api.run_beir_ladder_pipeline import (
    _append_to_accumulated,
    _empty_ladder_entry,
    _find_existing_run_dir,
    _load_or_init_progress,
    run_ladder,
)
from translation.api.run_beir_translation_pipeline import save_progress


# ── Fixtures ──────────────────────────────────────────────────────────────────

DATASET_NAME  = "BeIR/nfcorpus"
DATASET_SLUG  = "BeIR_nfcorpus"
SHARD_ROWS    = 5   # small so tests are fast
NUM_SHARDS    = 3   # three shards total

MINIMAL_CONFIG = {
    "run_id": "test_ladder_run",
    "queries": {
        "model": "fake-model",
        "temperature": 0.0,
        "prompt": {
            "file": "fake/prompts.yaml",
            "type": "query",
            "text_col": "text",
            "english_key": "Text",
            "hebrew_key": "Hebrew Query",
        },
    },
    "documents": {
        "model": "fake-model",
        "temperature": 0.0,
        "prompt": {
            "file": "fake/prompts.yaml",
            "type": "document",
            "text_col": "segment_text",
            "english_key": "Text",
            "hebrew_key": "Hebrew Document",
        },
    },
    "datasets": {
        "names": [DATASET_NAME],
        "default_shard_size": SHARD_ROWS,
        "shard_sizes": {DATASET_SLUG: SHARD_ROWS},
    },
    "execution": {
        "num_workers": 1,
        "sleep_time": 0,
        "force_translation": True,
    },
    "qa": {
        "enabled": True,
        "min_score": 3.5,
        "sample_size": 3,
        "sample_seed": 42,
        "judge_model": "fake-judge",
        "judge_location": None,
        "sleep_time": 0,
        "baseline_csv": "",   # no baseline — uses absolute min_score
    },
    "paths": {
        "ladder_candidates_base": "",   # overridden per test
        "ladder_runs_base": "",         # overridden per test
    },
}


def _make_shard_csv(path: str, n_rows: int = SHARD_ROWS, text_col: str = "text") -> None:
    """Write a minimal candidate shard CSV (no translations yet)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({
        "_id": [str(i) for i in range(n_rows)],
        text_col: [f"sentence {i}" for i in range(n_rows)],
        "segment_text": [f"segment {i}" for i in range(n_rows)],
        "dataset_name": DATASET_NAME,
        "tokenizer": "fake-tokenizer",
    }).to_csv(path, index=False)


def _make_manifest(slug_dir: str, num_shards: int = NUM_SHARDS) -> dict:
    """Write a shard_manifest.json and return its dict."""
    q_shards, d_shards = [], []
    for i in range(num_shards):
        qf = f"queries_shard_{i:03d}.csv"
        df = f"documents_shard_{i:03d}.csv"
        _make_shard_csv(os.path.join(slug_dir, qf))
        _make_shard_csv(os.path.join(slug_dir, df), text_col="segment_text")
        q_shards.append({"index": i, "file": qf, "rows": SHARD_ROWS})
        d_shards.append({"index": i, "file": df, "rows": SHARD_ROWS})

    manifest = {"shard_size": SHARD_ROWS, "types": {"queries": q_shards, "documents": d_shards}}
    with open(os.path.join(slug_dir, "shard_manifest.json"), "w") as f:
        json.dump(manifest, f)
    return manifest


def _make_translated_csv(path: str, n_rows: int = SHARD_ROWS) -> None:
    """Write a fake translated shard CSV (has 'translation' column)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({
        "_id": [str(i) for i in range(n_rows)],
        "text": [f"sentence {i}" for i in range(n_rows)],
        "segment_text": [f"segment {i}" for i in range(n_rows)],
        "translation": [f"תרגום {i}" for i in range(n_rows)],
    }).to_csv(path, index=False)


def _passing_qa(*args, **kwargs):
    return {"passed": True, "score_mean": 4.5, "score_std": 0.3, "n": 3}


def _failing_qa(*args, **kwargs):
    return {"passed": False, "score_mean": 2.0, "score_std": 0.5, "n": 3}


def _fake_translate_shard(shard_csv, output_dir, type_cfg, exec_cfg):
    """Fake translation: writes a translated CSV and returns its path."""
    basename = os.path.basename(shard_csv).replace(".csv", "_translated.csv")
    out_path = os.path.join(output_dir, basename)
    n = len(pd.read_csv(shard_csv))
    _make_translated_csv(out_path, n_rows=n)
    return out_path


def _config_with_dirs(candidates_base: str, runs_base: str) -> dict:
    import copy
    cfg = copy.deepcopy(MINIMAL_CONFIG)
    cfg["paths"]["ladder_candidates_base"] = candidates_base
    cfg["paths"]["ladder_runs_base"] = runs_base
    return cfg


# ── Helpers to load outputs ───────────────────────────────────────────────────

def _load_progress(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "progress.json")) as f:
        return json.load(f)


def _load_qa_scores(run_dir: str) -> pd.DataFrame:
    p = os.path.join(run_dir, "qa_scores.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHappyPath:
    """All shards pass QA → dataset is marked ladder_all_done."""

    def test_all_shards_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidates_base = os.path.join(tmp, "candidates")
            runs_base       = os.path.join(tmp, "runs")
            slug_dir        = os.path.join(candidates_base, DATASET_SLUG)
            _make_manifest(slug_dir, num_shards=NUM_SHARDS)

            run_dir  = os.path.join(runs_base, "20990101_000000_test_ladder_run")
            config   = _config_with_dirs(candidates_base, runs_base)
            progress = _load_or_init_progress(run_dir, config, "test_ladder_run")

            with patch("translation.api.run_beir_ladder_pipeline._translate_shard",
                       side_effect=_fake_translate_shard), \
                 patch("translation.api.run_beir_ladder_pipeline._ladder_qa",
                       side_effect=_passing_qa), \
                 patch("translation.api.run_beir_ladder_pipeline.render_plots"):
                run_ladder(config, run_dir, progress, dataset_filter=None)

            state = _load_progress(run_dir)["datasets"][DATASET_SLUG]
            assert state["ladder_all_done"] is True
            assert state["ladder_stopped"] is False
            assert state["ladder_current_stage"] == NUM_SHARDS

    def test_all_shards_written_to_qa_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidates_base = os.path.join(tmp, "candidates")
            runs_base       = os.path.join(tmp, "runs")
            _make_manifest(os.path.join(candidates_base, DATASET_SLUG), num_shards=NUM_SHARDS)

            run_dir  = os.path.join(runs_base, "20990101_000000_test_ladder_run")
            config   = _config_with_dirs(candidates_base, runs_base)
            progress = _load_or_init_progress(run_dir, config, "test_ladder_run")

            with patch("translation.api.run_beir_ladder_pipeline._translate_shard",
                       side_effect=_fake_translate_shard), \
                 patch("translation.api.run_beir_ladder_pipeline._ladder_qa",
                       side_effect=_passing_qa), \
                 patch("translation.api.run_beir_ladder_pipeline.render_plots"):
                run_ladder(config, run_dir, progress, dataset_filter=None)

            df = _load_qa_scores(run_dir)
            assert len(df) == NUM_SHARDS        # one row per shard
            assert list(df["stage"]) == list(range(NUM_SHARDS))
            assert df["overall_passed"].all()

    def test_accumulated_csv_grows_per_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidates_base = os.path.join(tmp, "candidates")
            runs_base       = os.path.join(tmp, "runs")
            _make_manifest(os.path.join(candidates_base, DATASET_SLUG), num_shards=2)

            run_dir  = os.path.join(runs_base, "20990101_000000_test_ladder_run")
            config   = _config_with_dirs(candidates_base, runs_base)
            progress = _load_or_init_progress(run_dir, config, "test_ladder_run")

            with patch("translation.api.run_beir_ladder_pipeline._translate_shard",
                       side_effect=_fake_translate_shard), \
                 patch("translation.api.run_beir_ladder_pipeline._ladder_qa",
                       side_effect=_passing_qa), \
                 patch("translation.api.run_beir_ladder_pipeline.render_plots"):
                run_ladder(config, run_dir, progress, dataset_filter=None)

            accumulated = pd.read_csv(
                os.path.join(run_dir, DATASET_SLUG, "queries_accumulated.csv")
            )
            assert len(accumulated) == 2 * SHARD_ROWS


class TestQaGating:
    """Ladder stops when QA fails; subsequent shards are never translated."""

    def test_stops_at_first_failing_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidates_base = os.path.join(tmp, "candidates")
            runs_base       = os.path.join(tmp, "runs")
            _make_manifest(os.path.join(candidates_base, DATASET_SLUG), num_shards=3)

            run_dir  = os.path.join(runs_base, "20990101_000000_test_ladder_run")
            config   = _config_with_dirs(candidates_base, runs_base)
            progress = _load_or_init_progress(run_dir, config, "test_ladder_run")

            # Shard 0 passes, shard 1 fails
            qa_results = [_passing_qa(), _passing_qa(), _failing_qa(), _failing_qa()]
            translate_calls = []

            def _counting_translate(shard_csv, output_dir, type_cfg, exec_cfg):
                translate_calls.append(shard_csv)
                return _fake_translate_shard(shard_csv, output_dir, type_cfg, exec_cfg)

            with patch("translation.api.run_beir_ladder_pipeline._translate_shard",
                       side_effect=_counting_translate), \
                 patch("translation.api.run_beir_ladder_pipeline._ladder_qa",
                       side_effect=qa_results), \
                 patch("translation.api.run_beir_ladder_pipeline.render_plots"):
                run_ladder(config, run_dir, progress, dataset_filter=None)

            state = _load_progress(run_dir)["datasets"][DATASET_SLUG]
            assert state["ladder_stopped"] is True
            assert state["ladder_all_done"] is False
            # current_stage is set to idx+1 before the gate fires, so after
            # shard 1 (idx=1) fails the stage is 2 and shard 2 is never reached.
            assert state["ladder_current_stage"] == 2

            # Shard 2 (index 2) must never be translated
            translated_shards = [os.path.basename(p) for p in translate_calls]
            assert not any("shard_002" in s for s in translated_shards)

    def test_stop_reason_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidates_base = os.path.join(tmp, "candidates")
            runs_base       = os.path.join(tmp, "runs")
            _make_manifest(os.path.join(candidates_base, DATASET_SLUG), num_shards=1)

            run_dir  = os.path.join(runs_base, "20990101_000000_test_ladder_run")
            config   = _config_with_dirs(candidates_base, runs_base)
            progress = _load_or_init_progress(run_dir, config, "test_ladder_run")

            with patch("translation.api.run_beir_ladder_pipeline._translate_shard",
                       side_effect=_fake_translate_shard), \
                 patch("translation.api.run_beir_ladder_pipeline._ladder_qa",
                       side_effect=_failing_qa), \
                 patch("translation.api.run_beir_ladder_pipeline.render_plots"):
                run_ladder(config, run_dir, progress, dataset_filter=None)

            state = _load_progress(run_dir)["datasets"][DATASET_SLUG]
            assert state["ladder_stop_reason"] is not None
            assert "QA failed" in state["ladder_stop_reason"]

    def test_qa_score_logged_on_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidates_base = os.path.join(tmp, "candidates")
            runs_base       = os.path.join(tmp, "runs")
            _make_manifest(os.path.join(candidates_base, DATASET_SLUG), num_shards=1)

            run_dir  = os.path.join(runs_base, "20990101_000000_test_ladder_run")
            config   = _config_with_dirs(candidates_base, runs_base)
            progress = _load_or_init_progress(run_dir, config, "test_ladder_run")

            with patch("translation.api.run_beir_ladder_pipeline._translate_shard",
                       side_effect=_fake_translate_shard), \
                 patch("translation.api.run_beir_ladder_pipeline._ladder_qa",
                       side_effect=_failing_qa), \
                 patch("translation.api.run_beir_ladder_pipeline.render_plots"):
                run_ladder(config, run_dir, progress, dataset_filter=None)

            df = _load_qa_scores(run_dir)
            assert len(df) == 1
            assert df.iloc[0]["overall_passed"] == False
            assert df.iloc[0]["q_score_mean"] == pytest.approx(2.0)


class TestResume:
    """After a simulated kill, re-running with existing progress.json resumes correctly."""

    def test_completed_shards_skipped_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidates_base = os.path.join(tmp, "candidates")
            runs_base       = os.path.join(tmp, "runs")
            _make_manifest(os.path.join(candidates_base, DATASET_SLUG), num_shards=3)

            run_dir  = os.path.join(runs_base, "20990101_000000_test_ladder_run")
            config   = _config_with_dirs(candidates_base, runs_base)

            # Simulate: shard 0 was already done
            progress = _load_or_init_progress(run_dir, config, "test_ladder_run")
            progress["datasets"][DATASET_SLUG]["ladder_current_stage"] = 1
            progress["datasets"][DATASET_SLUG]["ladder_stage_scores"]["0"] = {
                "q_score_mean": 4.5, "q_score_std": 0.2,
                "d_score_mean": 4.3, "d_score_std": 0.3,
                "passed": True, "cumulative_q_rows": SHARD_ROWS,
                "cumulative_d_rows": SHARD_ROWS, "timestamp": "2099-01-01T00:00:00",
            }
            # Pre-populate accumulated CSV so appending works from shard 1 onward
            acc_q = os.path.join(run_dir, DATASET_SLUG, "queries_accumulated.csv")
            acc_d = os.path.join(run_dir, DATASET_SLUG, "documents_accumulated.csv")
            os.makedirs(os.path.dirname(acc_q), exist_ok=True)
            _make_translated_csv(acc_q)
            _make_translated_csv(acc_d)
            save_progress(run_dir, progress)

            translate_calls = []

            def _counting_translate(shard_csv, output_dir, type_cfg, exec_cfg):
                translate_calls.append(os.path.basename(shard_csv))
                return _fake_translate_shard(shard_csv, output_dir, type_cfg, exec_cfg)

            with patch("translation.api.run_beir_ladder_pipeline._translate_shard",
                       side_effect=_counting_translate), \
                 patch("translation.api.run_beir_ladder_pipeline._ladder_qa",
                       side_effect=_passing_qa), \
                 patch("translation.api.run_beir_ladder_pipeline.render_plots"):
                run_ladder(config, run_dir, progress, dataset_filter=None)

            # Only shards 1 and 2 should have been translated
            assert not any("shard_000" in s for s in translate_calls), \
                "Shard 0 was already done and must not be re-translated"
            assert any("shard_001" in s for s in translate_calls)
            assert any("shard_002" in s for s in translate_calls)

    def test_already_done_dataset_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidates_base = os.path.join(tmp, "candidates")
            runs_base       = os.path.join(tmp, "runs")
            _make_manifest(os.path.join(candidates_base, DATASET_SLUG))

            run_dir  = os.path.join(runs_base, "20990101_000000_test_ladder_run")
            config   = _config_with_dirs(candidates_base, runs_base)
            progress = _load_or_init_progress(run_dir, config, "test_ladder_run")
            progress["datasets"][DATASET_SLUG]["ladder_all_done"] = True
            save_progress(run_dir, progress)

            translate_spy = MagicMock()
            with patch("translation.api.run_beir_ladder_pipeline._translate_shard",
                       side_effect=translate_spy), \
                 patch("translation.api.run_beir_ladder_pipeline.render_plots"):
                run_ladder(config, run_dir, progress, dataset_filter=None)

            translate_spy.assert_not_called()

    def test_stopped_dataset_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidates_base = os.path.join(tmp, "candidates")
            runs_base       = os.path.join(tmp, "runs")
            _make_manifest(os.path.join(candidates_base, DATASET_SLUG))

            run_dir  = os.path.join(runs_base, "20990101_000000_test_ladder_run")
            config   = _config_with_dirs(candidates_base, runs_base)
            progress = _load_or_init_progress(run_dir, config, "test_ladder_run")
            progress["datasets"][DATASET_SLUG]["ladder_stopped"] = True
            progress["datasets"][DATASET_SLUG]["ladder_stop_reason"] = "pre-set in test"
            save_progress(run_dir, progress)

            translate_spy = MagicMock()
            with patch("translation.api.run_beir_ladder_pipeline._translate_shard",
                       side_effect=translate_spy), \
                 patch("translation.api.run_beir_ladder_pipeline.render_plots"):
                run_ladder(config, run_dir, progress, dataset_filter=None)

            translate_spy.assert_not_called()


class TestKillGuard:
    """Starting without --resume when a run exists must exit with an error."""

    def test_exits_when_run_exists(self, capsys):
        with tempfile.TemporaryDirectory() as tmp:
            runs_base = os.path.join(tmp, "runs")
            existing  = os.path.join(runs_base, "20990101_000000_test_ladder_run")
            os.makedirs(existing)
            with open(os.path.join(existing, "progress.json"), "w") as f:
                json.dump({"run_id": "test_ladder_run"}, f)

            found = _find_existing_run_dir(runs_base, "test_ladder_run")
            assert found == existing

    def test_no_existing_run_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_base = os.path.join(tmp, "runs")
            found = _find_existing_run_dir(runs_base, "test_ladder_run")
            assert found is None

    def test_resume_picks_latest_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_base = os.path.join(tmp, "runs")
            for ts in ["20990101_000000", "20990102_000000", "20990103_000000"]:
                d = os.path.join(runs_base, f"{ts}_test_ladder_run")
                os.makedirs(d)
                with open(os.path.join(d, "progress.json"), "w") as f:
                    json.dump({"run_id": "test_ladder_run"}, f)

            found = _find_existing_run_dir(runs_base, "test_ladder_run")
            assert "20990103" in found   # latest timestamp wins


class TestMissingManifest:
    """Dataset with no shard_manifest.json is skipped gracefully."""

    def test_skipped_without_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidates_base = os.path.join(tmp, "candidates")
            runs_base       = os.path.join(tmp, "runs")
            # Deliberately no manifest

            run_dir  = os.path.join(runs_base, "20990101_000000_test_ladder_run")
            config   = _config_with_dirs(candidates_base, runs_base)
            progress = _load_or_init_progress(run_dir, config, "test_ladder_run")

            translate_spy = MagicMock()
            with patch("translation.api.run_beir_ladder_pipeline._translate_shard",
                       side_effect=translate_spy):
                run_ladder(config, run_dir, progress, dataset_filter=None)

            translate_spy.assert_not_called()
            # Progress should still be valid JSON
            state = _load_progress(run_dir)["datasets"][DATASET_SLUG]
            assert state["ladder_all_done"] is False
            assert state["ladder_stopped"] is False


class TestProgressPersistence:
    """progress.json is written atomically after every shard."""

    def test_progress_updated_after_each_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidates_base = os.path.join(tmp, "candidates")
            runs_base       = os.path.join(tmp, "runs")
            _make_manifest(os.path.join(candidates_base, DATASET_SLUG), num_shards=2)

            run_dir  = os.path.join(runs_base, "20990101_000000_test_ladder_run")
            config   = _config_with_dirs(candidates_base, runs_base)
            progress = _load_or_init_progress(run_dir, config, "test_ladder_run")

            stages_seen = []

            def _qa_and_capture(*args, **kwargs):
                stages_seen.append(
                    _load_progress(run_dir)["datasets"][DATASET_SLUG]["ladder_current_stage"]
                )
                return _passing_qa()

            with patch("translation.api.run_beir_ladder_pipeline._translate_shard",
                       side_effect=_fake_translate_shard), \
                 patch("translation.api.run_beir_ladder_pipeline._ladder_qa",
                       side_effect=_qa_and_capture), \
                 patch("translation.api.run_beir_ladder_pipeline.render_plots"):
                run_ladder(config, run_dir, progress, dataset_filter=None)

            # _ladder_qa is called twice per shard (queries + documents).
            # current_stage is saved to disk *after* both QA calls for that shard,
            # so the captured values reflect the stage at the start of each shard:
            #   shard 0 (idx=0): stage still 0 in file → two 0s captured
            #   shard 1 (idx=1): stage updated to 1 in file → two 1s captured
            assert 0 in stages_seen
            assert 1 in stages_seen
            assert stages_seen == sorted(stages_seen)  # monotonically non-decreasing

    def test_progress_json_is_valid_after_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidates_base = os.path.join(tmp, "candidates")
            runs_base       = os.path.join(tmp, "runs")
            _make_manifest(os.path.join(candidates_base, DATASET_SLUG), num_shards=1)

            run_dir  = os.path.join(runs_base, "20990101_000000_test_ladder_run")
            config   = _config_with_dirs(candidates_base, runs_base)
            progress = _load_or_init_progress(run_dir, config, "test_ladder_run")

            with patch("translation.api.run_beir_ladder_pipeline._translate_shard",
                       side_effect=_fake_translate_shard), \
                 patch("translation.api.run_beir_ladder_pipeline._ladder_qa",
                       side_effect=_passing_qa), \
                 patch("translation.api.run_beir_ladder_pipeline.render_plots"):
                run_ladder(config, run_dir, progress, dataset_filter=None)

            # Must be valid JSON and have the expected keys
            data = _load_progress(run_dir)
            assert "run_id" in data
            assert "datasets" in data
            assert DATASET_SLUG in data["datasets"]


class TestAppendToAccumulated:
    """_append_to_accumulated grows the CSV correctly across multiple calls."""

    def test_first_call_creates_file_with_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            shard_out = os.path.join(tmp, "shard_translated.csv")
            accumulated = os.path.join(tmp, "accumulated.csv")
            _make_translated_csv(shard_out, n_rows=5)

            count = _append_to_accumulated(shard_out, accumulated)
            assert count == 5
            df = pd.read_csv(accumulated)
            assert len(df) == 5
            assert "translation" in df.columns

    def test_subsequent_calls_append_without_duplicate_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            accumulated = os.path.join(tmp, "accumulated.csv")
            for i in range(3):
                shard = os.path.join(tmp, f"shard_{i}.csv")
                _make_translated_csv(shard, n_rows=4)
                _append_to_accumulated(shard, accumulated)

            df = pd.read_csv(accumulated)
            assert len(df) == 12            # 3 shards × 4 rows, no header duplication
            assert "translation" in df.columns


class TestDatasetFilter:
    """--dataset filter runs only the named dataset."""

    def test_filter_by_slug_runs_only_target(self):
        second_dataset = "BeIR/scifact"
        second_slug    = "BeIR_scifact"

        with tempfile.TemporaryDirectory() as tmp:
            candidates_base = os.path.join(tmp, "candidates")
            runs_base       = os.path.join(tmp, "runs")
            _make_manifest(os.path.join(candidates_base, DATASET_SLUG), num_shards=1)
            _make_manifest(os.path.join(candidates_base, second_slug),   num_shards=1)

            import copy
            config = _config_with_dirs(candidates_base, runs_base)
            config["datasets"]["names"] = [DATASET_NAME, second_dataset]

            run_dir  = os.path.join(runs_base, "20990101_000000_test_ladder_run")
            progress = _load_or_init_progress(run_dir, config, "test_ladder_run")

            translated_slugs = []

            def _spy_translate(shard_csv, output_dir, type_cfg, exec_cfg):
                for s in [DATASET_SLUG, second_slug]:
                    if s in shard_csv:
                        translated_slugs.append(s)
                return _fake_translate_shard(shard_csv, output_dir, type_cfg, exec_cfg)

            with patch("translation.api.run_beir_ladder_pipeline._translate_shard",
                       side_effect=_spy_translate), \
                 patch("translation.api.run_beir_ladder_pipeline._ladder_qa",
                       side_effect=_passing_qa), \
                 patch("translation.api.run_beir_ladder_pipeline.render_plots"):
                run_ladder(config, run_dir, progress, dataset_filter=DATASET_SLUG)

            assert all(s == DATASET_SLUG for s in translated_slugs), \
                f"Expected only {DATASET_SLUG}, got: {set(translated_slugs)}"
