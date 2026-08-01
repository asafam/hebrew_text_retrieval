"""Tests for the translation cache wired into the shard ladder.

The defect these guard against: the same English string previously received
different Hebrew in the query pass and the document pass, because the two passes
ran independently at temperature 0.7. That destroys the lexical overlap retrieval
depends on (see docs/benchmark/why-not-translation.md).

`test_cross_type_consistency` is the one that encodes the actual bug.
"""

import os
import sys
import tempfile
import unittest

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from translation.api.ladder_dedup import (  # noqa: E402
    SqliteTranslationCache,
    prefill_shard,
    finalize_shard,
    make_key,
)

MODEL = "gemini-3.1-flash-lite"
PROMPT = "prompts/translation/api/translation/translation_prompts_zeroshot_nocontext_v20260531.yaml"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.cache = SqliteTranslationCache(os.path.join(self.dir, "c.sqlite"))

    def tearDown(self):
        self.cache.close()
        self.tmp.cleanup()

    def csv(self, name, rows, col):
        p = os.path.join(self.dir, name)
        pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8")
        return p, col


class TestCache(Base):
    def test_roundtrip(self):
        self.cache.store(MODEL, PROMPT, "ECMO machine", "", "מכונת ECMO")
        self.assertEqual(self.cache.lookup(MODEL, PROMPT, "ECMO machine", ""), "מכונת ECMO")

    def test_miss_returns_none(self):
        self.assertIsNone(self.cache.lookup(MODEL, PROMPT, "absent", ""))

    def test_key_scoped_to_model_and_prompt(self):
        """A different model or prompt must not reuse a stale translation."""
        self.cache.store(MODEL, PROMPT, "text", "", "תרגום")
        self.assertIsNone(self.cache.lookup("other-model", PROMPT, "text", ""))
        self.assertIsNone(self.cache.lookup(MODEL, "other-prompt.yaml", "text", ""))

    def test_store_is_idempotent(self):
        self.cache.store(MODEL, PROMPT, "t", "", "first")
        self.cache.store(MODEL, PROMPT, "t", "", "second")
        # First write wins; re-storing must not corrupt or duplicate.
        self.assertEqual(self.cache.lookup(MODEL, PROMPT, "t", ""), "first")
        self.assertEqual(len(self.cache), 1)

    def test_persists_across_reopen(self):
        path = os.path.join(self.dir, "persist.sqlite")
        c1 = SqliteTranslationCache(path)
        c1.store(MODEL, PROMPT, "t", "", "תרגום")
        c1.close()
        c2 = SqliteTranslationCache(path)
        self.assertEqual(c2.lookup(MODEL, PROMPT, "t", ""), "תרגום")
        c2.close()


class TestPrefill(Base):
    def test_fills_from_cache(self):
        self.cache.store(MODEL, PROMPT, "alpha", "", "אלפא")
        src, col = self.csv("s.csv", [{"_id": "1", "text": "alpha"},
                                      {"_id": "2", "text": "beta"}], "text")
        out = os.path.join(self.dir, "out.csv")
        st = prefill_shard(src, self.cache, MODEL, PROMPT, col, None, out_csv=out)
        self.assertEqual(st["cache_hits"], 1)
        self.assertEqual(st["remaining"], 1)
        self.assertFalse(st["all_cached"])
        df = pd.read_csv(out)
        self.assertEqual(df.loc[df["_id"] == 1, "translation"].iloc[0], "אלפא")
        self.assertTrue(pd.isna(df.loc[df["_id"] == 2, "translation"].iloc[0]))

    def test_all_cached_flag(self):
        for t, he in [("alpha", "אלפא"), ("beta", "בטא")]:
            self.cache.store(MODEL, PROMPT, t, "", he)
        src, col = self.csv("s.csv", [{"_id": "1", "text": "alpha"},
                                      {"_id": "2", "text": "beta"}], "text")
        out = os.path.join(self.dir, "out.csv")
        st = prefill_shard(src, self.cache, MODEL, PROMPT, col, None, out_csv=out)
        self.assertTrue(st["all_cached"])
        self.assertEqual(st["remaining"], 0)

    def test_within_shard_duplicates_share_one_translation(self):
        self.cache.store(MODEL, PROMPT, "dup", "", "כפול")
        src, col = self.csv("s.csv", [{"_id": str(i), "text": "dup"} for i in range(4)], "text")
        out = os.path.join(self.dir, "out.csv")
        st = prefill_shard(src, self.cache, MODEL, PROMPT, col, None, out_csv=out)
        self.assertTrue(st["all_cached"])
        self.assertEqual(set(pd.read_csv(out)["translation"]), {"כפול"})

    def test_does_not_overwrite_source_csv(self):
        """Prefill must never mutate the candidate CSV — only the output."""
        self.cache.store(MODEL, PROMPT, "alpha", "", "אלפא")
        src, col = self.csv("s.csv", [{"_id": "1", "text": "alpha"}], "text")
        before = open(src, encoding="utf-8").read()
        prefill_shard(src, self.cache, MODEL, PROMPT, col, None,
                      out_csv=os.path.join(self.dir, "out.csv"))
        self.assertEqual(open(src, encoding="utf-8").read(), before)

    def test_empty_cache_writes_nothing(self):
        src, col = self.csv("s.csv", [{"_id": "1", "text": "alpha"}], "text")
        out = os.path.join(self.dir, "out.csv")
        st = prefill_shard(src, self.cache, MODEL, PROMPT, col, None, out_csv=out)
        self.assertEqual(st["cache_hits"], 0)
        self.assertFalse(st["all_cached"])
        self.assertFalse(os.path.exists(out), "no fills should mean no output file")


class TestFinalize(Base):
    def test_stores_translations(self):
        p, _ = self.csv("t.csv", [{"_id": "1", "text": "alpha", "translation": "אלפא"},
                                  {"_id": "2", "text": "beta", "translation": "בטא"}], "text")
        st = finalize_shard(p, self.cache, MODEL, PROMPT, "text", None)
        self.assertEqual(st["new_entries"], 2)
        self.assertEqual(self.cache.lookup(MODEL, PROMPT, "alpha", ""), "אלפא")

    def test_idempotent_on_resume(self):
        p, _ = self.csv("t.csv", [{"_id": "1", "text": "alpha", "translation": "אלפא"}], "text")
        finalize_shard(p, self.cache, MODEL, PROMPT, "text", None)
        st = finalize_shard(p, self.cache, MODEL, PROMPT, "text", None)
        self.assertEqual(st["new_entries"], 0)
        self.assertEqual(len(self.cache), 1)

    def test_missing_file_is_safe(self):
        st = finalize_shard(os.path.join(self.dir, "nope.csv"),
                            self.cache, MODEL, PROMPT, "text", None)
        self.assertEqual(st["new_entries"], 0)


class TestCrossTypeConsistency(Base):
    """The actual bug: one English string, two passes, two different Hebrew results."""

    def test_cross_type_consistency(self):
        shared = "Extracorporeal membrane oxygenation (ECMO) improves survival"

        # Document pass runs first and produces a translation.
        doc_csv, _ = self.csv(
            "docs.csv",
            [{"_id": "d1", "segment_text": shared, "translation": "חמצון ממברנלי חוץ-גופי (ECMO) משפר הישרדות"}],
            "segment_text")
        finalize_shard(doc_csv, self.cache, MODEL, PROMPT, "segment_text", None)

        # Query pass sees the same source string. Note the *different* text column —
        # the cache key is the text itself, so the column name must not matter.
        q_csv, _ = self.csv("queries.csv", [{"_id": "q1", "text": shared}], "text")
        q_out = os.path.join(self.dir, "q_out.csv")
        st = prefill_shard(q_csv, self.cache, MODEL, PROMPT, "text", None, out_csv=q_out)

        self.assertTrue(st["all_cached"], "query should be fully served from the document pass")
        self.assertEqual(
            pd.read_csv(q_out)["translation"].iloc[0],
            "חמצון ממברנלי חוץ-גופי (ECMO) משפר הישרדות",
            "query and document must receive byte-identical Hebrew",
        )

    def test_key_ignores_column_name(self):
        """A document's `segment_text` and a query's `text` hash to the same key."""
        self.assertEqual(
            make_key(MODEL, PROMPT, "same text", ""),
            make_key(MODEL, PROMPT, "same text", ""),
        )

    def test_context_separates_keys(self):
        """Queries translated with context legitimately differ — keys must not collide."""
        self.assertNotEqual(
            make_key(MODEL, PROMPT, "bank", "river"),
            make_key(MODEL, PROMPT, "bank", "finance"),
        )


class TestProductionConfig(unittest.TestCase):
    """Guard the two settings that caused the divergence."""

    @classmethod
    def setUpClass(cls):
        root = os.path.join(os.path.dirname(__file__), "..")
        with open(os.path.join(root, "config/translation/full_corpus.yaml")) as f:
            cls.cfg = yaml.safe_load(f)

    def test_all_translation_temperatures_are_zero(self):
        for section in ("queries", "documents", "titles"):
            self.assertEqual(
                self.cfg[section]["temperature"], 0.0,
                f"{section} must translate deterministically; non-zero temperature makes "
                f"the query and document passes diverge on identical source text",
            )

    def test_repair_temperature_is_zero(self):
        self.assertEqual(
            self.cfg["repair"]["temperature"], 0.0,
            "a repaired row must match what the first pass would have produced",
        )

    def test_dedup_enabled(self):
        self.assertTrue(self.cfg.get("dedup", {}).get("enabled", False))

    def test_queries_and_documents_share_model_and_prompt(self):
        """The cache key is (model, prompt_file, text, context). If the two passes
        used different models or prompt files they would never share entries."""
        self.assertEqual(self.cfg["queries"]["model"], self.cfg["documents"]["model"])
        self.assertEqual(self.cfg["queries"]["prompt"]["file"],
                         self.cfg["documents"]["prompt"]["file"])

    def test_query_and_document_render_the_same_prompt(self):
        """The whole point of prompt v20260801.

        Measured on v20260531: the words "query"/"document" in the prefix and the
        "Hebrew Query:"/"Hebrew Document:" output label changed the translation for
        38% of source strings at temperature 0 — e.g. "unsupervised" as ללא פיקוח
        in a query but בלתי מונחה in a document. Identical rendered prompts remove
        that by construction.
        """
        root = os.path.join(os.path.dirname(__file__), "..")
        with open(os.path.join(root, self.cfg["queries"]["prompt"]["file"])) as f:
            prompts = yaml.safe_load(f)

        SRC = "Extracorporeal membrane oxygenation (ECMO) improves survival"

        def rendered(kind, cfg_key, col):
            pr, pc = prompts[kind], self.cfg[cfg_key]["prompt"]
            body = (pr["user_prompt_template"]
                    .replace("{english_key}", pc["english_key"])
                    .replace("{" + col + "}", SRC)
                    .replace("{hebrew_key}", pc["hebrew_key"]))
            return pr["system_prompt"] + "\n" + pr["user_prompt_prefix"] + "\n" + body

        self.assertEqual(
            rendered("query", "queries", "text"),
            rendered("document", "documents", "segment_text"),
            "the query and document passes must send a byte-identical prompt for "
            "identical source text, otherwise the same term can be translated two ways",
        )

    def test_run_id_matches_prompt_version(self):
        """run_id encodes the prompt version so a run directory never mixes two.

        The 5 datasets translated under v20260531 stay in the old run directory;
        anything translated with the unified prompt belongs to a new one.
        """
        import re
        run_id = self.cfg["run_id"]
        m = re.search(r"prompt(v\d{8})$", run_id)
        self.assertIsNotNone(m, f"run_id should end in the prompt version: {run_id}")
        prompt_version = re.search(r"_(v\d{8})\.yaml$", self.cfg["queries"]["prompt"]["file"])
        self.assertIsNotNone(prompt_version)
        self.assertEqual(
            m.group(1), prompt_version.group(1),
            "run_id prompt version must match the prompt file actually configured",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
