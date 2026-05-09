"""
Translation cache for BeIR pipeline.

Maps (model_name, prompt_file, text, context) → translation, persisted as an
append-only JSONL file. Enables two layers of deduplication:

  1. Cross-run / cross-dataset: if segment_text X was already translated for
     dataset A (or a previous run), skip re-translating it for dataset B.

  2. Within-file: multiple document segments with identical text (common in
     large corpora like MS MARCO) are translated once and results propagated.

Cache keys are scoped to (model_name, prompt_file) so switching models or
prompts never reuses a stale translation.

For queries, context_text is included in the key because the same query text
can legitimately produce different Hebrew translations depending on document
context. Passing context="" treats the query as context-free.

Scale note: for very large datasets (millions of entries) a SQLite backend
would be more efficient. The JSONL format works well up to ~500K entries.
"""

import hashlib
import json
import os
from typing import Optional


class TranslationCache:
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self._cache: dict[str, str] = {}
        self._load()

    def lookup(
        self,
        model_name: str,
        prompt_file: str,
        text: str,
        context: str = "",
    ) -> Optional[str]:
        key = self._make_key(model_name, prompt_file, text, context)
        return self._cache.get(key)

    def store(
        self,
        model_name: str,
        prompt_file: str,
        text: str,
        context: str,
        translation: str,
    ) -> None:
        key = self._make_key(model_name, prompt_file, text, context)
        if key in self._cache:
            return
        self._cache[key] = translation
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "translation": translation}, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._cache)

    def _load(self) -> None:
        if not os.path.exists(self.cache_path):
            return
        with open(self.cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        self._cache[entry["key"]] = entry["translation"]
                    except (json.JSONDecodeError, KeyError):
                        pass  # Skip malformed lines

    @staticmethod
    def _make_key(model_name: str, prompt_file: str, text: str, context: str) -> str:
        content = f"{model_name}\x00{prompt_file}\x00{text}\x00{context}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
