"""
GCS-backed Gemini Batch API primitives.

Handles building/uploading input JSONL to GCS, submitting Vertex AI batch jobs
with GCS src/dest, and downloading/parsing prediction shards from GCS output.

Requires Vertex AI credentials (gcloud auth application-default login).
GEMINI_API_KEY must be unset; this module uses ADC only.
"""

from typing import Optional
import io
import json
import os

import pandas as pd
from google.cloud import storage

from translation.api.translate_batch_gemini import _get_client, _build_inline_request
from translation.api.utils import get_prompts, Translation


# ── GCS client ────────────────────────────────────────────────────────────────

def get_gcs_client(project: Optional[str] = None) -> storage.Client:
    """Create a GCS client using Application Default Credentials."""
    return storage.Client(project=project or os.environ.get("GEMINI_PROJECT"))


def _strip_gs_uri(uri: str) -> tuple:
    """Parse 'gs://bucket/path' → (bucket, path)."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    parts = uri[5:].split("/", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


# ── Input JSONL ───────────────────────────────────────────────────────────────

def build_and_upload_input(
    df: pd.DataFrame,
    id_columns: list,
    prompt_file: str,
    prompt_type: str,
    model_name: str,
    temperature: float,
    gcs_client: storage.Client,
    bucket: str,
    gcs_prefix: str,
    **kwargs,
) -> str:
    """
    Build JSONL from pending rows, upload to GCS, return gs:// URI.

    df must be pre-filtered to only pending rows (translation is NaN).
    Each line: {"_id": "<composite_id>", "request": {<inline_request>}}

    The _id embeds all id_columns joined by '__' so we can verify ordering
    at collect time by re-reading the input JSONL from GCS.
    """
    batch_data = get_prompts(prompt_file, prompt_type, df, id_columns, **kwargs)

    buffer = io.BytesIO()
    for item in batch_data:
        composite_id = "__".join(str(item[col]) for col in id_columns if col in item)
        request = _build_inline_request(
            item["system_prompt"],
            item["user_prompt"],
            temperature,
            Translation,
        )
        line = json.dumps({"_id": composite_id, "request": request}, ensure_ascii=False)
        buffer.write((line + "\n").encode("utf-8"))

    buffer.seek(0)
    blob_path = f"{gcs_prefix}/input.jsonl"
    gcs_client.bucket(bucket).blob(blob_path).upload_from_file(
        buffer, content_type="application/jsonl"
    )

    uri = f"gs://{bucket}/{blob_path}"
    print(f"  Uploaded {len(batch_data):,} requests → {uri}")
    return uri


# ── Batch job submission ──────────────────────────────────────────────────────

def submit_gcs_batch_job(
    gemini_client,
    model_name: str,
    input_uri: str,
    output_prefix: str,
    display_name: str,
) -> str:
    """
    Submit a Vertex AI Gemini batch job with GCS src/dest.
    Returns the fully-qualified job name (e.g. 'projects/.../batchPredictionJobs/...').
    """
    batch_job = gemini_client.batches.create(
        model=model_name,
        src=input_uri,
        config={
            "display_name": display_name,
            "dest": output_prefix,
        },
    )
    print(f"  Submitted batch job: {batch_job.name}")
    return batch_job.name


def check_job_status(gemini_client, job_name: str) -> str:
    """Returns the current state string for a batch job (e.g. 'JOB_STATE_RUNNING')."""
    return str(gemini_client.batches.get(name=job_name).state)


# ── Result download ───────────────────────────────────────────────────────────

def download_and_parse_results(
    gcs_client: storage.Client,
    bucket: str,
    gcs_output_prefix: str,
) -> list:
    """
    Download prediction shards from GCS and return translations in input order.

    Vertex AI writes shards named prediction-*.jsonl under gcs_output_prefix.
    Responses are 1:1 ordered with the input JSONL — we rely on this guarantee.

    gcs_output_prefix: path WITHOUT gs:// prefix (e.g. 'beir/run_id/slug/queries/output').

    Returns list of {"translation": str}, one per input row.
    """
    bucket_obj = gcs_client.bucket(bucket)
    blobs = sorted(
        [
            b for b in bucket_obj.list_blobs(prefix=gcs_output_prefix)
            if "prediction" in os.path.basename(b.name) and b.name.endswith(".jsonl")
        ],
        key=lambda b: b.name,
    )

    if not blobs:
        raise RuntimeError(
            f"No prediction shards found at gs://{bucket}/{gcs_output_prefix}. "
            f"Job may still be running or output prefix is wrong."
        )

    results = []
    for blob in blobs:
        for raw_line in blob.download_as_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            try:
                obj = json.loads(raw_line)
                text = obj["response"]["candidates"][0]["content"]["parts"][0]["text"]
                translation = json.loads(text).get("translation", text)
            except Exception:
                translation = ""
            results.append({"translation": translation})

    return results


# ── CSV write-back ────────────────────────────────────────────────────────────

def write_translated_csv(
    translations: list,
    source_csv: str,
    output_csv: str,
) -> None:
    """
    Write translations back into output_csv by matching pending rows positionally.

    Pending rows are those where 'translation' is NaN — the same filter applied
    at submit time via load_data(..., ignore_populated_column='translation').
    The contract: nothing writes to output_csv between submit and collect.

    Raises ValueError on row-count mismatch (CSV was modified, or wrong output prefix).
    """
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv, encoding="utf-8")
    else:
        df = pd.read_csv(source_csv, encoding="utf-8")

    if "translation" not in df.columns:
        df["translation"] = None

    pending_indices = df.index[df["translation"].isna()].tolist()

    if len(translations) != len(pending_indices):
        raise ValueError(
            f"Result count mismatch: received {len(translations)} translations "
            f"but {len(pending_indices)} rows are still pending in "
            f"{output_csv if os.path.exists(output_csv) else source_csv}. "
            f"The file may have been modified between submit and collect."
        )

    for idx, t in zip(pending_indices, translations):
        df.at[idx, "translation"] = t.get("translation", "")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"  Wrote {len(translations):,} translations → {output_csv}")
