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
    return get_job_info(gemini_client, job_name)["state"]


def get_job_info(gemini_client, job_name: str) -> dict:
    """Fetch a batch job's full status info.

    Returns a dict with: state, create_time, start_time, end_time,
    successful_count, failed_count, error_message. Fields are None when the
    Vertex response hasn't populated them yet (e.g. start_time is None while
    the job is still queued).
    """
    job = gemini_client.batches.get(name=job_name)

    def _attr(obj, *names):
        for n in names:
            v = getattr(obj, n, None)
            if v is not None:
                return v
        return None

    def _as_int(v):
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    # google-genai returns a JobState enum. Different SDK versions expose the
    # enum differently — .name might already be "JOB_STATE_SUCCEEDED" or just
    # "SUCCEEDED", and str() gives "JobState.SUCCEEDED". Normalize to the
    # canonical "JOB_STATE_*" form the rest of the codebase compares against.
    state_raw = _attr(job, "state")
    if hasattr(state_raw, "name"):
        name = state_raw.name
    elif state_raw is not None:
        name = str(state_raw).rsplit(".", 1)[-1]
    else:
        name = ""
    state_str = name if name.startswith("JOB_STATE_") else f"JOB_STATE_{name}"

    info = {
        "state":            state_str,
        "create_time":      _attr(job, "create_time", "createTime"),
        "start_time":       _attr(job, "start_time", "startTime"),
        "end_time":         _attr(job, "end_time", "endTime"),
        "successful_count": None,
        "failed_count":     None,
        "error_message":    None,
        "output_dir":       None,
    }
    stats = _attr(job, "completion_stats", "completionStats")
    if stats is not None:
        info["successful_count"] = _as_int(_attr(stats, "successful_count", "successfulCount"))
        info["failed_count"]     = _as_int(_attr(stats, "failed_count", "failedCount"))
    err = _attr(job, "error")
    if err is not None:
        info["error_message"] = _attr(err, "message")
    # The precise output directory for THIS job (e.g. .../output/prediction-model-<ts>/).
    # Crucial when multiple jobs share the parent gcs_output_prefix — without this,
    # _download_shard_results would mix predictions from orphaned earlier jobs.
    output_info = _attr(job, "output_info", "outputInfo")
    if output_info is not None:
        info["output_dir"] = _attr(output_info, "gcs_output_directory", "gcsOutputDirectory")
    return info


# ── Result download ───────────────────────────────────────────────────────────

def download_and_parse_results(
    gcs_client: storage.Client,
    bucket: str,
    gcs_output_prefix: str,
) -> list:
    """
    Download prediction shards from GCS and return translations tagged with their _id.

    Vertex AI does NOT guarantee that output order matches input order — predictions
    must be matched back to source rows via the composite _id embedded in each line.

    gcs_output_prefix: path WITHOUT gs:// prefix (e.g. 'beir/run_id/slug/queries/output').

    Returns list of {"_id": composite_id, "translation": str}, in JSONL order.
    Pass to write_translated_csv, which now matches by _id.
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
            obj = json.loads(raw_line)
            composite_id = obj.get("_id", "")
            try:
                text = obj["response"]["candidates"][0]["content"]["parts"][0]["text"]
                translation = json.loads(text).get("translation", text)
            except Exception:
                translation = ""
            results.append({"_id": composite_id, "translation": translation})

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

    pending_mask = df["translation"].isna()
    pending_indices = df.index[pending_mask].tolist()

    if len(translations) != len(pending_indices):
        raise ValueError(
            f"Result count mismatch: received {len(translations)} translations "
            f"but {len(pending_indices)} rows are still pending in "
            f"{output_csv if os.path.exists(output_csv) else source_csv}. "
            f"The file may have been modified between submit and collect."
        )

    # Match by composite _id when the result objects carry one — Vertex doesn't
    # guarantee output order matches input order. Fall back to positional zip
    # only when no _id is present (sync / inline path).
    have_ids = translations and all(t.get("_id") for t in translations)
    if have_ids:
        # Composite id is "<_id>" for queries, "<_id>__<segment_id>" for docs.
        df_composite = df["_id"].astype(str)
        if "segment_id" in df.columns:
            df_composite = df_composite + "__" + df["segment_id"].astype(str)
        id_to_idx = {cid: idx for idx, cid in df_composite.items() if pending_mask.loc[idx]}
        unmatched = []
        for t in translations:
            idx = id_to_idx.get(str(t["_id"]))
            if idx is None:
                unmatched.append(t["_id"])
                continue
            df.at[idx, "translation"] = t.get("translation", "")
        if unmatched:
            raise ValueError(
                f"{len(unmatched)} prediction _id(s) had no matching pending row in "
                f"{output_csv if os.path.exists(output_csv) else source_csv}. "
                f"First few unmatched: {unmatched[:5]}"
            )
    else:
        for idx, t in zip(pending_indices, translations):
            df.at[idx, "translation"] = t.get("translation", "")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"  Wrote {len(translations):,} translations → {output_csv}")
