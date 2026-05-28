from typing import List, Optional
import pandas as pd
import os
import json
from google import genai
from translation.api.utils import load_data, get_prompts, get_output_file, Translation

DEFAULT_TRACKING_DIR = "jobs"
JOB_TRACKING_FILE = "gemini_batch_jobs.json"

TERMINAL_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
FAILED_STATES = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    # Fall back to gcloud Application Default Credentials (run: gcloud auth application-default login)
    return genai.Client(
        vertexai=True,
        project=os.environ["GEMINI_PROJECT"],
        location=os.environ.get("GEMINI_LOCATION", "us-central1"),
    )


def _tracking_file(tracking_dir: str) -> str:
    return os.path.join(tracking_dir, JOB_TRACKING_FILE)


def _build_inline_request(system_prompt: str, user_prompt: str,
                           temperature: float, response_format) -> dict:
    generation_config = {"temperature": temperature}
    if response_format is not None:
        generation_config["response_mime_type"] = "application/json"
        generation_config["response_schema"] = response_format.model_json_schema()
    return {
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "generation_config": generation_config,
    }


def translate(system_prompts: List[str],
              user_prompts: List[str],
              model_name: str,
              temperature: float = 0.0,
              response_format=Translation) -> dict:
    client = _get_client()

    requests = [
        _build_inline_request(sp, up, temperature, response_format)
        for sp, up in zip(system_prompts, user_prompts)
    ]

    batch_job = client.batches.create(
        model=model_name,
        src=requests,
        config={"display_name": f"translation-{model_name}"},
    )

    return {
        "job_name": batch_job.name,
        "status": str(batch_job.state),
        "model_name": model_name,
    }


def run_translation_pipeline(source_file_path: str,
                              prompt_file_name: str,
                              model_name: str,
                              output_dir: str,
                              limit: int = 0,
                              force: bool = False,
                              tracking_dir: str = DEFAULT_TRACKING_DIR,
                              **kwargs) -> Optional[str]:
    """Submit a Gemini batch translation job. Returns the job name, or None if nothing to translate."""
    translation_output_file_path = get_output_file(
        source_file_path, output_dir, **kwargs
    ).replace(".csv", "_translated.csv")

    file_path = (translation_output_file_path
                 if os.path.exists(translation_output_file_path)
                 else source_file_path)
    filtered_df = load_data(file_path, limit, force, ignore_populated_column="translation")
    if filtered_df is None or filtered_df.empty:
        return None

    id_columns = ["id"]
    if "segment_id" in filtered_df.columns:
        id_columns.append("segment_id")

    prompt_type = "query" if source_file_path.endswith("queries.csv") else "document"
    batch_data = get_prompts(prompt_file_name, prompt_type, filtered_df, id_columns, **kwargs)

    response_format = kwargs.get("response_format", Translation)

    job_metadata = translate(
        system_prompts=[item["system_prompt"] for item in batch_data],
        user_prompts=[item["user_prompt"] for item in batch_data],
        model_name=model_name,
        response_format=response_format,
    )

    job_metadata.update({
        "row_indices": filtered_df[id_columns].to_dict(orient="records"),
        "source_file_path": source_file_path,
        "translation_output_file_path": translation_output_file_path,
    })

    _save_job(job_metadata, tracking_dir)
    return job_metadata["job_name"]


def check_batch_status(job_names: Optional[List[str]] = None,
                       tracking_dir: str = DEFAULT_TRACKING_DIR) -> list:
    """Refresh status for all (or a specific subset of) tracked Gemini batch jobs."""
    client = _get_client()
    jobs_metadata = _load_jobs_metadata(tracking_dir)

    if job_names is not None:
        jobs_metadata = [j for j in jobs_metadata if j.get("job_name") in job_names]

    for job_metadata in jobs_metadata:
        batch_job = client.batches.get(name=job_metadata["job_name"])
        job_metadata["status"] = str(batch_job.state)
        print(f"Job {job_metadata['job_name']} Status: {job_metadata['status']}")

    _flush_jobs_metadata(jobs_metadata, tracking_dir)
    return jobs_metadata


def retrieve_batch_results(job_names: Optional[List[str]] = None,
                           tracking_dir: str = DEFAULT_TRACKING_DIR) -> None:
    """Download and write results for all succeeded (or specified) Gemini batch jobs."""
    client = _get_client()
    jobs_metadata = check_batch_status(job_names=job_names, tracking_dir=tracking_dir)

    for job_metadata in jobs_metadata:
        if job_metadata["status"] not in ("JOB_STATE_SUCCEEDED",):
            print(f"Skipping job {job_metadata['job_name']} (status: {job_metadata['status']})")
            continue

        batch_job = client.batches.get(name=job_metadata["job_name"])
        inlined = list(batch_job.inlined_responses)

        translation_output_file_path = job_metadata["translation_output_file_path"]
        source_file_path = job_metadata["source_file_path"]

        if os.path.exists(translation_output_file_path):
            df = pd.read_csv(translation_output_file_path, encoding="utf-8")
        else:
            df = pd.read_csv(source_file_path, encoding="utf-8")

        row_indices = job_metadata["row_indices"]
        if len(inlined) != len(row_indices):
            print(f"Warning: result count mismatch for {job_metadata['job_name']} "
                  f"({len(inlined)} vs {len(row_indices)} expected)")
            continue

        for row_idx_dict, inlined_response in zip(row_indices, inlined):
            try:
                text = inlined_response.response.candidates[0].content.parts[0].text
                parsed = json.loads(text)
                translation = parsed.get("translation", text)
            except Exception:
                translation = ""

            mask = pd.Series([True] * len(df))
            for col, val in row_idx_dict.items():
                if col in df.columns:
                    mask &= df[col] == val
            df.loc[mask, "translation"] = translation

        os.makedirs(os.path.dirname(translation_output_file_path) or ".", exist_ok=True)
        df.to_csv(translation_output_file_path, encoding="utf-8", index=False)
        print(f"Saved results to {translation_output_file_path}")


def _save_job(job_metadata: dict, tracking_dir: str = DEFAULT_TRACKING_DIR) -> None:
    os.makedirs(tracking_dir, exist_ok=True)
    with open(_tracking_file(tracking_dir), "a") as f:
        f.write(json.dumps(job_metadata) + "\n")
    print(f"Created Gemini batch job: {job_metadata['job_name']}")


def _load_jobs_metadata(tracking_dir: str = DEFAULT_TRACKING_DIR) -> list:
    try:
        with open(_tracking_file(tracking_dir), "r") as f:
            return [json.loads(line) for line in f if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _flush_jobs_metadata(jobs_metadata: list, tracking_dir: str = DEFAULT_TRACKING_DIR) -> None:
    """Overwrite the tracking file with updated job metadata."""
    os.makedirs(tracking_dir, exist_ok=True)
    with open(_tracking_file(tracking_dir), "w") as f:
        for job_metadata in jobs_metadata:
            f.write(json.dumps(job_metadata) + "\n")
