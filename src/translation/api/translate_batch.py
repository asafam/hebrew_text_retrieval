from typing import List 
import pandas as pd
import os
import io
import json
from translation.api.utils import *


JOB_TRACKING_PATH = "jobs"
JOB_TRACKING_FILE = "batch_jobs.json"
JOB_TRACKING_FILE_MD = "batch_jobs.json"
    

def translate(system_prompts: List[str],
              user_prompts: List[str],
              model_name: str,
              temperature=0.0,
              response_format=Translation):
    # Load the OpenAI client
    client = get_openai_client()

    # Define the messages
    batch_input_data = []
    for i, (system_prompt, user_prompt) in enumerate(system_prompts, user_prompts):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        batch_input_data_item = dict(
            custom_id=f"request-{i}",
            method="POST",
            url="/v1/chat/completions",
            body=dict(
                model=model_name,
                messages=messages,
                temperature=temperature,
                response_format=response_format
            )
        )
        batch_input_data.append(batch_input_data_item)

    # Convert JSON list to JSONL format
    jsonl_data = "\n".join(json.dumps(obj) for obj in batch_input_data)

    # Convert to bytes and create an in-memory file-like object
    batch_input_stream = io.BytesIO(jsonl_data.encode('utf-8'))

    # Run chat completion inference
    file_obj = client.files.create(file=batch_input_stream, purpose="batch")
    batch_job = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
        }
    )

    # Store job tracking info
    job_metadata = {
        "job_id": batch_job.id,
        "file_id": file_obj.id,
        "status": batch_job.status,
        "created_at": batch_job.created_at,
        "model_name": model_name,
    }

    return job_metadata


def run_translation_pipeline(source_file_path: str,
                             prompt_file_name: str,
                             model_name: str,
                             limit: int = 0,
                             force: bool = False,
                             **kwargs):
    # Determine the output file path
    translation_output_file_path = get_output_file(source_file_path, 
                                                   model_name.replace("/", "_"),
                                                   **kwargs)

    # Load the data
    file_path = translation_output_file_path if os.path.exists(translation_output_file_path) else source_file_path
    filtered_df = load_data(file_path, limit, force, ignore_populated_column='translation')

   # Get the ID columns
    id_columns = ['id']
    if 'segment_id' in filtered_df.columns:
        id_columns.append('segment_id')

    # Get the batch data
    prompt_type = 'query' if source_file_path.endswith('queries.csv') else 'document'
    batch_data = get_prompts(prompt_file_name, prompt_type, filtered_df, id_columns, **kwargs)
    
    # Define the response format
    response_format = kwargs.get('response_format', Translation)

    # Translate batch
    job_metadata = translate(system_prompts=[item['system_prompt'] for item in batch_data], 
                             user_prompts=[item['user_prompt'] for item in batch_data],
                             model_name=model_name,
                             response_format=response_format)
    
    job_metadata = {
        **job_metadata,
        'row_indices': filtered_df[id_columns].tolist(),
        'source_file_path': source_file_path,
        'translation_output_file_path': translation_output_file_path
    }

    # Save job metadata
    _save_job(job_metadata)
    

def check_batch_status(to_md=True):
    jobs_metadata = _load_jobs_metadata()

    for job_metadata in jobs_metadata:
        client = get_openai_client()
        batch_job = client.batches.retrieve(job_metadata["job_id"])
        job_metadata["status"] = batch_job.status
        print(f"Job {job_metadata['job_id']} Status: {job_metadata['status']}")

    # Overwrite the file with updated statuses
    jobs_tracking_file_path = os.path.join(JOB_TRACKING_PATH, JOB_TRACKING_FILE)
    with open(jobs_tracking_file_path, "w") as f:
        for job_metadata in jobs_metadata:
            f.write(json.dumps(job_metadata, indent=2) + "\n")
    
    if to_md:
        jobs_tracking_file_path_md = os.path.join(JOB_TRACKING_PATH, JOB_TRACKING_FILE_MD)
        with open(jobs_tracking_file_path_md, "w") as f:
            f.write("# OpenAI Batch Jobs Status\n\n")
            for job_metadata in jobs_metadata:
                f.write(f"""
### Batch Job: {job_metadata['job_id']}
- **File ID:**      {job_metadata['file_id']}
- **Status:**       {job_metadata['status']}
- **Created At:**   {job_metadata['created_at']}
- **Model Name:**   {job_metadata['model_name']}
- **Source File:**  {job_metadata['source_file_path']}
- **Output File:**  {job_metadata['translation_output_file_path']}

---
"""
                )
    return jobs_metadata

    
def retrieve_batch_results():
    # Check the status of all jobs
    jobs_metadata = check_batch_status()

    # Retrieve results for completed jobs
    for job_metadata in jobs_metadata:
        if job_metadata["status"] == "completed":
            # Fetch results from OpenAI
            client = get_openai_client()
            results = client.files.content(job_metadata["file_id"]).text

            # Load original DataFrame
            translation_output_file_path = job_metadata["translation_output_file_path"]
            df = pd.read_csv(translation_output_file_path, encoding='utf-8')

            # Convert results back into a dictionary
            results_list = [json.loads(line) for line in results.split("\n") if line.strip()]

            # Ensure row indices match result count
            if len(results_list) != len(job_metadata["row_indices"]):
                print(f"Warning: Mismatch in result count for Job {job_metadata['job_id']}")
                continue

            # Update the original DataFrame
            for idx, result in zip(job_metadata["row_indices"], results_list):
                df.loc[idx, "translation"] = result.get("generated_text", "")
            
            # Save the updated DataFrame back to the CSV
            os.makedirs(os.path.dirname(translation_output_file_path), exist_ok=True)
            df.to_csv(translation_output_file_path, encoding='utf-8', index=False)


def _save_job(job_metadata):
    jobs_tracking_file_path = os.path.join(JOB_TRACKING_PATH, JOB_TRACKING_FILE)
    with open(jobs_tracking_file_path, "a") as f:
        f.write(json.dumps(job_metadata, indent=2) + "\n")
    print(f"Created batch job: {job_metadata['job_id']}")


def _load_jobs_metadata():
    try:
        jobs_tracking_file_path = os.path.join(JOB_TRACKING_PATH, JOB_TRACKING_FILE)
        with open(jobs_tracking_file_path, "r") as f:
            return [json.loads(line) for line in f]
    except (FileNotFoundError, json.JSONDecodeError):
        return []  # No previous jobs found