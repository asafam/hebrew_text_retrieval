import json
import os

from google import genai
from google.genai import types

_REQUEST_TIMEOUT = int(os.environ.get("GEMINI_TIMEOUT", "300"))


def _get_genai_client() -> genai.Client:
    http_options = types.HttpOptions(timeout=_REQUEST_TIMEOUT * 1000)
    project = os.environ.get("GEMINI_PROJECT")
    if project:
        # Prefer Vertex AI ADC when a project is configured (more reliable in batch pipelines)
        return genai.Client(
            vertexai=True,
            project=project,
            location=os.environ.get("GEMINI_LOCATION", "us-central1"),
            http_options=http_options,
        )
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key, http_options=http_options)
    raise RuntimeError(
        "Gemini client not configured. Set GEMINI_PROJECT (Vertex AI ADC) "
        "or GEMINI_API_KEY in .env"
    )


class GeminiLLM:
    def __init__(self):
        self.client = _get_genai_client()

    def completions(self, model_name, messages, response_format, temperature=0.7):
        system_prompt = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        user_content = "\n".join(
            m["content"] for m in messages if m["role"] == "user"
        )

        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=response_format.model_json_schema(),
            system_instruction=system_prompt,
        )

        response = self.client.models.generate_content(
            model=model_name,
            contents=user_content,
            config=config,
        )

        text = response.text
        parsed = response_format.model_validate(json.loads(text))

        return dict(
            completion=parsed,
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count,
        )
