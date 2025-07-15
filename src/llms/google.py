import os
from llms.openai import OpenAILLM

class GeminiLLM(OpenAILLM):
    def __init__(self):
        super().__init__(
            organization=None,
            api_key=os.environ['GEMINI_API_KEY'],
            project=os.environ['GEMINI_PROJECT_ID'],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )