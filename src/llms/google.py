import os
from llms.openai import OpenAI

class Gemini(OpenAI):
    def __init__(self):
        self.client = super().__init__(
            organization=None,
            api_key=os.environ['GEMINI_API_KEY'],
            project=os.environ['GEMINI_PROJECT'],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )