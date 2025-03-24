import os
from llms.openai import OpenAI

class TogetherAI(OpenAI):
    def __init__(self):
        self.client = super().__init__(
            organization=None,
            api_key=os.environ['TOGETHER_API_KEY'],
            project=None,
            base_url="https://api.together.xyz/v1"
        )