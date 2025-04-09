import os
import json
from llms.openai import OpenAILLM

class TogetherAILLM(OpenAILLM):
    def __init__(self):
        super().__init__(
            organization=None,
            api_key=os.environ['TOGETHER_API_KEY'],
            project=None,
            base_url="https://api.together.xyz/v1"
        )

    def completions(
            self, 
            model_name, 
            messages, 
            response_format,
            temperature=0.7, 
            **kwargs
        ):
        # Run chat completion inference
        together = self.client
        completion = together.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            response_format={
                "type": "json_object",
                "schema": response_format.model_json_schema(),
            }
        )

        if model_name in [
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "meta-llama/Llama-3.2-3B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-V3"
        ]:
            output = json.loads(completion.choices[0].message.content)
        else:
            output = completion.choices[0].message.content.strip()
        result = dict(
            completion = output,
            input_tokens = completion.usage.prompt_tokens,
            output_tokens = completion.usage.completion_tokens
        )
        return result