from openai import OpenAI
import os

class OpenAILLM():
    def __init__(self, 
                 organization=os.environ['OPENAI_API_ORG'],
                 api_key=os.environ['OPENAI_API_KEY'],
                 project=os.environ['OPENAI_PROJECT'],
                 base_url=None):
        args = dict(
            organization=organization,
            api_key=api_key,
            project=project
        )
        if base_url:
            args['base_url'] = base_url
        self.client = OpenAI(**args)

    def completions(
            self, 
            model_name, 
            messages, 
            response_format,
            temperature=0.7, 
        ):
        # Run chat completion inference
        completion = self.client.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            temperature=temperature,
            response_format=response_format
        )
        result = dict(
            completion = completion.choices[0].message.parsed,
            input_tokens = completion.usage.prompt_tokens,
            output_tokens = completion.usage.completion_tokens
        )
        return result