import os
import json
import re
from llms.openai import OpenAILLM

class OpenRouterLLM(OpenAILLM):
    def __init__(self):
        super().__init__(
            organization=None,
            api_key=os.environ['OPENROUTER_API_KEY'],
            project=None,
            base_url="https://openrouter.ai/api/v1"
        )

    def completions(
            self,
            model_name,
            messages,
            response_format,
            temperature=0.7,
            **kwargs
        ):
        schema = response_format.model_json_schema()
        fields = list(schema.get('properties', {}).keys())

        # Inject explicit JSON instruction since structured-outputs are not supported
        json_instruction = f'Respond with a valid JSON object with exactly these fields: {fields}. No markdown fences, no extra text.'
        augmented = [
            {**m, 'content': m['content'] + '\n\n' + json_instruction}
            if m['role'] == 'system' else m
            for m in messages
        ]

        completion = self.client.chat.completions.create(
            model=model_name,
            messages=augmented,
            temperature=temperature,
        )

        content = completion.choices[0].message.content or ''

        # Strip markdown fences if present
        content = re.sub(r'^```(?:json)?\s*', '', content.strip())
        content = re.sub(r'\s*```$', '', content)

        # Extract first JSON object from the response
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in response: {content!r}")
        data = json.loads(match.group())

        try:
            output = response_format(**data)
        except Exception:
            if len(fields) == 1:
                field = fields[0]
                aliases = [field, 'answer', 'result', 'output', 'translation', 'hebrew', 'critique', 'score']
                for alias in aliases:
                    if alias in data and data[alias] is not None:
                        output = response_format(**{field: data[alias]})
                        break
                else:
                    candidates = [(k, v) for k, v in data.items() if isinstance(v, str) and v]
                    if candidates:
                        _, v = min(candidates, key=lambda x: len(x[1]))
                        output = response_format(**{field: v})
                    else:
                        raise ValueError(f"Could not extract '{field}' from: {data}")
            else:
                raise ValueError(f"OpenRouter response fields don't match schema {fields}: {data}")

        return dict(
            completion=output,
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
        )
