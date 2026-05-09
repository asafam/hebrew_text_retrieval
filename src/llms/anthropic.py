import os
import anthropic


class AnthropicLLM:
    def __init__(self, api_key: str = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ['ANTHROPIC_API_KEY']
        )

    def completions(self, model_name, messages, response_format, temperature=0.7):
        system_prompt = ""
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

        tool_name = response_format.__name__
        tool_def = {
            "name": tool_name,
            "description": f"Return a structured {tool_name} object.",
            "input_schema": response_format.model_json_schema(),
        }
        kwargs = dict(
            model=model_name,
            max_tokens=4096,
            temperature=temperature,
            tools=[tool_def],
            tool_choice={"type": "tool", "name": tool_name},
            messages=anthropic_messages,
        )
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        if response.stop_reason == "max_tokens":
            raise ValueError(f"Response truncated (max_tokens={kwargs['max_tokens']}). Consider increasing max_tokens.")
        parsed = response_format(**response.content[0].input)

        return {
            "completion": parsed,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
