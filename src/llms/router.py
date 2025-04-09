import re
from llms.google import GeminiLLM
from llms.openai import OpenAILLM
from llms.together_ai import TogetherAILLM

def get_llm(model_name):
    if re.match(r'.*gpt.*', model_name):
        return OpenAILLM()
    elif re.match(r'.*gemini.*', model_name):
        return GeminiLLM()
    else:
        return TogetherAILLM()