import re
from llms.google import Gemini
from llms.openai import OpenAI

def get_llm(model_name):
    if re.match(r'.*gpt.*', model_name):
        return OpenAI
    elif re.match(r'.*gemini.*', model_name):
        return Gemini
    else:
        raise ValueError("Model not supported")