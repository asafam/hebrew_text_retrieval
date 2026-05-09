import re
from llms.google import GeminiLLM
from llms.openai import OpenAILLM, AzureOpenAILLM
from llms.together_ai import TogetherAILLM
from llms.anthropic import AnthropicLLM
from llms.openrouter import OpenRouterLLM

def get_llm(model_name, provider=None):
    if provider == 'azure':
        return AzureOpenAILLM()
    if re.match(r'.*gpt.*', model_name):
        return OpenAILLM()
    elif re.match(r'.*moonshotai.*|.*kimi.*|google/.*', model_name):
        return OpenRouterLLM()
    elif re.match(r'.*gemini.*', model_name):
        return GeminiLLM()
    elif re.match(r'.*claude.*', model_name):
        return AnthropicLLM()
    else:
        return TogetherAILLM()