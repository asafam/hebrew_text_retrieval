import tiktoken


def count_tokens(text, model_name="gpt-4"):
    if model_name.startswith("gpt"):
        encoder = tiktoken.encoding_for_model(model_name)
        return len(encoder.encode(text))
    else:
        raise ValueError(f"Model {model_name} not supported for token counting")

