from transformers import AutoTokenizer
import tiktoken
import hashlib
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

def count_tokens(text, model_or_tokenizer_name="gpt-4"):
    if model_or_tokenizer_name.startswith("gpt"):
        encoder = tiktoken.encoding_for_model(model_or_tokenizer_name)
        return len(encoder.encode(text))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_or_tokenizer_name)
        tokens = tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=tokenizer.model_max_length)
        return len(tokens["input_ids"][0]) if "input_ids" in tokens else 0
    

def hash(text):
    hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    return hash


def split_document_by_segments(document: dict, model_name_or_path: str, context_field: str = "text", max_tokens: int = 256):
        # Split document into sentences
        sentences = sent_tokenize(document[context_field])

        # Split into segments based on max tokens
        segments = []
        current_segment = ""
        current_segment_tokens = 0

        for sentence in sentences:
            # Tokenize the sentence and count tokens
            sentence_tokens = count_tokens(sentence, model_name_or_path)
            
            # Check if adding this sentence would exceed max tokens
            if current_segment_tokens + sentence_tokens <= max_tokens:
                current_segment += " " + sentence
                current_segment_tokens += sentence_tokens
            else:
                # Add the current segment to the list
                segments.append(current_segment.strip())
                # Start a new segment with the current sentence
                current_segment = sentence
                current_segment_tokens = sentence_tokens

        # Add the last segment
        if current_segment:
            segments.append(current_segment.strip())

        # Print segments
        document_segments = []
        for idx, segment in enumerate(segments):
            document_segments.append({
                **document,
                'text': document['text'],
                'segment_id': idx,
                'segment_text': segment
            })
        return document_segments

    

