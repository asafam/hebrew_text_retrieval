


def translate_sampled_queries_from_dataset(dataset_name: str, 
                                           model_name: str, 
                                           prompt_file_name: str, 
                                           n: int, 
                                           batch_size: int = 512,
                                           split: str = 'test',
                                           device: str = 'cuda',
                                           random_seed: int = 42):
    beir_dataset_loader = 
    corpus_dataset, queries_dataset, qrels_dataset = load_bier_dataset(dataset_name)
    sampled_qrels, sampled_queries, sampled_documents = sample_data_from_dataset(n, corpus_dataset, queries_dataset, qrels_dataset)

    with open(prompt_file_name, 'r') as file:
        prompt_data = yaml.safe_load(file)

    prompt_prefix = prompt_data['prompt_prefix']
    texts = [{
            'id': query['_id'],
            'text': prompt_data['prompt_template'].format(query=query['text'], context=query['context']).strip()
        } for query in sampled_queries]
    translations, resources_usage = translation_pipeline(model_name, prompt_prefix, texts, batch_size, ids=['id'], device=device)
    
    extra_args = {
        'task': 'query_translation',
        'num_queries': len(sampled_queries),
        'timestamp': datetime.now(),
        'prompt_file_name': prompt_file_name,
        'model': model_name,
        'dataset_name': dataset_name,
        'random_seed': random_seed
    }
    translations = [{**translation, **extra_args} for translation in translations]
    resources_usage = [{**resource_usage, **extra_args} for resource_usage in resources_usage]
    return translations, resources_usage


def translate_sampled_documents_from_dataset(dataset_name: str, 
                                             model_name: str, 
                                             prompt_file_name: str, 
                                             n: int, 
                                             batch_size: int = 512,
                                             split: str = 'test',
                                             device: str = "cuda",
                                             max_tokens: int = 64,
                                             random_seed: int = 42):
    # Load the data
    corpus_dataset, queries_dataset, qrels_dataset = load_bier_dataset(dataset_name)

    # Sample the data
    sampled_qrels, sampled_queries, sampled_documents = sample_data_from_dataset(n, corpus_dataset, queries_dataset, qrels_dataset)

    # Get the prompt
    with open(prompt_file_name, 'r') as file:
        prompt_data = yaml.safe_load(file)

    # Split long documents to shorter passages
    passages = []
    for document in sampled_documents:
        passages.extend(split_document_by_sentences(document, tokenizer, max_tokens=max_tokens))

    prompt_prefix = prompt_data['prompt_prefix']
    texts = [{
            'id': passage['_id'],
            'passage_id': passage['passage_id'],
            'text': prompt_data['prompt_template'].format(document=passage['text']).strip()
        } for passage in passages]
    
    # Translate
    translations, resources_usage = translation_pipeline(model_name, prompt_prefix, texts, batch_size, ids=['id', 'passage_id'], device=device)
    
    # Save translations
    extra_args = {
        'task': 'query_translation',
        'max_tokens': max_tokens,
        'num_documents': len(sampled_documents),
        'num_passages': len(passages),
        'timestamp': datetime.now(),
        'prompt_file_name': prompt_file_name,
        'model': model_name,
        'dataset_name': dataset_name,
        'random_seed': random_seed,
    }
    translations = [{**translation, **extra_args} for translation in translations]
    resources_usage = [{**resource_usage, **extra_args} for resource_usage in resources_usage]
    return translations, resources_usage


def save_data(data, csv_file_path, encoding='utf-8'):
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    if os.path.isfile(csv_file_path):
        df.to_csv(csv_file_path, mode='a', header=False, index=False, encoding=encoding)
    df.to_csv(csv_file_path, index=False, encoding=encoding)