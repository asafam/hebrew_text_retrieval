import os
from transformers import AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch
from model.dual_encoder.models import InfoNCEDualEncoderConfig, InfoNCEDualEncoder
from transformers import AutoConfig
from datasets import load_dataset
import numpy as np
import warnings
from sklearn.metrics import ndcg_score
from tqdm import tqdm
from pathlib import Path


def batched_encode(model, encoder, input_ids, attention_mask, device, batch_size=64):
        model.eval()
        dataset = TensorDataset(input_ids, attention_mask)
        loader = DataLoader(dataset, batch_size=batch_size)
        all_embeddings = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Encoding batches"):
                b_input_ids = batch[0].to(device)
                b_attention_mask = batch[1].to(device)
                emb = model.encode(encoder, b_input_ids, b_attention_mask)  # shape: [B, H]
                all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)


def encode(model_name_or_path: str,
           tokenizer_name_or_path: str,
           queries: list,
           documents: list,
           batch_size: int = 1024,
           max_length: int = 512,
           documents_embeddings_file: str = None,
           force_reencode: bool = False):
    # Load model
    if os.path.isdir(model_name_or_path):
        print("Loading model from checkpoint...")
        config = InfoNCEDualEncoderConfig.from_pretrained(model_name_or_path)
        model = InfoNCEDualEncoder.from_pretrained(model_name_or_path, config=config)
    else:
        print("Loading model from config...")
        config = InfoNCEDualEncoderConfig(query_model_name=model_name_or_path,
                                          doc_model_name=model_name_or_path, 
                                          query_tokenizer_path=tokenizer_name_or_path,
                                          doc_tokenizer_path=tokenizer_name_or_path,
                                          pooling='cls',
                                          temperature=0.05)
        model = InfoNCEDualEncoder(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load tokenizer
    tokenizer_q = AutoTokenizer.from_pretrained(tokenizer_name_or_path)  # or your actual model name
    tokenizer_d = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    print(f"Tokenizing {len(queries):,} queries...")
    q_batch = tokenizer_q(queries, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    # Encode queries and documents in batches
    print("Encoding queries...")
    q_emb = batched_encode(model, model.query_encoder, q_batch['input_ids'], q_batch['attention_mask'], device=device, batch_size=batch_size)


    print(f"Tokenizing {len(documents):,} documents...")
    d_batch = tokenizer_d(documents, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    
    print("Encoding documents...")
    if force_reencode or not documents_embeddings_file or not os.path.exists(documents_embeddings_file):
        d_emb = batched_encode(model, model.doc_encoder, d_batch['input_ids'], d_batch['attention_mask'], device=device, batch_size=batch_size)
        if documents_embeddings_file:
            Path(os.path.dirname(documents_embeddings_file)).mkdir(parents=True, exist_ok=True)
            print(f"Saving document embeddings to {documents_embeddings_file}")
            torch.save(d_emb.cpu(), documents_embeddings_file)
    else:
        d_emb = torch.load(documents_embeddings_file)
        print(f"Loaded document embeddings from {documents_embeddings_file}")

    return q_emb, d_emb


def measure_performance(q_emb: torch.Tensor,
                        d_emb: torch.Tensor,
                        documents: list,
                        gold: list,
                        query_context_id_field: str = "context_guid",
                        document_id_field: str = "guid"):
    assert q_emb.shape[0] == len(gold), "Number of queries in embeddings and gold data must match."
    
    sim_matrix = torch.matmul(q_emb, d_emb.t())
    sim_scores = sim_matrix.cpu().numpy()
    
    # `contexts` is already defined

    ranks = []
    ndcg_targets = []
    ndcg_scores = []
    missing_count = 0

    gold_context_ids_lists = [item[query_context_id_field] for item in gold]
    for i, gold_ids_list in tqdm(enumerate(gold_context_ids_lists), desc="Measuring performance"):
        scores = sim_scores[i]
        ranked_doc_indices = np.argsort(scores)[::-1]

        # Match each gold to its index in corpus
        gold_indices = []
        if type(gold_ids_list) is not list:
            gold_ids_list = [gold_ids_list]
        for gold_id in gold_ids_list:
            try:
                # find the index of the gold context in the documents list
                idx = next((i for i, doc in enumerate(documents) if doc[document_id_field] == gold_id), None)
                gold_indices.append(idx)
            except ValueError:
                continue  # this gold not found

        if not gold_indices:
            warnings.warn(f"[Warning] No gold contexts for query index {i} were found in the documents list.")
            missing_count += 1
            continue

        # Compute best rank
        rank_positions = [np.where(ranked_doc_indices == gi)[0][0] + 1 for gi in gold_indices]
        best_rank = min(rank_positions)
        ranks.append(best_rank)

        # Build binary relevance vector for nDCG
        relevance = np.zeros(len(documents))
        for gi in gold_indices:
            relevance[gi] = 1.0
        ndcg_targets.append(relevance)
        ndcg_scores.append(scores)

    # Filter valid entries
    valid_ranks = ranks
    valid_ndcg_targets = ndcg_targets
    valid_ndcg_scores = ndcg_scores

    # Basic metrics
    accuracy = np.mean([r == 1 for r in valid_ranks])
    mrr = np.mean([1.0 / r for r in valid_ranks])
    recall_at_5 = np.mean([r <= 5 for r in valid_ranks])
    recall_at_10 = np.mean([r <= 10 for r in valid_ranks])
    recall_at_100 = np.mean([r <= 100 for r in valid_ranks])

    # nDCG metrics
    ndcg_at_5 = np.mean([
        ndcg_score([target], [score], k=5)
        for target, score in zip(valid_ndcg_targets, valid_ndcg_scores)
    ])
    ndcg_at_10 = np.mean([
        ndcg_score([target], [score], k=10)
        for target, score in zip(valid_ndcg_targets, valid_ndcg_scores)
    ])
    ndcg_at_100 = np.mean([
        ndcg_score([target], [score], k=100)
        for target, score in zip(valid_ndcg_targets, valid_ndcg_scores)
    ])

    # Print results
    print(f"Top-1 Accuracy     : {accuracy:.4f}")
    print(f"MRR                : {mrr:.4f}")
    print(f"Recall@5           : {recall_at_5:.4f}")
    print(f"Recall@10          : {recall_at_10:.4f}")
    print(f"Recall@100         : {recall_at_100:.4f}")
    print(f"nDCG@5             : {ndcg_at_5:.4f}")
    print(f"nDCG@10            : {ndcg_at_10:.4f}")
    print(f"nDCG@100           : {ndcg_at_100:.4f}")

    if missing_count > 0:
        print(f"\n⚠️  {missing_count} queries had no gold contexts found in the documents list.")

    return {
        "accuracy": accuracy,
        "mrr": mrr,
        "recall_at_5": recall_at_5,
        "recall_at_10": recall_at_10,
        "recall_at_100": recall_at_100,
        "ndcg_at_5": ndcg_at_5,
        "ndcg_at_10": ndcg_at_10,
        "ndcg_at_100": ndcg_at_100
    }


def main(model_name_or_path: str,
         tokenizer_name_or_path: str,
         queries_path: str,
         documents_path: str,
         output_file: str,
         documents_embeddings_file: str = None,
         batch_size: int = 1024,
         max_length: int = 512,
         query_text_field: str = "text",
         query_context_field: str = "context",
         query_context_id_field: str = "context_guid",
         document_text_field: str = "text",
         document_source_field: str = "_source",
         main_source: str = "heq"):
    # Load datasets
    data_files = {
        "queries": {
            "test": queries_path
        },
        "documents": {
            "test": documents_path
        }
    }

    queries_dataset = load_dataset("json", data_files=data_files["queries"], split="test")
    documents_dataset = load_dataset("json", data_files=data_files["documents"], split="test")

    questions = [item[query_text_field] for item in tqdm(queries_dataset, desc="Loading queries")]
    gold = [{
        query_context_id_field: item[query_context_id_field],
        query_context_field: item[query_context_field]
    } for item in tqdm(queries_dataset, desc="Loading gold contexts")] # list of lists

    unique_contexts = set()
    deduped_contexts = []
    contexts = [item for item in tqdm(documents_dataset, desc="Loading documents")]
    # Deduplicate contexts
    contexts.sort(key=lambda x: 0 if x[document_source_field] == main_source else 1)
    for i, context in tqdm(enumerate(contexts), desc="Deduplicating contexts"):
        if context["guid"] not in unique_contexts:
            unique_contexts.add(context["guid"])
            deduped_contexts.append(context)
    contexts = [{
        document_text_field: c[document_text_field],
        "guid": c["guid"]
    } for c in deduped_contexts]
    print(f"Found {len(deduped_contexts)} unique contexts.")

    # Find how many gold contexts are in the documents
    gold_contexts = [c[query_context_id_field] for c in gold]
    contexts_ids = [c["guid"] for c in contexts]
    found_gold_contexts = [c for c in set(gold_contexts) if c in contexts_ids]
    print(f"Found {len(found_gold_contexts)} gold contexts in the documents ({len(set(gold_contexts))}).")
    assert len(found_gold_contexts) == len(set(gold_contexts)), \
        "Not all gold contexts were found in the documents. Please check your data."

    # Encode queries and documents
    q_emb, d_emb = encode(model_name_or_path=model_name_or_path,
                          tokenizer_name_or_path=tokenizer_name_or_path,
                          queries=questions,
                          documents=[c[document_text_field] for c in contexts],
                          max_length=max_length,
                          documents_embeddings_file=documents_embeddings_file,
                          batch_size=batch_size)
    print(f"Encoded {len(q_emb)} queries and {len(d_emb)} documents.")
    print(f"Query embeddings shape: {q_emb.shape}")
    print(f"Document embeddings shape: {d_emb.shape}")
    
    # Measure performance
    metrics = measure_performance(q_emb,
                                  d_emb,
                                  documents=contexts,
                                  gold=gold,
                                  query_context_id_field=query_context_id_field,
                                  document_id_field=document_text_field)
    
    # Save evaluation results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"Evaluation results saved to {output_file}")


if __name__ == "__main__":
    print("(eval_retrieval.py) Running evaluation script...")
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate retrieval model performance.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True, help="Path to the tokenizer directory.")
    parser.add_argument("--queries_path", type=str, required=True, help="Path to the queries JSON file.")
    parser.add_argument("--documents_path", type=str, required=True, help="Path to the documents JSON file.")
    parser.add_argument("--output_file", type=str, default="eval_results.txt", help="File to save evaluation results.")
    parser.add_argument("--documents_embeddings_file", type=str, default=None,
                        help="Path to save or load document embeddings. If None, embeddings will be re-encoded.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for encoding.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length for tokenization.")
    parser.add_argument("--query_text_field", type=str, default="text", help="Field name for query text.")
    parser.add_argument("--query_context_field", type=str, default="context", help="Field name for query context.")
    parser.add_argument("--query_context_id_field", type=str, default="context_id", help="Field name for query context id.")
    parser.add_argument("--document_text_field", type=str, default="text", help="Field name for document text.")
    parser.add_argument("--document_source_field", type=str, default="_source",
                        help="Field name for document source (to deduplicate).")
    parser.add_argument("--main_source", type=str, default="heq",
                        help="Main source to prioritize when deduplicating documents.")

    args = parser.parse_args()

    main(model_name_or_path=args.model_name_or_path,
         tokenizer_name_or_path=args.tokenizer_name_or_path,
         queries_path=args.queries_path,
         documents_path=args.documents_path,
         output_file=args.output_file,
         documents_embeddings_file=args.documents_embeddings_file,
         batch_size=args.batch_size,
         max_length=args.max_length,
         query_text_field=args.query_text_field,
         query_context_field=args.query_context_field,
         document_text_field=args.document_text_field,
         document_source_field=args.document_source_field,
         main_source=args.main_source)