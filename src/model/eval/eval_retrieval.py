import os
from pyexpat import model
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


def batched_encode(
    model, encoder, tokenizer, texts, device,
    max_length=512, batch_size=64,
    file=None
):
    model.eval()
    all_embeddings = []

    start_batch = 0
    total_batches = (len(texts) + batch_size - 1) // batch_size

    # Resume support: load partial if it exists
    partial_file = file.replace(".pt", "_partial.pt") if file else None
    if partial_file and os.path.exists(partial_file):
        print(f"[Resuming] Loading partial embeddings from {partial_file}")
        all_embeddings = list(torch.load(partial_file))
        start_batch = len(all_embeddings)
        print(f"[Resuming] {start_batch} batches already encoded.")

    for batch_num in tqdm(range(start_batch, total_batches), desc="Tokenizing and Encoding", unit="batch"):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]

        tokenized = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        # CUDA/log check - only print for the first batch, or every N batches
        if batch_num == start_batch:
            msg = (
                f"[Batch {batch_num}] input_ids.device: {input_ids.device}, "
                f"model device: {next(model.parameters()).device}"
            )
            if input_ids.device.type != "cuda" or next(model.parameters()).device.type != "cuda":
                tqdm.write(f"⚠️ WARNING: Not running on CUDA! " + msg)
            else:
                tqdm.write(f"[CUDA OK] " + msg)

        with torch.no_grad():
            emb = model.encode(encoder, input_ids, attention_mask)  # shape: [B, H]
            all_embeddings.append(emb.cpu())

        # Save after each batch
        if partial_file:
            Path(os.path.dirname(partial_file)).mkdir(parents=True, exist_ok=True)
            torch.save(all_embeddings, partial_file)

    # Concatenate all embeddings
    final_embeddings = torch.cat(all_embeddings, dim=0)

    # Save final embeddings if a file is specified
    if file:
        Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
        print(f"Saving document embeddings to {file}")
        torch.save(final_embeddings.cpu(), file)

    if partial_file:
        print(f"Removing partial file {partial_file}")
        os.remove(partial_file)
    
    return final_embeddings


def encode(model_name_or_path: str,
           tokenizer_name_or_path: str,
           queries: list,
           documents: list,
           batch_size: int = 1024,
           max_length: int = 512,
           embeddings_files_path: str = None,
           pretrain: bool = True,
           force_reencode: bool = False):
    # Load model
    if pretrain and os.path.isdir(model_name_or_path):
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
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.query_encoder = model.query_encoder.to(device)
    model.doc_encoder = model.doc_encoder.to(device)
    print("Model device:", next(model.parameters()).device)
    print("Query encoder device:", next(model.query_encoder.parameters()).device)
    print("Doc encoder device:", next(model.doc_encoder.parameters()).device)

    # Load tokenizer
    tokenizer_q = AutoTokenizer.from_pretrained(tokenizer_name_or_path)  # or your actual model name
    tokenizer_d = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    print(f"Encoding {len(queries):,} queries...")
    queries_embeddings_file = os.path.join(embeddings_files_path, "query_embeddings.pt") if embeddings_files_path else None
    if force_reencode or not queries_embeddings_file or not os.path.exists(queries_embeddings_file):
        q_emb = batched_encode(model=model, 
                               encoder=model.query_encoder, 
                               tokenizer=tokenizer_q, 
                               texts=queries, 
                               device=device,
                               max_length=max_length,
                               batch_size=batch_size,
                               file=queries_embeddings_file)

    print(f"Encoding {len(documents):,} documents...")
    documents_embeddings_file = os.path.join(embeddings_files_path, "doc_embeddings.pt") if embeddings_files_path else None
    if force_reencode or not documents_embeddings_file or not os.path.exists(documents_embeddings_file):
        d_emb = batched_encode(model=model, 
                               encoder=model.doc_encoder, 
                               tokenizer=tokenizer_d, 
                               texts=documents, 
                               device=device,
                               max_length=max_length, 
                               batch_size=batch_size,
                               file=documents_embeddings_file)
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

    for i, gold_list in tqdm(enumerate(gold), desc="Measuring performance"):
        scores = sim_scores[i]
        ranked_doc_indices = np.argsort(scores)[::-1]

        # Match each gold to its index in corpus
        gold_indices = []
        if type(gold_list) is not list:
            gold_list = [gold_list]
        for gold_item in gold_list:
            try:
                # find the index of the gold context in the documents list
                idx = next((i for i, doc in enumerate(documents) if doc[document_id_field] == gold_item[query_context_id_field]), None)
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
         embeddings_files_path: str = None,
         batch_size: int = 1024,
         max_length: int = 512,
         query_text_field: str = "text",
         query_context_field: str = "context",
         query_context_id_field: str = "context_guid",
         document_text_field: str = "text",
         document_id_field: str = "guid",
         document_source_field: str = "_source",
         main_source: str = "heq",
         pretrain: bool = True):
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
        if context[document_id_field] not in unique_contexts:
            unique_contexts.add(context[document_id_field])
            deduped_contexts.append(context)
    contexts = [{
        document_text_field: c[document_text_field],
        document_id_field: c[document_id_field]
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
                          embeddings_files_path=embeddings_files_path,
                          pretrain=pretrain,
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
                                  document_id_field=document_id_field)
    
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
    parser.add_argument("--embeddings_files_path", type=str, default=None,
                        help="Path to save or load the query and document embeddings. If None, embeddings will be re-encoded.")
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
    parser.add_argument("--pretrain", action='store_true', help="Whether to load the model from a pre-trained checkpoint.")

    args = parser.parse_args()

    main(model_name_or_path=args.model_name_or_path,
         tokenizer_name_or_path=args.tokenizer_name_or_path,
         queries_path=args.queries_path,
         documents_path=args.documents_path,
         output_file=args.output_file,
         embeddings_files_path=args.embeddings_files_path,
         batch_size=args.batch_size,
         max_length=args.max_length,
         query_text_field=args.query_text_field,
         query_context_field=args.query_context_field,
         document_text_field=args.document_text_field,
         document_source_field=args.document_source_field,
         main_source=args.main_source,
         pretrain=args.pretrain)