# Long-Context Retrieval Evaluation — Hard-Negative Patching

Companion to [EVALUATION.md](EVALUATION.md) (standard BeIR retrieval eval).

## What this measures

Standard retrieval eval ranks a query against a corpus of short, individual passages.
This eval asks a harder question: **what happens to retrieval quality when each document
is padded with semantically similar (hard-negative) passages?**

For every document `d_i` in the corpus:
1. Encode all documents with the model → FAISS index.
2. Find the K nearest-neighbour documents to `d_i` (excluding itself) — these are the hard negatives: similar topic, but not the gold answer.
3. Concatenate `d_i` with those K neighbours into one longer "patched" document (`d_i || --- || hn_1 || --- || ... || hn_K`).
4. Re-encode the patched corpus and run retrieval.

At K=0 this is identical to the standard eval (baseline). As K grows the model must
compress a longer, noisier sequence into a single CLS embedding and still surface the
right document.

This stress-tests two things:
- **Embedding dilution** — can the CLS vector retain enough signal from the positive span?
- **Hard-negative proximity** — the injected neighbours are the *most similar* documents in
  embedding space, so they are the worst-case distractors.

---

## Datasets

Same five BeIR datasets used in the standard eval (see EVALUATION.md):

| Dataset | Domain | Test queries | Corpus |
|---------|--------|-------------:|-------:|
| `BeIR/scifact` | Scientific fact verification | 300 | 5,183 |
| `BeIR/scidocs` | Scientific citation | 1,000 | 25,657 |
| `BeIR/nfcorpus` | Bio-medical IR | 323 | 3,633 |
| `BeIR/arguana` | Argument retrieval | 1,401 | 8,674 |
| `BeIR/fiqa` | Financial QA | 648 | 57,600 |

---

## Metrics

- **Acc@1** — fraction of queries where the patched gold document is the top-1 result
- **MRR** — mean reciprocal rank
- **Recall@5 / @10** — fraction of queries where the patched gold document appears in the top-5/10

K=0 results are the baseline (no patching) and should match standard retrieval numbers.

---

## Code

| File | Purpose |
|------|---------|
| `src/data/long_context/patch_documents.py` | FAISS NN search + document concatenation |
| `src/model/eval/eval_long_context_hn_patching.py` | End-to-end eval script (loads BeIR, encodes, sweeps K) |

### `patch_documents.py` API

```python
from data.long_context.patch_documents import build_patched_corpus

patched_texts, positive_positions = build_patched_corpus(
    documents=passages,       # List[str]
    embeddings=d_emb_orig,    # torch.Tensor (N, D), L2-normalised
    k=3,                      # hard negatives per document
    positive_position='random',  # 'first' | 'last' | 'random'
    seed=42,
)
```

### `eval_long_context_hn_patching.py` CLI

```bash
python src/model/eval/eval_long_context_hn_patching.py \
    --model_name_or_path  <path_or_hub_name> \
    --dataset_name        BeIR/scifact \
    --k_values            0,1,3,5 \
    --positive_position   random \
    --batch_size          128 \
    --max_length          512 \
    --output_dir          outputs/eval/dual_encoder/beir_long_context/<model>
```

Outputs one JSON file per dataset to `--output_dir`.

---

## Running

Scripts follow the same convention as the standard eval (`eval_dual_encoder_retrieval_{model}.sh`),
located under `scripts/model/dual_encoder/beir_long_context/eval/`.

| Script | Model | Source |
|--------|-------|--------|
| `eval_dual_encoder_retrieval_neodictabert.sh` | NeoDictaBERT (BeIR fine-tuned) | local checkpoint |
| `eval_dual_encoder_retrieval_hebmodernbert.sh` | HMB (BeIR + hard-neg fine-tuned) | local checkpoint |
| `eval_dual_encoder_retrieval_multilingual-e5-large.sh` | mE5-large | HuggingFace hub (zero-shot) |
| `eval_dual_encoder_retrieval_multilingual-e5-base.sh` | mE5-base | HuggingFace hub (zero-shot) |

```bash
# Submit one model
sbatch scripts/model/dual_encoder/beir_long_context/eval/eval_dual_encoder_retrieval_neodictabert.sh

# Or run locally (no SLURM)
bash scripts/model/dual_encoder/beir_long_context/eval/eval_dual_encoder_retrieval_neodictabert.sh
```

> **HMB checkpoint note:** `eval_dual_encoder_retrieval_hebmodernbert.sh` points to the
> `beir_hebrew_hn/hebmodernbert/ModernBERT-Hebrew-base_v2` checkpoint (trained by
> `train_dual_encoder_beir_hn_hebmodernbert.sh`). If you want to test a different HMB
> variant (e.g. the 20250622 checkpoint), update `MODEL_PATH` and `TOKENIZER_PATH` in
> that script before submitting.

---

## Output format

Each run writes one JSON per dataset:

```
outputs/eval/dual_encoder/beir_long_context/<model>/
├── BeIR_scifact.json
├── BeIR_scidocs.json
├── BeIR_nfcorpus.json
├── BeIR_arguana.json
└── BeIR_fiqa.json
```

Each JSON:

```json
{
  "model": "...",
  "dataset": "BeIR/scifact",
  "num_queries": 300,
  "num_passages": 5183,
  "k_values": [0, 1, 3, 5],
  "results": {
    "0": {"acc@1": 0.62, "mrr": 0.70, "recall@5": 0.84, "recall@10": 0.89},
    "1": {"acc@1": 0.55, ...},
    "3": {"acc@1": 0.44, ...},
    "5": {"acc@1": 0.38, ...}
  }
}
```

---

## Results

> Fill in after running. K=0 should match EVALUATION.md NDCG@10 figures (different metric, expect same ranking).

| Model | Dataset | K=0 Acc@1 | K=1 Acc@1 | K=3 Acc@1 | K=5 Acc@1 |
|-------|---------|----------:|----------:|----------:|----------:|
| NeoDictaBERT | scifact | — | — | — | — |
| NeoDictaBERT | nfcorpus | — | — | — | — |
| mE5-large | scifact | — | — | — | — |
| mE5-large | nfcorpus | — | — | — | — |
