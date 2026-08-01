# Documentation

## Benchmark — the translated Hebrew BeIR datasets and model results

| Doc | What it covers |
|---|---|
| [benchmark/tasks.md](benchmark/tasks.md) | **Start here.** What each of the 5 datasets asks a retriever to do — query, document, what counts as a match, with examples. |
| [benchmark/datasets.md](benchmark/datasets.md) | Detailed per-dataset reference: relevance scales, judgment counts, length profiles, splits, hard negatives, per-dataset pitfalls. Includes the 10 pending datasets as a forward checklist. |
| [benchmark/results.md](benchmark/results.md) | Model scores, standings, and the analysis behind the "ship NeoDictaBERT" decision. |
| [benchmark/runbook.md](benchmark/runbook.md) | Summary results table plus how to run or re-run an evaluation. |
| [benchmark/why-not-translation.md](benchmark/why-not-translation.md) | **The case that translation is not what limits our Hebrew scores** — the evidence, assembled, and the one narrow place where it is. Read this for the argument and the decision. |
| [benchmark/failure-analysis.md](benchmark/failure-analysis.md) | The lab notebook behind the above: method, per-dataset breakdowns, statistics, scripts. |
| [benchmark/long-context.md](benchmark/long-context.md) | Long-context retrieval evaluation with hard-negative patching. |

## Translation — building the Hebrew corpora

| Doc | What it covers |
|---|---|
| [translation/pipeline.md](translation/pipeline.md) | The current production pipeline: Gemini Batch API on Vertex AI, pilot → submit → collect. |
| [translation/ledger.md](translation/ledger.md) | Running status of which BeIR datasets are translated, the approval gate, and the exact prompt/model settings used. |
| [translation/candidate-generation.md](translation/candidate-generation.md) | How candidate query/document sets are sampled and sharded. |
| [translation/prompt-experiments.md](translation/prompt-experiments.md) | Prompt-strategy and judge-model comparison experiments. |
| [translation/archive/](translation/archive/) | Superseded pipeline docs, kept for historical reference. |

## Process

| Doc | What it covers |
|---|---|
| [experiments.md](experiments.md) | Step-by-step guide for running the BeIR Hebrew translation experiments end to end. |

---

Project overview and experiment history: [../README.md](../README.md).
Repository conventions for Claude Code: [../CLAUDE.md](../CLAUDE.md).
