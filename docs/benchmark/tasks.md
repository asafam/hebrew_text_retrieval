# The 5 Hebrew BeIR Retrieval Tasks

What each dataset asks the model to do, in concrete terms.

The setup is the same everywhere: **the model receives one Hebrew query and must
rank the entire Hebrew corpus, returning the documents that match.** What differs
between datasets is what the query is, what a document is, and what counts as a match.

These are the 5 datasets translated so far. The remaining 10 (msmarco, nq, hotpotqa,
fever, climate-fever, quora, trec-covid, dbpedia-entity, cqadupstack,
webis-touche2020) are planned.

Detailed statistics and known issues: `datasets.md`. Translation status:
`../translation/ledger.md`.

---

## 1. nfcorpus — health topic → supporting medical studies

**Task:** given the title of a consumer-health article, find the medical research
papers that provide its evidence base.

| | |
|---|---|
| **Query** | A health topic or claim. Very short — 2 words typically. |
| **Document** | A PubMed abstract (title + abstract text), ~193 words. |
| **Match means** | The paper is cited as evidence for that topic, or is on the same subject. |
| **Corpus** | 3,633 abstracts |
| **Correct answers per query** | ~38 |

**Example**

| | |
|---|---|
| Query | `תאי סרטן השד ניזונים מכולסטרול` — "Breast Cancer Cells Feed on Cholesterol" |
| A correct document | "Statin Use and Breast Cancer Survival: A Nationwide Cohort Study from Finland" |

**The catch:** relevance has two levels — **2** = the paper is directly cited as
evidence, **1** = merely on the same topic. 95% of judgments are level 1, so a
randomly chosen correct pair is usually a loose topical match. Also, the training
split contains *only* level-1 judgments, so training and testing define "relevant"
differently.

---

## 2. fiqa — financial question → answer

**Task:** given a personal-finance question, find the forum post that answers it.

| | |
|---|---|
| **Query** | A natural question about money, taxes, or investing. ~9 words. |
| **Document** | A StackExchange or Reddit answer, ~74 words. No titles. |
| **Match means** | The post answers the question. |
| **Corpus** | 57,600 answers |
| **Correct answers per query** | ~3 |

**Example**

| | |
|---|---|
| Query | `מה נחשב להוצאה עסקית בנסיעת עסקים?` — "What is considered a business expense on a business trip?" |
| A correct document | The answer post explaining which travel costs are deductible |

**The catch:** none of consequence. Short question, medium answer, few correct
answers, clean train/validation/test splits. This is the most conventional retrieval
task of the five and the best one to sanity-check a model against.

---

## 3. scifact — scientific claim → the paper that tests it

**Task:** given a scientific claim, find the paper abstract containing the evidence
that bears on it.

| | |
|---|---|
| **Query** | A one-sentence scientific claim. ~11 words. |
| **Document** | A scientific paper abstract, ~165 words. |
| **Match means** | The abstract reports evidence that supports **or refutes** the claim. |
| **Corpus** | 5,183 abstracts |
| **Correct answers per query** | ~1 |

**Example**

| | |
|---|---|
| Query | `לחומרים ביולוגיים 0-ממדיים חסרות תכונות השראתיות` — "0-dimensional biomaterials lack inductive properties" |
| A correct document | The abstract whose experimental findings address that property |

**The catch:** the model is not asked whether the claim is true. An abstract that
**refutes** the claim is exactly as correct as one that supports it — the task is
finding the paper that settles the question, not settling it. Also, the answer key
points at a whole abstract rather than the specific sentence doing the work, so a
correct pair can look only loosely related.

---

## 4. arguana — argument → the argument that rebuts it

**Task:** given an argument, find the counter-argument written against it.

| | |
|---|---|
| **Query** | A complete argument — a full paragraph, ~140 words. |
| **Document** | Another argument of the same kind and length, ~117 words. |
| **Match means** | The document is the designated rebuttal to that specific argument. |
| **Corpus** | 8,674 debate passages |
| **Correct answers per query** | exactly 1 |

**Example**

| | |
|---|---|
| Query | `להיות צמחוני עוזר לסביבה. חקלאות מודרנית היא אחד המקורות העיקריים לזיהום בנהרות שלנו...` — "Being vegetarian helps the environment. Modern farming is a major source of river pollution..." |
| The correct document | `אתם לא חייבים להיות צמחונים כדי להיות ירוקים. סביבות מיוחדות רבות נוצרו על ידי גידול בעלי חיים...` — "You don't have to be vegetarian to be green. Many special environments were created by animal farming..." |

**The catch:** two things make this unlike the others. First, the query is a full
paragraph rather than a short question, so this is long-text-to-long-text matching.
Second, the correct answer is a passage that **disagrees** with the query — while
retrieval models are generally trained to find text that is *similar* to the query.
The two objectives work against each other.

---

## 5. scidocs — paper title → the papers it cites

**Task:** given the title of a scientific paper, find the papers appearing in its
bibliography.

| | |
|---|---|
| **Query** | A paper title. ~9 words. |
| **Document** | A scientific paper abstract, ~129 words. |
| **Match means** | The query paper cites that paper. |
| **Corpus** | 25,313 abstracts |
| **Correct answers per query** | ~5 |

**Example**

| | |
|---|---|
| Query | `שיטת חיפוש ישירה לפתרון בעיית שיגור כלכלי עם אפקט נקודת שסתום` — "A Direct Search Method to solve Economic Dispatch Problem with Valve-Point Effect" |
| A correct document | "A hybrid of genetic algorithm and particle swarm optimization for recurrent network design" |

**The catch:** the answer key also lists **25 deliberately wrong answers per query** —
same-field papers that are *not* cited, included because this dataset was built for
reranking. Any code reading this file must keep only rows with `score > 0`, or it
will treat non-matches as matches. (This already caused a real bug in the review
spreadsheet.) The upside: those 25 make excellent ready-made hard negatives.

---

## Side by side

| Dataset | Query | Document | Match means | Answers/query |
|---|---|---|---|---:|
| **nfcorpus** | health topic (2 words) | PubMed abstract | cited as evidence, or on-topic | ~38 |
| **fiqa** | finance question (9 words) | forum answer | answers the question | ~3 |
| **scifact** | scientific claim (11 words) | paper abstract | contains supporting or refuting evidence | ~1 |
| **arguana** | full argument (140 words) | another argument | is the rebuttal to it | 1 |
| **scidocs** | paper title (9 words) | paper abstract | is cited by the query paper | ~5 |

Two differences matter most when reading scores:

- **Query length.** Four datasets give the model a short phrase; arguana gives it a
  whole paragraph.
- **Number of correct answers.** Ranges from exactly 1 (arguana) to ~38 (nfcorpus).
  A good NDCG on scifact and a good NDCG on nfcorpus mean quite different things.
