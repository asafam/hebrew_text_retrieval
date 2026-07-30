# The 5 Hebrew BeIR Retrieval Tasks — Plain Explanation

One page. What goes in, what comes out, for each of the 5 translated datasets.

In every case the setup is the same: **the model gets one Hebrew query and must rank the whole Hebrew corpus, returning the documents that match.** What changes between datasets is what a "query" is, what a "document" is, and what "matches" means.

These are the 5 datasets translated so far. The remaining 10 (msmarco, nq, hotpotqa, fever, climate-fever, quora, trec-covid, dbpedia-entity, cqadupstack, webis-touche2020) are planned — add a section here as each one lands.

For the detailed version — statistics, relevance scales, known issues — see `BEIR_DATASETS.md`. Translation status: `BEIR_TRANSLATION.md`.

---

## 1. nfcorpus — "find the studies behind this health claim"

| | |
|---|---|
| **Input** | A short health topic, ~2 words |
| **Output** | Medical research abstracts that support it |
| **Corpus** | 3,633 PubMed abstracts |

**Example**
> **Input:** `תאי סרטן השד ניזונים מכולסטרול`
> ("Breast Cancer Cells Feed on Cholesterol")
>
> **Output:** the abstract *"Statin Use and Breast Cancer Survival: A Nationwide Cohort Study from Finland"* — plus ~37 other relevant abstracts.

**The catch:** ~38 documents are correct per query, and they come in two strengths — strongly relevant (directly cited) and weakly relevant (just on-topic). 95% are the weak kind.

---

## 2. fiqa — "answer this money question"

| | |
|---|---|
| **Input** | A personal-finance question, ~9 words |
| **Output** | The forum answer that answers it |
| **Corpus** | 57,600 StackExchange / Reddit answers |

**Example**
> **Input:** `מה נחשב להוצאה עסקית בנסיעת עסקים?`
> ("What is considered a business expense on a business trip?")
>
> **Output:** the answer post explaining which travel costs are deductible. ~2–3 answers count as correct.

**The catch:** none, really. Short question, medium answer, few correct answers, clean train/val/test splits. This is the most normal task of the five — use it as your sanity check.

---

## 3. scifact — "find the paper that settles this claim"

| | |
|---|---|
| **Input** | A one-sentence scientific claim, ~11 words |
| **Output** | The paper abstract holding the evidence |
| **Corpus** | 5,183 scientific abstracts |

**Example**
> **Input:** `לחומרים ביולוגיים 0-ממדיים חסרות תכונות השראתיות`
> ("0-dimensional biomaterials lack inductive properties")
>
> **Output:** the one abstract whose findings bear on that claim. Usually exactly 1 correct answer.

**The catch:** the model is **not** asked whether the claim is true. An abstract that *refutes* the claim is just as correct as one that supports it — the job is finding the paper that settles the question. Also, the match is to the whole abstract, not to the specific sentence, so correct pairs often look only loosely related.

---

## 4. arguana — "write the rebuttal… no, *find* it"

| | |
|---|---|
| **Input** | A full argument, ~140 words |
| **Output** | The argument that argues the opposite |
| **Corpus** | 8,674 debate passages |

**Example**
> **Input:** `להיות צמחוני עוזר לסביבה. חקלאות מודרנית היא אחד המקורות העיקריים לזיהום בנהרות שלנו...`
> ("Being vegetarian helps the environment. Modern farming is a major source of river pollution...")
>
> **Output:** `אתם לא חייבים להיות צמחונים כדי להיות ירוקים. סביבות מיוחדות רבות נוצרו על ידי גידול בעלי חיים...`
> ("You don't have to be vegetarian to be green. Many special environments were created by animal farming...")
>
> Exactly 1 correct answer per query.

**The catch:** this is the odd one out. The input is **long** — a whole paragraph, not a question — and the output is the same kind of text. So it's long-vs-long matching, not short-question-vs-passage. And it rewards finding text that **disagrees** with the input, while most retrieval models are trained to find text that's *similar* to the input. Those two goals pull in opposite directions.

---

## 5. scidocs — "which papers does this paper cite?"

| | |
|---|---|
| **Input** | A paper title, ~9 words |
| **Output** | Abstracts of the papers it cites |
| **Corpus** | 25,313 scientific abstracts |

**Example**
> **Input:** `שיטת חיפוש ישירה לפתרון בעיית שיגור כלכלי עם אפקט נקודת שסתום`
> ("A Direct Search Method to solve Economic Dispatch Problem with Valve-Point Effect")
>
> **Output:** the ~5 abstracts that paper cites, e.g. *"A hybrid of genetic algorithm and particle swarm optimization for recurrent network design"*.

**The catch:** the answer key also lists **25 wrong answers per query** — deliberately chosen same-field papers that are *not* cited. Anything reading this dataset must keep only rows with `score > 0`, or it will treat non-matches as matches. (This already caused a real bug in the review spreadsheet.) The upside: those 25 are excellent ready-made hard negatives.

---

## Side by side

| | Input | Output | Correct answers per query |
|---|---|---|---|
| **nfcorpus** | 2-word health topic | medical abstracts | ~38 |
| **fiqa** | finance question | forum answer | ~3 |
| **scifact** | scientific claim | paper abstract | ~1 |
| **arguana** | 140-word argument | the counter-argument | exactly 1 |
| **scidocs** | paper title | cited papers' abstracts | ~5 |

The single biggest difference is the **input length**: four datasets give the model a short phrase, arguana gives it a whole paragraph. The second biggest is **how many answers are correct** — which is why a good score on scifact and a good score on nfcorpus mean quite different things.
