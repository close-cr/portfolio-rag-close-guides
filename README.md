🇬🇧 English · 🇪🇸 [Español](README.es.md)

# RAG bot over the CLOSE guides (with evaluation)

A bot that answers questions **only from the public guides on [closecr.com](https://closecr.com)**, cites its source, and says "I don't have that information" when the question is unrelated. It ships with an evaluation on questions with known answers, which is the part that matters most.

**Cost: zero. No network, no API keys, standard library only (Python 3.10+).** It runs in under a second.

> The guides are written in Spanish and the bot answers Spanish-speaking customers, so the corpus, the example questions and the bot's replies are in Spanish. The code, comments and documentation are in English.

## What it does

```
HTML guides ──extract_corpus.py──▶ text ──split by section (~110 words, 1-line overlap)──▶ 59 chunks
                                                                                              │
question ──tokenize──▶ BM25 (+ synonyms) ──▶ top-3 ──▶ is coverage high enough? ──no──▶ "No tengo esa información"
                                                                  │ yes
                                                                  ▼
                                                  cited answer  /  prompt ready for an LLM
```

| File | Purpose |
|---|---|
| `extract_corpus.py` | Turns the 5 HTML guides into plain text (visible text only). |
| `rag.py` | Chunking, a hand-written BM25 index, refusal logic and prompt building. |
| `ask.py` | Ask from the terminal (`--prompt` prints the prompt you would send to an LLM). |
| `evaluate.py` | Measures hit@1, hit@3, MRR and correct refusal. |
| `questions_dev.json` | 15 in-scope + 5 out-of-scope questions (development set). |
| `questions_holdout.json` | 10 + 5 questions written **before** adding synonyms (verification set). |
| `questions_third.json` | 5 + 3 more, written before running the experiments. |
| `experiments.py` | Refusal-threshold and synonym experiments (results below). |

## Run it

```bash
python3 ask.py "¿Cuánto dura la ventana en la que puedo contestar libremente en WhatsApp?"
python3 ask.py "¿Cuál es la capital de Francia?"            # refuses
python3 evaluate.py                                          # full evaluation
python3 evaluate.py questions_holdout.json --no-synonyms     # baseline version
```

## Design decisions (and why)

- **BM25 instead of embeddings.** For 5 documents and 59 chunks, BM25 is free, deterministic, and explainable in 40 lines. Embeddings are the natural next step (see below).
- **Chunk by section** (`##` headings), not by fixed size: each chunk keeps its section title, which is a strong retrieval signal.
- **Refuse before answering.** The bot computes an *idf-weighted coverage*: what fraction of the question's important terms appear in the best chunk. A term that exists nowhere in the corpus (e.g. "iPhone") gets the maximum weight, so it lowers coverage. Threshold: 0.5.
- **Extractive answer by default**, so there is no dependency on a paid LLM. `build_prompt()` builds the prompt you would send to one, instructing it to answer only from the context and to cite.

## Evaluation results

Measured with `evaluate.py`, exactly as in this repo.

| Set | Version | hit@1 | hit@3 | MRR | False refusals | Correct refusal (off-topic) |
|---|---|---|---|---|---|---|
| Development (15 + 5) | BM25 baseline | 53 % | 73 % | 0.63 | 3/15 | 5/5 |
| Development (15 + 5) | + synonyms | 60 % | 93 % | 0.77 | 2/15 | 5/5 |
| **Holdout (10 + 5)** | BM25 baseline | 50 % | 60 % | 0.53 | 2/10 | 4/5 |
| **Holdout (10 + 5)** | + synonyms | 40 % | 70 % | 0.53 | 2/10 | 5/5 |

**How to read this honestly**

- The synonym list was written while looking at the errors of the development set, so those numbers are **inflated**. The holdout set, written earlier, tells the truth: synonyms raise hit@3 (60 → 70 %) but **lower hit@1 (50 → 40 %)**, and MRR does not change. With only 10 questions that difference is closer to noise than signal.
- What is solid: **refusal works** (5/5 on both sets with synonyms). "What is the dollar price today?" slipped through in the baseline because "dólares" does appear in the guides, a good example of the lexical limitation.
- The retrieval errors share one pattern: **the question and the guide use different words for the same thing** ("how is it charged?" vs "how much does it cost", "register twice" vs "avoids duplicates", "cell phone" vs "telephone"). This is the known limitation of keyword search.

## Experiments

Run with `python3 experiments.py`. Real output. The third set was written **before** running anything.

### 1. What does the refusal threshold do? (with synonyms)

| Threshold | Set | False refusals (valid questions refused) | Correct refusal (off-topic) |
|---|---|---|---|
| 0.3 | dev | 0/15 | 4/5 |
| 0.3 | holdout | 0/10 | 3/5 |
| 0.3 | third | 0/5 | 1/3 |
| **0.5** | dev | 2/15 | 5/5 |
| **0.5** | holdout | 2/10 | 5/5 |
| **0.5** | third | 0/5 | 3/3 |
| 0.7 | dev | 7/15 | 5/5 |
| 0.7 | holdout | 4/10 | 5/5 |
| 0.7 | third | 3/5 | 3/3 |

**Reading:** it is a clear trade-off. A low threshold (0.3) never refuses a valid question but lets off-topic ones through (only 1 of 3 refused on the third set). A high one (0.7) refuses everything off-topic but also 7 of 15 valid questions. 0.5 balances both; it was chosen before these experiments, but with so few questions it cannot be claimed to be optimal.

### 2. Third set (threshold 0.5)

| Version | hit@1 | hit@3 | False refusals | Correct refusal |
|---|---|---|---|---|
| BM25 baseline | 4/5 | 4/5 | 0/5 | 3/3 |
| + synonyms | 4/5 | 4/5 | 0/5 | 3/3 |

**Reading:** here synonyms change nothing. Together with the holdout set (hit@1 went down, hit@3 went up), the evidence is that **handwritten synonyms help little and inconsistently**; the real next step is embeddings.

## Limitations

- Lexical search only: it does not understand paraphrases. The ~8 synonym groups are a manual patch.
- The evaluation set is small (43 questions in three sets) and was written by the person who wrote the system.
- Generation with a real LLM is not included, so **answer wording quality was not measured**, only retrieval and refusal.
- With a larger corpus the 0.5 refusal threshold would need re-tuning.

## Next steps

1. **Hybrid search**: BM25 + embeddings (for example local `sentence-transformers`, no per-call cost) and merge the rankings.
2. A **reranker** over the top 10 chunks.
3. Plug an LLM into `answer()` and measure faithfulness: is the answer supported by the cited chunk?
4. Grow the question set with real users' questions and split it into dev and holdout **before** touching the system.

## Data source

The guides are public CLOSE content (closecr.com), copied as text into `corpus/` with their URL. If they change on the web, re-run `extract_corpus.py`.
