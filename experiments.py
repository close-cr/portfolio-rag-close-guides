"""README experiments: effect of the refusal threshold and of synonyms, plus a third question set.

Usage: python experiments.py
The third set (questions_third.json) was written BEFORE running this script.
"""
import json

import rag
from rag import REFUSAL, Index, answer, load_chunks, normalize

CHUNKS = load_chunks("corpus")


def evaluate(file, synonyms=True, threshold=0.5):
    rag.COVERAGE_THRESHOLD = threshold
    ix = Index(CHUNKS, synonyms=synonyms)
    data = json.load(open(file, encoding="utf-8"))
    n = len(data["in_scope"])
    hit1 = hit3 = false_refusals = 0
    for p in data["in_scope"]:
        res, _ = ix.search(p["question"], 3)
        pos = next((i for i, (_s, c) in enumerate(res, 1) if c.doc in p["docs"] and any(x in normalize(c.text) for x in p["contains"])), None)
        hit1 += pos == 1
        hit3 += pos is not None
        false_refusals += answer(ix, p["question"])[0] == REFUSAL
    m = len(data["out_of_scope"])
    ok = sum(answer(ix, q)[0] == REFUSAL for q in data["out_of_scope"])
    return dict(hit1=f"{hit1}/{n}", hit3=f"{hit3}/{n}", false_refusals=f"{false_refusals}/{n}", correct_refusal=f"{ok}/{m}")


print("## Experiment 1: refusal threshold (with synonyms)")
print("| Threshold | Set | False refusals (valid questions refused) | Correct refusal (off-topic) |")
print("|---|---|---|---|")
for u in (0.3, 0.5, 0.7):
    for name, f in (("dev", "questions_dev.json"), ("holdout", "questions_holdout.json"), ("third", "questions_third.json")):
        r = evaluate(f, True, u)
        print(f"| {u} | {name} | {r['false_refusals']} | {r['correct_refusal']} |")

print("\n## Experiment 2: third set (written before running), threshold 0.5")
print("| Version | hit@1 | hit@3 | False refusals | Correct refusal |")
print("|---|---|---|---|---|")
for name, s in (("BM25 baseline", False), ("+ synonyms", True)):
    r = evaluate("questions_third.json", s, 0.5)
    print(f"| {name} | {r['hit1']} | {r['hit3']} | {r['false_refusals']} | {r['correct_refusal']} |")
