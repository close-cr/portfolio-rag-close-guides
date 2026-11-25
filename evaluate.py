"""Evaluate the bot with questions that have a known answer.

Retrieval metrics (did the right chunk come back?):
  - hit@1 : the best chunk is from the right document AND contains the expected phrase
  - hit@3 : any of the top 3 chunks does
  - MRR   : mean of 1/rank of the first correct chunk
Refusal metric: how many out-of-scope questions are refused (should be all of them),
and how many valid questions are refused by mistake (should be none).

Usage: python evaluate.py [questions.json] [--no-synonyms]
"""
import json
import sys

from rag import REFUSAL, Index, answer, load_chunks, normalize

args = [a for a in sys.argv[1:] if not a.startswith("--")]
index = Index(load_chunks("corpus"), synonyms="--no-synonyms" not in sys.argv)
data = json.load(open(args[0] if args else "questions_dev.json", encoding="utf-8"))

print(f"Chunks indexed: {index.n}\n")
hit1 = hit3 = 0
rr = []
false_refusals = 0
for p in data["in_scope"]:
    results, _q = index.search(p["question"], 3)
    pos = None
    for i, (_s, c) in enumerate(results, 1):
        if c.doc in p["docs"] and any(x in normalize(c.text) for x in p["contains"]):
            pos = i
            break
    hit1 += pos == 1
    hit3 += pos is not None
    rr.append(1 / pos if pos else 0)
    text, _ = answer(index, p["question"])
    refused = text == REFUSAL
    false_refusals += refused
    mark = "✓" if pos == 1 else ("~" if pos else "✗")
    print(f"{mark} rank={pos} refused={'YES' if refused else 'no'}  {p['question']}")

n = len(data["in_scope"])
print(f"\nhit@1 = {hit1}/{n} ({hit1 / n:.0%})   hit@3 = {hit3}/{n} ({hit3 / n:.0%})   MRR = {sum(rr) / n:.2f}")
print(f"False refusals (valid questions the bot refused): {false_refusals}/{n}\n")

ok = 0
for question in data["out_of_scope"]:
    text, _ = answer(index, question)
    refused = text == REFUSAL
    ok += refused
    print(f"{'✓' if refused else '✗'} {'refused' if refused else 'ANSWERED (wrong)'}  {question}")
m = len(data["out_of_scope"])
print(f"\nCorrect refusal of out-of-scope questions: {ok}/{m} ({ok / m:.0%})")
