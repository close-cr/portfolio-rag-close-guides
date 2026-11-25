"""Tiny dependency-free RAG: chunking, BM25 retrieval, refusal and prompt building.

Designed to cost nothing and run offline:
  - Lexical retrieval with BM25 (implemented here in ~40 lines).
  - Explicit refusal when the question looks like nothing in the corpus.
  - The generation step is pluggable: by default the bot returns the cited chunk (extractive answer);
    `build_prompt` builds the prompt you would send to an LLM.

The corpus (public guides from closecr.com) is in Spanish, so stopwords, the refusal message and the
LLM instructions are Spanish on purpose: the bot answers Spanish-speaking customers.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SPANISH_STOPWORDS = set(
    """a al algo ante como con cual cuales cuando cuanto cuanta cuantos cuantas de del donde el ella
    ellos en es esta este esto estos fue ha hace hacen hay la las le les lo los me mi mis mas muy no o
    para pero por que se si sin sobre son su sus te tu tus un una uno unos unas y ya puedo puede pueden
    tiene tienen hacer ser esta estan quien quienes cual cuales""".split()
)


def normalize(text: str) -> str:
    """Lowercase and strip accents."""
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def tokenize(text: str) -> list[str]:
    tokens = []
    for t in re.findall(r"[a-z0-9]+", normalize(text)):
        if t in SPANISH_STOPWORDS or len(t) < 2:
            continue
        if len(t) > 4:  # minimal "stemming": simple plurals
            t = re.sub(r"(es|s)$", "", t)
        tokens.append(t)
    return tokens


# Generic Spanish synonyms (handwritten, ~8 groups). A cheap patch for BM25's lexical limitation;
# the real fix is embeddings (see README).
SYNONYM_GROUPS = [
    ["cobrar", "costar", "cuesta", "precio", "pagar", "pagas", "tarifa"],
    ["falta", "necesitar", "necesito", "necesitas", "requiere", "requisitos"],
    ["generar", "genera", "incluir", "incluye", "produce", "entrega", "trae"],
    ["duplicado", "repetir", "repite", "vez", "veces", "doble"],
    ["celular", "telefono", "movil", "app"],
    ["tardar", "tarda", "tardaba", "demora", "duracion", "dura"],
    ["error", "falla", "fallar"],
    ["empezar", "iniciar", "comenzar", "arrancar", "marcha"],
]


def _synonym_map() -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for group in SYNONYM_GROUPS:
        toks = {t for w in group for t in tokenize(w)}
        for t in toks:
            mapping.setdefault(t, set()).update(toks - {t})
    return mapping


@dataclass
class Chunk:
    doc: str
    section: str
    text: str
    url: str
    tokens: list[str]


def load_chunks(folder: str | Path, max_words: int = 110) -> list[Chunk]:
    """Split each document by section (## headings) and cut long sections into smaller chunks."""
    out: list[Chunk] = []
    for f in sorted(Path(folder).glob("*.txt")):
        lines = f.read_text(encoding="utf-8").splitlines()
        url = lines[0].replace("URL:", "").strip()
        section, buffer = "Introducción", []

        def flush():
            nonlocal buffer
            if not buffer:
                return
            current: list[str] = []
            for line in buffer:
                if current and sum(len(x.split()) for x in current) + len(line.split()) > max_words:
                    text = " ".join(current)
                    out.append(Chunk(f.stem, section, text, url, tokenize(section + " " + text)))
                    current = current[-1:]  # one-line overlap so ideas are not cut in half
                current.append(line)
            text = " ".join(current)
            out.append(Chunk(f.stem, section, text, url, tokenize(section + " " + text)))
            buffer = []

        for line in lines[3:]:
            if line.startswith("## ") or line.startswith("### "):
                flush()
                section = line.lstrip("# ").strip()
            elif line.strip() and not line.startswith("# "):
                buffer.append(line.strip())
        flush()
    return out


class Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75, synonyms: bool = True):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.syn = _synonym_map() if synonyms else {}
        self.n = len(chunks)
        self.avg_len = sum(len(c.tokens) for c in chunks) / self.n
        df = Counter(t for c in chunks for t in set(c.tokens))
        self.idf = {t: math.log(1 + (self.n - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.idf_max = max(self.idf.values())  # weight of a term that exists nowhere in the corpus
        self.tf = [Counter(c.tokens) for c in chunks]

    def score(self, query: list[str], i: int) -> float:
        s, tf, length = 0.0, self.tf[i], len(self.chunks[i].tokens)
        for t in query:
            if t in tf:
                num = tf[t] * (self.k1 + 1)
                den = tf[t] + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                s += self.idf[t] * num / den
        return s

    def search(self, question: str, k: int = 3):
        q = tokenize(question)
        expanded = q + [x for t in q for x in sorted(self.syn.get(t, ())) if x not in q]
        ranking = sorted(((self.score(expanded, i), i) for i in range(self.n)), reverse=True)[:k]
        return [(s, self.chunks[i]) for s, i in ranking], q

    def coverage(self, q: list[str], chunk: Chunk) -> float:
        """Idf-weighted fraction of the question's terms that appear in the chunk.
        A term missing from the WHOLE corpus gets the maximum weight, so «iPhone» lowers coverage."""
        if not q:
            return 0.0
        weight = lambda t: self.idf.get(t, self.idf_max)
        total = sum(weight(t) for t in q)
        present = set(chunk.tokens)
        return sum(weight(t) for t in q if t in present or self.syn.get(t, set()) & present) / total


COVERAGE_THRESHOLD = 0.5  # below this the bot says "I don't know"
REFUSAL = "No tengo esa información en las guías de CLOSE."  # the bot answers Spanish-speaking customers

INSTRUCTIONS = (
    "Responde SOLO con el contexto de abajo. Si la respuesta no está en el contexto, "
    f"responde exactamente: «{REFUSAL}» "
    "Cita la fuente entre corchetes."
)


def answer(index: Index, question: str, k: int = 3):
    """Return (text, chunks_used). Refuses when the best chunk's coverage is low."""
    results, q = index.search(question, k)
    if not results or results[0][0] <= 0 or index.coverage(q, results[0][1]) < COVERAGE_THRESHOLD:
        return REFUSAL, []
    best = results[0][1]
    return f"{best.text}\n[Fuente: {best.url} · «{best.section}»]", [r[1] for r in results]


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(f"[{c.url} · {c.section}]\n{c.text}" for c in chunks) or "(sin contexto relevante)"
    return f"{INSTRUCTIONS}\n\nContexto:\n{context}\n\nPregunta: {question}"
