"""Mini RAG sin dependencias: fragmentación, búsqueda BM25, rechazo y armado del prompt.

Diseño pensado para costar cero y correr sin red:
  - Recuperación léxica con BM25 (implementada aquí, ~40 líneas).
  - Rechazo explícito cuando la pregunta no se parece a nada del corpus.
  - El paso «Generation» es intercambiable: por defecto devuelve el fragmento citado
    (respuesta extractiva); `construir_prompt` arma el prompt que se le daría a un LLM.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PALABRAS_VACIAS = set(
    """a al algo ante como con cual cuales cuando cuanto cuanta cuantos cuantas de del donde el ella
    ellos en es esta este esto estos fue ha hace hacen hay la las le les lo los me mi mis mas muy no o
    para pero por que se si sin sobre son su sus te tu tus un una uno unos unas y ya puedo puede pueden
    tiene tienen hacer ser esta estan quien quienes cual cuales""".split()
)


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def tokenizar(texto: str) -> list[str]:
    toks = []
    for t in re.findall(r"[a-z0-9]+", normalizar(texto)):
        if t in PALABRAS_VACIAS or len(t) < 2:
            continue
        if len(t) > 4:  # «stemming» mínimo: plural simple
            t = re.sub(r"(es|s)$", "", t)
        toks.append(t)
    return toks


# Sinónimos genéricos del español (escritos a mano, ~8 grupos). Es un parche barato a la
# limitación léxica de BM25; la solución de fondo son los embeddings (ver README).
GRUPOS_SINONIMOS = [
    ["cobrar", "costar", "cuesta", "precio", "pagar", "pagas", "tarifa"],
    ["falta", "necesitar", "necesito", "necesitas", "requiere", "requisitos"],
    ["generar", "genera", "incluir", "incluye", "produce", "entrega", "trae"],
    ["duplicado", "repetir", "repite", "vez", "veces", "doble"],
    ["celular", "telefono", "movil", "app"],
    ["tardar", "tarda", "tardaba", "demora", "duracion", "dura"],
    ["error", "falla", "fallar"],
    ["empezar", "iniciar", "comenzar", "arrancar", "marcha"],
]


def _mapa_sinonimos() -> dict[str, set[str]]:
    mapa: dict[str, set[str]] = {}
    for grupo in GRUPOS_SINONIMOS:
        toks = {t for w in grupo for t in tokenizar(w)}
        for t in toks:
            mapa.setdefault(t, set()).update(toks - {t})
    return mapa


@dataclass
class Fragmento:
    doc: str
    seccion: str
    texto: str
    url: str
    tokens: list[str]


def cargar_fragmentos(carpeta: str | Path, max_palabras: int = 110) -> list[Fragmento]:
    """Parte cada documento por secciones (## títulos) y las trocea si son largas."""
    out: list[Fragmento] = []
    for f in sorted(Path(carpeta).glob("*.txt")):
        lineas = f.read_text(encoding="utf-8").splitlines()
        url = lineas[0].replace("URL:", "").strip()
        seccion, buffer = "Introducción", []

        def cerrar():
            nonlocal buffer
            if not buffer:
                return
            actual: list[str] = []
            for linea in buffer:
                if actual and sum(len(x.split()) for x in actual) + len(linea.split()) > max_palabras:
                    texto = " ".join(actual)
                    out.append(Fragmento(f.stem, seccion, texto, url, tokenizar(seccion + " " + texto)))
                    actual = actual[-1:]  # solape de una línea para no cortar ideas
                actual.append(linea)
            texto = " ".join(actual)
            out.append(Fragmento(f.stem, seccion, texto, url, tokenizar(seccion + " " + texto)))
            buffer = []

        for linea in lineas[3:]:
            if linea.startswith("## ") or linea.startswith("### "):
                cerrar()
                seccion = linea.lstrip("# ").strip()
            elif linea.strip() and not linea.startswith("# "):
                buffer.append(linea.strip())
        cerrar()
    return out


class Indice:
    def __init__(self, fragmentos: list[Fragmento], k1: float = 1.5, b: float = 0.75, sinonimos: bool = True):
        self.f, self.k1, self.b = fragmentos, k1, b
        self.sin = _mapa_sinonimos() if sinonimos else {}
        self.N = len(fragmentos)
        self.largo_prom = sum(len(x.tokens) for x in fragmentos) / self.N
        df = Counter(t for x in fragmentos for t in set(x.tokens))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.idf_max = max(self.idf.values())  # peso de un término que no existe en el corpus
        self.tf = [Counter(x.tokens) for x in fragmentos]

    def puntuar(self, consulta: list[str], i: int) -> float:
        s, tf, largo = 0.0, self.tf[i], len(self.f[i].tokens)
        for t in consulta:
            if t in tf:
                num = tf[t] * (self.k1 + 1)
                den = tf[t] + self.k1 * (1 - self.b + self.b * largo / self.largo_prom)
                s += self.idf[t] * num / den
        return s

    def buscar(self, pregunta: str, k: int = 3):
        q = tokenizar(pregunta)
        expandida = q + [x for t in q for x in sorted(self.sin.get(t, ())) if x not in q]
        ranking = sorted(((self.puntuar(expandida, i), i) for i in range(self.N)), reverse=True)[:k]
        return [(s, self.f[i]) for s, i in ranking], q

    def cobertura(self, q: list[str], frag: Fragmento) -> float:
        """Fracción (ponderada por idf) de los términos de la pregunta que aparecen en el fragmento.
        Un término ausente de TODO el corpus pesa lo máximo: así «iPhone» baja la cobertura."""
        if not q:
            return 0.0
        peso = lambda t: self.idf.get(t, self.idf_max)
        total = sum(peso(t) for t in q)
        presentes = set(frag.tokens)
        return sum(peso(t) for t in q if t in presentes or self.sin.get(t, set()) & presentes) / total


UMBRAL_COBERTURA = 0.5  # por debajo, el bot dice «no lo sé»

INSTRUCCIONES = (
    "Responde SOLO con el contexto de abajo. Si la respuesta no está en el contexto, "
    "responde exactamente: «No tengo esa información en las guías de CLOSE.» "
    "Cita la fuente entre corchetes."
)


def responder(indice: Indice, pregunta: str, k: int = 3):
    """Devuelve (texto, fragmentos_usados). Rechaza si la cobertura del mejor fragmento es baja."""
    resultados, q = indice.buscar(pregunta, k)
    if not resultados or resultados[0][0] <= 0 or indice.cobertura(q, resultados[0][1]) < UMBRAL_COBERTURA:
        return "No tengo esa información en las guías de CLOSE.", []
    mejor = resultados[0][1]
    return f"{mejor.texto}\n[Fuente: {mejor.url} · «{mejor.seccion}»]", [r[1] for r in resultados]


def construir_prompt(pregunta: str, fragmentos: list[Fragmento]) -> str:
    ctx = "\n\n".join(f"[{f.url} · {f.seccion}]\n{f.texto}" for f in fragmentos) or "(sin contexto relevante)"
    return f"{INSTRUCCIONES}\n\nContexto:\n{ctx}\n\nPregunta: {pregunta}"
