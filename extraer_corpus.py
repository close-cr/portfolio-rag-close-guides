"""Convierte las guías públicas de closecr.com (HTML) en texto plano para el RAG.

Uso: python extraer_corpus.py <carpeta_con_html> <carpeta_salida>
Solo extrae texto visible (sin scripts, estilos ni menús) y conserva los títulos.
"""
import re, sys
from html.parser import HTMLParser
from pathlib import Path

SALTAR = {"script", "style", "nav", "footer", "header", "svg", "noscript", "form", "button"}
TITULOS = {"h1": "# ", "h2": "## ", "h3": "### "}
BLOQUES = {"p", "li", "h1", "h2", "h3", "div", "section", "article", "tr", "br"}


class Extractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.pila, self.salida, self.actual = [], [], ""

    def _cerrar(self):
        t = re.sub(r"\s+", " ", self.actual).strip()
        if t:
            self.salida.append(t)
        self.actual = ""

    def handle_starttag(self, tag, attrs):
        self.pila.append(tag)
        if tag in BLOQUES:
            self._cerrar()
            if tag in TITULOS:
                self.actual = TITULOS[tag]

    def handle_endtag(self, tag):
        if tag in BLOQUES:
            self._cerrar()
        while self.pila and self.pila.pop() != tag:
            pass

    def handle_data(self, data):
        if not any(t in SALTAR for t in self.pila):
            self.actual += data


def main(origen, destino):
    Path(destino).mkdir(exist_ok=True)
    for f in sorted(Path(origen).glob("*.html")):
        if f.stem.startswith("google") or f.stem.startswith("index") or f.stem in {"privacidad", "recursos-github"}:
            continue
        html = f.read_text(encoding="utf-8")
        titulo = re.search(r"<title>(.*?)</title>", html, re.S)
        ex = Extractor()
        ex.feed(html)
        ex._cerrar()
        cuerpo = "\n".join(ex.salida)
        url = f"https://closecr.com/{f.name}"
        (Path(destino) / f"{f.stem}.txt").write_text(f"URL: {url}\nTITULO: {titulo.group(1).strip() if titulo else f.stem}\n\n{cuerpo}\n", encoding="utf-8")
        print(f"{f.name}: {len(cuerpo.split())} palabras")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
