"""Convert the public closecr.com guides (HTML) into plain text for the RAG.

Usage: python extract_corpus.py <html_folder> <output_folder>
Extracts visible text only (no scripts, styles or menus) and keeps the headings.
"""
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

SKIP = {"script", "style", "nav", "footer", "header", "svg", "noscript", "form", "button"}
HEADINGS = {"h1": "# ", "h2": "## ", "h3": "### "}
BLOCKS = {"p", "li", "h1", "h2", "h3", "div", "section", "article", "tr", "br"}


class Extractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.out, self.current = [], [], ""

    def _close(self):
        t = re.sub(r"\s+", " ", self.current).strip()
        if t:
            self.out.append(t)
        self.current = ""

    def handle_starttag(self, tag, attrs):
        self.stack.append(tag)
        if tag in BLOCKS:
            self._close()
            if tag in HEADINGS:
                self.current = HEADINGS[tag]

    def handle_endtag(self, tag):
        if tag in BLOCKS:
            self._close()
        while self.stack and self.stack.pop() != tag:
            pass

    def handle_data(self, data):
        if not any(t in SKIP for t in self.stack):
            self.current += data


def main(source, target):
    Path(target).mkdir(exist_ok=True)
    for f in sorted(Path(source).glob("*.html")):
        if f.stem.startswith("google") or f.stem.startswith("index") or f.stem in {"privacidad", "recursos-github"}:
            continue
        html = f.read_text(encoding="utf-8")
        title = re.search(r"<title>(.*?)</title>", html, re.S)
        ex = Extractor()
        ex.feed(html)
        ex._close()
        body = "\n".join(ex.out)
        url = f"https://closecr.com/{f.name}"
        (Path(target) / f"{f.stem}.txt").write_text(f"URL: {url}\nTITULO: {title.group(1).strip() if title else f.stem}\n\n{body}\n", encoding="utf-8")
        print(f"{f.name}: {len(body.split())} words")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
