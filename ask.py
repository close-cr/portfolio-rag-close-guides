"""Hace una pregunta al bot desde la terminal.

  python ask.py "¿Cuánto dura la ventana de WhatsApp?"
  python ask.py "..." --prompt     # muestra el prompt que se enviaría a un LLM
"""
import sys

from rag import Indice, cargar_fragmentos, construir_prompt, responder

if len(sys.argv) < 2:
    sys.exit(__doc__)
indice = Indice(cargar_fragmentos("corpus"))
pregunta = sys.argv[1]
texto, usados = responder(indice, pregunta)
print(texto)
if "--prompt" in sys.argv:
    print("\n--- PROMPT PARA UN LLM ---\n" + construir_prompt(pregunta, usados))
