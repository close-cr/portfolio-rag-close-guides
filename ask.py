"""Ask the bot a question from the terminal.

  python ask.py "¿Cuánto dura la ventana de WhatsApp?"
  python ask.py "..." --prompt     # also print the prompt you would send to an LLM
"""
import sys

from rag import Index, build_prompt, load_chunks, answer

if len(sys.argv) < 2:
    sys.exit(__doc__)
index = Index(load_chunks("corpus"))
question = sys.argv[1]
text, used = answer(index, question)
print(text)
if "--prompt" in sys.argv:
    print("\n--- PROMPT FOR AN LLM ---\n" + build_prompt(question, used))
