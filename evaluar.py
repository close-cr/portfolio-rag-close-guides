"""Evalúa el bot con preguntas de respuesta conocida (preguntas.json).

Métricas de recuperación (¿llegó el fragmento correcto?):
  - hit@1  : el mejor fragmento es del documento correcto y contiene la frase esperada
  - hit@3  : alguno de los 3 mejores cumple lo mismo
  - MRR    : promedio de 1/posición del primer fragmento correcto
Métrica de rechazo: de las preguntas fuera de alcance, cuántas rechaza (debe rechazarlas todas)
y, de las de dentro, cuántas rechaza por error (no debería rechazar ninguna).
"""
import json
import sys

from rag import Indice, cargar_fragmentos, normalizar, responder

# Uso: python evaluar.py [archivo.json] [--sin-sinonimos]
args = [a for a in sys.argv[1:] if not a.startswith("--")]
indice = Indice(cargar_fragmentos("corpus"), sinonimos="--sin-sinonimos" not in sys.argv)
datos = json.load(open(args[0] if args else "preguntas.json", encoding="utf-8"))

print(f"Fragmentos indexados: {indice.N}\n")
hit1 = hit3 = 0
rr = []
falsos_rechazos = 0
for p in datos["dentro_del_alcance"]:
    res, _q = indice.buscar(p["pregunta"], 3)
    pos = None
    for i, (_s, f) in enumerate(res, 1):
        if f.doc in p["docs"] and any(c in normalizar(f.texto) for c in p["contiene"]):
            pos = i
            break
    hit1 += pos == 1
    hit3 += pos is not None
    rr.append(1 / pos if pos else 0)
    texto, _ = responder(indice, p["pregunta"])
    rechazo = texto.startswith("No tengo esa información")
    falsos_rechazos += rechazo
    marca = "✓" if pos == 1 else ("~" if pos else "✗")
    print(f"{marca} pos={pos} rechazo={'SÍ' if rechazo else 'no'}  {p['pregunta']}")

n = len(datos["dentro_del_alcance"])
print(f"\nhit@1 = {hit1}/{n} ({hit1 / n:.0%})   hit@3 = {hit3}/{n} ({hit3 / n:.0%})   MRR = {sum(rr) / n:.2f}")
print(f"Falsos rechazos (preguntas válidas que el bot rechazó): {falsos_rechazos}/{n}\n")

ok = 0
for pregunta in datos["fuera_de_alcance"]:
    texto, _ = responder(indice, pregunta)
    rechazo = texto.startswith("No tengo esa información")
    ok += rechazo
    print(f"{'✓' if rechazo else '✗'} {'rechaza' if rechazo else 'RESPONDE (mal)'}  {pregunta}")
m = len(datos["fuera_de_alcance"])
print(f"\nRechazo correcto de preguntas fuera de alcance: {ok}/{m} ({ok / m:.0%})")
