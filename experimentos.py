"""Experimentos del README: efecto del umbral de rechazo y de los sinónimos, y un tercer conjunto de preguntas.

Uso: python3 experimentos.py
El tercer conjunto (preguntas_3.json) se escribió ANTES de correr este script.
"""
import json

import rag
from rag import Indice, cargar_fragmentos, normalizar, responder

FRAGS = cargar_fragmentos("corpus")
RECHAZO = "No tengo esa información"


def evaluar(archivo, sinonimos=True, umbral=0.5):
    rag.UMBRAL_COBERTURA = umbral
    ix = Indice(FRAGS, sinonimos=sinonimos)
    datos = json.load(open(archivo, encoding="utf-8"))
    n = len(datos["dentro_del_alcance"])
    hit1 = hit3 = falsos = 0
    for p in datos["dentro_del_alcance"]:
        res, _ = ix.buscar(p["pregunta"], 3)
        pos = next((i for i, (_s, f) in enumerate(res, 1) if f.doc in p["docs"] and any(c in normalizar(f.texto) for c in p["contiene"])), None)
        hit1 += pos == 1
        hit3 += pos is not None
        falsos += responder(ix, p["pregunta"])[0].startswith(RECHAZO)
    m = len(datos["fuera_de_alcance"])
    ok = sum(responder(ix, q)[0].startswith(RECHAZO) for q in datos["fuera_de_alcance"])
    return dict(hit1=f"{hit1}/{n}", hit3=f"{hit3}/{n}", falsos_rechazos=f"{falsos}/{n}", rechazo_correcto=f"{ok}/{m}")


print("## Experimento 1: umbral de cobertura (con sinónimos)")
print("| Umbral | Conjunto | Falsos rechazos (válidas rechazadas) | Rechazo correcto (fuera de tema) |")
print("|---|---|---|---|")
for u in (0.3, 0.5, 0.7):
    for nombre, f in (("desarrollo", "preguntas.json"), ("verificación", "preguntas_nuevas.json"), ("tercer conjunto", "preguntas_3.json")):
        r = evaluar(f, True, u)
        print(f"| {u} | {nombre} | {r['falsos_rechazos']} | {r['rechazo_correcto']} |")

print("\n## Experimento 2: tercer conjunto (escrito antes de correr), umbral 0,5")
print("| Versión | hit@1 | hit@3 | Falsos rechazos | Rechazo correcto |")
print("|---|---|---|---|---|")
for nombre, s in (("BM25 base", False), ("+ sinónimos", True)):
    r = evaluar("preguntas_3.json", s, 0.5)
    print(f"| {nombre} | {r['hit1']} | {r['hit3']} | {r['falsos_rechazos']} | {r['rechazo_correcto']} |")
