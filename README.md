# Bot RAG sobre las guías de CLOSE (con evaluación)

Un bot que responde preguntas **solo con el contenido de las guías públicas de [closecr.com](https://closecr.com)**, cita la fuente y dice «no tengo esa información» cuando la pregunta no tiene que ver con ellas. Incluye una evaluación con preguntas de respuesta conocida, que es la parte que más importa.

**Costo: cero. Sin red, sin API keys, solo la biblioteca estándar de Python (3.10+).** Corre en menos de un segundo.

## Qué hace

```
guías HTML ──extraer_corpus.py──▶ texto ──fragmentar por secciones (~110 palabras, solape de 1 línea)──▶ 59 fragmentos
                                                                                                           │
pregunta ──tokenizar──▶ BM25 (+ sinónimos) ──▶ top-3 ──▶ ¿la cobertura es suficiente? ──no──▶ «No tengo esa información»
                                                                         │ sí
                                                                         ▼
                                                    respuesta citada  /  prompt listo para un LLM
```

| Archivo | Para qué sirve |
|---|---|
| `extraer_corpus.py` | Convierte las 5 guías HTML en texto plano (solo texto visible). |
| `rag.py` | Fragmentación, índice BM25 implementado a mano, rechazo y armado del prompt. |
| `ask.py` | Preguntar desde la terminal (`--prompt` muestra el prompt que iría a un LLM). |
| `evaluar.py` | Mide hit@1, hit@3, MRR y rechazo correcto. |
| `preguntas.json` | 15 preguntas de dentro del alcance + 5 de fuera (conjunto de desarrollo). |
| `preguntas_nuevas.json` | 10 + 5 preguntas escritas **antes** de añadir los sinónimos (conjunto de verificación). |

## Cómo correrlo

```bash
python3 ask.py "¿Cuánto dura la ventana en la que puedo contestar libremente en WhatsApp?"
python3 ask.py "¿Cuál es la capital de Francia?"       # rechaza
python3 evaluar.py                                       # evaluación completa
python3 evaluar.py preguntas_nuevas.json --sin-sinonimos # versión base
```

## Decisiones de diseño (y por qué)

- **BM25 en lugar de embeddings.** Para un corpus de 5 documentos y 59 fragmentos, BM25 es gratis, determinista y se explica en 40 líneas. Los embeddings serían el siguiente paso (ver abajo).
- **Fragmentar por secciones** (`##`), no por tamaño fijo: cada fragmento conserva el título de su sección, que es una señal fuerte para la búsqueda.
- **Rechazar antes de responder.** Se calcula la *cobertura ponderada por idf*: qué fracción de los términos importantes de la pregunta aparece en el mejor fragmento. Un término que no existe en todo el corpus (por ejemplo «iPhone») pesa lo máximo, así que baja la cobertura. Umbral: 0,5.
- **Respuesta extractiva por defecto**, para no depender de un LLM de pago. `construir_prompt()` arma el prompt que se le daría a uno, con la instrucción de responder solo con el contexto y citar.

## Resultados de la evaluación

Medidos con `evaluar.py`, tal como están en el repo.

| Conjunto | Versión | hit@1 | hit@3 | MRR | Falsos rechazos | Rechazo correcto (fuera de alcance) |
|---|---|---|---|---|---|---|
| Desarrollo (15 + 5) | BM25 base | 53 % | 73 % | 0,63 | 3/15 | 5/5 |
| Desarrollo (15 + 5) | + sinónimos | 60 % | 93 % | 0,77 | 2/15 | 5/5 |
| **Verificación (10 + 5)** | BM25 base | 50 % | 60 % | 0,53 | 2/10 | 4/5 |
| **Verificación (10 + 5)** | + sinónimos | 40 % | 70 % | 0,53 | 2/10 | 5/5 |

**Cómo leer esto con honestidad**

- La lista de sinónimos se escribió viendo los errores del conjunto de desarrollo, así que esas cifras están **infladas**. El conjunto de verificación, escrito antes, es el que dice la verdad: los sinónimos suben hit@3 (60 → 70 %) pero **bajan hit@1 (50 → 40 %)** y el MRR no cambia. Con solo 10 preguntas, esa diferencia es ruido más que señal.
- Lo que sí es sólido: **el rechazo funciona** (5/5 en ambos conjuntos con sinónimos). La pregunta «precio del dólar hoy» se coló en la versión base porque «dólares» sí aparece en las guías: un buen ejemplo de la limitación léxica.
- Los errores de recuperación tienen un patrón claro: **la pregunta y la guía usan palabras distintas para lo mismo** («¿cómo se cobra?» vs «cuánto cuesta», «registrar dos veces» vs «evita duplicados», «celular» vs «teléfono»). Es la limitación conocida de la búsqueda por palabras.

## Qué limita esta versión

- Búsqueda solo léxica: no entiende paráfrasis. Los ~8 grupos de sinónimos son un parche manual.
- El conjunto de evaluación es pequeño (25 preguntas) y lo escribió quien escribió el sistema.
- El paso de generación no usa un LLM real, así que **no se midió la calidad de la redacción**, solo la recuperación y el rechazo.
- Las guías son 5 páginas; con un corpus mayor habría que revisar el umbral de rechazo.

## Siguientes pasos

1. Búsqueda **híbrida**: BM25 + embeddings (por ejemplo `sentence-transformers` en local, sin costo por llamada) y combinar los rankings.
2. **Reranker** sobre los 10 mejores fragmentos.
3. Conectar un LLM en `responder()` y medir fidelidad: ¿la respuesta se apoya en el fragmento citado?
4. Crecer el conjunto de preguntas con las que hagan personas reales y repartirlas en desarrollo y verificación **antes** de tocar el sistema.

## Fuente de los datos

Las guías son contenido público de CLOSE (closecr.com), copiado como texto en `corpus/` con su URL. Si cambian en la web, se vuelve a correr `extraer_corpus.py`.
