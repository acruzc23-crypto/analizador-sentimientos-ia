"""Genera un dataset sintético de reseñas de productos/servicios en español.

Se combinan plantillas, aspectos y opiniones para obtener frases variadas.
Uso: python generar_datos.py
"""
import csv
import random
from pathlib import Path

ASPECTOS = ["la atención", "el producto", "la entrega", "la calidad", "el precio", "el servicio",
            "la aplicación", "el envío", "el soporte técnico", "la comida", "el hotel", "el curso"]

POSITIVAS = ["excelente", "muy buena", "increíble", "rápida y eficiente", "maravillosa", "perfecta",
             "mejor de lo que esperaba", "de muy buena calidad", "genial", "súper recomendable"]
NEGATIVAS = ["pésima", "muy mala", "terrible", "lenta y deficiente", "horrible", "decepcionante",
             "peor de lo que esperaba", "de muy mala calidad", "un desastre", "nada recomendable"]

PLANTILLAS = [
    "{a} fue {o}",
    "me pareció que {a} es {o}",
    "sinceramente {a} estuvo {o}",
    "en general {a} resultó {o}",
    "{a} es {o}, {extra}",
    "la verdad {a} fue {o}",
]
EXTRA_POS = ["volveré a comprar", "lo recomiendo a todos", "estoy muy satisfecho", "gracias por todo"]
EXTRA_NEG = ["no volveré a comprar", "no lo recomiendo", "estoy muy molesto", "quiero mi dinero de vuelta"]

FRASES_POS = [
    "me encantó, cinco estrellas", "todo llegó a tiempo y en perfecto estado", "el personal fue muy amable",
    "funciona de maravilla", "vale cada centavo", "superó mis expectativas", "estoy feliz con mi compra",
    "no tuve ningún problema, todo bien", "muy contento con el resultado", "atención rápida y amable",
]
FRASES_NEG = [
    "no me gustó para nada", "llegó roto y tarde", "nadie responde los mensajes", "dejó de funcionar al día siguiente",
    "es un robo, carísimo", "no cumple lo que promete", "me arrepiento de esta compra",
    "no funciona, muy mal", "estoy decepcionado con el resultado", "atención lenta y grosera",
]


def generar(n_por_clase: int = 150, semilla: int = 42) -> list[tuple[str, str]]:
    rnd = random.Random(semilla)
    filas = set()
    while len([f for f in filas if f[1] == "positivo"]) < n_por_clase:
        filas.add((_frase(rnd, POSITIVAS, EXTRA_POS, FRASES_POS), "positivo"))
    while len([f for f in filas if f[1] == "negativo"]) < n_por_clase:
        filas.add((_frase(rnd, NEGATIVAS, EXTRA_NEG, FRASES_NEG), "negativo"))
    filas = sorted(filas)
    rnd.shuffle(filas)
    return filas


def _frase(rnd, opiniones, extras, frases):
    if rnd.random() < 0.3:
        return rnd.choice(frases).capitalize() + "."
    texto = rnd.choice(PLANTILLAS).format(a=rnd.choice(ASPECTOS), o=rnd.choice(opiniones), extra=rnd.choice(extras))
    return texto[0].upper() + texto[1:] + "."


def guardar_csv(ruta: str = "datos/resenas.csv") -> str:
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["texto", "sentimiento"])
        w.writerows(generar())
    return ruta


def asegurar_dataset(ruta: str = "datos/resenas.csv") -> str:
    """Genera el dataset solo si todavía no existe."""
    if not Path(ruta).exists():
        guardar_csv(ruta)
    return ruta


if __name__ == "__main__":
    print(f"Dataset guardado en {guardar_csv()}")
