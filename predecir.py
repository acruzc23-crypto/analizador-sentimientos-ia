"""Clasifica textos desde la terminal.

Uso:
  python predecir.py "El envío llegó rapidísimo, excelente"
  python predecir.py            # modo interactivo
"""
import sys

from sentimientos import AnalizadorSentimientos

ICONOS = {"positivo": "😀", "negativo": "😞"}


def mostrar(modelo, texto):
    r = modelo.predecir(texto)
    print(f"{ICONOS.get(r['sentimiento'], '')} {r['sentimiento'].upper()} (confianza {r['confianza']:.0%})")


def main():
    modelo = AnalizadorSentimientos.cargar("modelos/sentimientos.joblib")
    if len(sys.argv) > 1:
        mostrar(modelo, " ".join(sys.argv[1:]))
        return
    print("Escribe una reseña (Enter vacío para salir):")
    while texto := input("> ").strip():
        mostrar(modelo, texto)


if __name__ == "__main__":
    main()
