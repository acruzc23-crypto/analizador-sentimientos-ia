"""Entrena el modelo y muestra métricas.

Uso: python entrenar.py [--datos datos/resenas.csv] [--salida modelos/sentimientos.joblib]
"""
import argparse

from generar_datos import asegurar_dataset
from sentimientos import AnalizadorSentimientos


def main():
    parser = argparse.ArgumentParser(description="Entrena el analizador de sentimientos")
    parser.add_argument("--datos", default="datos/resenas.csv")
    parser.add_argument("--salida", default="modelos/sentimientos.joblib")
    args = parser.parse_args()

    asegurar_dataset(args.datos)
    modelo = AnalizadorSentimientos()
    m = modelo.entrenar_desde_csv(args.datos)
    print(f"Entrenamiento: {m['n_entrenamiento']} ejemplos | Prueba: {m['n_prueba']} ejemplos")
    print(f"Exactitud: {m['exactitud']:.2%}\n")
    print(m["reporte"])
    print("Matriz de confusión", modelo.clases, ":", m["matriz_confusion"])
    print("\nPalabras más influyentes:")
    for clase, palabras in modelo.palabras_clave(8).items():
        print(f"  {clase}: {', '.join(palabras)}")
    modelo.guardar(args.salida)
    print(f"\nModelo guardado en {args.salida}")


if __name__ == "__main__":
    main()
