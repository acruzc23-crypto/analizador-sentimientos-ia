"""Limpieza y normalización de texto en español."""
import re
import unicodedata

NEGACIONES = {"no", "nunca", "jamas", "nada", "ni", "tampoco"}

STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "y", "o", "a", "en",
    "que", "es", "fue", "por", "con", "para", "se", "lo", "me", "mi", "su", "sus", "este", "esta",
    "estuvo", "resulto", "parecio", "verdad", "general", "sinceramente", "todo", "todos",
}


def quitar_acentos(texto: str) -> str:
    normalizado = unicodedata.normalize("NFD", texto)
    return "".join(c for c in normalizado if unicodedata.category(c) != "Mn")


def limpiar(texto: str) -> str:
    """Minúsculas, sin acentos, sin URLs ni signos, y marca de negación.

    La negación se une a la palabra siguiente ("no funciona" -> "no_funciona")
    para que el modelo distinga "funciona" de "no funciona".
    """
    texto = quitar_acentos(texto.lower())
    texto = re.sub(r"https?://\S+", " ", texto)
    texto = re.sub(r"[^a-zñ\s]", " ", texto)
    palabras = texto.split()

    resultado, negar = [], False
    for p in palabras:
        if p in NEGACIONES:
            negar = True
            continue
        if p in STOPWORDS:
            continue
        resultado.append(f"no_{p}" if negar else p)
        negar = False
    return " ".join(resultado)
