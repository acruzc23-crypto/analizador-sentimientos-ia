from sentimientos.preprocesamiento import limpiar, quitar_acentos


def test_quita_acentos():
    assert quitar_acentos("atención rápida") == "atencion rapida"


def test_limpia_signos_urls_y_minusculas():
    assert limpiar("¡EXCELENTE! Visita https://x.com YA") == "excelente visita ya"


def test_marca_negaciones():
    assert limpiar("No funciona") == "no_funciona"
    assert limpiar("No lo recomiendo") == "no_recomiendo"


def test_elimina_stopwords():
    assert limpiar("el producto es de la mejor calidad") == "producto mejor calidad"
