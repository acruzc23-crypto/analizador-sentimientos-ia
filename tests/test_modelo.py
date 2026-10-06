import pytest

from generar_datos import asegurar_dataset
from sentimientos import AnalizadorSentimientos

DATOS = "datos/resenas.csv"


@pytest.fixture(scope="module")
def entrenado():
    modelo = AnalizadorSentimientos()
    metricas = modelo.entrenar_desde_csv(asegurar_dataset(DATOS))
    return modelo, metricas


def test_exactitud_minima(entrenado):
    _, metricas = entrenado
    assert metricas["exactitud"] >= 0.9


@pytest.mark.parametrize("texto,esperado", [
    ("El servicio fue excelente, lo recomiendo", "positivo"),
    ("Llegó roto, pésima experiencia", "negativo"),
    ("Me encantó la comida", "positivo"),
    ("No me gustó para nada", "negativo"),
])
def test_predicciones(entrenado, texto, esperado):
    modelo, _ = entrenado
    assert modelo.predecir(texto)["sentimiento"] == esperado


def test_probabilidades_suman_uno(entrenado):
    modelo, _ = entrenado
    r = modelo.predecir("Muy buena calidad")
    assert abs(sum(r["probabilidades"].values()) - 1) < 0.01
    assert 0.5 <= r["confianza"] <= 1


def test_texto_vacio(entrenado):
    modelo, _ = entrenado
    with pytest.raises(ValueError):
        modelo.predecir("   ")


def test_guardar_y_cargar(entrenado, tmp_path):
    modelo, _ = entrenado
    ruta = tmp_path / "m.joblib"
    modelo.guardar(ruta)
    copia = AnalizadorSentimientos.cargar(ruta)
    assert copia.predecir("horrible")["sentimiento"] == modelo.predecir("horrible")["sentimiento"]
