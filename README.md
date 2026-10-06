# 🤖 Analizador de Sentimientos en Español · IA con scikit-learn

![CI](https://github.com/acruzc23-crypto/analizador-sentimientos-ia/actions/workflows/ci.yml/badge.svg)

Modelo de **Machine Learning / Procesamiento de Lenguaje Natural (NLP)** que clasifica reseñas de clientes en español como **positivas** o **negativas**, usando **TF-IDF + Regresión Logística** con scikit-learn.

```bash
$ python predecir.py "El envío llegó rapidísimo, súper contento"
😀 POSITIVO (confianza 80%)

$ python predecir.py "la app no funciona y nadie me ayuda"
😞 NEGATIVO (confianza 69%)
```

## ✨ Características

- **Preprocesamiento en español**: minúsculas, eliminación de acentos, URLs, signos y *stopwords*
- **Manejo de negaciones**: `"no funciona"` → `no_funciona`, para que el modelo distinga el sentido
- **Pipeline de scikit-learn**: TF-IDF con unigramas y bigramas + Regresión Logística
- **Evaluación**: exactitud, precisión, recall, F1 y matriz de confusión
- **Interpretabilidad**: muestra las palabras que más influyen en cada clase
- **Persistencia** del modelo entrenado con `joblib`
- CLI para predecir desde la terminal (modo directo o interactivo)

## 📊 Resultados

| Métrica | Valor |
|---------|-------|
| Ejemplos de entrenamiento | 240 |
| Ejemplos de prueba | 60 |
| Exactitud (accuracy) | **98 %** |
| F1-score (macro) | 0.98 |

Palabras más influyentes aprendidas por el modelo:
- **Negativo:** horrible, decepcionante, terrible, pésima, desastre
- **Positivo:** genial, increíble, maravillosa, excelente, perfecta

> ⚠️ El dataset es **sintético** (generado con `generar_datos.py` a partir de plantillas), por lo que la exactitud con reseñas reales sería menor. El siguiente paso es entrenarlo con un dataset real de reseñas.

## 🛠️ Tecnologías

Python 3.12 · scikit-learn · pandas · joblib · pytest · GitHub Actions

## 🚀 Cómo ejecutarlo

```bash
git clone https://github.com/acruzc23-crypto/analizador-sentimientos-ia.git
cd analizador-sentimientos-ia
pip install -r requirements.txt

python generar_datos.py      # genera el dataset (300 reseñas)
python entrenar.py           # entrena, muestra métricas y guarda el modelo
python predecir.py "Excelente atención, volveré"   # predice
python predecir.py           # modo interactivo
pytest -v                    # ejecuta las pruebas
```

## 📁 Estructura

```
sentimientos/
├── preprocesamiento.py   # limpieza de texto y manejo de negaciones
└── modelo.py             # clase AnalizadorSentimientos (entrenar, predecir, guardar)
datos/                    # dataset (se genera con generar_datos.py)
entrenar.py               # script de entrenamiento y métricas
predecir.py               # CLI de predicción
tests/                    # 12 pruebas con pytest
```

## 🧠 ¿Cómo funciona?

1. **Limpieza:** `"¡No me gustó NADA!"` → `"no_gusto"`
2. **Vectorización TF-IDF:** cada texto se convierte en un vector numérico donde las palabras frecuentes en ese texto pero raras en el resto pesan más.
3. **Clasificación:** la Regresión Logística aprende un peso por palabra; la suma ponderada da la probabilidad de cada sentimiento.

## 📚 Lo que aprendí

- Flujo completo de un proyecto de ML: datos → preprocesamiento → entrenamiento → evaluación → despliegue
- Por qué separar datos de entrenamiento y prueba (evitar sobreajuste)
- Interpretar métricas de clasificación y la matriz de confusión
- Probar modelos de ML con pytest

---

👤 **Desarrollado por Alfredo Cruz**, con asistencia de IA (Claude).
Estudiante de Ingeniería en Software — UNEMI, Ecuador.
[GitHub](https://github.com/acruzc23-crypto) · [LinkedIn](https://www.linkedin.com/in/alfredo-cruz-dev)
