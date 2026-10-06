"""Modelo de clasificación: TF-IDF + Regresión Logística."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocesamiento import limpiar


class AnalizadorSentimientos:
    def __init__(self, pipeline: Pipeline | None = None):
        self.pipeline = pipeline or Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=limpiar, ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=5.0)),
        ])

    # ---------- entrenamiento ----------
    def entrenar(self, textos, etiquetas, test_size: float = 0.2, semilla: int = 42) -> dict:
        x_train, x_test, y_train, y_test = train_test_split(
            list(textos), list(etiquetas), test_size=test_size, random_state=semilla, stratify=list(etiquetas)
        )
        self.pipeline.fit(x_train, y_train)
        pred = self.pipeline.predict(x_test)
        return {
            "exactitud": accuracy_score(y_test, pred),
            "reporte": classification_report(y_test, pred, zero_division=0),
            "matriz_confusion": confusion_matrix(y_test, pred, labels=self.clases).tolist(),
            "n_entrenamiento": len(x_train),
            "n_prueba": len(x_test),
        }

    def entrenar_desde_csv(self, ruta: str | Path, **kwargs) -> dict:
        df = pd.read_csv(ruta).dropna(subset=["texto", "sentimiento"])
        return self.entrenar(df["texto"], df["sentimiento"], **kwargs)

    # ---------- predicción ----------
    @property
    def clases(self) -> list[str]:
        return [str(c) for c in self.pipeline.classes_]

    def predecir(self, texto: str) -> dict:
        if not texto or not texto.strip():
            raise ValueError("El texto no puede estar vacío")
        probs = self.pipeline.predict_proba([texto])[0]
        indice = probs.argmax()
        return {
            "texto": texto,
            "sentimiento": self.clases[indice],
            "confianza": round(float(probs[indice]), 3),
            "probabilidades": {c: round(float(p), 3) for c, p in zip(self.clases, probs)},
        }

    def palabras_clave(self, n: int = 10) -> dict[str, list[str]]:
        """Palabras con más peso hacia cada clase (interpretabilidad)."""
        vocab = self.pipeline.named_steps["tfidf"].get_feature_names_out()
        coef = self.pipeline.named_steps["clf"].coef_[0]
        orden = coef.argsort()
        return {self.clases[0]: list(vocab[orden[:n]]), self.clases[1]: list(vocab[orden[-n:][::-1]])}

    # ---------- persistencia ----------
    def guardar(self, ruta: str | Path) -> None:
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, ruta)

    @classmethod
    def cargar(cls, ruta: str | Path) -> "AnalizadorSentimientos":
        return cls(joblib.load(ruta))
