"""
core/training.py
----------------
Orquesta el entrenamiento de varios modelos híbridos a la vez, arma la tabla
comparativa y determina el mejor modelo. Reutiliza core/models.py.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from . import models as M
from . import config


def entrenar_varios(model_keys, X, y, test_size=0.2, epoch=None, pop_size=None,
                    n_splits=5, progreso=None):
    """Entrena cada modelo de la lista y devuelve (bundles, tabla_comparativa,
    mejor_key).

    progreso: callable(frac, texto) para actualizar la barra de la UI.
    """
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    bundles = {}
    total = len(model_keys)
    for i, key in enumerate(model_keys):
        if progreso:
            progreso(i / total, f"Entrenando {M.REGISTRY[key]['label']}...")
        bundles[key] = M.entrenar_hibrido(
            key, Xtr, Xte, ytr, yte,
            epoch=epoch, pop_size=pop_size, n_splits=n_splits,
        )
    if progreso:
        progreso(1.0, "Entrenamiento completo")

    tabla = tabla_comparativa(bundles)
    # Mejor modelo según la métrica cruda (no la columna con mayúsculas)
    menor = config.METRICA_MENOR_MEJOR
    mejor_key = min(
        bundles,
        key=lambda k: bundles[k]["metrics"][config.METRICA_RANKING]
    ) if menor else max(
        bundles,
        key=lambda k: bundles[k]["metrics"][config.METRICA_RANKING]
    )
    return bundles, tabla, mejor_key


def entrenar_varios_curva(model_keys, X, y, test_size=0.2, epoch=None,
                          pop_size=None, n_splits=5, progreso=None):
    """Entrena varios modelos de curva (multi-salida) y devuelve
    (bundles, tabla_comparativa, mejor_key)."""
    from . import curve_models as CM
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    bundles = {}
    total = len(model_keys)
    for i, key in enumerate(model_keys):
        if progreso:
            progreso(i / total, f"Entrenando {CM.REGISTRY[key]['label']}...")
        bundles[key] = CM.entrenar_curva_hibrido(
            key, Xtr, Xte, ytr, yte,
            epoch=epoch, pop_size=pop_size, n_splits=n_splits)
    if progreso:
        progreso(1.0, "Entrenamiento completo")

    tabla = tabla_comparativa(bundles)
    mejor_key = min(bundles,
                    key=lambda k: bundles[k]["metrics"][config.METRICA_RANKING])
    return bundles, tabla, mejor_key


def tabla_comparativa(bundles: dict) -> pd.DataFrame:
    filas = []
    for key, b in bundles.items():
        m = b["metrics"]
        fila = {
            "key": key,
            "Modelo": b["label"],
            "R2_test": round(m.get("r2_test", float("nan")), 4),
            "RMSE_test": round(m.get("rmse_test", float("nan")), 4),
            "MAE_test": round(m.get("mae_test", float("nan")), 4),
            "RMSE_cv": round(m.get("rmse_cv", float("nan")), 4),
        }
        if "r2_train" in m:
            fila["R2_train"] = round(m["r2_train"], 4)
        filas.append(fila)
    df = pd.DataFrame(filas)
    return df.sort_values("RMSE_test").reset_index(drop=True)
