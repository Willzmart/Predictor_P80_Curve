"""
core/curve_models.py
--------------------
Modelos híbridos para predecir la CURVA GRANULOMÉTRICA completa (percentiles
P10..P100) como regresión multi-salida, portados del notebook de tesis de curva.

6 modelos: LightGBM/XGBoost × (FFA / PSO / GA). Cada uno envuelve el regresor
base en MultiOutputRegressor y optimiza sus hiperparámetros minimizando el RMSE
promedio sobre los 10 percentiles (validación cruzada 5-fold).

Incluye corrección de monotonía: la curva acumulada no puede decrecer, así que
se fuerza P10 <= P20 <= ... <= P100 en las predicciones.
"""
import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    root_mean_squared_error, mean_absolute_error, r2_score,
)
from mealpy import FFA, PSO, GA, FloatVar, IntegerVar

from . import preprocessing as P


# --------------------------------------------------------------------------- #
#  Constructores de estimadores multi-salida                                  #
# --------------------------------------------------------------------------- #
def _build_lgbm(p):
    return MultiOutputRegressor(LGBMRegressor(
        n_estimators=int(p[0]), max_depth=int(p[1]), learning_rate=p[2],
        subsample=p[3], colsample_bytree=p[4], num_leaves=int(p[5]),
        random_state=42, n_jobs=-1, verbose=-1,
    ))


def _build_xgb(p):
    return MultiOutputRegressor(XGBRegressor(
        n_estimators=int(p[0]), max_depth=int(p[1]), learning_rate=p[2],
        subsample=p[3], colsample_bytree=p[4], min_child_weight=int(p[5]),
        objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0,
    ))


def _opt_ffa(epoch, pop_size):
    return FFA.OriginalFFA(epoch=epoch, pop_size=pop_size)


def _opt_pso(epoch, pop_size):
    return PSO.OriginalPSO(epoch=epoch, pop_size=pop_size, c1=2.05, c2=2.05, w=0.4)


def _opt_ga(epoch, pop_size):
    return GA.BaseGA(epoch=epoch, pop_size=pop_size, pc=0.95, pm=0.025)


_BOUNDS_LGBM = lambda: [
    IntegerVar(lb=50, ub=300), IntegerVar(lb=3, ub=8), FloatVar(lb=0.01, ub=0.2),
    FloatVar(lb=0.6, ub=1.0), FloatVar(lb=0.6, ub=1.0), IntegerVar(lb=20, ub=100),
]
_BOUNDS_XGB = lambda: [
    IntegerVar(lb=50, ub=300), IntegerVar(lb=3, ub=8), FloatVar(lb=0.01, ub=0.2),
    FloatVar(lb=0.6, ub=1.0), FloatVar(lb=0.6, ub=1.0), IntegerVar(lb=1, ub=10),
]

_PARAMS_LGBM = ["n_estimators", "max_depth", "learning_rate", "subsample",
                "colsample_bytree", "num_leaves"]
_PARAMS_XGB = ["n_estimators", "max_depth", "learning_rate", "subsample",
               "colsample_bytree", "min_child_weight"]


REGISTRY = {
    "lgbm_ffa": {"label": "LightGBM + FFA", "build": _build_lgbm,
                 "make_bounds": _BOUNDS_LGBM, "make_optimizer": _opt_ffa,
                 "param_names": _PARAMS_LGBM, "epoch": 20, "pop_size": 10},
    "lgbm_pso": {"label": "LightGBM + PSO", "build": _build_lgbm,
                 "make_bounds": _BOUNDS_LGBM, "make_optimizer": _opt_pso,
                 "param_names": _PARAMS_LGBM, "epoch": 20, "pop_size": 10},
    "lgbm_ga": {"label": "LightGBM + GA", "build": _build_lgbm,
                "make_bounds": _BOUNDS_LGBM, "make_optimizer": _opt_ga,
                "param_names": _PARAMS_LGBM, "epoch": 20, "pop_size": 10},
    "xgb_ffa": {"label": "XGBoost + FFA", "build": _build_xgb,
                "make_bounds": _BOUNDS_XGB, "make_optimizer": _opt_ffa,
                "param_names": _PARAMS_XGB, "epoch": 20, "pop_size": 10},
    "xgb_pso": {"label": "XGBoost + PSO", "build": _build_xgb,
                "make_bounds": _BOUNDS_XGB, "make_optimizer": _opt_pso,
                "param_names": _PARAMS_XGB, "epoch": 20, "pop_size": 10},
    "xgb_ga": {"label": "XGBoost + GA", "build": _build_xgb,
               "make_bounds": _BOUNDS_XGB, "make_optimizer": _opt_ga,
               "param_names": _PARAMS_XGB, "epoch": 20, "pop_size": 10},
}


def listar_modelos():
    return [(k, v["label"]) for k, v in REGISTRY.items()]


def corregir_monotonia(y_pred):
    """Fuerza que cada curva sea no decreciente (P10 <= P20 <= ... <= P100)."""
    y = np.array(y_pred, dtype=float)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    for i in range(1, y.shape[1]):
        y[:, i] = np.maximum(y[:, i], y[:, i - 1])
    return y


# --------------------------------------------------------------------------- #
#  Entrenamiento de un modelo de curva                                        #
# --------------------------------------------------------------------------- #
def entrenar_curva_hibrido(modelo_key, X_train, X_test, y_train, y_test,
                           epoch=None, pop_size=None, n_splits=5):
    """Optimiza, entrena y evalúa un modelo de curva multi-salida. Devuelve un
    bundle listo para guardar y predecir."""
    cfg = REGISTRY[modelo_key]
    epoch = max(int(epoch or cfg["epoch"]), 2)
    pop_size = max(int(pop_size or cfg["pop_size"]), 10)
    percentiles = list(y_train.columns)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def fitness(solution):
        model = cfg["build"](solution)
        rmse_folds = []
        for tr_idx, val_idx in cv.split(X_train):
            Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            model.fit(Xtr, ytr)
            yhat = model.predict(Xval)
            rmse_p = [root_mean_squared_error(yval.iloc[:, i], yhat[:, i])
                      for i in range(len(percentiles))]
            rmse_folds.append(np.mean(rmse_p))
        return float(np.mean(rmse_folds))

    problem = {"obj_func": fitness, "bounds": cfg["make_bounds"](),
               "minmax": "min", "log_to": None}
    optimizer = cfg["make_optimizer"](epoch, pop_size)
    g_best = optimizer.solve(problem)
    best = g_best.solution
    best_cv_rmse = float(g_best.target.fitness)

    # Compilar y entrenar el mejor modelo
    model = cfg["build"](best)
    model.fit(X_train, y_train)
    y_pred_test = corregir_monotonia(model.predict(X_test))
    y_pred_train = corregir_monotonia(model.predict(X_train))

    # Métricas por percentil y promedios
    por_percentil = []
    for i, p in enumerate(percentiles):
        por_percentil.append({
            "percentil": p,
            "rmse_test": round(root_mean_squared_error(y_test.iloc[:, i], y_pred_test[:, i]), 4),
            "mae_test": round(mean_absolute_error(y_test.iloc[:, i], y_pred_test[:, i]), 4),
            "r2_test": round(r2_score(y_test.iloc[:, i], y_pred_test[:, i]), 4),
        })
    rmse_prom = float(np.mean([d["rmse_test"] for d in por_percentil]))
    mae_prom = float(np.mean([d["mae_test"] for d in por_percentil]))
    r2_prom = float(np.mean([d["r2_test"] for d in por_percentil]))

    best_params = {}
    for name, val in zip(cfg["param_names"], best):
        best_params[name] = int(val) if name in (
            "n_estimators", "max_depth", "num_leaves", "min_child_weight"
        ) else round(float(val), 5)

    try:
        convergence = list(optimizer.history.list_global_best_fit)
    except Exception:
        convergence = []

    return {
        "modelo_key": modelo_key,
        "label": cfg["label"],
        "model": model,
        "needs_scaling": False,
        "features": list(X_train.columns),
        "percentiles": percentiles,
        "best_params": best_params,
        "metrics": {"rmse_test": round(rmse_prom, 4),
                    "mae_test": round(mae_prom, 4),
                    "r2_test": round(r2_prom, 4),
                    "rmse_cv": round(best_cv_rmse, 4)},
        "por_percentil": por_percentil,
        "convergence": convergence,
        # muestra para graficar en resultados
        "y_test_muestra": np.asarray(y_test.iloc[:min(5, len(y_test))]),
        "y_pred_muestra": y_pred_test[:min(5, len(y_pred_test))],
    }


def predecir_curva(bundle, X_nuevo):
    """Predice la curva (P10..P100) para datos nuevos, con monotonía corregida."""
    X = X_nuevo[bundle["features"]].copy()
    pred = bundle["model"].predict(X.values)
    return corregir_monotonia(pred)
