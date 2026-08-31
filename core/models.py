"""
models.py
---------
Registro de modelos híbridos (ML + metaheurística) para predecir P80, portados
1:1 desde el notebook de tesis. Cada modelo define:

  - needs_scaling : si usa datos estandarizados (SVR, ANN) o no (árboles).
  - param_names   : nombres de los hiperparámetros que optimiza la metaheurística.
  - make_bounds() : límites de búsqueda (mealpy FloatVar / IntegerVar).
  - build(params) : construye el estimador sklearn/xgboost con la solución.
  - make_optimizer(epoch, pop_size) : instancia del optimizador (FFA/PSO/GA).

La función `entrenar_hibrido()` corre la optimización de hiperparámetros,
compila el mejor modelo, lo evalúa en train/test y devuelve todo en un "bundle"
listo para guardar y para predecir.

Nota: se incluyen los 7 modelos basados en sklearn/xgboost (100% serializables
con joblib). El ANN+PSO del notebook requiere TensorFlow/Keras y un guardado
distinto; se puede añadir aparte si se necesita.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from mealpy import FFA, PSO, GA, FloatVar, IntegerVar


# --------------------------------------------------------------------------- #
#  Constructores de estimadores (a partir del vector solución de la metaheur.) #
# --------------------------------------------------------------------------- #
def _build_gbm(p):
    return GradientBoostingRegressor(
        n_estimators=int(p[0]), max_depth=int(p[1]), learning_rate=p[2],
        subsample=p[3], min_samples_leaf=int(p[4]), min_samples_split=int(p[5]),
        random_state=42,
    )


def _build_svr(p):
    return SVR(kernel="rbf", C=p[0], epsilon=p[1], gamma=p[2])


def _build_xgb6(p):  # XGB con 6 hiperparámetros (FFA)
    return XGBRegressor(
        n_estimators=int(p[0]), max_depth=int(p[1]), learning_rate=p[2],
        subsample=p[3], colsample_bytree=p[4], min_child_weight=int(p[5]),
        objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0,
    )


def _build_xgb5(p):  # XGB con 5 hiperparámetros (PSO / GA)
    return XGBRegressor(
        n_estimators=int(p[0]), max_depth=int(p[1]), learning_rate=p[2],
        subsample=p[3], colsample_bytree=p[4],
        objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0,
    )


def _build_rf(p):
    return RandomForestRegressor(
        n_estimators=int(p[0]), max_depth=int(p[1]) if int(p[1]) > 0 else None,
        min_samples_split=int(p[2]), min_samples_leaf=int(p[3]),
        max_features=p[4], random_state=42, n_jobs=-1,
    )


# --------------------------------------------------------------------------- #
#  Fábricas de optimizadores                                                  #
# --------------------------------------------------------------------------- #
def _opt_ffa(epoch, pop_size):
    return FFA.OriginalFFA(epoch=epoch, pop_size=pop_size)


def _opt_pso(epoch, pop_size):
    return PSO.OriginalPSO(epoch=epoch, pop_size=pop_size)


def _opt_ga(epoch, pop_size):
    return GA.BaseGA(epoch=epoch, pop_size=pop_size, pc=0.9, pm=0.2, k_way=0.5)


# --------------------------------------------------------------------------- #
#  Registro de modelos                                                        #
# --------------------------------------------------------------------------- #
REGISTRY = {
    "gbm_ffa": {
        "label": "GBM + FFA",
        "needs_scaling": False,
        "param_names": ["n_estimators", "max_depth", "learning_rate",
                        "subsample", "min_samples_leaf", "min_samples_split"],
        "make_bounds": lambda: [
            IntegerVar(lb=100, ub=300), IntegerVar(lb=2, ub=3),
            FloatVar(lb=0.01, ub=0.5), FloatVar(lb=0.5, ub=1.0),
            IntegerVar(lb=5, ub=30), IntegerVar(lb=10, ub=50),
        ],
        "build": _build_gbm, "make_optimizer": _opt_ffa,
        "epoch": 10, "pop_size": 15,
    },
    "svr_pso": {
        "label": "SVR + PSO",
        "needs_scaling": True,
        "param_names": ["C", "epsilon", "gamma"],
        "make_bounds": lambda: [
            FloatVar(lb=0.1, ub=100), FloatVar(lb=0.001, ub=1.0),
            FloatVar(lb=0.001, ub=1.0),
        ],
        "build": _build_svr, "make_optimizer": _opt_pso,
        "epoch": 30, "pop_size": 10,
    },
    "svr_ga": {
        "label": "SVR + GA",
        "needs_scaling": True,
        "param_names": ["C", "epsilon", "gamma"],
        "make_bounds": lambda: [
            FloatVar(lb=0.1, ub=200), FloatVar(lb=0.001, ub=1.0),
            FloatVar(lb=0.001, ub=1.0),
        ],
        "build": _build_svr, "make_optimizer": _opt_ga,
        "epoch": 20, "pop_size": 30,
    },
    "xgb_ffa": {
        "label": "XGBoost + FFA",
        "needs_scaling": False,
        "param_names": ["n_estimators", "max_depth", "learning_rate",
                        "subsample", "colsample_bytree", "min_child_weight"],
        "make_bounds": lambda: [
            IntegerVar(lb=50, ub=200), IntegerVar(lb=2, ub=8),
            FloatVar(lb=0.01, ub=0.2), FloatVar(lb=0.6, ub=1.0),
            FloatVar(lb=0.6, ub=1.0), IntegerVar(lb=1, ub=10),
        ],
        "build": _build_xgb6, "make_optimizer": _opt_ffa,
        "epoch": 15, "pop_size": 15,
    },
    "xgb_pso": {
        "label": "XGBoost + PSO",
        "needs_scaling": False,
        "param_names": ["n_estimators", "max_depth", "learning_rate",
                        "subsample", "colsample_bytree"],
        "make_bounds": lambda: [
            IntegerVar(lb=50, ub=400), IntegerVar(lb=3, ub=5),
            FloatVar(lb=0.03, ub=0.2), FloatVar(lb=0.6, ub=1.0),
            FloatVar(lb=0.6, ub=1.0),
        ],
        "build": _build_xgb5, "make_optimizer": _opt_pso,
        "epoch": 15, "pop_size": 15,
    },
    "xgb_ga": {
        "label": "XGBoost + GA",
        "needs_scaling": False,
        "param_names": ["n_estimators", "max_depth", "learning_rate",
                        "subsample", "colsample_bytree"],
        "make_bounds": lambda: [
            IntegerVar(lb=50, ub=200), IntegerVar(lb=2, ub=6),
            FloatVar(lb=0.03, ub=0.2), FloatVar(lb=0.6, ub=1.0),
            FloatVar(lb=0.6, ub=1.0),
        ],
        "build": _build_xgb5, "make_optimizer": _opt_ga,
        "epoch": 20, "pop_size": 30,
    },
    "rf_ga": {
        "label": "Random Forest + GA",
        "needs_scaling": False,
        "param_names": ["n_estimators", "max_depth", "min_samples_split",
                        "min_samples_leaf", "max_features"],
        "make_bounds": lambda: [
            IntegerVar(lb=50, ub=200), IntegerVar(lb=2, ub=10),
            IntegerVar(lb=2, ub=10), IntegerVar(lb=1, ub=5),
            FloatVar(lb=0.5, ub=1.0),
        ],
        "build": _build_rf, "make_optimizer": _opt_ga,
        "epoch": 20, "pop_size": 30,
    },
}


def listar_modelos():
    """Devuelve [(key, label), ...] para poblar el selector de la app."""
    return [(k, v["label"]) for k, v in REGISTRY.items()]


# --------------------------------------------------------------------------- #
#  Entrenamiento híbrido                                                      #
# --------------------------------------------------------------------------- #
def entrenar_hibrido(
    modelo_key,
    X_train, X_test, y_train, y_test,
    epoch=None, pop_size=None, n_splits=5,
    log=None,
):
    """Optimiza hiperparámetros con la metaheurística, compila el mejor modelo
    y lo evalúa. Devuelve un dict "bundle" con todo lo necesario para predecir.

    log: callable opcional para reportar progreso a la UI (recibe un str).
    """
    cfg = REGISTRY[modelo_key]
    epoch = epoch or cfg["epoch"]
    pop_size = pop_size or cfg["pop_size"]
    # Piso de seguridad: mealpy (sobre todo GA) falla con poblaciones diminutas.
    pop_size = max(int(pop_size), 10)
    epoch = max(int(epoch), 2)
    needs_scaling = cfg["needs_scaling"]

    def _say(msg):
        if log:
            log(msg)

    # Escalado (siempre ajustamos el scaler; se aplica según el modelo)
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train) if needs_scaling else X_train.values
    Xte = scaler.transform(X_test) if needs_scaling else X_test.values

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Función objetivo: RMSE promedio en validación cruzada
    def fitness(solution):
        model = cfg["build"](solution)
        scores = cross_val_score(
            model, Xtr, y_train, cv=cv, scoring="neg_mean_squared_error"
        )
        return float(np.sqrt(-scores.mean()))

    problem = {
        "obj_func": fitness,
        "bounds": cfg["make_bounds"](),
        "minmax": "min",
        "log_to": None,   # silencia el logger de mealpy
    }

    _say(f"Optimizando {cfg['label']}  (epoch={epoch}, pop_size={pop_size})...")
    optimizer = cfg["make_optimizer"](epoch, pop_size)
    g_best = optimizer.solve(problem)
    best_solution = g_best.solution
    best_cv_rmse = float(g_best.target.fitness)
    _say(f"Optimización terminada. RMSE (CV) = {best_cv_rmse:.4f}")

    # Compilar el mejor modelo y entrenarlo con todo el train
    best_model = cfg["build"](best_solution)
    best_model.fit(Xtr, y_train)

    y_pred_train = best_model.predict(Xtr)
    y_pred_test = best_model.predict(Xte)

    metrics = {
        "rmse_train": root_mean_squared_error(y_train, y_pred_train),
        "rmse_test": root_mean_squared_error(y_test, y_pred_test),
        "mae_train": mean_absolute_error(y_train, y_pred_train),
        "mae_test": mean_absolute_error(y_test, y_pred_test),
        "r2_train": r2_score(y_train, y_pred_train),
        "r2_test": r2_score(y_test, y_pred_test),
        "rmse_cv": best_cv_rmse,
    }

    # Historial de convergencia (para graficar)
    try:
        convergence = list(optimizer.history.list_global_best_fit)
    except Exception:
        convergence = []

    # Mejores hiperparámetros con nombres legibles
    best_params = {}
    for name, val in zip(cfg["param_names"], best_solution):
        best_params[name] = int(val) if name in (
            "n_estimators", "max_depth", "min_samples_leaf",
            "min_samples_split", "min_child_weight",
        ) else round(float(val), 5)

    bundle = {
        "modelo_key": modelo_key,
        "label": cfg["label"],
        "model": best_model,
        "scaler": scaler,
        "needs_scaling": needs_scaling,
        "features": list(X_train.columns),
        "target": y_train.name,
        "best_params": best_params,
        "metrics": metrics,
        "convergence": convergence,
        "y_test": np.asarray(y_test),
        "y_pred_test": np.asarray(y_pred_test),
        "X_test": X_test.reset_index(drop=True),
    }
    return bundle


def predecir(bundle, X_nuevo):
    """Aplica un bundle entrenado a datos nuevos (DataFrame con las features
    en el orden correcto) y devuelve el arreglo de predicciones de P80."""
    X = X_nuevo[bundle["features"]].copy()
    X = bundle["scaler"].transform(X) if bundle["needs_scaling"] else X.values
    return bundle["model"].predict(X)
