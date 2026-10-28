"""
core/optimization.py
--------------------
Diseño inverso de la malla de voladura usando los modelos de P80 ya entrenados
como "surrogate" (modelo sustituto) dentro de una metaheurística (FFA/PSO/GA).

Idea
    El usuario fija un P80 objetivo. El optimizador propone una geometría
    (burden B, taco T, longitud de taladro L y, opcionalmente, la densidad del
    explosivo). En CADA evaluación:
        1. se completa la malla:   S = k·B   (k = 1 cuadrada, 2/√3 triangular)
                                    J = L − H
                                    Lc = L − T
        2. se calcula el FP con la fórmula empírica:
               Q  = c · ρe · (π·d²/4) · Lc          [kg por taladro]
               FP = Q / (B · S · H)                 [kg/m³]
           (c = factor de calibración con la base de la mina, ver `calibrar_fp`)
        3. se arma el vector de entrada del modelo y se predice el P80.
    La función objetivo penaliza el error respecto del P80 objetivo, el costo
    (FP) y las mallas fuera del rango de la base o de las reglas empíricas.

Generalidad (la app sirve para cualquier mina)
    Cada base nombra distinto sus columnas ('Taco', 'Retacado_m', 'T_B'...).
    `detectar_roles` asigna a cada variable del modelo un "rol" físico
    (B, S, H, L, J, T, Lc, d, ρe, Q, FP o relaciones S/B, H/B, B/D, T/B, J/B).
    Las variables sin rol (UCS, VOD, litología...) quedan fijas.
"""
import re
import unicodedata

import numpy as np
import pandas as pd
from mealpy import FloatVar

from . import models as M

# --------------------------------------------------------------------------- #
#  Roles físicos                                                              #
# --------------------------------------------------------------------------- #
FIJA = "—"          # la variable no es de diseño: se ingresa como valor fijo

ROLES = {
    "B": "Burden (m)",
    "S": "Espaciamiento (m)",
    "H": "Altura de banco (m)",
    "L": "Longitud / profundidad de taladro (m)",
    "J": "Sobreperforación (m)",
    "T": "Taco / retacado (m)",
    "Lc": "Longitud de carga (m)",
    "d": "Diámetro de perforación",
    "rho": "Densidad del explosivo (g/cm³)",
    "Q": "Carga por taladro (kg)",
    "FP": "Factor de potencia (kg/m³)",
    "S_B": "Relación S/B",
    "H_B": "Relación H/B (índice de rigidez)",
    "B_D": "Relación B/D",
    "T_B": "Relación T/B",
    "J_B": "Relación J/B",
}

# Orden importa: la primera coincidencia gana (p. ej. 'longitud_taco' es T,
# 'altura_de_taladro' es L y no H).
_PATRONES = [
    ("S_B", r"^s_?b$"),
    ("H_B", r"^h_?b$|rigidez"),
    ("B_D", r"^b_?d$"),
    ("T_B", r"^t_?b$"),
    ("J_B", r"^[ju]_?b$"),
    ("FP", r"^(fp|pf)(_|$)|factor_(de_)?potencia|powder_factor"),
    ("Q", r"carga_(por_|de_|del_)?taladro|^carga(_kg)?$|^q(_kg)?$"),
    ("Lc", r"longitud_(de_)?carga|^columna(_|$)"),
    ("T", r"^taco|retacado|stemming|longitud_(de_|del_)?taco"),
    ("J", r"sobre_?perforacion|subdrill|sobreperf"),
    ("L", r"(profundidad|prof|altura|longitud)_(de_|del_)?taladro|^profundidad"),
    ("H", r"altura_(de_|del_)?banco|bench_height"),
    ("d", r"diametro"),
    ("rho", r"densidad_(de_|del_)?explosivo"),
    ("B", r"^burden"),
    ("S", r"^espaciamiento(_m)?$"),
]

K_PATRON = {"Triangular equilátera": 2 / np.sqrt(3),   # S = 1.155·B
            "Cuadrada": 1.0}                            # S = B

_A_D = {"mm": 1e-3, "pulg": 0.0254, "m": 1.0}           # unidad diámetro -> m


def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def detectar_roles(features) -> dict:
    """{feature: rol} con la primera coincidencia de nombre; cada rol se asigna
    a una sola variable. Las no reconocidas quedan como FIJA."""
    usados, out = set(), {}
    for f in features:
        n = _norm(f)
        rol = FIJA
        for r, pat in _PATRONES:
            if r not in usados and re.search(pat, n):
                rol = r
                break
        if rol != FIJA:
            usados.add(rol)
        out[f] = rol
    return out


def detectar_unidad_diametro(nombre, serie=None) -> str:
    n = _norm(nombre)
    if "pulg" in n or re.search(r"(^|_)(in|inch|pulgadas?)(_|$)", n):
        return "pulg"
    if re.search(r"(^|_)mm(_|$)", n):
        return "mm"
    if serie is not None:
        med = pd.to_numeric(serie, errors="coerce").median()
        if pd.notna(med):
            return "pulg" if med <= 30 else "mm"
    return "mm"


def detectar_patron(df, mapa) -> str:
    """Elige el patrón más parecido a la relación S/B de la base."""
    inv = {r: f for f, r in mapa.items() if r != FIJA}
    sb = None
    if "S_B" in inv and inv["S_B"] in df:
        sb = pd.to_numeric(df[inv["S_B"]], errors="coerce").median()
    elif "S" in inv and "B" in inv and inv["S"] in df and inv["B"] in df:
        sb = (pd.to_numeric(df[inv["S"]], errors="coerce")
              / pd.to_numeric(df[inv["B"]], errors="coerce")).median()
    if sb is None or pd.isna(sb):
        return "Triangular equilátera"
    return min(K_PATRON, key=lambda k: abs(K_PATRON[k] - sb))


# --------------------------------------------------------------------------- #
#  Geometría de la base (para calibrar el FP y obtener rangos)                #
# --------------------------------------------------------------------------- #
def geometria_desde_datos(df, mapa, fijos) -> pd.DataFrame:
    """Reconstruye B, S, H, L, J, T, Lc, d, ρe y el FP de fórmula (sin
    calibrar) de cada voladura de la base. Lo que la base no trae se toma de
    `fijos` (H, d_m, rho, k_patron)."""
    inv = {r: f for f, r in mapa.items() if r != FIJA and f in df.columns}
    n = len(df)

    def g(rol):
        return (pd.to_numeric(df[inv[rol]], errors="coerce").astype(float)
                .to_numpy() if rol in inv else None)

    def const(v):
        return np.full(n, float(v))

    d = g("d")
    d = d * _A_D[fijos.get("unidad_d_col", "mm")] if d is not None else const(fijos["d_m"])
    B = g("B")
    if B is None and g("B_D") is not None:
        B = g("B_D") * d
    if B is None:
        return pd.DataFrame()                      # sin burden no hay geometría
    S = g("S")
    if S is None:
        S = g("S_B") * B if g("S_B") is not None else fijos["k_patron"] * B
    L0, J0 = g("L"), g("J")
    if J0 is None and g("J_B") is not None:
        J0 = g("J_B") * B
    H = g("H")
    if H is None:
        if g("H_B") is not None:
            H = g("H_B") * B
        elif L0 is not None and J0 is not None:
            H = L0 - J0
        else:
            H = const(fijos["H"])
    J = J0 if J0 is not None else (L0 - H if L0 is not None else const(fijos.get("J", 0.0)))
    L = L0 if L0 is not None else H + J
    T = g("T")
    if T is None:
        if g("T_B") is not None:
            T = g("T_B") * B
        elif g("Lc") is not None:
            T = L - g("Lc")
        else:
            T = np.full(n, np.nan)
    Lc = g("Lc") if g("Lc") is not None else L - T
    rho = g("rho") if g("rho") is not None else const(fijos["rho"])
    Q = rho * 1000.0 * np.pi * d ** 2 / 4.0 * Lc
    with np.errstate(divide="ignore", invalid="ignore"):
        FP = Q / (B * S * H)
    out = pd.DataFrame({"B": B, "S": S, "H": H, "L": L, "J": J, "T": T,
                        "Lc": Lc, "d": d, "rho": rho, "Q_formula": Q,
                        "FP_formula": FP})
    if "FP" in inv:
        out["FP_base"] = pd.to_numeric(df[inv["FP"]], errors="coerce").to_numpy()
    return out


def calibrar_fp(geo: pd.DataFrame) -> dict:
    """Compara el FP de la base con el de la fórmula. Devuelve el factor c
    (mediana de FP_base / FP_formula) y la calidad del ajuste."""
    if geo.empty or "FP_base" not in geo:
        return {"c": 1.0, "n": 0, "r2": None, "mape": None, "disponible": False}
    v = geo[["FP_base", "FP_formula"]].replace([np.inf, -np.inf], np.nan).dropna()
    v = v[(v["FP_base"] > 0) & (v["FP_formula"] > 0)]
    if len(v) < 3:
        return {"c": 1.0, "n": len(v), "r2": None, "mape": None, "disponible": False}
    c = float(np.median(v["FP_base"] / v["FP_formula"]))
    pred = c * v["FP_formula"]
    ss_res = float(((v["FP_base"] - pred) ** 2).sum())
    ss_tot = float(((v["FP_base"] - v["FP_base"].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    mape = float((np.abs(v["FP_base"] - pred) / v["FP_base"]).mean() * 100)
    return {"c": c, "n": len(v), "r2": r2, "mape": mape, "disponible": True}


def rango(serie, completo=False):
    s = pd.Series(serie).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return None
    return (float(s.min()), float(s.max())) if completo else \
        (float(s.quantile(0.05)), float(s.quantile(0.95)))


# --------------------------------------------------------------------------- #
#  Malla propuesta -> vector de entrada del modelo                            #
# --------------------------------------------------------------------------- #
def completar_malla(B, T, L, rho, cfg) -> dict:
    """Geometría completa + FP calibrado a partir de las variables de diseño."""
    H, d, k, c = cfg["H"], cfg["d_m"], cfg["k_patron"], cfg["c"]
    S = k * B
    J = L - H
    Lc = L - T
    Q = c * rho * 1000.0 * np.pi * d ** 2 / 4.0 * Lc
    FP = Q / (B * S * H)
    return {"B": B, "S": S, "H": H, "L": L, "J": J, "T": T, "Lc": Lc,
            "d": d, "rho": rho, "Q": Q, "FP": FP,
            "S_B": S / B, "H_B": H / B, "B_D": B / d, "T_B": T / B, "J_B": J / B}


def vector_modelo(malla, features, mapa, fijas, unidad_d_col="mm") -> dict:
    fila = {}
    for f in features:
        rol = mapa.get(f, FIJA)
        if rol == FIJA:
            fila[f] = fijas[f]
        elif rol == "d":
            fila[f] = malla["d"] / _A_D[unidad_d_col]
        else:
            fila[f] = malla[rol]
    return fila


def _predictor_rapido(bundle):
    """Las predicciones de 1 fila con n_jobs=-1 son lentas (sobrecosto de
    paralelizar); para optimizar se usa n_jobs=1."""
    m = bundle["model"]
    try:
        if "n_jobs" in m.get_params():
            m.set_params(n_jobs=1)
    except Exception:
        pass
    return bundle


# --------------------------------------------------------------------------- #
#  Optimización                                                               #
# --------------------------------------------------------------------------- #
def optimizar(bundles, cfg, log=None):
    """
    bundles : lista de bundles de P80 (si son varios, se usa el promedio).
    cfg     : dict con
        objetivo, modo ('igual'|'maximo'), tol
        variables: {'B': (lo, hi) | valor_fijo, 'T': ..., 'L': ..., 'rho': ...}
        H, d_m, k_patron, c, unidad_d_col
        mapa {feature: rol}, fijas {feature: valor}
        rangos_base {feature: (lo, hi)} -> zona de confianza del modelo
        reglas {'T_B': (lo, hi), 'J_B': (lo, hi), 'H_B': (lo, None)}
        lam (peso del costo), fp_ref, algoritmo, epoch, pop_size
    """
    bundles = [_predictor_rapido(b) for b in bundles]
    features = bundles[0]["features"]
    mapa, fijas = cfg["mapa"], cfg["fijas"]
    orden = ["B", "T", "L", "rho"]
    libres = [v for v in orden if isinstance(cfg["variables"][v], tuple)]
    if not libres:
        raise ValueError("Elige al menos una variable de diseño a optimizar.")

    def decodificar(x):
        vals = {v: cfg["variables"][v] for v in orden if v not in libres}
        vals.update({v: float(x[i]) for i, v in enumerate(libres)})
        return vals

    def evaluar(vals):
        malla = completar_malla(vals["B"], vals["T"], vals["L"], vals["rho"], cfg)
        fila = vector_modelo(malla, features, mapa, fijas, cfg["unidad_d_col"])
        X = pd.DataFrame([fila])
        preds = [float(M.predecir(b, X)[0]) for b in bundles]
        return malla, preds

    obj, tol = cfg["objetivo"], cfg.get("tol", 0.0)
    registro = []

    def fitness(x):
        vals = decodificar(x)
        malla, preds = evaluar(vals)
        p80 = float(np.mean(preds))
        if cfg["modo"] == "maximo":
            e = max(0.0, p80 - obj - tol)
        else:
            e = max(0.0, abs(p80 - obj) - tol)
        f = 10.0 * e / obj

        pen = 0.0
        if malla["Lc"] <= 0.5:                       # sin columna de carga
            pen += 100.0 + (0.5 - malla["Lc"])
        for rol, (lo, hi) in cfg.get("reglas", {}).items():
            v = malla[rol]
            if lo is not None and v < lo:
                pen += 5.0 * (lo - v) / max(abs(lo), 1e-6)
            if hi is not None and v > hi:
                pen += 5.0 * (v - hi) / max(abs(hi), 1e-6)
        for feat, (lo, hi) in cfg.get("rangos_base", {}).items():
            rol = mapa.get(feat, FIJA)
            if rol == FIJA:
                continue
            v = malla["d"] / _A_D[cfg["unidad_d_col"]] if rol == "d" else malla[rol]
            ancho = max(hi - lo, 1e-6)
            pen += 2.0 * (max(0.0, lo - v) + max(0.0, v - hi)) / ancho

        costo = cfg.get("lam", 0.0) * malla["FP"] / max(cfg.get("fp_ref", 1.0), 1e-6)
        total = f + costo + pen
        registro.append((total, e, pen, p80, list(x)))
        return total

    bounds = [FloatVar(lb=cfg["variables"][v][0], ub=cfg["variables"][v][1])
              for v in libres]
    problem = {"obj_func": fitness, "bounds": bounds, "minmax": "min",
               "log_to": None}

    epoch = max(int(cfg.get("epoch", 40)), 2)
    pop = max(int(cfg.get("pop_size", 30)), 10)
    if pop % 2:
        pop += 1                                     # GA de mealpy exige par
    fabrica = {"FFA": M._opt_ffa, "PSO": M._opt_pso, "GA": M._opt_ga}
    optimizer = fabrica[cfg.get("algoritmo", "PSO")](epoch, pop)
    if log:
        log(f"Optimizando con {cfg.get('algoritmo', 'PSO')} "
            f"(epoch={epoch}, pop={pop}, variables={', '.join(libres)})...")
    try:
        g_best = optimizer.solve(problem, seed=42)
    except TypeError:                                 # mealpy sin 'seed'
        g_best = optimizer.solve(problem)

    try:
        convergencia = list(optimizer.history.list_global_best_fit)
    except Exception:
        convergencia = []

    mejor_vals = decodificar(g_best.solution)
    malla, preds = evaluar(mejor_vals)

    # Alternativas: soluciones distintas entre sí con buen puntaje
    reg = sorted(registro, key=lambda r: r[0])
    span = np.array([cfg["variables"][v][1] - cfg["variables"][v][0] for v in libres])
    span[span == 0] = 1.0
    elegidas = []
    for total, e, pen, p80, x in reg:
        x = np.asarray(x)
        if any(np.max(np.abs(x - y) / span) < 0.08 for y in elegidas):
            continue
        elegidas.append(x)
        if len(elegidas) >= 6:
            break
    alternativas = []
    for x in elegidas:
        v = decodificar(x)
        m, p = evaluar(v)
        alternativas.append({**m, "P80": float(np.mean(p))})

    return {
        "malla": malla,
        "pred_por_modelo": preds,
        "p80": float(np.mean(preds)),
        "fitness": float(g_best.target.fitness),
        "convergencia": convergencia,
        "alternativas": alternativas,
        "libres": libres,
        "n_eval": len(registro),
    }
