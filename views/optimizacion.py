"""views/optimizacion.py — Optimización de la malla (diseño inverso del P80).

El usuario indica el P80 requerido; una metaheurística (FFA/PSO/GA) busca el
burden, el taco y la longitud de taladro usando el/los modelo(s) de P80 ya
entrenados. El factor de potencia NO se optimiza libremente: se calcula en cada
evaluación con la fórmula empírica de carga (ver core/optimization.py).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from core import db, versioning as V, optimization as O
from views.prediccion import _defaults_desde_datos, _rmse_modelo, _tarjeta_resultado

_VARS = {
    "B": ("Burden", "m"),
    "T": ("Taco / retacado", "m"),
    "L": ("Longitud de taladro", "m"),
    "rho": ("Densidad del explosivo", "g/cm³"),
}
_OPC_FIJA = "— (variable fija, no es de diseño)"


def _opcion(rol):
    return _OPC_FIJA if rol == O.FIJA else f"{rol} · {O.ROLES[rol]}"


def _rol_de(opcion):
    return O.FIJA if opcion == _OPC_FIJA else opcion.split(" · ")[0]


def _med(serie, default):
    s = pd.Series(serie).replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.median()) if not s.empty else float(default)


def page_optimizacion():
    st.title("🧩 Optimización de la malla")
    st.caption(
        "Diseño inverso: indica el P80 que necesitas y el optimizador busca el "
        "**burden**, el **taco** y la **longitud de taladro** usando tus modelos "
        "entrenados. El **factor de potencia** se calcula en cada intento con "
        "Q = ρe·(π·d²/4)·(L − T) y FP = Q / (B·S·H).")

    va = V.version_activa("p80")
    if va is None:
        st.warning("No hay modelos de P80 entrenados. Entrena y guarda una "
                   "versión en **Entrenamiento**.")
        return
    if not db.tiene_datos():
        st.warning("No hay datos en la base. Impórtalos en **Datos**.")
        return

    meta = V.cargar_metadata(va, "p80")
    df = db.cargar_datos().drop(columns=["id"], errors="ignore")
    unidad = db.meta_get("unidad_p80", "cm")
    target = meta["target"]
    st.caption(f"Versión activa (P80): **{va}** · {meta['fecha']}")

    # ------------------------------------------------------------------ #
    # 1 · Objetivo                                                        #
    # ------------------------------------------------------------------ #
    st.subheader("1 · P80 objetivo")
    keys = meta["modelos"]
    labels = [meta["labels"][k] for k in keys]
    c1, c2, c3 = st.columns([1, 1.3, 2])
    p80_base = pd.to_numeric(df[target], errors="coerce") if target in df else pd.Series(dtype=float)
    objetivo = c1.number_input(f"P80 requerido ({unidad})", min_value=0.001,
                               value=round(_med(p80_base, 10.0), 2), key="opt_obj")
    modo_txt = c2.radio("Criterio", ["No superar el objetivo (P80 ≤ objetivo)",
                                     "Igualar el objetivo (± tolerancia)"],
                        key="opt_modo")
    modo = "maximo" if modo_txt.startswith("No superar") else "igual"
    sel = c3.multiselect(
        "Modelo(s) a usar", labels,
        default=[meta["labels"][meta["mejor_modelo"]]], key="opt_mdl",
        help="Con varios modelos se optimiza sobre el promedio de sus "
             "predicciones (más robusto que confiar en uno solo).")
    if not sel:
        st.info("Elige al menos un modelo.")
        return
    bundles = [V.cargar_bundle(va, keys[labels.index(s)], "p80") for s in sel]
    rmses = [r for r in (_rmse_modelo(b) for b in bundles) if r is not None]
    rmse = float(np.mean(rmses)) if rmses else None

    tol = 0.0
    if modo == "igual":
        tol = st.number_input(
            f"Tolerancia ± ({unidad})", min_value=0.0,
            value=round(0.05 * objetivo, 2), key="opt_tol",
            help="Dentro de esta banda se elige la malla de menor FP. "
                 + (f"Referencia: el RMSE del modelo es {rmse:.2f} {unidad}; "
                    "diferencias menores al RMSE no son distinguibles."
                    if rmse else ""))
    if not p80_base.dropna().empty:
        st.caption(f"En tu base el P80 va de {p80_base.min():.2f} a "
                   f"{p80_base.max():.2f} {unidad}. Objetivos fuera de ese rango "
                   "no son alcanzables de forma confiable.")

    # ------------------------------------------------------------------ #
    # 2 · Correspondencia de variables                                    #
    # ------------------------------------------------------------------ #
    features = bundles[0]["features"]
    auto = O.detectar_roles(features)
    with st.expander("2 · Variables de diseño reconocidas en tu base", expanded=False):
        st.caption("Cada base nombra distinto sus columnas. Revisa que cada "
                   "variable del modelo tenga el rol correcto; las que no son de "
                   "diseño (UCS, VOD, litología…) quedan como fijas.")
        opciones = [_OPC_FIJA] + [_opcion(r) for r in O.ROLES]
        tabla = pd.DataFrame({"Variable del modelo": features,
                              "Rol": [_opcion(auto[f]) for f in features]})
        editada = st.data_editor(
            tabla, hide_index=True, width="stretch", key=f"opt_map_{va}",
            disabled=["Variable del modelo"],
            column_config={"Rol": st.column_config.SelectboxColumn(
                "Rol", options=opciones, required=True)})
    mapa = {f: _rol_de(r) for f, r in zip(editada["Variable del modelo"], editada["Rol"])}
    inv = {r: f for f, r in mapa.items() if r != O.FIJA}

    if "FP" not in inv:
        st.warning("El modelo no usa el factor de potencia como variable de "
                   "entrada; el FP se calculará solo como resultado del diseño.")
    if not ({"B", "B_D"} & inv.keys()):
        st.warning("No se reconoció el **burden** entre las variables del modelo. "
                   "Revisa la tabla del paso 2.")

    # ------------------------------------------------------------------ #
    # 3 · Parámetros fijos de la malla                                    #
    # ------------------------------------------------------------------ #
    st.subheader("3 · Parámetros fijos de la malla")
    col_d = inv.get("d")
    unidad_d_col = O.detectar_unidad_diametro(col_d, df.get(col_d)) if col_d else "mm"

    p1, p2, p3, p4 = st.columns(4)
    patrones = list(O.K_PATRON)
    patron = p1.radio("Diseño de malla", patrones,
                      index=patrones.index(O.detectar_patron(df, mapa)),
                      key="opt_patron",
                      help="Triangular equilátera: S = 1.155·B · Cuadrada: S = B")
    k_patron = O.K_PATRON[patron]

    def _serie(rol):
        f = inv.get(rol)
        return pd.to_numeric(df[f], errors="coerce") if f in df else None

    # Diámetro y explosivo
    unidad_d_in = unidad_d_col if col_d else p3.selectbox(
        "Unidad del diámetro", ["mm", "pulg"], key="opt_ud")
    d_def = _med(_serie("d"), 165) if col_d else (165.0 if unidad_d_in == "mm" else 6.5)
    d_val = p3.number_input(f"Diámetro ({unidad_d_in})", min_value=0.1,
                            value=round(d_def, 2), key="opt_d")
    d_m = d_val * O._A_D[unidad_d_in]
    if not col_d:
        p3.caption("Tu modelo no usa el diámetro (no está en la base o era "
                   "constante): ingresa el de tu perforadora.")

    rho_fijo = p4.number_input(
        "Densidad del explosivo (g/cm³)", min_value=0.3, max_value=2.0,
        value=round(_med(_serie("rho"), 1.0), 3), key="opt_rho",
        help="Referencias: ANFO ≈ 0.80 · Heavy ANFO ≈ 1.10–1.30 · Emulsión ≈ 1.15–1.25")

    # Sobreperforación de referencia (solo si la base no la permite deducir)
    b_ser = _serie("B") if _serie("B") is not None else (
        _serie("B_D") * d_m if _serie("B_D") is not None else None)
    j_fijo = round(0.3 * _med(b_ser, 5.0), 2)
    if not ({"L", "J", "J_B"} & inv.keys()):
        j_fijo = st.number_input(
            "Sobreperforación típica J (m)", min_value=0.0, value=j_fijo,
            key="opt_J",
            help="Tu base no trae longitud de taladro ni sobreperforación; se usa "
                 "este valor para reconstruir la geometría histórica.")

    # Altura de banco: se deduce de la base si es posible (H, H/B·B o L − J)
    base = {"d_m": d_m, "rho": rho_fijo, "k_patron": k_patron,
            "unidad_d_col": unidad_d_col, "J": j_fijo}
    h_en_base = ({"H", "H_B"} & inv.keys()) or ({"L", "J"} <= inv.keys())
    if h_en_base:
        h_def = _med(O.geometria_desde_datos(df, mapa, {**base, "H": 10.0}).get("H"), 10)
    elif _serie("L") is not None:
        h_def = round(_med(_serie("L"), 11) / 1.1 * 2) / 2
    else:
        h_def = 10.0
    H = p2.number_input("Altura de banco H (m)", min_value=1.0,
                        value=round(h_def, 2), key="opt_H",
                        help=None if h_en_base else
                        "Tu base no trae la altura de banco: verifica este valor.")

    fijos = {**base, "H": H}
    geo = O.geometria_desde_datos(df, mapa, fijos)
    if geo.empty:
        st.error("No se pudo reconstruir la geometría de la base (falta el burden).")
        return

    # ------------------------------------------------------------------ #
    # 4 · Calibración del FP                                              #
    # ------------------------------------------------------------------ #
    cal = O.calibrar_fp(geo)
    with st.expander("4 · Calibración de la fórmula del FP con tu base",
                     expanded=False):
        if not cal["disponible"]:
            st.info("La base no tiene una columna de FP para comparar; se usa la "
                    "fórmula sin corrección (c = 1).")
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Factor c", f"{cal['c']:.3f}")
            k2.metric("R² (FP base vs. fórmula·c)",
                      f"{cal['r2']:.2f}" if cal["r2"] is not None else "—")
            k3.metric("Error medio", f"{cal['mape']:.1f} %")
            k4.metric("Voladuras usadas", cal["n"])
            st.caption(
                "c = mediana de FP_base / FP_fórmula. Corrige diferencias "
                "sistemáticas (densidad real, cebos, decks, definición del FP en "
                "la mina) para que el FP calculado quede en la misma escala que "
                "aprendió el modelo.")
            if cal["r2"] is not None and cal["r2"] < 0.5:
                st.warning(
                    "La geometría explica poco del FP registrado: voladuras con la "
                    "misma malla tienen FP muy distintos. Lo más probable es que "
                    "varíe el explosivo (densidad) o la forma de carguío. Ingresa "
                    "la densidad del explosivo que vas a usar o actívala como "
                    "variable a optimizar.")
            v = geo[["FP_base", "FP_formula"]].dropna()
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            ax.scatter(v["FP_base"], cal["c"] * v["FP_formula"], s=14, alpha=0.7)
            lim = [0, float(max(v["FP_base"].max(), (cal["c"] * v["FP_formula"]).max())) * 1.05]
            ax.plot(lim, lim, "--", color="gray", lw=1)
            ax.set_xlabel("FP de la base (kg/m³)")
            ax.set_ylabel("FP calculado · c (kg/m³)")
            ax.grid(True, ls="--", alpha=0.4)
            g1, _ = st.columns([1, 1])
            g1.pyplot(fig)
    aplicar_c = st.checkbox("Aplicar el factor de calibración del FP",
                            value=cal["disponible"], key="opt_cal",
                            disabled=not cal["disponible"])
    c = cal["c"] if (aplicar_c and cal["disponible"]) else 1.0

    # ------------------------------------------------------------------ #
    # 5 · Variables a optimizar                                           #
    # ------------------------------------------------------------------ #
    st.subheader("5 · Variables a optimizar")
    completo = st.toggle("Usar el rango completo de la base (mín–máx) en vez de "
                         "percentiles 5–95", value=False, key="opt_rng")
    variables = {}
    for v, (nombre, u) in _VARS.items():
        serie = geo[v] if v != "rho" else (_serie("rho") if _serie("rho") is not None else None)
        rng = O.rango(serie, completo) if serie is not None else None
        if v == "rho" and rng is None:
            rng = (0.80, 1.30)
        if rng is None:
            rng = (1.0, 10.0)
        cA, cB, cC = st.columns([1.4, 1, 1])
        opt = cA.checkbox(f"Optimizar {nombre.lower()} ({u})", value=(v != "rho"),
                          key=f"opt_chk_{v}")
        if opt:
            lo = cB.number_input("mín", value=round(rng[0], 3), key=f"opt_lo_{v}")
            hi = cC.number_input("máx", value=round(rng[1], 3), key=f"opt_hi_{v}")
            if hi <= lo:
                cA.caption("⚠️ En tu base esta variable casi no varía; se fija en "
                           f"{lo:.2f} {u}.")
                variables[v] = float(lo)
            else:
                variables[v] = (float(lo), float(hi))
        else:
            fijo_def = rho_fijo if v == "rho" else _med(serie, np.mean(rng))
            variables[v] = float(cB.number_input(
                "valor fijo", value=round(fijo_def, 3), key=f"opt_fix_{v}"))

    # ------------------------------------------------------------------ #
    # 6 · Condiciones fijas (macizo, explosivo, etc.)                     #
    # ------------------------------------------------------------------ #
    feats_fijas = [f for f in features if mapa[f] == O.FIJA]
    fijas = {}
    if feats_fijas:
        with st.expander(f"6 · Condiciones del sitio ({len(feats_fijas)} variables fijas)",
                         expanded=False):
            st.caption("Propiedades de la roca, del explosivo y otras variables que "
                       "no se optimizan. Vienen con la mediana de tu base (0/1 en "
                       "litología y material); ajústalas a la zona que vas a volar.")
            defaults, binarias = _defaults_desde_datos(feats_fijas)
            grid = st.columns(3)
            for i, f in enumerate(feats_fijas):
                with grid[i % 3]:
                    if binarias.get(f):
                        fijas[f] = st.selectbox(f"{f} (0/1)", [0, 1],
                                                index=int(defaults[f]) if int(defaults[f]) in (0, 1) else 0,
                                                key=f"opt_in_{f}")
                    else:
                        fijas[f] = st.number_input(f, value=round(float(defaults[f]), 4),
                                                   format="%.4f", key=f"opt_in_{f}")

    # ------------------------------------------------------------------ #
    # 7 · Restricciones, costo y algoritmo                                #
    # ------------------------------------------------------------------ #
    with st.expander("7 · Restricciones, costo y algoritmo", expanded=False):
        dentro = st.checkbox(
            "Mantener la malla dentro del rango de la base (recomendado)", value=True,
            key="opt_dentro",
            help="Fuera del rango de entrenamiento los modelos de árboles se "
                 "aplanan y el SVR extrapola mal: el P80 predicho deja de ser confiable.")
        reglas = {}
        r1, r2, r3 = st.columns(3)
        if r1.checkbox("Taco/Burden (T/B)", value=True, key="opt_r_tb"):
            reglas["T_B"] = r1.slider("T/B", 0.3, 2.0, (0.7, 1.3), 0.05, key="opt_tb")
        if r2.checkbox("Sobreperforación/Burden (J/B)", value=True, key="opt_r_jb"):
            reglas["J_B"] = r2.slider("J/B", 0.0, 1.0, (0.1, 0.5), 0.05, key="opt_jb")
        if r3.checkbox("Rigidez H/B mínima", value=False, key="opt_r_hb"):
            reglas["H_B"] = (r3.number_input("H/B ≥", 1.0, 6.0, 2.0, 0.1,
                                             key="opt_hb"), None)
        st.caption("Reglas de práctica (Ash, Konya): T ≈ 0.7–1.3·B · J ≈ 0.2–0.5·B · "
                   "H/B ≥ 2 para buena fragmentación. Se aplican como penalización.")
        lam = st.slider("Peso del ahorro de explosivo (costo)", 0.0, 1.0, 0.2, 0.05,
                        key="opt_lam",
                        help="0 = solo importa cumplir el P80. Mayor valor = entre "
                             "las mallas que cumplen, prefiere la de menor FP.")
        a1, a2, a3 = st.columns(3)
        algoritmo = a1.selectbox("Metaheurística", ["PSO", "FFA", "GA"], key="opt_alg")
        epoch = a2.number_input("Épocas", 5, 300, 40, key="opt_ep")
        pop = a3.number_input("Población", 10, 100, 30, key="opt_pop")

    rangos_base = {}
    if dentro:
        for f, rol in mapa.items():
            if rol == O.FIJA or rol == "d" or f not in df:
                continue
            if rol == "rho" and not isinstance(variables["rho"], tuple):
                continue
            r = O.rango(pd.to_numeric(df[f], errors="coerce"), completo)
            if r is not None and r[1] > r[0]:
                rangos_base[f] = r

    fp_ref = _med(geo["FP_base"], 0) if "FP_base" in geo else 0
    if fp_ref <= 0:
        fp_ref = _med(c * geo["FP_formula"], 1.0)

    # ------------------------------------------------------------------ #
    # Ejecutar                                                            #
    # ------------------------------------------------------------------ #
    if st.button("🧩 Optimizar malla", type="primary"):
        cfg = {"objetivo": float(objetivo), "modo": modo, "tol": float(tol),
               "variables": variables, "H": H, "d_m": d_m, "k_patron": k_patron,
               "c": c, "unidad_d_col": unidad_d_col, "mapa": mapa, "fijas": fijas,
               "rangos_base": rangos_base, "reglas": reglas, "lam": lam,
               "fp_ref": fp_ref, "algoritmo": algoritmo, "epoch": int(epoch),
               "pop_size": int(pop)}
        try:
            with st.spinner("Buscando la malla óptima..."):
                res = O.optimizar(bundles, cfg)
            st.session_state["_opt"] = {"res": res, "cfg": cfg, "rmse": rmse,
                                        "unidad": unidad, "patron": patron,
                                        "modelos": sel, "unidad_d": unidad_d_in}
        except Exception as e:
            st.error(f"Error en la optimización: {e}")

    if "_opt" in st.session_state:
        _mostrar_resultado(st.session_state["_opt"])


# =========================================================================== #
#  Resultados                                                                 #
# =========================================================================== #
def _mostrar_resultado(s):
    res, cfg, rmse, unidad = s["res"], s["cfg"], s["rmse"], s["unidad"]
    m, p80, obj = res["malla"], res["p80"], cfg["objetivo"]
    ud = s["unidad_d"]
    d_show = m["d"] / O._A_D[ud]

    st.divider()
    st.subheader("Malla recomendada")
    if cfg["modo"] == "maximo":
        cumple = p80 <= obj + 1e-9
    else:
        cumple = abs(p80 - obj) <= cfg["tol"] + 1e-9
    if cumple:
        st.success(f"✅ Se alcanzó el objetivo: P80 predicho {p80:.2f} {unidad} "
                   f"(objetivo {obj:.2f} {unidad}).")
    else:
        st.warning(
            f"⚠️ No se alcanzó el objetivo: lo más cercano fue {p80:.2f} {unidad} "
            f"(objetivo {obj:.2f} {unidad}) sin salir de los rangos y reglas "
            "fijados. Prueba ampliar los rangos, cambiar el explosivo o relajar "
            "las restricciones del paso 7.")

    c1, c2, c3 = st.columns(3)
    with c1:
        txt = f"{p80:.2f} ± {rmse:.2f} {unidad}" if rmse else f"{p80:.2f} {unidad}"
        _tarjeta_resultado("P80 esperado", txt,
                           "± RMSE en test del modelo" if rmse else None)
    with c2:
        _tarjeta_resultado("Malla (B × S)", f"{m['B']:.2f} × {m['S']:.2f} m",
                           f"{s['patron']} · taco {m['T']:.2f} m · taladro {m['L']:.2f} m")
    with c3:
        _tarjeta_resultado("Factor de potencia", f"{m['FP']:.3f} kg/m³",
                           f"Carga por taladro: {m['Q']:.1f} kg")

    tabla = pd.DataFrame([
        ("Burden (B)", m["B"], "m"),
        ("Espaciamiento (S)", m["S"], "m"),
        ("Altura de banco (H)", m["H"], "m"),
        ("Longitud de taladro (L)", m["L"], "m"),
        ("Sobreperforación (J = L − H)", m["J"], "m"),
        ("Taco (T)", m["T"], "m"),
        ("Longitud de carga (L − T)", m["Lc"], "m"),
        (f"Diámetro", d_show, ud),
        ("Densidad del explosivo", m["rho"], "g/cm³"),
        ("Carga por taladro (Q)", m["Q"], "kg"),
        ("Factor de potencia (FP)", m["FP"], "kg/m³"),
        ("T/B", m["T_B"], "—"),
        ("J/B", m["J_B"], "—"),
        ("H/B (rigidez)", m["H_B"], "—"),
        ("B/D", m["B_D"], "—"),
    ], columns=["Parámetro", "Valor", "Unidad"])
    tabla["Valor"] = tabla["Valor"].astype(float).round(3)
    cA, cB = st.columns([1.1, 1])
    with cA:
        st.dataframe(tabla, hide_index=True, width="stretch", height=560)
    with cB:
        if len(res["pred_por_modelo"]) > 1:
            st.caption("Predicción de cada modelo para esta malla")
            st.dataframe(pd.DataFrame({
                "Modelo": s["modelos"],
                f"P80 ({unidad})": np.round(res["pred_por_modelo"], 3)}),
                hide_index=True, width="stretch")
        if res["convergencia"]:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.plot(res["convergencia"], color="#1f4e79")
            ax.set_xlabel("Época")
            ax.set_ylabel("Función objetivo")
            ax.set_title(f"Convergencia ({cfg['algoritmo']}, {res['n_eval']} evaluaciones)")
            ax.grid(True, ls="--", alpha=0.4)
            st.pyplot(fig)
        st.caption(f"Variables optimizadas: {', '.join(_VARS[v][0] for v in res['libres'])}. "
                   f"FP calculado con c = {cfg['c']:.3f}.")

    st.subheader("Alternativas")
    st.caption("Otras mallas con buen puntaje, ordenadas de menor a mayor FP. "
               "Útiles si la recomendada no es práctica en campo.")
    alt = pd.DataFrame(res["alternativas"])
    if not alt.empty:
        alt = alt[["B", "S", "T", "L", "J", "Lc", "rho", "Q", "FP", "P80"]]
        alt.columns = ["B (m)", "S (m)", "Taco (m)", "L (m)", "J (m)", "Carga (m)",
                       "ρe (g/cm³)", "Q (kg)", "FP (kg/m³)", f"P80 ({unidad})"]
        alt = alt.sort_values("FP (kg/m³)").round(3)
        st.dataframe(alt, hide_index=True, width="stretch")
        st.download_button(
            "⬇️ Descargar malla y alternativas (CSV)",
            (tabla.to_csv(index=False) + "\nAlternativas\n"
             + alt.to_csv(index=False)).encode("utf-8"),
            "malla_optimizada.csv", "text/csv")

    st.info("El resultado es una recomendación basada en el modelo: valídala con "
            "una voladura de prueba y su análisis granulométrico antes de "
            "adoptarla como estándar.")
