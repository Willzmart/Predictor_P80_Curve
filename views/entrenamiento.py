"""views/entrenamiento.py — Entrenamiento de modelos: P80 o Curva granulométrica."""
import re
import unicodedata
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
 
from core import (db, models as M, curve_models as CM, training,
                  versioning as V, preprocessing as P, autotune as AT)
 
 
# --------------------------------------------------------------------------- #
#  Utilidades para trabajar con datasets de nombres variables                 #
# --------------------------------------------------------------------------- #
def _norm(s) -> str:
    """Normaliza un nombre de columna: sin acentos, minúsculas, sin espacios.
    'P80 (in)' -> 'p80(in)' ; 'Pmáx (in)' -> 'pmax(in)'."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return s.strip().lower().replace(" ", "")
 
 
def _detectar_target_p80(cols):
    """Devuelve la columna que representa el P80, aunque tenga sufijos
    ('P80', 'P80 (in)', 'p80', 'P80.'). None si no se encuentra ninguna."""
    if P.TARGET_P80 in cols:                 # coincidencia exacta ("P80")
        return P.TARGET_P80
    for c in cols:
        if _norm(c).startswith("p80"):       # "P80 (in)", "p80", ...
            return c
    return None
 
 
def _es_salida_fragmentacion(col) -> bool:
    """True si la columna es una SALIDA de la voladura (otro percentil, Pmáx o
    % de finos) y por tanto no debe usarse como predictor del P80: usarla sería
    fuga de información. Reconoce nombres con sufijos como '(in)' o '(%)'."""
    n = _norm(col)
    if "finos" in n:                         # "Porcentaje de finos <1 (%)"
        return True
    if n.startswith("pmax"):                 # "Pmáx (in)" -> "pmax(in)"
        return True
    if re.match(r"^p\d{1,3}(\D|$)", n):       # p10, p50, p80, p100, "p80(in)"...
        return True
    if re.match(r"^x\d{1,3}(\D|$)", n):       # x50_ref, x80... (tamaños de la curva)
        return True
    if "uniformidad" in n or n.startswith("topsize"):  # índice n de Rosin-Rammler, Top Size
        return True
    return False
 
 
def _nota_diagnostico():
    st.caption(
        "**Épocas**: las que usó cada modelo hasta que su error de validación "
        "dejó de mejorar. **Diagnóstico**: compara R² de entrenamiento y de test; "
        f"una brecha mayor a {AT.BRECHA_R2_ALERTA:.2f} indica que el modelo "
        "memoriza el entrenamiento más de lo que generaliza. Ante un empate "
        "práctico de RMSE, prefiere el modelo que generaliza bien.")


def page_entrenamiento():
    st.title("🧠 Entrenamiento de modelos")
 
    if not db.tiene_datos():
        st.warning("Primero importa datos en la sección **Datos**.")
        return
 
    df = db.cargar_datos().drop(columns=["id"], errors="ignore")
    hay_curva = P.tiene_columnas_curva(df)
 
    tarea = st.radio(
        "¿Qué deseas entrenar?",
        ["Predicción de P80", "Curva granulométrica (P10–P100)"],
        horizontal=True,
        help="La curva solo está disponible si la base tiene las columnas P10 a P100.",
    )
 
    if tarea.startswith("Curva") and not hay_curva:
        st.warning("Tu base no tiene las columnas de percentiles (P10…P100), así "
                   "que la curva no está disponible. Importa una base que las "
                   "incluya para habilitar esta opción.")
        return
 
    if tarea.startswith("Predicción"):
        _entrenar_p80(df, hay_curva)
    else:
        _entrenar_curva(df)
 
 
# =========================================================================== #
#  P80                                                                        #
# =========================================================================== #
def _entrenar_p80(df, hay_curva):
    st.subheader("1 · Selección de modelos a entrenar")
    opciones = M.listar_modelos()
    cols = st.columns(2)
    seleccion = []
    for i, (key, label) in enumerate(opciones):
        with cols[i % 2]:
            if st.checkbox(label, value=(key in ("svr_pso", "xgb_ga", "rf_ga")),
                           key=f"chk_p80_{key}"):
                seleccion.append(key)
 
    st.subheader("2 · Configuración")
    cols_all = df.columns.tolist()
 
    # La variable objetivo la ELIGE el usuario (con autodetección de P80). Antes
    # se caía a la última columna cuando no existía una llamada exactamente
    # "P80", lo que hacía que el modelo predijera una columna equivocada.
    sugerido = _detectar_target_p80(cols_all)
    c1, c2 = st.columns(2)
    target = c1.selectbox(
        "Variable objetivo (P80)", cols_all,
        index=cols_all.index(sugerido) if sugerido else len(cols_all) - 1,
        key="target_p80",
        help="Columna que contiene el P80. Se autoselecciona si se detecta "
             "(admite 'P80', 'P80 (in)', 'p80', ...).")
    if sugerido is None:
        c1.warning("No se detectó una columna 'P80'. Selecciónala manualmente "
                   "antes de entrenar.")
    test_size = c2.slider("Proporción de test", 0.1, 0.4, 0.2, 0.05, key="ts_p80")
 
    # Excluir por defecto las SALIDAS de fragmentación (otros percentiles, Pmáx,
    # % de finos) para evitar fuga de información. Reconoce sufijos como "(in)".
    posibles = [c for c in cols_all if c != target]
    excl_def = [c for c in posibles if _es_salida_fragmentacion(c)]
    excluir = st.multiselect(
        "Excluir de las variables predictoras", posibles,
        default=excl_def, key="excl_p80",
        help="Se excluyen otros percentiles, Pmáx y % de finos (son resultados de "
             "la voladura, no parámetros de diseño) para evitar fuga de información.")
    features = [c for c in posibles if c not in excluir]
 
    st.info("⚙️ **Configuración automática** · validación cruzada k = 5 · "
            f"población = {AT.POP_SIZE} · épocas: cada modelo se detiene solo cuando su error "
            "de validación deja de mejorar (parada temprana, entre "
            f"{AT.MIN_EPOCH} y {AT.MAX_EPOCH}). La búsqueda penaliza la brecha entre validación y "
            "entrenamiento para evitar sobreajuste.")
    st.info(f"**{target}** ← {len(features)} variables · {len(seleccion)} modelos")
 
    if st.button("🚀 Entrenar (P80)", type="primary", disabled=not seleccion):
        X, y, _ = P.separar_x_y(df, target, excluir=excluir)
        barra = st.progress(0.0, text="Iniciando...")
        with st.spinner("Optimizando y entrenando..."):
            bundles, tabla, mejor = training.entrenar_varios(
                seleccion, X, y, test_size=test_size,
                progreso=lambda f, t: barra.progress(f, text=t))
        st.session_state["_p80"] = {"bundles": bundles, "tabla": tabla,
                                    "mejor": mejor, "nobs": len(df), "target": target}
 
    if "_p80" in st.session_state:
        s = st.session_state["_p80"]
        _mostrar_resultado_p80(s)
 
 
def _mostrar_resultado_p80(s):
    bundles, tabla, mejor = s["bundles"], s["tabla"], s["mejor"]
    st.subheader("Evaluación y comparación")
    st.dataframe(tabla.drop(columns=["key"]), width="stretch", hide_index=True)
    _nota_diagnostico()
    st.success(f"🏆 Mejor: **{M.REGISTRY[mejor]['label']}** "
               f"(RMSE test {bundles[mejor]['metrics']['rmse_test']:.4f})")
    st.bar_chart(tabla.set_index("Modelo")[["R2_test"]])
 
    va = V.version_activa("p80")
    if va is None:
        st.info("No hay versión previa de P80: será la Versión 1.")
    elif V.mejora_respecto_activa(bundles, "p80"):
        st.success("✅ Mejora respecto a la versión activa de P80.")
    else:
        st.warning("⚠️ No mejora respecto a la versión activa de P80.")
 
    if st.button("💾 Guardar como nueva versión (P80)", type="primary"):
        v = V.guardar_version(bundles, s["nobs"], s["target"], kind="p80")
        st.success(f"Guardado como Versión {v} (P80).")
        st.session_state.pop("_p80", None)
        st.rerun()
 
 
# =========================================================================== #
#  Curva granulométrica                                                       #
# =========================================================================== #
def _entrenar_curva(df):
    st.subheader("1 · Selección de modelos a entrenar")
    opciones = CM.listar_modelos()
    cols = st.columns(2)
    seleccion = []
    for i, (key, label) in enumerate(opciones):
        with cols[i % 2]:
            if st.checkbox(label, value=(key in ("lgbm_ffa", "xgb_ga")),
                           key=f"chk_cur_{key}"):
                seleccion.append(key)
 
    st.subheader("2 · Configuración")
    st.caption("Objetivo (multi-salida): " + ", ".join(P.PERCENTILES))
    # Detecta las columnas reales de percentiles (admite sufijos como '_cm').
    mapa_pct = P.mapa_percentiles(df)
    cols_pct = list(mapa_pct.values())
    # Features = todo menos las columnas de percentiles y las salidas de
    # fragmentación (Pmáx, % de finos), reconociendo nombres con sufijos.
    posibles = [c for c in df.columns if c not in cols_pct]
    excl_def = [c for c in posibles if _es_salida_fragmentacion(c)]
    excluir = st.multiselect("Excluir de las variables predictoras", posibles,
                             default=excl_def, key="excl_cur")
    features = [c for c in posibles if c not in excluir]
 
    c1, _ = st.columns(2)
    test_size = c1.slider("Proporción de test", 0.1, 0.4, 0.2, 0.05, key="ts_cur")
    st.info("⚙️ **Configuración automática** · validación cruzada k = 5 · "
            f"población = {AT.POP_SIZE} · épocas: cada modelo se detiene solo cuando su error "
            "de validación deja de mejorar (parada temprana, entre "
            f"{AT.MIN_EPOCH} y {AT.MAX_EPOCH}). La búsqueda penaliza la brecha entre validación y "
            "entrenamiento para evitar sobreajuste.")
    st.info(f"Curva (10 percentiles) ← {len(features)} variables · "
            f"{len(seleccion)} modelos")
 
    if st.button("🚀 Entrenar (Curva)", type="primary", disabled=not seleccion):
        X = df[features]
        # Toma las columnas reales en orden y las renombra a los canónicos
        # (P10..P100) para el resto del pipeline.
        y = df[[mapa_pct[p] for p in P.PERCENTILES]].copy()
        y.columns = P.PERCENTILES
        barra = st.progress(0.0, text="Iniciando...")
        with st.spinner("Optimizando y entrenando (multi-salida, más lento)..."):
            bundles, tabla, mejor = training.entrenar_varios_curva(
                seleccion, X, y, test_size=test_size,
                progreso=lambda f, t: barra.progress(f, text=t))
        st.session_state["_cur"] = {"bundles": bundles, "tabla": tabla,
                                    "mejor": mejor, "nobs": len(df)}
 
    if "_cur" in st.session_state:
        _mostrar_resultado_curva(st.session_state["_cur"])
 
 
def _mostrar_resultado_curva(s):
    bundles, tabla, mejor = s["bundles"], s["tabla"], s["mejor"]
    st.subheader("Evaluación y comparación")
    st.dataframe(tabla.drop(columns=["key"]), width="stretch", hide_index=True)
    _nota_diagnostico()
    st.success(f"🏆 Mejor: **{CM.REGISTRY[mejor]['label']}** "
               f"(RMSE promedio test {bundles[mejor]['metrics']['rmse_test']:.4f})")
 
    b = bundles[mejor]
    cA, cB = st.columns(2)
    with cA:
        st.caption("Métricas por percentil (mejor modelo)")
        st.dataframe(pd.DataFrame(b["por_percentil"]), width="stretch",
                     hide_index=True, height=280)
    with cB:
        st.caption("Curvas de muestra (real vs. predicho)")
        fig, ax = plt.subplots(figsize=(5, 4))
        pct = [int(p[1:]) for p in b["percentiles"]]
        yreal = np.asarray(b["y_test_muestra"])
        ypred = np.asarray(b["y_pred_muestra"])
        for i in range(min(3, yreal.shape[0])):
            ax.plot(yreal[i], pct, "o-", alpha=0.6)
            ax.plot(ypred[i], pct, "s--", alpha=0.6)
        ax.set_xlabel("Tamaño (cm)"); ax.set_ylabel("% pasante acumulado")
        ax.set_title("Real (○) vs. predicho (□)")
        ax.grid(True, ls="--", alpha=0.4)
        st.pyplot(fig)
 
    va = V.version_activa("curva")
    if va is None:
        st.info("No hay versión previa de curva: será la Versión 1.")
    elif V.mejora_respecto_activa(bundles, "curva"):
        st.success("✅ Mejora respecto a la versión activa de curva.")
    else:
        st.warning("⚠️ No mejora respecto a la versión activa de curva.")
 
    if st.button("💾 Guardar como nueva versión (Curva)", type="primary"):
        v = V.guardar_version(bundles, s["nobs"], P.PERCENTILES, kind="curva")
        st.success(f"Guardado como Versión {v} (Curva).")
        st.session_state.pop("_cur", None)
        st.rerun()
 
