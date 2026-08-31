"""views/prediccion.py — Predicción de P80 o de la Curva granulométrica."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from core import (db, models as M, curve_models as CM,
                  versioning as V, preprocessing as P)


def page_prediccion():
    st.title("🎯 Predicción")

    hay_p80 = V.version_activa("p80") is not None
    hay_curva = V.version_activa("curva") is not None

    if not hay_p80 and not hay_curva:
        st.warning("No hay modelos entrenados. Entrena y guarda una versión en "
                   "**Entrenamiento**.")
        return

    disponibles = []
    if hay_p80:
        disponibles.append("Predicción de P80")
    if hay_curva:
        disponibles.append("Curva granulométrica (P10–P100)")
    tarea = st.radio("¿Qué deseas predecir?", disponibles, horizontal=True)

    if tarea.startswith("Predicción"):
        _predecir_p80()
    else:
        _predecir_curva()


# =========================================================================== #
#  P80                                                                        #
# =========================================================================== #
def _predecir_p80():
    va = V.version_activa("p80")
    meta = V.cargar_metadata(va, "p80")
    st.caption(f"Versión activa (P80): **{va}** · {meta['fecha']}")

    keys = meta["modelos"]
    labels = [meta["labels"][k] for k in keys]
    sel = st.selectbox("Modelo", labels, index=keys.index(meta["mejor_modelo"]),
                       key="mdl_p80")
    modelo_key = keys[labels.index(sel)]
    bundle = V.cargar_bundle(va, modelo_key, "p80")
    target = bundle["target"]

    modo = st.radio("Modo", ["Ingreso manual", "Archivo por lotes"],
                    horizontal=True, key="modo_p80")

    if modo == "Ingreso manual":
        vals = _formulario_features(bundle["features"], "p80")
        if st.button("🔮 Predecir P80", type="primary"):
            pred = float(M.predecir(bundle, pd.DataFrame([vals]))[0])
            _tarjeta_resultado(f"{target} estimado", f"{pred:.2f}")
            db.log_prediccion(va, modelo_key, vals, pred)
    else:
        _prediccion_lote_p80(bundle, va, modelo_key, target)

    st.divider()
    with st.expander("🕘 Historial de predicciones"):
        hist = db.historial_predicciones(tipo="p80")
        if hist.empty:
            st.caption("Sin predicciones registradas.")
        else:
            st.dataframe(hist[["fecha", "version", "modelo", "p80_predicho"]],
                         width="stretch", hide_index=True)


def _prediccion_lote_p80(bundle, va, modelo_key, target):
    arch = st.file_uploader("CSV / Excel", type=["xlsx", "xls", "csv"], key="f_p80")
    aplicar = st.checkbox("Aplicar pipeline de tesis", value=True, key="ap_p80")
    if arch is not None:
        try:
            dfn = P.cargar_datos(arch, arch.name)
            if aplicar:
                dfn = P.preprocess_thesis(dfn, conservar_curva=True)
            faltan = [c for c in bundle["features"] if c not in dfn.columns]
            if faltan:
                st.error("Faltan columnas: " + ", ".join(faltan))
            else:
                out = dfn.copy()
                out[f"{target}_predicho"] = M.predecir(bundle, dfn)
                st.success(f"{len(out)} predicciones generadas.")
                st.dataframe(out.head(30), width="stretch")
                st.download_button("⬇️ Descargar (CSV)",
                                   out.to_csv(index=False).encode("utf-8"),
                                   "predicciones_p80.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")


# =========================================================================== #
#  Curva granulométrica                                                       #
# =========================================================================== #
def _predecir_curva():
    va = V.version_activa("curva")
    meta = V.cargar_metadata(va, "curva")
    st.caption(f"Versión activa (Curva): **{va}** · {meta['fecha']}")

    keys = meta["modelos"]
    labels = [meta["labels"][k] for k in keys]
    sel = st.selectbox("Modelo", labels, index=keys.index(meta["mejor_modelo"]),
                       key="mdl_cur")
    modelo_key = keys[labels.index(sel)]
    bundle = V.cargar_bundle(va, modelo_key, "curva")
    percentiles = bundle["percentiles"]
    pct = [int(p[1:]) for p in percentiles]

    modo = st.radio("Modo", ["Ingreso manual", "Archivo por lotes"],
                    horizontal=True, key="modo_cur")

    if modo == "Ingreso manual":
        vals = _formulario_features(bundle["features"], "cur")
        if st.button("🔮 Predecir curva", type="primary"):
            curva = CM.predecir_curva(bundle, pd.DataFrame([vals]))[0]
            p80_curva = float(curva[percentiles.index("P80")]) \
                if "P80" in percentiles else None
            db.log_prediccion(va, modelo_key, vals, p80_curva,
                              tipo="curva", curva=[round(float(x), 4) for x in curva])
            c1, c2 = st.columns([3, 2])
            with c1:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(curva, pct, "s--", color="#d1701c", label="Predicho")
                ax.set_xlabel("Tamaño de partícula (cm)")
                ax.set_ylabel("% pasante acumulado")
                ax.set_title("Curva granulométrica predicha")
                ax.grid(True, ls="--", alpha=0.4)
                ax.legend()
                st.pyplot(fig)
            with c2:
                tab = pd.DataFrame({"Percentil": percentiles,
                                    "Tamaño (cm)": np.round(curva, 3)})
                st.dataframe(tab, width="stretch", hide_index=True)
                # P80 leído de la curva
                if "P80" in percentiles:
                    st.metric("P80 (de la curva)",
                              f"{curva[percentiles.index('P80')]:.2f}")
                st.download_button("⬇️ Descargar curva (CSV)",
                                   tab.to_csv(index=False).encode("utf-8"),
                                   "curva_predicha.csv", "text/csv")
    else:
        _prediccion_lote_curva(bundle, percentiles)

    st.divider()
    with st.expander("🕘 Historial de predicciones de curva"):
        hist = db.historial_predicciones(tipo="curva")
        if hist.empty:
            st.caption("Sin curvas registradas todavía.")
        else:
            vista = hist[["fecha", "version", "modelo", "p80_predicho"]].copy()
            vista = vista.rename(columns={"p80_predicho": "P80 (de la curva)"})
            st.dataframe(vista, width="stretch", hide_index=True)
            st.caption("Cada registro guarda además la curva completa (10 "
                       "percentiles). Descárgala con detalle abajo.")
            import json as _json
            filas = []
            for _, r in hist.iterrows():
                fila = {"fecha": r["fecha"], "modelo": r["modelo"]}
                if r.get("curva"):
                    for p, v in zip(percentiles, _json.loads(r["curva"])):
                        fila[p] = v
                filas.append(fila)
            det = pd.DataFrame(filas)
            st.download_button("⬇️ Descargar historial de curvas (CSV)",
                               det.to_csv(index=False).encode("utf-8"),
                               "historial_curvas.csv", "text/csv")


def _prediccion_lote_curva(bundle, percentiles):
    arch = st.file_uploader("CSV / Excel", type=["xlsx", "xls", "csv"], key="f_cur")
    aplicar = st.checkbox("Aplicar pipeline de tesis", value=True, key="ap_cur")
    if arch is not None:
        try:
            dfn = P.cargar_datos(arch, arch.name)
            if aplicar:
                dfn = P.preprocess_thesis(dfn, conservar_curva=True)
            faltan = [c for c in bundle["features"] if c not in dfn.columns]
            if faltan:
                st.error("Faltan columnas: " + ", ".join(faltan))
            else:
                curvas = CM.predecir_curva(bundle, dfn)
                out = dfn.copy()
                for j, p in enumerate(percentiles):
                    out[f"{p}_pred"] = np.round(curvas[:, j], 3)
                st.success(f"{len(out)} curvas generadas.")
                st.dataframe(out.head(30), width="stretch")
                st.download_button("⬇️ Descargar (CSV)",
                                   out.to_csv(index=False).encode("utf-8"),
                                   "curvas_predichas.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")


# =========================================================================== #
#  Utilidades                                                                 #
# =========================================================================== #
def _formulario_features(features, prefijo):
    st.write("Parámetros de voladura:")
    vals = {}
    grid = st.columns(3)
    for i, feat in enumerate(features):
        with grid[i % 3]:
            vals[feat] = st.number_input(feat, value=0.0, format="%.4f",
                                         key=f"in_{prefijo}_{feat}")
    return vals


def _tarjeta_resultado(titulo, valor):
    st.markdown(
        f"<div style='text-align:center;padding:1.4rem;border:2px solid #4c78a8;"
        f"border-radius:12px'><div style='font-size:1rem;color:#666'>{titulo}"
        f"</div><div style='font-size:2.6rem;font-weight:700;color:#1f4e79'>"
        f"{valor}</div></div>", unsafe_allow_html=True)
