"""views/resultados.py — Resultados y evaluación de modelos."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.inspection import permutation_importance

from core import db, models as M, versioning as V


def page_resultados():
    st.title("📈 Resultados y evaluación")

    hay_p80 = V.version_activa("p80") is not None
    hay_curva = V.version_activa("curva") is not None
    if not hay_p80 and not hay_curva:
        st.warning("No hay versiones entrenadas todavía.")
        return

    disponibles = []
    if hay_p80:
        disponibles.append("Predicción de P80")
    if hay_curva:
        disponibles.append("Curva granulométrica (P10–P100)")
    tarea = st.radio("Resultados de:", disponibles, horizontal=True)

    if tarea.startswith("Predicción"):
        _resultados_p80()
    else:
        _resultados_curva()


def _resultados_p80():
    va = V.version_activa("p80")
    meta = V.cargar_metadata(va, "p80")

    # --- 1) Resumen de desempeño ------------------------------------------ #
    st.subheader(f"1 · Resumen de desempeño · Versión {va}")
    filas = []
    for k in meta["modelos"]:
        m = meta["metricas"][k]
        filas.append({
            "Modelo": meta["labels"][k],
            "R2_test": round(m["r2_test"], 4),
            "RMSE_test": round(m["rmse_test"], 4),
            "MAE_test": round(m["mae_test"], 4),
            "RMSE_cv": round(m["rmse_cv"], 4),
        })
    tabla = pd.DataFrame(filas).sort_values("RMSE_test").reset_index(drop=True)
    tabla.insert(0, "Ranking", range(1, len(tabla) + 1))
    st.dataframe(tabla, width='stretch', hide_index=True)
    st.success(f"🏆 Mejor modelo: **{meta['labels'][meta['mejor_modelo']]}**")

    cta, ctb = st.columns(2)
    cta.bar_chart(tabla.set_index("Modelo")[["R2_test"]])
    ctb.bar_chart(tabla.set_index("Modelo")[["RMSE_test"]])

    # --- 2) Análisis detallado -------------------------------------------- #
    st.subheader("2 · Análisis detallado")
    keys = meta["modelos"]
    labels = [meta["labels"][k] for k in keys]
    sel = st.selectbox("Modelo a analizar", labels,
                       index=keys.index(meta["mejor_modelo"]))
    key = keys[labels.index(sel)]
    bundle = V.cargar_bundle(va, key)

    y_test = np.asarray(bundle["y_test"])
    y_pred = np.asarray(bundle["y_pred_test"])
    resid = y_test - y_pred

    g1, g2 = st.columns(2)
    with g1:
        st.caption("Predicho vs. Real (test)")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test, y_pred, alpha=0.7, edgecolor="k")
        lo, hi = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel("Real"); ax.set_ylabel("Predicho")
        st.pyplot(fig)
    with g2:
        st.caption("Residuos vs. Predicho")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.scatter(y_pred, resid, alpha=0.7, edgecolor="k")
        ax2.axhline(0, color="r", ls="--", lw=1)
        ax2.set_xlabel("Predicho"); ax2.set_ylabel("Residuo")
        st.pyplot(fig2)

    # Importancia de variables (permutación sobre test — versión ligera, sin SHAP)
    st.caption("Importancia de variables (permutación sobre test)")
    try:
        imp = _importancia(bundle)
        st.bar_chart(imp.set_index("variable"))
    except Exception as e:
        st.caption(f"No disponible: {e}")

    # --- 3) Historial de entrenamientos ----------------------------------- #
    st.subheader("3 · Historial de entrenamientos")
    he = db.historial_entrenamientos()
    if not he.empty and "tipo" in he.columns:
        he = he[he["tipo"] == "p80"]
    if he.empty:
        st.caption("Sin registros.")
    else:
        st.dataframe(he[["fecha", "version", "n_obs", "mejor_modelo"]],
                     width='stretch', hide_index=True)

    # --- 4) Historial de predicciones ------------------------------------- #
    st.subheader("4 · Historial de predicciones")
    hp = db.historial_predicciones()
    if hp.empty:
        st.caption("Sin registros.")
    else:
        st.dataframe(hp[["fecha", "version", "modelo", "p80_predicho"]],
                     width='stretch', hide_index=True)

    # --- 5) Exportar ------------------------------------------------------ #
    st.subheader("5 · Exportar resultados")
    st.download_button("⬇️ Comparativa (CSV)",
                       tabla.to_csv(index=False).encode("utf-8"),
                       f"resultados_version_{va}.csv", "text/csv")


def _resultados_curva():
    from core import curve_models as CM
    va = V.version_activa("curva")
    meta = V.cargar_metadata(va, "curva")

    st.subheader(f"1 · Resumen de desempeño · Curva · Versión {va}")
    filas = []
    for k in meta["modelos"]:
        m = meta["metricas"][k]
        filas.append({"Modelo": meta["labels"][k],
                      "R2_test": round(m["r2_test"], 4),
                      "RMSE_test": round(m["rmse_test"], 4),
                      "MAE_test": round(m["mae_test"], 4),
                      "RMSE_cv": round(m["rmse_cv"], 4)})
    tabla = pd.DataFrame(filas).sort_values("RMSE_test").reset_index(drop=True)
    tabla.insert(0, "Ranking", range(1, len(tabla) + 1))
    st.dataframe(tabla, width="stretch", hide_index=True)
    st.success(f"🏆 Mejor modelo: **{meta['labels'][meta['mejor_modelo']]}** "
               "(RMSE promedio sobre los 10 percentiles)")
    st.bar_chart(tabla.set_index("Modelo")[["RMSE_test"]])

    st.subheader("2 · Análisis detallado")
    keys = meta["modelos"]
    labels = [meta["labels"][k] for k in keys]
    sel = st.selectbox("Modelo a analizar", labels,
                       index=keys.index(meta["mejor_modelo"]), key="cur_res")
    key = keys[labels.index(sel)]
    bundle = V.cargar_bundle(va, key, "curva")

    g1, g2 = st.columns(2)
    with g1:
        st.caption("Métricas por percentil")
        st.dataframe(pd.DataFrame(bundle["por_percentil"]), width="stretch",
                     hide_index=True, height=300)
    with g2:
        st.caption("Curvas de muestra (real vs. predicho)")
        pct = [int(p[1:]) for p in bundle["percentiles"]]
        yreal = np.asarray(bundle["y_test_muestra"])
        ypred = np.asarray(bundle["y_pred_muestra"])
        fig, ax = plt.subplots(figsize=(5, 4))
        for i in range(min(3, yreal.shape[0])):
            ax.plot(yreal[i], pct, "o-", alpha=0.6)
            ax.plot(ypred[i], pct, "s--", alpha=0.6)
        ax.set_xlabel("Tamaño (cm)"); ax.set_ylabel("% pasante")
        ax.set_title("Real (○) vs. predicho (□)")
        ax.grid(True, ls="--", alpha=0.4)
        st.pyplot(fig)

    st.subheader("3 · Historial de entrenamientos")
    he = db.historial_entrenamientos()
    if not he.empty and "tipo" in he.columns:
        he = he[he["tipo"] == "curva"]
    st.dataframe(he[["fecha", "version", "n_obs", "mejor_modelo"]],
                 width="stretch", hide_index=True) if not he.empty \
        else st.caption("Sin registros.")

    st.subheader("4 · Exportar")
    st.download_button("⬇️ Comparativa (CSV)",
                       tabla.to_csv(index=False).encode("utf-8"),
                       f"resultados_curva_v{va}.csv", "text/csv")


def _importancia(bundle):
    """Importancia por permutación sobre el test guardado. Funciona con
    cualquier modelo (incluido SVR rbf) porque no depende de coeficientes."""
    feats = bundle["features"]
    X_test = bundle.get("X_test")
    y_test = np.asarray(bundle["y_test"])
    if X_test is None:
        # Compatibilidad con modelos guardados antes de almacenar X_test
        model = bundle["model"]
        if hasattr(model, "feature_importances_"):
            return pd.DataFrame({"variable": feats,
                                 "importancia": model.feature_importances_})
        raise ValueError("Reentrena el modelo para calcular importancias.")

    Xt = bundle["scaler"].transform(X_test) if bundle["needs_scaling"] else X_test.values
    r = permutation_importance(bundle["model"], Xt, y_test,
                               n_repeats=10, random_state=42,
                               scoring="neg_root_mean_squared_error")
    return pd.DataFrame({"variable": feats,
                         "importancia": r.importances_mean}
                        ).sort_values("importancia", ascending=False)
