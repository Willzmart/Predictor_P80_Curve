"""views/configuracion.py — Configuración, gestión de versiones y ayuda."""
import pandas as pd
import streamlit as st

from core import db, versioning as V


def page_configuracion():
    st.title("⚙️ Configuración y ayuda")

    # --- Configuración general -------------------------------------------- #
    st.subheader("Configuración general")
    c1, c2 = st.columns(2)
    unidad = c1.selectbox("Unidad de P80", ["cm", "mm", "in", "m"],
                          index=["cm", "mm", "in", "m"].index(
                              db.meta_get("unidad_p80", "cm")))
    if unidad != db.meta_get("unidad_p80", "cm"):
        db.meta_set("unidad_p80", unidad)
    c2.metric("Semilla aleatoria", "42 (fija)")
    st.caption("Ruta de almacenamiento: `storage/` (BD SQLite + modelos versionados).")

    # --- Gestión de versiones --------------------------------------------- #
    st.subheader("Gestión de versiones de modelos")
    tipo_lbl = st.radio("Tipo de modelo", ["P80", "Curva granulométrica"],
                        horizontal=True)
    kind = "p80" if tipo_lbl == "P80" else "curva"
    versiones = V.listar_versiones(kind)
    if not versiones:
        st.info(f"Aún no hay versiones guardadas de {tipo_lbl}.")
    else:
        activa = V.version_activa(kind)
        tabla = pd.DataFrame([{
            "Versión": m["version"],
            "Fecha": m["fecha"],
            "N° obs": m["n_obs"],
            "Mejor modelo": m["labels"][m["mejor_modelo"]],
            "RMSE_test (mejor)": round(
                m["metricas"][m["mejor_modelo"]]["rmse_test"], 4),
            "Activa": "✅" if m["version"] == activa else "",
        } for m in versiones])
        st.dataframe(tabla, width='stretch', hide_index=True)

        nums = [m["version"] for m in versiones]
        ca, cb = st.columns(2)
        with ca:
            v_act = st.selectbox("Activar versión", nums,
                                 index=nums.index(activa) if activa in nums else 0)
            if st.button("Activar"):
                V.activar_version(v_act, kind)
                st.success(f"Versión {v_act} activada.")
                st.rerun()
        with cb:
            v_del = st.selectbox("Eliminar versión", nums, key="del")
            if st.button("Eliminar", type="secondary"):
                V.eliminar_version(v_del, kind)
                st.warning(f"Versión {v_del} eliminada.")
                st.rerun()

        # Comparar dos versiones
        if len(nums) >= 2:
            st.markdown("**Comparar versiones**")
            cc1, cc2 = st.columns(2)
            v1 = cc1.selectbox("Versión A", nums, index=0, key="cmpA")
            v2 = cc2.selectbox("Versión B", nums, index=len(nums) - 1, key="cmpB")
            if v1 != v2:
                m1, m2 = V.cargar_metadata(v1, kind), V.cargar_metadata(v2, kind)
                comp = pd.DataFrame({
                    "Versión": [v1, v2],
                    "Mejor modelo": [m1["labels"][m1["mejor_modelo"]],
                                     m2["labels"][m2["mejor_modelo"]]],
                    "RMSE_test": [
                        round(m1["metricas"][m1["mejor_modelo"]]["rmse_test"], 4),
                        round(m2["metricas"][m2["mejor_modelo"]]["rmse_test"], 4)],
                    "R2_test": [
                        round(m1["metricas"][m1["mejor_modelo"]]["r2_test"], 4),
                        round(m2["metricas"][m2["mejor_modelo"]]["r2_test"], 4)],
                })
                st.dataframe(comp, width='stretch', hide_index=True)

    # --- Información de la app --------------------------------------------- #
    st.subheader("Información de la aplicación")
    st.markdown(
        "**Descripción.** Sistema de predicción de fragmentación (P80) a partir "
        "de variables de voladura, con modelos híbridos ML + metaheurística.\n\n"
        "**Metodología.** Cada modelo optimiza sus hiperparámetros con una "
        "metaheurística (PSO, GA o FFA) minimizando el RMSE en validación cruzada; "
        "luego se compara y se versiona el mejor.\n\n"
        "**Modelos.** GBM+FFA, SVR+PSO, SVR+GA, XGBoost+FFA, XGBoost+PSO, "
        "XGBoost+GA, Random Forest+GA.\n\n"
        "**Créditos.** Derivado del proyecto de tesis de predicción de P80 y % Finos.")
    with st.expander("Manual de uso rápido"):
        st.markdown(
            "1. **Datos** — importa tu Excel/CSV, revisa calidad, edita si hace falta.\n"
            "2. **Entrenamiento** — elige modelos, configura y entrena; guarda la versión.\n"
            "3. **Predicción** — ingresa una voladura o sube un lote y obtén el P80.\n"
            "4. **Resultados** — compara modelos, revisa residuos e importancias.\n"
            "5. **Configuración** — administra versiones (activar, eliminar, comparar).")
