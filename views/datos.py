"""views/datos.py — Gestión de datos: importar, ver, editar, calidad, exportar."""
import io
import numpy as np
import pandas as pd
import streamlit as st

from core import db, preprocessing as P


def page_datos():
    st.title("📊 Gestión de datos")
    st.caption("Base de datos de voladuras (SQLite)")

    # --- Importar ---------------------------------------------------------- #
    with st.expander("📥 Importar datos (CSV / Excel)", expanded=not db.tiene_datos()):
        arch = st.file_uploader("Archivo", type=["xlsx", "xls", "csv"])
        aplicar = st.checkbox("Aplicar pipeline de tesis al importar", value=True,
                              help="Extrae Burden de 'Malla', codifica Material y "
                                   "Litología, elimina P10–P100, Fecha, etc.")
        modo = st.radio("Al importar:", ["Reemplazar base actual", "Añadir a la base"],
                        horizontal=True)
        if arch is not None and st.button("Importar", type="primary"):
            try:
                nuevo = P.cargar_datos(arch, arch.name)
                if aplicar:
                    # conservar_curva: mantiene P10..P100 si existen, para poder
                    # predecir también la curva granulométrica
                    nuevo = P.preprocess_thesis(nuevo, conservar_curva=True)
                if modo.startswith("Añadir") and db.tiene_datos():
                    base = db.cargar_datos().drop(columns=["id"], errors="ignore")
                    nuevo = pd.concat([base, nuevo], ignore_index=True)
                db.guardar_datos(nuevo)
                st.success(f"Importadas {len(nuevo)} filas.")
                st.rerun()
            except Exception as e:
                st.error(f"Error al importar: {e}")

    if not db.tiene_datos():
        st.info("Aún no hay datos. Importa un archivo para empezar.")
        return

    df = db.cargar_datos()

    # --- Información de la base -------------------------------------------- #
    estado = db.estado_base()
    color = {"Actualizado": "🟢", "Desactualizado": "🟡", "Sin entrenar": "⚪"}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", len(df))
    c2.metric("Variables", df.drop(columns=["id"]).shape[1])
    c3.metric("Últ. modificación", (db.meta_get("datos_modificados") or "—")[-8:])
    c4.metric("Estado", f"{color.get(estado,'')} {estado}")

    # --- Visualización / edición ------------------------------------------ #
    st.subheader("Tabla interactiva")
    st.caption("Edita celdas directamente y pulsa **Guardar cambios**. "
               "Marca filas y usa **Eliminar seleccionadas**.")
    editado = st.data_editor(df, width='stretch', num_rows="dynamic",
                             height=340, key="editor_datos", hide_index=True)

    cc1, cc2, _ = st.columns([1, 1, 3])
    if cc1.button("💾 Guardar cambios"):
        db.guardar_datos(editado.drop(columns=["id"], errors="ignore"))
        st.success("Cambios guardados.")
        st.rerun()

    with cc2.popover("🗑️ Eliminar por ID"):
        ids = st.multiselect("IDs a eliminar", df["id"].tolist())
        if st.button("Confirmar eliminación") and ids:
            db.eliminar_registros(ids)
            st.rerun()

    # --- Agregar registro -------------------------------------------------- #
    with st.expander("➕ Agregar registro manual"):
        cols = [c for c in df.columns if c != "id"]
        vals = {}
        grid = st.columns(3)
        for i, c in enumerate(cols):
            with grid[i % 3]:
                if pd.api.types.is_numeric_dtype(df[c]):
                    vals[c] = st.number_input(c, value=0.0, key=f"add_{c}")
                else:
                    vals[c] = st.text_input(c, key=f"add_{c}")
        if st.button("Agregar"):
            db.agregar_registro(vals)
            st.success("Registro agregado.")
            st.rerun()

    # --- Control de calidad ----------------------------------------------- #
    st.subheader("Validación y control de calidad")
    q = db.control_calidad(df)
    qc1, qc2, qc3 = st.columns(3)
    qc1.metric("Total de nulos", q["total_nulos"])
    qc2.metric("Filas duplicadas", q["duplicados"])
    qc3.metric("Columnas numéricas", len(q["outliers_por_columna"]))

    tab1, tab2, tab3 = st.tabs(["Nulos", "Outliers (IQR)", "Tipos"])
    with tab1:
        st.dataframe(pd.DataFrame(
            {"columna": list(q["nulos_por_columna"]),
             "nulos": list(q["nulos_por_columna"].values())}),
            width='stretch', hide_index=True)
    with tab2:
        st.dataframe(pd.DataFrame(
            {"columna": list(q["outliers_por_columna"]),
             "outliers": list(q["outliers_por_columna"].values())}),
            width='stretch', hide_index=True)
    with tab3:
        st.dataframe(pd.DataFrame(
            {"columna": list(q["tipos"]), "tipo": list(q["tipos"].values())}),
            width='stretch', hide_index=True)

    # --- Exportar --------------------------------------------------------- #
    st.subheader("Exportar")
    e1, e2 = st.columns(2)
    csv = df.to_csv(index=False).encode("utf-8")
    e1.download_button("⬇️ CSV", csv, "voladuras.csv", "text/csv")
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="voladuras")
    e2.download_button("⬇️ Excel", xbuf.getvalue(), "voladuras.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
