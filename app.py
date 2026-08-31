"""
app.py — Aplicación Streamlit
Predicción y Optimización de Fragmentación (P80)
Sistema de gestión de datos, entrenamiento y predicción de modelos ML híbridos.

Ejecutar:  streamlit run app.py
"""
import streamlit as st

from core import db, versioning as V

st.set_page_config(page_title="Predicción de Fragmentación P80",
                   page_icon="⛏️", layout="wide")

# Inicializa la base de datos y el almacenamiento
db.init_db()

from views.datos import page_datos
from views.entrenamiento import page_entrenamiento
from views.prediccion import page_prediccion
from views.resultados import page_resultados
from views.configuracion import page_configuracion


def page_inicio():
    st.title("⛏️ Predicción y Optimización de Fragmentación (P80)")
    st.caption("Sistema de gestión de datos, entrenamiento y predicción de modelos ML híbridos")
    st.divider()

    estado = db.estado_base() if db.tiene_datos() else "Sin datos"
    va = V.version_activa("p80")
    va_c = V.version_activa("curva")
    n_reg = len(db.cargar_datos()) if db.tiene_datos() else 0
    n_ver = len(V.listar_versiones("p80"))
    n_ver_c = len(V.listar_versiones("curva"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros en la base", n_reg)
    c2.metric("Versiones P80", n_ver)
    c3.metric("Versiones Curva", n_ver_c)
    c4.metric("Estado", estado)

    st.divider()
    st.subheader("Flujo general")
    f = st.columns(5)
    pasos = ["1 · Cargar / gestionar datos", "2 · Entrenar / reentrenar",
             "3 · Evaluar y elegir mejor", "4 · Predecir P80", "5 · Reportes y decisiones"]
    for col, txt in zip(f, pasos):
        col.info(txt)

    st.subheader("Secciones")
    st.markdown(
        "- 📊 **Datos** — base de voladuras: importar, ver, editar, calidad, exportar.\n"
        "- 🧠 **Entrenamiento** — seleccionar modelos, entrenar, comparar y versionar.\n"
        "- 🎯 **Predicción** — estimar el P80 o la **curva granulométrica** de nuevas voladuras.\n"
        "- 📈 **Resultados** — ranking, predicho-vs-real, residuos, importancias, historiales.\n"
        "- ⚙️ **Configuración** — gestión de versiones y ayuda.")

    if not db.tiene_datos():
        st.warning("Empieza importando tu base de datos en la sección **Datos**.")


# ---- Navegación (dashboard) ---------------------------------------------- #
nav = st.navigation({
    "Principal": [
        st.Page(page_inicio, title="Dashboard", icon="🏠", default=True),
    ],
    "Módulos": [
        st.Page(page_datos, title="Datos", icon="📊", url_path="datos"),
        st.Page(page_entrenamiento, title="Entrenamiento", icon="🧠", url_path="entrenamiento"),
        st.Page(page_prediccion, title="Predicción", icon="🎯", url_path="prediccion"),
        st.Page(page_resultados, title="Resultados", icon="📈", url_path="resultados"),
        st.Page(page_configuracion, title="Configuración", icon="⚙️", url_path="configuracion"),
    ],
})
nav.run()
