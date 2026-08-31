"""
core/db.py
----------
Capa de datos SQLite del sistema. Maneja:
  - tabla de datos de voladuras (dinámica: se adapta a las columnas del archivo)
  - metadatos clave-valor (estado, versión activa, marca de tiempo de datos)
  - historial de entrenamientos
  - historial de predicciones
  - control de calidad (nulos, duplicados, outliers, tipos)
"""
import sqlite3
import json
from datetime import datetime

import pandas as pd

from . import config

config.ensure_dirs()


# --------------------------------------------------------------------------- #
#  Conexión e inicialización                                                  #
# --------------------------------------------------------------------------- #
def get_conn():
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    """Crea las tablas de sistema si no existen."""
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_meta (
                clave TEXT PRIMARY KEY,
                valor TEXT
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha TEXT,
                version INTEGER,
                n_obs INTEGER,
                modelos TEXT,
                mejor_modelo TEXT,
                metricas TEXT
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fecha TEXT,
                version INTEGER,
                modelo TEXT,
                parametros TEXT,
                p80_predicho REAL
            )""")
        # Migración: columna 'tipo' en training_history (p80 / curva)
        cols = [r[1] for r in conn.execute(
            "PRAGMA table_info(training_history)").fetchall()]
        if "tipo" not in cols:
            conn.execute(
                "ALTER TABLE training_history ADD COLUMN tipo TEXT DEFAULT 'p80'")
        # Migración: columnas 'tipo' y 'curva' en prediction_history
        pcols = [r[1] for r in conn.execute(
            "PRAGMA table_info(prediction_history)").fetchall()]
        if "tipo" not in pcols:
            conn.execute(
                "ALTER TABLE prediction_history ADD COLUMN tipo TEXT DEFAULT 'p80'")
        if "curva" not in pcols:
            # Guarda la curva completa (10 percentiles) como JSON cuando aplica
            conn.execute(
                "ALTER TABLE prediction_history ADD COLUMN curva TEXT")


# --------------------------------------------------------------------------- #
#  Metadatos clave-valor                                                      #
# --------------------------------------------------------------------------- #
def meta_set(clave, valor):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO app_meta(clave, valor) VALUES(?, ?) "
            "ON CONFLICT(clave) DO UPDATE SET valor=excluded.valor",
            (clave, str(valor)),
        )


def meta_get(clave, default=None):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT valor FROM app_meta WHERE clave=?", (clave,)
        ).fetchone()
    return row[0] if row else default


# --------------------------------------------------------------------------- #
#  Tabla de datos de voladuras                                                #
# --------------------------------------------------------------------------- #
def tiene_datos():
    with get_conn() as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (config.DATA_TABLE,),
        ).fetchone()
    return row is not None


def guardar_datos(df: pd.DataFrame):
    """Reemplaza la tabla de datos y marca la fecha de modificación."""
    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df.insert(0, "id", range(1, len(df) + 1))
    with get_conn() as conn:
        df.to_sql(config.DATA_TABLE, conn, if_exists="replace", index=False)
    meta_set("datos_modificados", datetime.now().isoformat(timespec="seconds"))


def cargar_datos() -> pd.DataFrame:
    if not tiene_datos():
        return pd.DataFrame()
    with get_conn() as conn:
        return pd.read_sql(f"SELECT * FROM {config.DATA_TABLE}", conn)


def eliminar_registros(ids):
    df = cargar_datos()
    if df.empty:
        return
    df = df[~df["id"].isin(list(ids))].reset_index(drop=True)
    guardar_datos(df)


def agregar_registro(dic: dict):
    df = cargar_datos()
    nuevo = pd.DataFrame([dic])
    df = pd.concat([df.drop(columns=["id"], errors="ignore"), nuevo],
                   ignore_index=True)
    guardar_datos(df)


# --------------------------------------------------------------------------- #
#  Control de calidad                                                         #
# --------------------------------------------------------------------------- #
def control_calidad(df: pd.DataFrame) -> dict:
    """Devuelve un resumen de calidad: nulos, duplicados, outliers, tipos."""
    if df.empty:
        return {}
    datos = df.drop(columns=["id"], errors="ignore")
    num = datos.select_dtypes(include="number")

    # Outliers por IQR
    outliers = {}
    for col in num.columns:
        q1, q3 = num[col].quantile(0.25), num[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers[col] = int(((num[col] < lo) | (num[col] > hi)).sum())

    return {
        "n_registros": len(datos),
        "n_variables": datos.shape[1],
        "nulos_por_columna": datos.isnull().sum().to_dict(),
        "total_nulos": int(datos.isnull().sum().sum()),
        "duplicados": int(datos.duplicated().sum()),
        "outliers_por_columna": outliers,
        "tipos": datos.dtypes.astype(str).to_dict(),
    }


# --------------------------------------------------------------------------- #
#  Estado de la base respecto a los modelos                                   #
# --------------------------------------------------------------------------- #
def estado_base():
    """Sin entrenar / Desactualizado / Actualizado."""
    version_activa = meta_get("version_activa")
    if not version_activa:
        return "Sin entrenar"
    ts_datos = meta_get("datos_modificados")
    ts_entren = meta_get("ultimo_entrenamiento")
    if ts_datos and ts_entren and ts_datos > ts_entren:
        return "Desactualizado"
    return "Actualizado"


# --------------------------------------------------------------------------- #
#  Historiales                                                                #
# --------------------------------------------------------------------------- #
def log_entrenamiento(version, n_obs, modelos, mejor_modelo, metricas, tipo="p80"):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO training_history"
            "(fecha, version, n_obs, modelos, mejor_modelo, metricas, tipo) "
            "VALUES(?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), int(version),
             int(n_obs), json.dumps(modelos), mejor_modelo,
             json.dumps(metricas), tipo),
        )
    meta_set("ultimo_entrenamiento",
             datetime.now().isoformat(timespec="seconds"))


def historial_entrenamientos() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql(
            "SELECT * FROM training_history ORDER BY id DESC", conn)


def log_prediccion(version, modelo, parametros, p80, tipo="p80", curva=None):
    """Registra una predicción. Para P80, 'p80' es el valor y 'curva' None.
    Para la curva, 'p80' es el P80 leído de la curva (o None) y 'curva' es la
    lista de percentiles predichos, que se guarda como JSON."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO prediction_history"
            "(fecha, version, modelo, parametros, p80_predicho, tipo, curva) "
            "VALUES(?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"),
             int(version) if version else None,
             modelo, json.dumps(parametros),
             float(p80) if p80 is not None else None,
             tipo,
             json.dumps(curva) if curva is not None else None),
        )


def historial_predicciones(tipo=None) -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT * FROM prediction_history ORDER BY id DESC", conn)
    if tipo is not None and not df.empty and "tipo" in df.columns:
        df = df[df["tipo"] == tipo].reset_index(drop=True)
    return df
