"""
preprocessing.py
----------------
Replica el preprocesamiento del notebook de tesis (P80 / % Finos) de forma
defensiva: solo aplica cada paso si las columnas necesarias existen, para que
también funcione con archivos ligeramente distintos.
 
Dos modos de uso desde la app:
  1) modo "tesis"  -> preprocess_thesis(df): toma el Excel/CSV crudo con las
     columnas originales (Malla, Material, Litologia, ...) y devuelve la tabla
     numérica lista para modelar.
  2) modo "datos limpios" -> el usuario sube una tabla ya numérica y solo elige
     target y features en la interfaz (no se llama a preprocess_thesis).
"""
 
import re
 
import pandas as pd
import numpy as np
 
# Objetivos del proyecto
TARGET_P80 = "P80"
TARGET_FINOS = "Porcentaje Finos"
 
# Percentiles de la curva granulométrica (target multi-salida)
PERCENTILES = ["P10", "P20", "P30", "P40", "P50", "P60", "P70", "P80", "P90", "P100"]
 
 
def _norm(s) -> str:
    """Normaliza un nombre de columna: minúsculas, sin espacios."""
    return str(s).strip().lower().replace(" ", "")
 
 
def mapa_percentiles(df: pd.DataFrame) -> dict:
    """Detecta las columnas de percentiles P10..P100 aunque tengan sufijos o
    prefijos ('P10_cm', 'P10 (cm)', 'p10', ...). Devuelve un dict que asocia
    cada percentil canónico con el nombre real de la columna en la base:
    {'P10': 'P10_cm', ...}. Solo incluye los que encuentra."""
    mapa = {}
    for p in PERCENTILES:                 # 'P10'..'P100'
        objetivo = p.lower()               # 'p10'
        for c in df.columns:
            m = re.match(r"^(p\d{1,3})", _norm(c))
            if m and m.group(1) == objetivo:
                mapa[p] = c
                break
    return mapa
 
 
def tiene_columnas_curva(df: pd.DataFrame) -> bool:
    """True si la base tiene los diez percentiles necesarios para la curva,
    reconociendo nombres con sufijos como '_cm' o '(cm)'."""
    return len(mapa_percentiles(df)) == len(PERCENTILES)
 
 
def columnas_curva_presentes(df: pd.DataFrame) -> list:
    """Percentiles canónicos presentes en la base (con detección tolerante)."""
    return list(mapa_percentiles(df).keys())
 
# Litologías base para el multi-label encoding
LITO_BASES = ["CZ", "DIO", "MZ", "SKARN", "DIQUE", "DOP"]
 
# Columnas que el notebook elimina (solo se borran las que existan)
COLS_A_ELIMINAR = [
    "Fecha", "Banco", "Proyecto", "Malla", "Diametro de perforación",
    "P10", "P20", "P30", "P40", "P50", "P60", "P70", "P80.", "P90", "P100",
    "MODELO", "Espaciamiento", "Top Size", "TIPO MALLA",
]
 
MAP_MATERIAL = {"MINERAL": 1, "DESMONTE": 0}
MAP_TIPO_MALLA = {"MIXTO": 1, "NORMAL": 0}
 
 
def cargar_datos(archivo, nombre: str) -> pd.DataFrame:
    """Lee un archivo subido (Excel o CSV) a DataFrame."""
    nombre = nombre.lower()
    if nombre.endswith((".xlsx", ".xls")):
        return pd.read_excel(archivo)
    if nombre.endswith(".csv"):
        # intenta separadores comunes
        try:
            return pd.read_csv(archivo)
        except Exception:
            archivo.seek(0)
            return pd.read_csv(archivo, sep=";")
    raise ValueError("Formato no soportado. Usa .xlsx, .xls o .csv")
 
 
def preprocess_thesis(df: pd.DataFrame, conservar_curva: bool = False) -> pd.DataFrame:
    """Aplica el pipeline de preprocesamiento del notebook, paso a paso y
    de forma tolerante a columnas faltantes.
 
    conservar_curva: si True, NO elimina P10–P100 (se necesitan como target
    de la curva granulométrica). Útil para el modo de curva."""
    df = df.copy()
 
    # 1) Burden / Espaciamiento desde 'Malla' (formato "B x E")
    if "Malla" in df.columns:
        partes = df["Malla"].astype(str).str.split(r"\s*x\s*", expand=True)
        if partes.shape[1] >= 2:
            df["Burden"] = pd.to_numeric(
                partes[0].str.replace(",", ".", regex=False), errors="coerce"
            )
            df["Espaciamiento"] = pd.to_numeric(
                partes[1].str.replace(",", ".", regex=False), errors="coerce"
            )
 
    # 2) Limpieza de texto
    if "Litologia" in df.columns:
        df["Litologia"] = df["Litologia"].astype(str).str.replace(" ", "", regex=False)
    if "TIPO MALLA" in df.columns:
        df["TIPO MALLA"] = df["TIPO MALLA"].astype(str).str.strip()
 
    # 3) Encoding categórico
    if "Material" in df.columns:
        df["Material"] = df["Material"].map(MAP_MATERIAL)
    if "TIPO MALLA" in df.columns:
        df["TIPO MALLA"] = df["TIPO MALLA"].map(MAP_TIPO_MALLA)
 
    # 4) Eliminar columnas no usadas (solo las presentes)
    cols_elim = list(COLS_A_ELIMINAR)
    if conservar_curva:
        # Para la curva se conservan los percentiles P10–P100 (son el target)
        cols_elim = [c for c in cols_elim if c not in PERCENTILES]
    a_eliminar = [c for c in cols_elim if c in df.columns]
    df = df.drop(columns=a_eliminar)
 
    # 5) Multi-label encoding de Litología
    if "Litologia" in df.columns:
        df["Litologia"] = df["Litologia"].astype(str).str.upper().str.strip()
        for lito in LITO_BASES:
            df[f"Lito_{lito}"] = df["Litologia"].apply(
                lambda x: 1 if lito in str(x) else 0
            )
        df = df.drop(columns=["Litologia"])
 
    return df
 
 
def separar_x_y(df: pd.DataFrame, target: str, excluir=None):
    """Devuelve X, y y la lista de columnas de features.
 
    excluir: columnas adicionales a dejar fuera de X (p. ej. el otro target).
    """
    excluir = set(excluir or [])
    excluir.add(target)
    features = [c for c in df.columns if c not in excluir]
    X = df[features].copy()
    y = df[target].copy()
    return X, y, features
 
 
def resumen_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla resumen de nulos por columna (para mostrar en la app)."""
    nulos = df.isnull().sum()
    return (
        pd.DataFrame({"columna": nulos.index, "nulos": nulos.values})
        .sort_values("nulos", ascending=False)
        .reset_index(drop=True)
    )
 
