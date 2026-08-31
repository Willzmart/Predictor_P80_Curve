"""
core/config.py
--------------
Rutas y constantes del sistema. Crea la estructura de almacenamiento del
diagrama: base de datos SQLite + carpetas de modelos versionados + auxiliares.

Importante: cuando la app corre empaquetada como .exe (PyInstaller, sys.frozen),
NO se puede guardar dentro del bundle (es temporal y de solo lectura en --onefile).
En ese caso el almacenamiento va JUNTO al ejecutable, en una carpeta persistente
que el usuario puede ver y respaldar. Si esa ubicacion no es escribible (p. ej.
el .exe esta en Archivos de programa), cae a la carpeta del usuario.
"""
import sys
from pathlib import Path


def _base_almacenamiento() -> Path:
    """Devuelve una carpeta escribible y persistente para 'storage/'."""
    if getattr(sys, "frozen", False):
        # Corriendo como .exe -> junto al ejecutable
        exe_dir = Path(sys.executable).resolve().parent
        try:
            prueba = exe_dir / ".permiso_escritura"
            prueba.touch()
            prueba.unlink()
            return exe_dir
        except Exception:
            # Sin permiso de escritura -> carpeta del usuario
            return Path.home() / "FragmentacionP80"
    # Modo desarrollo (streamlit run) -> raiz del proyecto
    return Path(__file__).resolve().parent.parent


BASE_DIR = _base_almacenamiento()
STORAGE_DIR = BASE_DIR / "storage"

DB_PATH = STORAGE_DIR / "voladuras.db"
MODELS_DIR = STORAGE_DIR / "models"          # models/version_XXX/...
DATA_IMPORTS_DIR = STORAGE_DIR / "data_imports"
REPORTS_DIR = STORAGE_DIR / "reports"
LOGS_DIR = STORAGE_DIR / "logs"

DATA_TABLE = "voladuras"

# Metrica usada para decidir "el mejor modelo" y si una version mejora
# (menor es mejor -> RMSE en test)
METRICA_RANKING = "rmse_test"
METRICA_MENOR_MEJOR = True


def ensure_dirs():
    """Crea todas las carpetas de almacenamiento si no existen."""
    for d in (STORAGE_DIR, MODELS_DIR, DATA_IMPORTS_DIR, REPORTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
