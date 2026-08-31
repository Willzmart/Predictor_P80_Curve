"""
core/versioning.py
------------------
Sistema de versionado de modelos, consciente del TIPO de tarea:

    storage/models/                 <- versiones de P80 (kind="p80")
        version_001/ ... metadata.json
    storage/models/curva/           <- versiones de la curva (kind="curva")
        version_001/ ... metadata.json

Cada .joblib es un "bundle" autocontenido. metadata.json guarda la comparación
y el mejor modelo. P80 conserva su ruta y clave originales por compatibilidad.
"""
import json
import shutil
from datetime import datetime

import joblib

from . import config, db

config.ensure_dirs()


def _dir_base(kind: str):
    return config.MODELS_DIR if kind == "p80" else config.MODELS_DIR / kind


def _meta_key(kind: str):
    return "version_activa" if kind == "p80" else f"version_activa_{kind}"


def _dir_version(version: int, kind: str):
    return _dir_base(kind) / f"version_{version:03d}"


def siguiente_version(kind: str = "p80") -> int:
    base = _dir_base(kind)
    base.mkdir(parents=True, exist_ok=True)
    existentes = [
        int(p.name.split("_")[1])
        for p in base.glob("version_*")
        if p.is_dir() and p.name.split("_")[1].isdigit()
    ]
    return (max(existentes) + 1) if existentes else 1


def _metrica(bundle):
    return bundle["metrics"].get(config.METRICA_RANKING, float("inf"))


def mejor_de(bundles: dict) -> str:
    """Devuelve la key del mejor modelo según la métrica de ranking."""
    items = list(bundles.items())
    items.sort(key=lambda kv: _metrica(kv[1]),
               reverse=not config.METRICA_MENOR_MEJOR)
    return items[0][0]


def guardar_version(bundles: dict, n_obs: int, target, kind: str = "p80") -> int:
    """Guarda un conjunto de modelos como una nueva versión y la activa.

    kind: "p80" (default) o "curva". target puede ser un nombre o una lista
    (percentiles) en el caso de la curva."""
    version = siguiente_version(kind)
    vdir = _dir_version(version, kind)
    vdir.mkdir(parents=True, exist_ok=True)

    for key, bundle in bundles.items():
        joblib.dump(bundle, vdir / f"{key}.joblib")

    best = mejor_de(bundles)
    metadata = {
        "version": version,
        "kind": kind,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "target": target,
        "n_obs": n_obs,
        "modelos": list(bundles.keys()),
        "labels": {k: b["label"] for k, b in bundles.items()},
        "mejor_modelo": best,
        "metricas": {k: b["metrics"] for k, b in bundles.items()},
        "features": bundles[best]["features"],
        "metrica_ranking": config.METRICA_RANKING,
    }
    if kind == "curva":
        metadata["percentiles"] = bundles[best].get("percentiles", [])
    with open(vdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    db.meta_set(_meta_key(kind), version)
    db.meta_set("ultimo_entrenamiento",
                datetime.now().isoformat(timespec="seconds"))
    db.log_entrenamiento(version, n_obs, list(bundles.keys()), best,
                         {k: v["metrics"] for k, v in bundles.items()},
                         tipo=kind)
    return version


def listar_versiones(kind: str = "p80") -> list:
    base = _dir_base(kind)
    metas = []
    if base.exists():
        for vdir in sorted(base.glob("version_*")):
            meta_path = vdir / "metadata.json"
            if meta_path.exists():
                with open(meta_path, encoding="utf-8") as f:
                    metas.append(json.load(f))
    return sorted(metas, key=lambda m: m["version"])


def cargar_metadata(version: int, kind: str = "p80"):
    meta_path = _dir_version(version, kind) / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def cargar_bundle(version: int, modelo_key: str, kind: str = "p80"):
    ruta = _dir_version(version, kind) / f"{modelo_key}.joblib"
    if not ruta.exists():
        return None
    return joblib.load(ruta)


def version_activa(kind: str = "p80"):
    v = db.meta_get(_meta_key(kind))
    return int(v) if v else None


def activar_version(version: int, kind: str = "p80"):
    db.meta_set(_meta_key(kind), version)


def eliminar_version(version: int, kind: str = "p80"):
    vdir = _dir_version(version, kind)
    if vdir.exists():
        shutil.rmtree(vdir)
    if version_activa(kind) == version:
        restantes = [m["version"] for m in listar_versiones(kind)]
        db.meta_set(_meta_key(kind), max(restantes) if restantes else "")


def mejor_metrica_version(version: int, kind: str = "p80"):
    meta = cargar_metadata(version, kind)
    if not meta:
        return None
    best = meta["mejor_modelo"]
    return meta["metricas"][best].get(config.METRICA_RANKING)


def mejora_respecto_activa(bundles: dict, kind: str = "p80") -> bool:
    va = version_activa(kind)
    if va is None:
        return True
    ref = mejor_metrica_version(va, kind)
    if ref is None:
        return True
    nuevo = _metrica(bundles[mejor_de(bundles)])
    return (nuevo < ref) if config.METRICA_MENOR_MEJOR else (nuevo > ref)
