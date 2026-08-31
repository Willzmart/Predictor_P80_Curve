# Predicción y Optimización de Fragmentación (P80) — App Streamlit

Sistema completo de **gestión de datos + entrenamiento + predicción + versionado**
de modelos híbridos ML (ML + metaheurística) para predecir el **P80** a partir de
variables de voladura. Implementa la arquitectura del diagrama, con las decisiones
acordadas: **sin ANN/TensorFlow** (los 7 modelos ligeros), **SQLite** para
persistencia, y análisis avanzado (SHAP / PDF) diferido a una fase posterior.

## Secciones (fiel al diagrama)

- **🏠 Dashboard** — resumen: registros, versiones, versión activa, estado.
- **📊 Datos** — importar CSV/Excel, tabla interactiva editable, agregar/eliminar
  registros, control de calidad (nulos, duplicados, outliers IQR, tipos),
  exportar CSV/Excel. Estado de la base: *Sin entrenar / Desactualizado / Actualizado*.
- **🧠 Entrenamiento** — seleccionar varios modelos, configurar (train/test, k-fold,
  semilla 42, épocas/población), entrenar, comparar métricas y **guardar como versión**
  (solo se activa si mejora, o manualmente).
- **🎯 Predicción** — elegir modelo de la versión activa, ingreso manual o por lotes,
  P80 estimado, historial de predicciones.
- **📈 Resultados** — ranking de modelos, predicho-vs-real, residuos, importancia de
  variables (por permutación), historial de entrenamientos y predicciones, exportar.
- **⚙️ Configuración** — gestión de versiones (activar, eliminar, comparar) y ayuda.

## Dos tareas de predicción

La app predice **dos cosas** a partir de la misma base:

1. **P80** (valor único) — siempre disponible.
2. **Curva granulométrica** (percentiles P10…P100) — disponible **solo si la base
   tiene las columnas P10 a P100**. Si no las tiene, la opción se deshabilita sola.

En **Entrenamiento** y **Predicción** hay un selector arriba para elegir la tarea.
Las versiones de P80 y de curva se guardan y administran por separado.

## Modelos incluidos

**P80** (valor único): GBM+FFA · SVR+PSO · SVR+GA · XGBoost+FFA · XGBoost+PSO ·
XGBoost+GA · Random Forest+GA

**Curva granulométrica** (multi-salida, 10 percentiles): LightGBM+FFA · LightGBM+PSO ·
LightGBM+GA · XGBoost+FFA · XGBoost+PSO · XGBoost+GA

Cada uno optimiza sus hiperparámetros con su metaheurística (FFA/PSO/GA) minimizando
el RMSE en validación cruzada (en la curva, el RMSE promedio sobre los 10 percentiles),
con los mismos rangos de los notebooks de tesis. Las curvas predichas se corrigen
para ser monótonas (P10 ≤ P20 ≤ … ≤ P100).

## Estructura del proyecto

```
frag_app/
├── app.py                    # entrada: dashboard + navegación
├── core/
│   ├── config.py             # rutas de almacenamiento
│   ├── db.py                 # capa SQLite: CRUD, calidad, historiales, estado
│   ├── preprocessing.py      # pipeline de datos de la tesis
│   ├── models.py             # registro de los 7 modelos híbridos + entrenamiento
│   ├── training.py           # orquestador multi-modelo + comparación
│   └── versioning.py         # versionado: guardar/activar/comparar/eliminar
├── views/                    # una vista por sección
│   ├── datos.py
│   ├── entrenamiento.py
│   ├── prediccion.py
│   ├── resultados.py
│   └── configuracion.py
├── requirements.txt
└── README.md

# Se crea en tiempo de ejecución:
storage/
├── voladuras.db              # SQLite (datos + historiales + metadatos)
├── models/version_00X/       # un .joblib por modelo + metadata.json
├── data_imports/  reports/  logs/
```

## Instalación y ejecución

```bash
python -m venv venv
# Windows: venv\Scripts\activate   |   Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Flujo típico

1. **Datos** → importa tu Excel (activa "pipeline de tesis") y revisa calidad.
2. **Entrenamiento** → marca 2–3 modelos, entrena, compara y guarda la versión.
3. **Predicción** → ingresa una voladura o sube un lote → P80 estimado.
4. **Resultados** → revisa ranking, residuos e importancias; exporta.
5. **Configuración** → administra versiones a medida que reentrenas.

## Versionado ("guardar solo si mejora")

Al entrenar, el sistema compara el mejor modelo nuevo (menor RMSE en test) contra
la versión activa. Si mejora, lo indica y puedes guardarlo como *Versión n+1*, lo
que la activa y marca la base como **Actualizado**. Cada versión guarda sus modelos
como bundles autocontenidos (modelo + scaler + features) más un `metadata.json`.

## Pendiente para la Fase 4 (acordado)

- SHAP para importancia de variables (además de la de permutación ya incluida).
- Reportes en PDF (hoy se exporta a CSV/Excel).
- ANN+PSO (requiere TensorFlow; excluido para mantener el deploy ligero).

## Notas

- Semilla fija (42) para reproducibilidad, igual que en la tesis.
- Escala de datos por modelo (SVR sí, árboles no); el scaler viaja dentro del bundle.
- Pensado para datos a escala de tesis (cientos de filas). Para volúmenes grandes,
  conviene migrar el CRUD a operaciones SQL incrementales.
