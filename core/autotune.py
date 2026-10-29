"""
core/autotune.py
----------------
Configuración AUTOMÁTICA del entrenamiento híbrido, para que el usuario no
tenga que elegir épocas, población ni k.

  * k (validación cruzada) = 5 y población = 20 quedan fijos (par: evita el
    fallo de GA de mealpy con poblaciones impares).
  * Las ÉPOCAS las decide cada modelo con "parada temprana" (early stopping):
    la metaheurística se detiene cuando el mejor RMSE de validación cruzada
    deja de mejorar de forma relevante durante PACIENCIA épocas seguidas
    (nunca antes de MIN_EPOCH ni después de MAX_EPOCH). Así cada modelo usa
    las épocas que su dataset necesita: ni corta la búsqueda antes de tiempo,
    ni la alarga sin sentido.

Control del sobreajuste (overfitting)
  1. La función objetivo es el RMSE en VALIDACIÓN cruzada (datos que el modelo
     no vio al entrenar en cada fold), nunca el RMSE de entrenamiento.
  2. Se penaliza la brecha entre validación y entrenamiento:
         objetivo = RMSE_val + λ · max(0, RMSE_val − RMSE_train)
     Entre dos combinaciones de hiperparámetros con validación parecida, gana
     la que generaliza mejor (menos memoriza). λ = 0.25.
  3. La parada temprana evita "sobre-optimizar" los hiperparámetros contra los
     mismos 5 folds (sobreajuste a la validación) cuando ya no hay mejora real.
  4. El conjunto de TEST no participa en nada de lo anterior; se usa solo al
     final para medir el desempeño real y diagnosticar sobreajuste.
"""
import numpy as np
from mealpy.utils.termination import Termination

K_FOLDS = 5
POP_SIZE = 20
MIN_EPOCH = 10          # nunca se busca menos que el valor manual anterior
MAX_EPOCH = 60          # tope de seguridad
MAX_SEG = 900           # tope de tiempo por modelo (15 min), útil en Streamlit Cloud
PACIENCIA = 5           # épocas seguidas sin mejora relevante -> detener
TOL_REL = 0.002         # "mejora relevante" = 0.2 % de la desviación estándar del objetivo
LAMBDA_BRECHA = 0.25    # peso de la penalización por brecha validación–entrenamiento

# Umbrales del diagnóstico final (train vs. test)
BRECHA_R2_ALERTA = 0.15


class _TerminacionConMinimo(Termination):
    """Igual que la parada temprana de mealpy, pero no permite detenerse antes
    de MIN_EPOCH épocas (evita cortar la búsqueda por una meseta inicial)."""

    def should_terminate(self, current_epoch, current_fe, current_time,
                         current_threshold):
        if current_epoch < MIN_EPOCH:
            current_threshold = 0           # la parada por tiempo sí aplica
        return super().should_terminate(current_epoch, current_fe,
                                        current_time, current_threshold)


def terminacion(y_train=None, escala=None) -> Termination:
    """Criterio de parada: MAX_EPOCH o PACIENCIA épocas sin mejorar más que
    TOL_REL·std(y). La tolerancia se escala con el dataset (cm, pulg, m...)."""
    if escala is None:
        escala = float(np.nanstd(np.asarray(y_train, dtype=float)))
    if not np.isfinite(escala) or escala <= 0:
        escala = 1.0
    t = _TerminacionConMinimo(max_epoch=MAX_EPOCH, max_early_stop=PACIENCIA,
                              max_time=MAX_SEG, log_to=None)
    t.epsilon = TOL_REL * escala
    return t


def objetivo(rmse_val, rmse_train) -> float:
    """RMSE de validación penalizado por la brecha con entrenamiento."""
    return float(rmse_val + LAMBDA_BRECHA * max(0.0, rmse_val - rmse_train))


def epocas_usadas(optimizer) -> int:
    try:
        return len(optimizer.history.list_global_best)
    except Exception:
        return 0


def diagnostico(r2_train, r2_test) -> str:
    """Etiqueta para la tabla comparativa."""
    if r2_train is None or r2_test is None or not np.isfinite([r2_train, r2_test]).all():
        return "—"
    brecha = r2_train - r2_test
    if r2_test < 0.5 and r2_train < 0.6:
        return "⚠️ Subajuste"
    if brecha > BRECHA_R2_ALERTA:
        return f"⚠️ Sobreajuste (brecha R² {brecha:.2f})"
    return "✅ Generaliza bien"
