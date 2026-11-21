"""
u4_tools.py - Funciones de la Unidad 4 (predictibilidad y error de predicción).

Contiene el código del apunte (sección "Implementación práctica en Python") y,
al final, algunas utilidades que usan los scripts de la unidad:
    rolling_mean_causal_nan   ventanas causales que toleran huecos (NaN)
    make_edges, entropy, conditional_entropy   (de la Unidad 3, para la Actividad 3)
    autocorr                  autocorrelación normalizada
    alarm_intervals           convierte una alarma booleana en intervalos

Solo requiere numpy:
    from u4_tools import predict_persistence, fit_ar, predict_ar, nll_gauss
"""
import numpy as np

def predict_persistence(x):
    """x_hat[t] = x[t-1]. Devuelve (predicciones, errores) para t = 1..T-1."""
    x = np.asarray(x, dtype=float)
    pred = x[:-1]
    return pred, x[1:] - pred

def _lag_matrix(x, p):
    """Matriz de regresores [x_{t-1}, ..., x_{t-p}, 1] para t = p..T-1."""
    cols = [x[p - k:len(x) - k] for k in range(1, p + 1)]
    return np.column_stack(cols + [np.ones(len(x) - p)])

def fit_ar(x_train, p):
    """Coeficientes [a_1, ..., a_p, c] de un AR(p) por mínimos cuadrados."""
    x_train = np.asarray(x_train, dtype=float)
    X = _lag_matrix(x_train, p)
    coef, *_ = np.linalg.lstsq(X, x_train[p:], rcond=None)
    return coef

def predict_ar(x, coef):
    """Predicción a un paso con un AR ya ajustado. Devuelve (pred, err) para t = p..T-1."""
    x = np.asarray(x, dtype=float)
    p = len(coef) - 1
    pred = _lag_matrix(x, p) @ coef
    return pred, x[p:] - pred


def error_metrics(e):
    """Sesgo, MSE, RMSE y MAE de un vector de errores."""
    e = np.asarray(e, dtype=float)
    mse = np.mean(e ** 2)
    return {'bias': np.mean(e), 'mse': mse, 'rmse': np.sqrt(mse),
            'mae': np.mean(np.abs(e))}

def skill(e_model, e_baseline):
    """1 - MSE_modelo / MSE_baseline (0: igual a la base; 1: perfecto)."""
    return 1.0 - np.mean(np.square(e_model)) / np.mean(np.square(e_baseline))


def estimate_sigma(e_ref, floor=1e-6):
    """
    sigma de los errores en el PERÍODO DE REFERENCIA (operación normal).
    El piso evita sigma = 0 si los errores de referencia son todos nulos;
    elegilo según la resolución de la señal, no como un número arbitrario.
    """
    s = np.sqrt(np.mean(np.square(e_ref)))
    return max(s, floor)

def nll_gauss(e, sigma):
    """NLL gaussiano instantáneo (nats) con sigma FIJA."""
    e = np.asarray(e, dtype=float)
    return 0.5 * np.log(2 * np.pi * sigma ** 2) + e ** 2 / (2 * sigma ** 2)


def rolling_mean_causal(x, window):
    """
    Media de x en la ventana [t-window+1, t], asignada al instante final t.
    Los primeros window-1 valores quedan en NaN (todavía no hay ventana completa).
    """
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[window - 1:] = (c[window:] - c[:-window]) / window
    return out


def threshold(ref_values, k=3.0, robust=False):
    """
    Umbral a partir de los valores del indicador en el período de referencia.
    robust=False: mu + k*sigma. robust=True: mediana + k*1.4826*MAD
    (no usar si el indicador tiene muchos empates: la MAD puede ser 0).
    """
    r = np.asarray(ref_values, dtype=float)
    r = r[~np.isnan(r)]
    if robust:
        med = np.median(r)
        return med + k * 1.4826 * np.median(np.abs(r - med))
    return np.mean(r) + k * np.std(r)

def persistent_alarm(above, n_consec=3):
    """True en t si el indicador superó el umbral en las últimas n_consec ventanas."""
    above = np.asarray(above, dtype=bool)
    run = np.zeros(len(above), dtype=int)
    for t in range(len(above)):
        run[t] = run[t - 1] + 1 if (above[t] and t > 0) else int(above[t])
    return run >= n_consec

def evaluate_detection(alarm, anomaly_start, anomaly_end, t0=0):
    """
    alarm: booleano por muestra (alineado causalmente).
    Devuelve el retardo (muestras) y las alarmas fuera del intervalo anómalo.
    """
    idx = np.flatnonzero(alarm) + t0
    inside = idx[(idx >= anomaly_start) & (idx < anomaly_end)]
    delay = (inside[0] - anomaly_start) if len(inside) else None
    false_alarms = int(np.sum(idx < anomaly_start))
    return {'delay': delay, 'false_alarms_before': false_alarms,
            'detected': len(inside) > 0}


# ---------------------------------------------------------------------------
# Utilidades adicionales (no están en el apunte)
# ---------------------------------------------------------------------------

def rolling_mean_causal_nan(x, window, min_frac=0.8):
    """
    Como rolling_mean_causal, pero ignora NaN (huecos del re-muestreo).
    La ventana [t-window+1, t] da NaN si tiene menos de min_frac*window valores válidos.
    """
    x = np.asarray(x, dtype=float)
    ok = ~np.isnan(x)
    c = np.cumsum(np.insert(np.where(ok, x, 0.0), 0, 0.0))
    n = np.cumsum(np.insert(ok.astype(float), 0, 0.0))
    out = np.full(len(x), np.nan)
    if len(x) >= window:
        s = c[window:] - c[:-window]
        k = n[window:] - n[:-window]
        with np.errstate(invalid='ignore', divide='ignore'):
            m = s / k
        m[k < min_frac * window] = np.nan
        out[window - 1:] = m
    return out


def make_edges(lo, hi, n_bins):
    """Bordes de una grilla fija de n_bins entre lo y hi (Unidad 3)."""
    return np.linspace(lo, hi, n_bins + 1)


def _entropy_counts(counts):
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    if n == 0:
        return np.nan
    p = counts[counts > 0] / n
    return float(-np.sum(p * np.log2(p))) + 0.0


def entropy(x, edges):
    """Entropía de Shannon (bits) con grilla fija; ignora NaN (Unidad 3)."""
    x = np.asarray(x, dtype=float)
    x = np.clip(x[~np.isnan(x)], edges[0], edges[-1])
    return _entropy_counts(np.histogram(x, bins=edges)[0])


def conditional_entropy(x, edges):
    """H(X_t+1 | X_t) en bits con grilla fija (Unidad 3)."""
    x = np.asarray(x, dtype=float)
    ok = ~(np.isnan(x[:-1]) | np.isnan(x[1:]))
    s = np.digitize(np.clip(np.nan_to_num(x), edges[0], edges[-1]), edges[1:-1])
    a, b = s[:-1][ok], s[1:][ok]
    nb = len(edges) - 1
    joint = np.zeros((nb, nb))
    np.add.at(joint, (a, b), 1)
    return _entropy_counts(joint.ravel()) - _entropy_counts(joint.sum(axis=1))


def autocorr(x, max_lag):
    """Autocorrelación normalizada (r_0 = 1) para retardos 0..max_lag; ignora NaN."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    x = x - x.mean()
    den = np.sum(x * x)
    if den == 0:
        return np.full(max_lag + 1, np.nan)
    return np.array([np.sum(x[:len(x) - k] * x[k:]) / den for k in range(max_lag + 1)])


def alarm_intervals(alarm):
    """Lista de (inicio, fin) de los tramos donde alarm es True (fin exclusivo)."""
    a = np.asarray(alarm, dtype=bool).astype(int)
    d = np.diff(np.r_[0, a, 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
