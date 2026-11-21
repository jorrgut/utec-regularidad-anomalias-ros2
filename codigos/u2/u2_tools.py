"""
u2_tools.py - Funciones de la Unidad 2 (ventanas y re-muestreo).

Es el mismo código que aparece en el apunte, sección "Implementación práctica
en Python", reunido en un módulo para poder importarlo:

    from u2_tools import resample_to_grid, sliding_windows, window_stats

Solo requiere numpy.

Funciones:
    prepare_series     ordena, resuelve timestamps repetidos, detecta retrocesos
    clip_physical      marca como NaN valores físicamente imposibles
    hampel_filter      detector de outliers con mediana/MAD locales
    resample_to_grid   re-muestreo 'linear' | 'nearest' | 'previous' (ZOH), con huecos en NaN
    sliding_windows    ventanas por cantidad de muestras
    time_windows       ventanas por duración temporal
    window_stats       estadísticos por ventana (tolera NaN)
    verify_resampling  controles de sanidad del re-muestreo
"""
import numpy as np

def prepare_series(t, x):
    """
    Verifica que el tiempo no retroceda y resuelve timestamps repetidos
    (dt = 0), quedándose con el último valor recibido en cada instante.
    Si hay tiempos que retroceden, falla: eso no se arregla ordenando.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if np.any(np.diff(t) < 0):
        raise ValueError("Hay timestamps que retroceden (¿reset de la "
                         "simulación?). Revisá el experimento.")
    # np.unique devuelve el primer índice de cada valor; invertimos
    # para quedarnos con el último mensaje de cada instante repetido
    t_rev, idx_rev = np.unique(t[::-1], return_index=True)
    idx = len(t) - 1 - idx_rev
    return t[idx], x[idx]


def clip_physical(x, limit):
    """Marca como NaN los valores físicamente imposibles."""
    x = np.asarray(x, dtype=float).copy()
    x[np.abs(x) > limit] = np.nan
    return x

def hampel_filter(x, half_window=5, c=3.0):
    """
    Filtro de Hampel deslizante: marca como outlier la muestra que se
    aleja de la mediana LOCAL más de c * 1.4826 * MAD local.
    Devuelve una máscara booleana (True = outlier).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    mask = np.zeros(n, dtype=bool)
    for k in range(n):
        lo, hi = max(0, k - half_window), min(n, k + half_window + 1)
        w = x[lo:hi]
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        if mad > 0 and abs(x[k] - med) > c * 1.4826 * mad:
            mask[k] = True
        elif mad == 0 and x[k] != med:
            # ventana local casi constante: cualquier valor distinto
            # de la mayoría es sospechoso solo si es aislado
            vecinos = np.delete(w, k - lo)
            if np.all(vecinos == med):
                mask[k] = True
    return mask


def resample_to_grid(t, x, target_freq=20.0, method='linear',
                     max_gap=None):
    """
    Re-muestrea una señal irregular a una grilla uniforme.

    Args:
        t, x: timestamps (s) y valores; t no decreciente
        target_freq: frecuencia de la grilla (Hz)
        method: 'linear', 'nearest' o 'previous' (ZOH)
        max_gap: hueco máximo tolerable (s). Los puntos de la grilla
                 dentro de un hueco mayor quedan en NaN. Si es None,
                 se usa 3 * mediana(dt).

    Returns:
        t_grid, x_grid
    """
    t, x = prepare_series(t, x)
    ok = ~np.isnan(x)                 # ignorar valores marcados
    t, x = t[ok], x[ok]

    dt = 1.0 / target_freq
    m = int(np.floor((t[-1] - t[0]) / dt + 1e-9))
    t_grid = t[0] + dt * np.arange(m + 1)   # evita acumular error

    if method == 'linear':
        x_grid = np.interp(t_grid, t, x)
    elif method == 'previous':
        idx = np.searchsorted(t, t_grid, side='right') - 1
        x_grid = x[idx]
    elif method == 'nearest':
        idx = np.searchsorted(t, t_grid)
        idx = np.clip(idx, 1, len(t) - 1)
        left_closer = (t_grid - t[idx - 1]) <= (t[idx] - t_grid)
        x_grid = np.where(left_closer, x[idx - 1], x[idx])
    else:
        raise ValueError(f"Método desconocido: {method}")

    # Marcar huecos
    if max_gap is None:
        max_gap = 3.0 * np.median(np.diff(t))
    k = np.searchsorted(t, t_grid, side='right') - 1
    k = np.clip(k, 0, len(t) - 2)
    in_gap = (t[k + 1] - t[k]) > max_gap
    x_grid = x_grid.astype(float)
    x_grid[in_gap] = np.nan
    return t_grid, x_grid


def sliding_windows(data, window_size, stride):
    """
    Ventanas por muestras. Devuelve un array (n_windows, window_size)
    y los índices de inicio de cada ventana.
    """
    data = np.asarray(data)
    if len(data) < window_size:
        return np.empty((0, window_size)), np.array([], dtype=int)
    starts = np.arange(0, len(data) - window_size + 1, stride)
    windows = np.stack([data[s:s + window_size] for s in starts])
    return windows, starts

def time_windows(t, x, tw, step):
    """
    Ventanas por tiempo: [t0, t0 + tw), [t0 + step, t0 + step + tw), ...
    Devuelve una lista de (t_inicio, valores_en_la_ventana).
    """
    t, x = np.asarray(t), np.asarray(x)
    out = []
    t_start = t[0]
    while t_start + tw <= t[-1]:
        sel = (t >= t_start) & (t < t_start + tw)
        out.append((t_start, x[sel]))
        t_start += step
    return out


def window_stats(windows):
    """
    Estadísticos por ventana. windows: array (n_windows, window_size).
    Usa funciones nan-aware: ventanas con NaN (huecos) no rompen el cálculo.
    """
    med = np.nanmedian(windows, axis=1, keepdims=True)
    return {
        'mean':   np.nanmean(windows, axis=1),
        'std':    np.nanstd(windows, axis=1, ddof=1),  # igual que la fórmula
        'median': med[:, 0],
        'mad':    np.nanmedian(np.abs(windows - med), axis=1),
        'min':    np.nanmin(windows, axis=1),
        'max':    np.nanmax(windows, axis=1),
        'range':  np.nanmax(windows, axis=1) - np.nanmin(windows, axis=1),
        'power':  np.nanmean(windows**2, axis=1),
        'n_nan':  np.sum(np.isnan(windows), axis=1),
    }


def verify_resampling(t_orig, x_orig, t_grid, x_grid, max_gap=None):
    """Controles de sanidad del re-muestreo. Lanza AssertionError si algo falla."""
    t_orig, x_orig = prepare_series(t_orig, x_orig)

    # 1. Paso constante
    d = np.diff(t_grid)
    assert np.allclose(d, d[0], rtol=0, atol=1e-9), "Paso de grilla no constante"

    # 2. Cobertura: la grilla no sale del intervalo original
    assert t_grid[0] >= t_orig[0] - 1e-9 and t_grid[-1] <= t_orig[-1] + 1e-9, \
        "La grilla extrapola fuera de los datos"

    # 3. Rango: lineal, nearest y previous nunca salen del rango original
    xv = x_grid[~np.isnan(x_grid)]
    assert xv.min() >= np.nanmin(x_orig) - 1e-12, "Mínimo fuera de rango"
    assert xv.max() <= np.nanmax(x_orig) + 1e-12, "Máximo fuera de rango"

    # 4. Media temporal: comparar con la media ponderada por duración
    #    (no con la media simple de muestras irregulares),
    #    excluyendo los huecos, igual que el re-muestreo
    w = np.diff(t_orig)
    if max_gap is None:
        max_gap = 3.0 * np.median(w)
    keep = w <= max_gap
    mean_time = np.sum(x_orig[:-1][keep] * w[keep]) / np.sum(w[keep])
    diff = abs(np.nanmean(x_grid) - mean_time)
    scale = max(np.nanmax(np.abs(x_orig)), 1e-12)        # evita dividir por ~0
    assert diff / scale < 0.05, f"La media cambió {100*diff/scale:.1f}% del rango"

    print(f"Re-muestreo verificado: {len(t_grid)} puntos, "
          f"{np.sum(np.isnan(x_grid))} en huecos")
