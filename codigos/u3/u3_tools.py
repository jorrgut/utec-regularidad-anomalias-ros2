"""
u3_tools.py - Funciones de la Unidad 3 (entropía de Shannon).

Es el mismo código del apunte, sección "Implementación práctica en Python",
reunido en un módulo, más dos utilidades (binary_entropy, spearman):

    from u3_tools import make_edges, entropy, entropy_sliding_windows

Solo requiere numpy.

Funciones:
    make_edges               grilla fija de bins
    entropy_from_counts      entropía a partir de conteos (opcional Miller-Madow)
    entropy                  entropía de una señal con grilla FIJA (ignora NaN)
    entropy_sliding_windows  entropía por ventana, misma grilla en todas
    conditional_entropy      H(X_t+1 | X_t)
    autocorr                 autocorrelación normalizada
    binary_entropy           h2(p)
    spearman                 correlación de rangos
"""
import numpy as np

def make_edges(lo, hi, n_bins):
    """Bordes de una grilla fija de n_bins entre lo y hi (incluidos)."""
    return np.linspace(lo, hi, n_bins + 1)

def entropy_from_counts(counts, miller_madow=False):
    """Entropía (bits) a partir de conteos por bin."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    if n == 0:
        return np.nan
    nz = counts[counts > 0]          # 0 * log2(0) = 0: se omiten bins vacíos
    p = nz / n
    H = -np.sum(p * np.log2(p))
    if miller_madow:
        H += (len(nz) - 1) / (2 * n * np.log(2))
    return float(H) + 0.0           # evita mostrar -0.0

def entropy(data, edges, miller_madow=False):
    """
    Entropía de Shannon (bits) de una señal, con una grilla de bins FIJA.
    Ignora NaN. Los valores fuera de [edges[0], edges[-1]] se asignan
    al bin extremo más cercano (no se descartan en silencio).
    """
    x = np.asarray(data, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    x = np.clip(x, edges[0], edges[-1])
    counts, _ = np.histogram(x, bins=edges)
    return entropy_from_counts(counts, miller_madow)


def entropy_sliding_windows(signal, window_size, stride, edges,
                            max_nan_frac=0.0, miller_madow=False):
    """
    Entropía por ventana, con la MISMA grilla de bins en todas.

    Returns:
        H:      array con la entropía de cada ventana (NaN si se descarta)
        starts: índice de inicio de cada ventana
    Las ventanas con más de max_nan_frac de valores NaN (huecos del
    re-muestreo de la Unidad 2) se descartan: con menos muestras, el
    sesgo del estimador cambia y la comparación deja de ser justa.
    """
    x = np.asarray(signal, dtype=float)
    starts = np.arange(0, len(x) - window_size + 1, stride)
    H = np.full(len(starts), np.nan)
    for j, s in enumerate(starts):
        w = x[s:s + window_size]
        if np.mean(np.isnan(w)) > max_nan_frac:
            continue
        H[j] = entropy(w, edges, miller_madow)
    return H, starts


def conditional_entropy(data, edges):
    """
    H(X_{t+1} | X_t) en bits, discretizando con la grilla 'edges'.
    Pares con algún NaN se ignoran.
    """
    x = np.asarray(data, dtype=float)
    x = np.clip(x, edges[0], edges[-1])
    s = np.digitize(x, edges[1:-1])          # índice de bin 0..n_bins-1
    a, b = s[:-1], s[1:]
    ok = ~(np.isnan(x[:-1]) | np.isnan(x[1:]))
    a, b = a[ok], b[ok]
    n_bins = len(edges) - 1
    joint = np.zeros((n_bins, n_bins))
    np.add.at(joint, (a, b), 1)              # conteos de pares
    H_joint = entropy_from_counts(joint.ravel())
    H_prev = entropy_from_counts(joint.sum(axis=1))
    return H_joint - H_prev


def autocorr(x, max_lag):
    """Autocorrelación normalizada (r_0 = 1) para retardos 0..max_lag."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()                 # sin restar la media, domina el nivel medio
    den = np.sum(x * x)
    return np.array([np.sum(x[:len(x) - k] * x[k:]) / den
                     for k in range(max_lag + 1)])


def binary_entropy(p):
    """h2(p) en bits, vectorizada (h2(0) = h2(1) = 0)."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return np.nan_to_num(h) + 0.0


def spearman(a, b):
    """Correlación de rangos de Spearman (sin scipy). Ignora pares con NaN."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3:
        return np.nan
    ra = _rank(a[ok])
    rb = _rank(b[ok])
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _rank(x):
    """Rangos promedio (empates comparten el rango medio)."""
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks
