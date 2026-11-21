"""
u6_data.py - Preparación de datos de la Unidad 6 (solo numpy y pandas).

Contenido:
  load_cmdvel        lee un CSV de /cmd_vel (t_sec, v, w) en orden de llegada
  resample_zoh       grilla regular con retención de orden cero ("previous", causal),
                     huecos en NaN y conteo de mensajes por celda (canal n_k)
  build_features     arma la matriz de entrada (v, w) o (v, w, n_k)
  make_windows       ventanas deslizantes de T muestras, sin huecos
  split_blocks       partición por TIEMPO (las ventanas se arman dentro de cada bloque)
  Standardizer       media y desvío por canal, calculados solo con entrenamiento
  entropy_windows    H(v) en ventanas de la grilla, con grilla de bins FIJA (Unidad 3)
  read_events        archivo de eventos: una línea "inicio fin" por evento
  save_json/load_json

Apunte: secciones "Qué ve un modelo" y "Protocolo de entrenamiento y validación".
"""
import json
import sys

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

CHANNELS = ('v', 'w')
CHANNELS_COUNT = ('v', 'w', 'n_k')


def load_cmdvel(path):
    """Devuelve (t, v, w) con t relativo al primer mensaje, en orden de llegada."""
    df = pd.read_csv(path)
    for c in ('t_sec', 'v', 'w'):
        if c not in df.columns:
            sys.exit(f'[error] {path}: falta la columna {c}')
    df = df[['t_sec', 'v', 'w']].dropna().sort_values('t_sec', kind='stable')
    t = df['t_sec'].to_numpy(dtype=float)
    if len(t) > 2 and np.ptp(np.diff(t)) < 1e-9:
        print(f'[aviso] {path} parece re-muestreado (Δt constante). La Unidad 6 re-muestrea '
              'por su cuenta y cuenta mensajes por celda: usá el CSV original del logger.')
    return t - t[0], df['v'].to_numpy(dtype=float), df['w'].to_numpy(dtype=float)


def resample_zoh(t, v, w, rate=10.0, max_gap=0.5):
    """
    Re-muestreo causal a una grilla de período 1/rate.
      - valor en g_j: el del ÚLTIMO mensaje con t <= g_j (retención de orden cero);
      - n_k en g_j: cantidad de mensajes recibidos en (g_{j-1}, g_j];
      - si el último mensaje es más viejo que max_gap, la celda es un hueco (NaN).
    Los bordes de la grilla se corren medio período respecto de la fase típica de llegada:
    si un borde coincidiera con los instantes de llegada, el jitter haría que algunas celdas
    tengan 0 mensajes y la siguiente 2, y n_k sería ruidoso aun en operación nominal.
    Devuelve un DataFrame con columnas t, v, w, n_k.
    """
    period = 1.0 / rate
    ang = 2 * np.pi * (t / period)
    phase = (np.angle(np.mean(np.exp(1j * ang))) / (2 * np.pi)) % 1.0 * period
    start = t[0] + ((phase + period / 2 - t[0]) % period)
    if start <= t[0]:
        start += period
    g = np.arange(start, t[-1] + 1e-12, period)
    idx = np.searchsorted(t, g, side='right') - 1           # último mensaje <= g
    vv, ww = v[idx].copy(), w[idx].copy()
    age = g - t[idx]
    gap = age > max_gap
    vv[gap], ww[gap] = np.nan, np.nan
    prev_idx = np.r_[np.searchsorted(t, g[0] - period, side='right') - 1, idx[:-1]]
    n_k = (idx - prev_idx).astype(float)
    return pd.DataFrame({'t': g, 'v': vv, 'w': ww, 'n_k': n_k})


def build_features(grid, counts=False):
    cols = CHANNELS_COUNT if counts else CHANNELS
    return grid[list(cols)].to_numpy(dtype=float)


def make_windows(X, T):
    """
    X: (N, d) sobre la grilla. Devuelve (W, t_end):
      W     (M, T, d) ventanas sin NaN
      t_end (M,) índice en la grilla del ÚLTIMO instante de cada ventana
    (el error de una ventana se asigna a ese instante: el indicador es causal).
    """
    if len(X) < T:
        return np.empty((0, T, X.shape[1])), np.empty(0, dtype=int)
    W = sliding_window_view(X, (T, X.shape[1]))[:, 0]
    ok = ~np.isnan(W).any(axis=(1, 2))
    t_end = np.arange(T - 1, len(X))
    return np.ascontiguousarray(W[ok]), t_end[ok]


def split_blocks(X, frac_train=0.8):
    """Partición por tiempo: primer tramo para entrenar, el último para validar."""
    n = int(len(X) * frac_train)
    return X[:n], X[n:]


class Standardizer:
    """(x - media) / desvío por canal, con estadísticas SOLO de entrenamiento."""

    def __init__(self, mean=None, std=None, floor=1e-3):
        self.mean = None if mean is None else np.asarray(mean, dtype=float)
        self.std = None if std is None else np.asarray(std, dtype=float)
        self.floor = floor

    def fit(self, X, floors=None):
        """
        floors: piso del desvío por canal. Un canal casi constante en nominal (n_k = 1 casi
        siempre) tendría desvío ~0: sin piso, un mensaje de más daría valores enormes.
        Por defecto, 0.5 mensajes para n_k (si X tiene 3 canales) y self.floor para el resto.
        """
        self.mean = np.nanmean(X, axis=0)
        if floors is None:
            floors = [self.floor] * X.shape[1]
            if X.shape[1] == len(CHANNELS_COUNT):
                floors[-1] = 0.5
        self.std = np.maximum(np.nanstd(X, axis=0), np.asarray(floors, dtype=float))
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def to_dict(self):
        return {'mean': self.mean.tolist(), 'std': self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d['mean'], d['std'])


def entropy_windows(v, edges, win):
    """H(v) en bits sobre ventanas de 'win' muestras de la grilla (NaN si hay huecos)."""
    out = np.full(len(v), np.nan)
    for i in range(win - 1, len(v)):
        seg = v[i - win + 1:i + 1]
        if np.isnan(seg).any():
            continue
        c = np.histogram(np.clip(seg, edges[0], edges[-1]), bins=edges)[0]
        p = c[c > 0] / c.sum()
        out[i] = float(-(p * np.log2(p)).sum()) + 0.0
    return out


def read_events(path):
    ev = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if line:
                a, b = line.split()[:2]
                ev.append((float(a), float(b)))
    return ev


def save_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def style_axes(ax):
    ax.grid(True, color='#e5e5e5', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#999999')
    ax.tick_params(colors='#444444', labelsize=9)


# Una tinta fija por modo (el color sigue al modo, no a su posición)
MODE_COLORS = {'structured': '#2a6fb0', 'shuffle_blocks': '#e07b39', 'iid': '#7a5195'}
