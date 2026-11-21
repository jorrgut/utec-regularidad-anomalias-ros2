"""
u5_core.py - Núcleo de cálculo de la Unidad 5 (score, umbral y persistencia).

Este módulo lo usan TANTO los nodos ROS 2 (entropy_metrics, cmdvel_predictor,
anomaly_detector) COMO los scripts fuera de línea (metrics_from_csv.py,
calibrate_detector.py, evaluate_detector.py). Así, lo que se prueba fuera de línea
es exactamente lo que calculan los nodos.

Solo requiere numpy (y PyYAML para leer/escribir la calibración, incluido en ROS 2).

Contenido:
  Indicadores en línea (ventanas por TIEMPO sobre los mensajes recibidos):
    OnlineIndicators          acumula (t, v, w) y calcula los indicadores en t_now
  Score (apunte, sección "El score de anomalía"):
    robust_center_scale, score
  Umbral y persistencia:
    quantile_threshold, k_consecutive, k_of_n, hysteresis, cusum
  Evaluación por eventos:
    intervals, evaluate_events
  Calibración:
    save_calibration, load_calibration
"""
from collections import deque

import numpy as np

# Indicadores que produce el sistema, con su dirección y peso por defecto.
#   'up'   : solo cuenta como anómalo que el indicador SUBA
#   'both' : cuenta cualquier desvío (|z|)
INDICATORS = {
    'mse_v':   {'direction': 'up',   'weight': 0.5, 'desc': 'Rolling MSE del error de persistencia de v'},
    'mse_w':   {'direction': 'up',   'weight': 0.5, 'desc': 'Rolling MSE del error de persistencia de w'},
    'Hcond_v': {'direction': 'up',   'weight': 1.0, 'desc': 'Entropía condicional H(v_t+1 | v_t)'},
    'H_dt':    {'direction': 'both', 'weight': 1.0, 'desc': 'Entropía de los intervalos Δt'},
    'H_v':     {'direction': 'both', 'weight': 1.0, 'desc': 'Entropía de Shannon de v'},
}
# Los dos MSE pesan 0.5 cada uno: juntos equivalen a UN indicador de predictibilidad.


# ---------------------------------------------------------------------------
# Entropías (Unidad 3) con grilla FIJA
# ---------------------------------------------------------------------------

def make_edges(lo, hi, n_bins):
    return np.linspace(lo, hi, n_bins + 1)


def _entropy_counts(counts):
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    if n == 0:
        return float('nan')
    p = counts[counts > 0] / n
    return float(-np.sum(p * np.log2(p))) + 0.0


def entropy(x, edges):
    x = np.clip(np.asarray(x, dtype=float), edges[0], edges[-1])
    return _entropy_counts(np.histogram(x, bins=edges)[0])


def conditional_entropy(x, edges):
    """H(X_t+1 | X_t) en bits."""
    x = np.clip(np.asarray(x, dtype=float), edges[0], edges[-1])
    if len(x) < 2:
        return float('nan')
    s = np.digitize(x, edges[1:-1])
    nb = len(edges) - 1
    joint = np.zeros((nb, nb))
    np.add.at(joint, (s[:-1], s[1:]), 1)
    return _entropy_counts(joint.ravel()) - _entropy_counts(joint.sum(axis=1))


# ---------------------------------------------------------------------------
# Indicadores en línea
# ---------------------------------------------------------------------------

class OnlineIndicators:
    """
    Acumula los mensajes recibidos (t, v, w) y calcula, para el instante t_now,
    los indicadores sobre la ventana (t_now - window_sec, t_now].

    Es CAUSAL: solo usa mensajes recibidos hasta t_now. El error de persistencia
    de cada mensaje es su valor menos el del mensaje anterior (equivale al ZOH).
    """

    def __init__(self, window_sec=5.0, min_samples=30, v_range=(-0.22, 0.22),
                 w_range=(-2.84, 2.84), n_bins=15, dt_max=0.3, n_bins_dt=20):
        self.window_sec = window_sec
        self.min_samples = min_samples
        self.edges_v = make_edges(*v_range, n_bins)
        self.edges_dt = make_edges(0.0, dt_max, n_bins_dt)
        self.buf = deque()          # (t, v, w, e_v, e_w, dt)
        self.last = None

    def add(self, t, v, w):
        if self.last is None:
            e_v = e_w = dt = float('nan')
        else:
            t0, v0, w0 = self.last
            e_v, e_w, dt = v - v0, w - w0, t - t0
            if dt < 0:                      # el tiempo retrocedió (reset de Gazebo)
                self.buf.clear()
                e_v = e_w = dt = float('nan')
        self.buf.append((t, v, w, e_v, e_w, dt))
        self.last = (t, v, w)

    def _prune(self, t_now):
        # conservar un margen para no perder mensajes en el borde
        while self.buf and self.buf[0][0] <= t_now - 2 * self.window_sec:
            self.buf.popleft()

    def compute(self, t_now):
        """Devuelve un dict con n_msgs y los indicadores (NaN si no hay datos suficientes)."""
        self._prune(t_now)
        rows = [r for r in self.buf if t_now - self.window_sec < r[0] <= t_now]
        out = {'n_msgs': len(rows)}
        nan = float('nan')
        if len(rows) < self.min_samples:
            out.update(H_v=nan, Hcond_v=nan, H_dt=nan, mse_v=nan, mse_w=nan)
            return out
        a = np.array(rows, dtype=float)
        v, e_v, e_w, dt = a[:, 1], a[:, 3], a[:, 4], a[:, 5]
        ok = ~np.isnan(e_v)
        out['H_v'] = entropy(v, self.edges_v)
        out['Hcond_v'] = conditional_entropy(v, self.edges_v)
        dts = dt[~np.isnan(dt)]
        out['H_dt'] = entropy(dts, self.edges_dt) if len(dts) else nan
        out['mse_v'] = float(np.mean(e_v[ok] ** 2)) if ok.any() else nan
        out['mse_w'] = float(np.mean(e_w[ok] ** 2)) if ok.any() else nan
        return out


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def robust_center_scale(x_nom, floor=1e-9):
    """Centro y escala robustos: 1.4826*MAD; si es ~0, desvío estándar; si no, el piso."""
    x = np.asarray(x_nom, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return 0.0, floor, 'sin_datos'
    c = float(np.median(x))
    s = 1.4826 * float(np.median(np.abs(x - c)))
    if s > floor:
        return c, s, 'mad'
    s = float(np.std(x))
    if s > floor:
        return c, s, 'std'
    return c, floor, 'piso'


def score(indicators, calib, weights, direction, z_max=None):
    """
    indicators: dict nombre -> array (o escalar).
    calib:      dict nombre -> (centro, escala).
    z_max:      saturación de cada término (None = sin saturar).
    Devuelve (S, contribuciones). Un indicador NaN aporta 0 (no hay evidencia).
    """
    contrib = {}
    for name, x in indicators.items():
        if name not in calib or weights.get(name, 0) == 0:
            continue
        c, s = calib[name]
        z = (np.asarray(x, dtype=float) - c) / s
        g = np.maximum(z, 0.0) if direction[name] == 'up' else np.abs(z)
        if z_max is not None:
            g = np.minimum(g, z_max)
        contrib[name] = weights[name] * np.nan_to_num(g, nan=0.0)
    S = np.sum(list(contrib.values()), axis=0) if contrib else 0.0
    return S, contrib


# ---------------------------------------------------------------------------
# Umbral y persistencia
# ---------------------------------------------------------------------------

def quantile_threshold(S_nom, p=0.99):
    S = np.asarray(S_nom, dtype=float)
    return float(np.quantile(S[~np.isnan(S)], p))


def k_consecutive(above, K):
    run, out = 0, np.zeros(len(above), dtype=bool)
    for t, a in enumerate(above):
        run = run + 1 if a else 0
        out[t] = run >= K
    return out


def k_of_n(above, k, n):
    a = np.asarray(above, dtype=float)
    c = np.cumsum(np.insert(a, 0, 0.0))
    cnt = np.array([c[t + 1] - c[max(0, t + 1 - n)] for t in range(len(a))])
    return cnt >= k


def hysteresis(S, tau_on, tau_off, K=1):
    state, run, out = False, 0, np.zeros(len(S), dtype=bool)
    for t, s in enumerate(S):
        if not state:
            run = run + 1 if s >= tau_on else 0
            state = run >= K
        elif s < tau_off:
            state, run = False, 0
        out[t] = state
    return out


def cusum(S, k, h, cap=None):
    """
    CUSUM de Page. cap (por defecto 2h) acota el acumulado: sin tope, un ataque fuerte
    lo lleva a valores enormes y la alarma tarda mucho en apagarse cuando termina.
    """
    cap = 2 * h if cap is None else cap
    C, out = 0.0, np.zeros(len(S), dtype=bool)
    for t, s in enumerate(S):
        C = min(cap, max(0.0, C + (0.0 if np.isnan(s) else s) - k))
        out[t] = C >= h
    return out


class OnlinePersistence:
    """
    Versión incremental de k_consecutive y cusum, para el nodo detector.
    El CUSUM trabaja sobre S - center (center = mediana del score nominal): sin centrar,
    un score nominal positivo acumula siempre y la alarma queda encendida.
    """

    def __init__(self, rule='consecutive', K=3, cusum_k=0.5, cusum_h=5.0, center=0.0):
        self.rule, self.K, self.k, self.h, self.center = rule, K, cusum_k, cusum_h, center
        self.run, self.C = 0, 0.0

    def update(self, S, tau):
        if self.rule == 'cusum':
            self.C = min(2 * self.h, max(0.0, self.C + (S - self.center) - self.k))
            return self.C >= self.h
        self.run = self.run + 1 if S >= tau else 0
        return self.run >= self.K


# ---------------------------------------------------------------------------
# Evaluación por eventos
# ---------------------------------------------------------------------------

def intervals(mask):
    d = np.diff(np.r_[0, np.asarray(mask, dtype=int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def evaluate_events(t, alarm, events, tolerance):
    t = np.asarray(t, dtype=float)
    alarm = np.asarray(alarm, dtype=bool)
    alarm_iv = [(t[a], t[b - 1]) for a, b in intervals(alarm)]
    detected, delays = 0, []
    for (e0, e1) in events:
        hits = t[alarm & (t >= e0) & (t <= e1 + tolerance)]
        if len(hits):
            detected += 1
            delays.append(hits[0] - e0)

    def overlaps(a0, a1):
        return any(a1 >= e0 and a0 <= e1 + tolerance for e0, e1 in events)

    false_iv = [iv for iv in alarm_iv if not overlaps(*iv)]
    hours = (t[-1] - t[0]) / 3600.0 if len(t) > 1 else 0.0
    nominal = np.ones(len(t), dtype=bool)
    for e0, e1 in events:
        nominal &= ~((t >= e0) & (t <= e1 + tolerance))
    frac = float(np.mean(alarm[nominal])) if nominal.any() else float('nan')
    return {'eventos': len(events), 'detectados': detected,
            'frac_alarma_nominal': frac,
            'retardo_medio': float(np.mean(delays)) if delays else None,
            'falsas_alarmas': len(false_iv),
            'falsas_por_hora': len(false_iv) / hours if hours > 0 else None,
            'intervalos_falsos': false_iv}


# ---------------------------------------------------------------------------
# Archivo de calibración (YAML)
# ---------------------------------------------------------------------------

def save_calibration(path, calib):
    import yaml
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(calib, f, sort_keys=False, allow_unicode=True)


def load_calibration(path):
    import yaml
    with open(path, encoding='utf-8') as f:
        c = yaml.safe_load(f)
    centers = {k: (v['center'], v['scale']) for k, v in c['indicators'].items()}
    weights = {k: v['weight'] for k, v in c['indicators'].items()}
    direction = {k: v['direction'] for k, v in c['indicators'].items()}
    return c, centers, weights, direction
