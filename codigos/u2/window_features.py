#!/usr/bin/env python3
"""
Descriptores por ventana deslizante (Unidad 2).

Modos:
    samples : ventanas por cantidad de muestras (--n, --stride)
    time    : ventanas por duración temporal   (--tw, --step)

Entrada: CSV con t_sec y las señales a analizar (por defecto v y w). Sirve tanto
para el CSV original de la Unidad 1 como para el re-muestreado (que puede tener
NaN en los huecos).

Salida (--out): una fila por ventana con
    i, t_start, t_end, T_span, n_samples, n_nan,
    y para cada señal: mean, std, median, mad, min, max, range, power

Además imprime un resumen (CV de T_span o de n_samples, ventanas anómalas) y,
con --plot, guarda una figura.

Uso:
    python3 window_features.py --csv data/cmdvel_u1.csv --mode samples \
        --n 20 --stride 10 --out data/features_samples.csv
    python3 window_features.py --csv data/cmdvel_u1.csv --mode time \
        --tw 2.0 --step 1.0 --out data/features_time.csv --plot
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from u2_tools import sliding_windows, time_windows, window_stats

# Ventanas totalmente en un hueco (todo NaN) generan avisos de numpy; el resultado
# (NaN) es el esperado, así que los silenciamos.
warnings.filterwarnings('ignore', category=RuntimeWarning)

C_DATA = '#2a6fb0'
C_FLAG = '#c0392b'
STAT_NAMES = ['mean', 'std', 'median', 'mad', 'min', 'max', 'range', 'power']


def style_axes(ax):
    ax.grid(True, color='#e5e5e5', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#999999')
    ax.tick_params(colors='#444444', labelsize=9)


def load(path, signals):
    df = pd.read_csv(path)
    missing = [c for c in ['t_sec', *signals] if c not in df.columns]
    if missing:
        sys.exit(f'[error] faltan columnas en el CSV: {missing}')
    df = df.dropna(subset=['t_sec']).reset_index(drop=True)
    t = df['t_sec'].to_numpy(dtype=float)
    if np.any(np.diff(t) < 0):
        sys.exit('[error] hay timestamps que retroceden; revisá el experimento')
    return df


def stats_or_nan(values):
    """window_stats para una sola ventana de largo variable; NaN si está vacía."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return {k: np.nan for k in STAT_NAMES}
    if np.sum(~np.isnan(values)) < 2:
        s = {k: np.nan for k in STAT_NAMES}
        v = values[~np.isnan(values)][0]
        s.update(mean=v, median=v, min=v, max=v, range=0.0, mad=0.0, power=v * v)
        return s
    st = window_stats(values[np.newaxis, :])
    return {k: float(st[k][0]) for k in STAT_NAMES}


def features_samples(df, signals, n, stride):
    t = df['t_sec'].to_numpy(dtype=float)
    _, starts = sliding_windows(t, n, stride)
    if len(starts) == 0:
        sys.exit(f'[error] la serie tiene {len(t)} muestras, menos que n = {n}')
    rows = []
    stats = {}
    for sig in signals:
        W, _ = sliding_windows(df[sig].to_numpy(dtype=float), n, stride)
        stats[sig] = window_stats(W)
    for j, s in enumerate(starts):
        row = {
            'i': j,
            't_start': t[s],
            't_end': t[s + n - 1],
            'T_span': t[s + n - 1] - t[s],
            'n_samples': n,
            'n_nan': int(sum(stats[sig]['n_nan'][j] for sig in signals)),
        }
        for sig in signals:
            for k in STAT_NAMES:
                row[f'{sig}_{k}'] = float(stats[sig][k][j])
        rows.append(row)
    return pd.DataFrame(rows)


def features_time(df, signals, tw, step):
    t = df['t_sec'].to_numpy(dtype=float)
    per_sig = {sig: time_windows(t, df[sig].to_numpy(dtype=float), tw, step) for sig in signals}
    t_wins = time_windows(t, t, tw, step)
    if len(t_wins) == 0:
        sys.exit(f'[error] la serie dura menos que una ventana de {tw} s')
    rows = []
    for j, (t0, tt) in enumerate(t_wins):
        row = {
            'i': j,
            't_start': t0,
            't_end': t0 + tw,
            'T_span': (tt[-1] - tt[0]) if len(tt) > 1 else 0.0,
            'n_samples': len(tt),
            'n_nan': int(sum(np.sum(np.isnan(per_sig[sig][j][1])) for sig in signals)),
        }
        for sig in signals:
            s = stats_or_nan(per_sig[sig][j][1])
            for k in STAT_NAMES:
                row[f'{sig}_{k}'] = s[k]
        rows.append(row)
    return pd.DataFrame(rows)


def cv(x):
    x = np.asarray(x, dtype=float)
    m = np.mean(x)
    return float(np.std(x, ddof=1) / m) if len(x) > 1 and m != 0 else float('nan')


def summarize(feat, mode, args):
    L = [f'Entrada: {args.csv}', f'Modo: {mode}']
    if mode == 'samples':
        L.append(f'Ventanas de n = {args.n} muestras, stride = {args.stride} '
                 f'(solapamiento {100 * max(0, 1 - args.stride / args.n):.0f}%)')
    else:
        L.append(f'Ventanas de T_w = {args.tw:g} s, paso = {args.step:g} s '
                 f'(solapamiento {100 * max(0, 1 - args.step / args.tw):.0f}%)')
    L.append(f'Cantidad de ventanas: {len(feat)}')
    L.append('')

    flagged = pd.Series(False, index=feat.index)
    if mode == 'samples':
        ts = feat['T_span']
        med = ts.median()
        L.append(f'T_span: mediana {med:.4f} s | min {ts.min():.4f} | max {ts.max():.4f} | '
                 f'CV {100 * cv(ts):.3f}%')
        flagged = ts > 1.05 * med
        L.append(f'Ventanas con T_span > 1.05 × mediana: {int(flagged.sum())}'
                 '  (posibles pérdidas o pausas)')
        c = 100 * cv(ts)
        if c < 1:
            L.append('  -> CV < 1%: duraciones comparables.')
        elif c < 5:
            L.append('  -> CV entre 1% y 5%: hay eventos; identificá qué ventanas los contienen.')
        else:
            L.append('  -> CV > 5%: re-muestreá o usá ventanas por tiempo.')
    else:
        ns = feat['n_samples']
        med = ns.median()
        L.append(f'Muestras por ventana: mediana {med:.0f} | min {ns.min()} | max {ns.max()} | '
                 f'CV {100 * cv(ns):.2f}%')
        flagged = ns < 0.8 * med
        L.append(f'Ventanas con < 80% de la mediana de muestras: {int(flagged.sum())}'
                 '  (posibles pérdidas o pausas)')
        n_empty = int((ns == 0).sum())
        if n_empty:
            L.append(f'Ventanas vacías: {n_empty}')

    n_with_nan = int((feat['n_nan'] > 0).sum())
    if n_with_nan:
        L.append(f'Ventanas que contienen NaN (huecos del re-muestreo): {n_with_nan}')

    if flagged.any():
        L.append('')
        L.append('Primeras ventanas marcadas (t relativo al inicio del registro):')
        t0 = feat['t_start'].iloc[0]
        for _, r in feat[flagged].head(10).iterrows():
            L.append(f'  i={int(r["i"]):>4}  t={r["t_start"] - t0:8.2f} s  '
                     f'T_span={r["T_span"]:.3f} s  n={int(r["n_samples"])}')
    return '\n'.join(L) + '\n', flagged


def plot_features(feat, mode, signals, flagged, path):
    t0 = feat['t_start'].iloc[0]
    tc = (feat['t_start'] + feat['t_end']) / 2 - t0
    sig = signals[0]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)

    if mode == 'samples':
        axes[0].plot(tc, feat['T_span'], color=C_DATA, linewidth=1, marker='.', markersize=3)
        axes[0].set_ylabel('T_span (s)')
        axes[0].set_title('Duración de cada ventana', loc='left', fontsize=10)
    else:
        axes[0].plot(tc, feat['n_samples'], color=C_DATA, linewidth=1, marker='.', markersize=3,
                     drawstyle='steps-mid')
        axes[0].set_ylabel('muestras')
        axes[0].set_title('Muestras por ventana', loc='left', fontsize=10)
    yv = feat['T_span'] if mode == 'samples' else feat['n_samples']
    if flagged.any():
        axes[0].scatter(tc[flagged], yv[flagged], s=30, color=C_FLAG, zorder=3,
                        edgecolor='white', linewidth=0.8, label='ventana marcada')
        axes[0].legend(fontsize=8, frameon=False, loc='upper right')

    axes[1].plot(tc, feat[f'{sig}_mean'], color=C_DATA, linewidth=1)
    axes[1].set_ylabel(f'media de {sig}')
    axes[1].set_title(f'Media por ventana ({sig})', loc='left', fontsize=10)

    axes[2].plot(tc, feat[f'{sig}_std'], color=C_DATA, linewidth=1)
    axes[2].set_ylabel(f'std de {sig}')
    axes[2].set_title(f'Desviación estándar por ventana ({sig})', loc='left', fontsize=10)
    axes[2].set_xlabel('centro de la ventana (s desde el inicio)')

    for ax in axes:
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='Descriptores por ventana deslizante')
    p.add_argument('--csv', required=True, help='CSV con t_sec y las señales')
    p.add_argument('--mode', choices=['samples', 'time'], required=True)
    p.add_argument('--n', type=int, help='[samples] tamaño de ventana (muestras)')
    p.add_argument('--stride', type=int, help='[samples] paso (muestras); por defecto n/2')
    p.add_argument('--tw', type=float, help='[time] ancho de ventana (s)')
    p.add_argument('--step', type=float, help='[time] paso (s); por defecto tw/2')
    p.add_argument('--signals', nargs='+', default=['v', 'w'], help='columnas a analizar')
    p.add_argument('--out', required=True, help='CSV de salida')
    p.add_argument('--plot', action='store_true',
                   help='guardar una figura junto al CSV (mismo nombre, .png)')
    args = p.parse_args()

    if args.mode == 'samples':
        if not args.n or args.n < 2:
            p.error('--mode samples requiere --n >= 2')
        args.stride = args.stride or max(1, args.n // 2)
    else:
        if not args.tw or args.tw <= 0:
            p.error('--mode time requiere --tw > 0')
        args.step = args.step or args.tw / 2

    df = load(args.csv, args.signals)
    if args.mode == 'samples':
        feat = features_samples(df, args.signals, args.n, args.stride)
    else:
        feat = features_time(df, args.signals, args.tw, args.step)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    feat.to_csv(args.out, index=False, float_format='%.6f')

    text, flagged = summarize(feat, args.mode, args)
    print(text)
    print(f'Features guardadas en: {args.out}')
    if args.plot:
        png = os.path.splitext(args.out)[0] + '.png'
        plot_features(feat, args.mode, args.signals, flagged, png)
        print(f'Figura guardada en: {png}')


if __name__ == '__main__':
    main()
