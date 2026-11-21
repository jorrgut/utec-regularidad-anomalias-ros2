#!/usr/bin/env python3
"""
Entropía de Shannon por ventana deslizante (Unidad 3, Actividad 1).

Entrada: CSV re-muestreado de la Unidad 2 (t_sec, v, w; puede tener NaN en huecos).

Para cada ventana y cada señal calcula, con una grilla de bins FIJA:
    H       entropía de Shannon (bits)
    Hnorm   H / log2(n_bins)
    Hcond   entropía condicional H(X_t+1 | X_t)
    mean, std, mad   (para comparar con la Unidad 2)

Salidas:
    --out                              CSV con una fila por ventana
    <outdir>/entropy_series.png        señal + serie de entropía de cada señal
    <outdir>/entropy_vs_std.png        dispersión H vs. desviación estándar
    <outdir>/entropy_compare.png       (con --compare) H con dos re-muestreos

Uso:
    python3 entropy_windows.py --csv data/cmdvel_20hz.csv --out outputs_u3a1/entropy_windows.csv
    python3 entropy_windows.py --csv data/cmdvel_20hz.csv --compare data/cmdvel_20hz_zoh.csv \
        --labels linear ZOH --out outputs_u3a1/entropy_windows.csv
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

from u3_tools import (make_edges, entropy, conditional_entropy, binary_entropy,
                      spearman)

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Límites físicos del TurtleBot3 burger: definen la grilla por defecto
DEFAULT_RANGES = {'v': (-0.22, 0.22), 'w': (-2.84, 2.84)}
UNITS = {'v': 'm/s', 'w': 'rad/s'}
LABELS = {'v': 'v', 'w': 'ω'}

C_DATA = '#2a6fb0'
C_ALT = '#e07b39'
C_REF = '#7f7f7f'


def style_axes(ax):
    ax.grid(True, color='#e5e5e5', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#999999')
    ax.tick_params(colors='#444444', labelsize=9)


def parse_ranges(items):
    ranges = dict(DEFAULT_RANGES)
    for it in items or []:
        try:
            name, rng = it.split('=')
            lo, hi = (float(x) for x in rng.split(','))
        except ValueError:
            sys.exit(f'[error] --range mal escrito: "{it}" (formato: v=-0.22,0.22)')
        if hi <= lo:
            sys.exit(f'[error] --range {name}: el máximo debe ser mayor que el mínimo')
        ranges[name] = (lo, hi)
    return ranges


def load(path, signals):
    df = pd.read_csv(path)
    missing = [c for c in ['t_sec', *signals] if c not in df.columns]
    if missing:
        sys.exit(f'[error] {path}: faltan columnas {missing}')
    t = df['t_sec'].to_numpy(dtype=float)
    dt = np.diff(t)
    if len(dt) and (np.any(dt <= 0) or np.ptp(dt) > 1e-6 * max(np.median(dt), 1e-12)):
        print(f'[aviso] {path}: el paso temporal no es constante. Esta unidad supone la '
              'grilla uniforme de la Unidad 2 (resample_cmdvel_20hz.py).')
    return df


def window_table(df, signals, edges, n, stride, max_nan, miller_madow):
    t = df['t_sec'].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t)))
    data = {s: df[s].to_numpy(dtype=float) for s in signals}
    starts = np.arange(0, len(t) - n + 1, stride)
    if len(starts) == 0:
        sys.exit(f'[error] la serie tiene {len(t)} muestras, menos que una ventana de {n}')
    rows = []
    for j, s0 in enumerate(starts):
        row = {'i': j, 't_start': t[s0] - t[0], 't_center': t[s0] - t[0] + n * dt / 2}
        nan_frac = max(np.mean(np.isnan(data[s][s0:s0 + n])) for s in signals)
        row['nan_frac'] = nan_frac
        for s in signals:
            w = data[s][s0:s0 + n]
            if nan_frac > max_nan:
                vals = dict(H=np.nan, Hnorm=np.nan, Hcond=np.nan,
                            mean=np.nan, std=np.nan, mad=np.nan)
            else:
                H = entropy(w, edges[s], miller_madow)
                med = np.nanmedian(w)
                vals = dict(
                    H=H,
                    Hnorm=H / np.log2(len(edges[s]) - 1),
                    Hcond=conditional_entropy(w, edges[s]),
                    mean=np.nanmean(w),
                    std=np.nanstd(w, ddof=1),
                    mad=np.nanmedian(np.abs(w - med)),
                )
            for k, val in vals.items():
                row[f'{s}_{k}'] = val
        rows.append(row)
    return pd.DataFrame(rows)


def distinct_levels(x, tol=1e-9):
    x = np.asarray(x, dtype=float)
    x = np.round(x[~np.isnan(x)] / tol) * tol
    return len(np.unique(x))


def summarize(tab, df, signals, edges, args):
    L = [f'Entrada: {args.csv}',
         f'Ventanas: n = {args.n} muestras, paso = {args.stride}, total = {len(tab)}',
         f'Descartadas por huecos (NaN > {100 * args.max_nan:.0f}%): '
         f'{int(tab[f"{signals[0]}_H"].isna().sum())}']
    for s in signals:
        e = edges[s]
        H = tab[f'{s}_H']
        L.append('')
        L.append(f'[{s}] grilla: {len(e) - 1} bins en [{e[0]:g}, {e[-1]:g}] {UNITS.get(s, "")}'
                 f'  (máximo posible: {np.log2(len(e) - 1):.2f} bits)')
        L.append(f'  H: media {H.mean():.3f} | std {H.std():.3f} | min {H.min():.3f} | '
                 f'max {H.max():.3f} bits')
        L.append(f'  Ventanas con H = 0 (un solo bin ocupado): {int((H < 1e-12).sum())}')
        L.append(f'  Correlación H vs. std: Pearson {tab[[f"{s}_H", f"{s}_std"]].corr().iloc[0, 1]:.3f}'
                 f' | Spearman {spearman(H, tab[f"{s}_std"]):.3f}')
        nlev = distinct_levels(df[s].to_numpy())
        if nlev <= 3:
            L.append(f'  Nota: la señal toma solo {nlev} valores distintos. H y std dependen '
                     'de la misma fracción de tiempo en cada nivel (apunte, "Una consecuencia '
                     'de que la señal tenga dos niveles").')
        top = tab.nlargest(3, f'{s}_H')
        L.append('  Ventanas de mayor entropía (t_start en s): ' +
                 ', '.join(f'{r.t_start:.1f} ({r[f"{s}_H"]:.2f} bits)' for _, r in top.iterrows()))
    return '\n'.join(L) + '\n'


def plot_series(df, tab, signals, path):
    t = df['t_sec'].to_numpy() - df['t_sec'].iloc[0]
    fig, axes = plt.subplots(1 + len(signals), 1, figsize=(11, 2.6 * (1 + len(signals))),
                             sharex=True)
    axes[0].plot(t, df[signals[0]], color=C_REF, linewidth=1)
    axes[0].set_ylabel(f'{LABELS.get(signals[0], signals[0])} ({UNITS.get(signals[0], "")})')
    axes[0].set_title(f'Señal {LABELS.get(signals[0], signals[0])}', loc='left', fontsize=10)
    for ax, s in zip(axes[1:], signals):
        ax.plot(tab['t_center'], tab[f'{s}_H'], color=C_DATA, linewidth=1.2,
                marker='.', markersize=3)
        ax.set_ylabel('H (bits)')
        ax.set_title(f'Entropía por ventana de {LABELS.get(s, s)}', loc='left', fontsize=10)
    axes[-1].set_xlabel('tiempo (s) — entropía ubicada en el centro de cada ventana')
    for ax in axes:
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_vs_std(df, tab, signals, path):
    fig, axes = plt.subplots(1, len(signals), figsize=(5.5 * len(signals), 4.2))
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, signals):
        ax.scatter(tab[f'{s}_std'], tab[f'{s}_H'], s=18, color=C_DATA, alpha=0.75,
                   edgecolor='white', linewidth=0.5, label='ventanas')
        # Curva teórica para una señal de dos niveles {0, V}
        x = df[s].to_numpy(dtype=float)
        V = np.nanmax(np.abs(x)) if np.any(~np.isnan(x)) else 0
        if V > 0:
            p = np.linspace(0, 1, 201)
            ax.plot(V * np.sqrt(p * (1 - p)), binary_entropy(p), color=C_REF,
                    linewidth=1.2, linestyle='--', label=f'dos niveles {{0, {V:g}}}')
        ax.set_xlabel(f'desviación estándar de {LABELS.get(s, s)} ({UNITS.get(s, "")})')
        ax.set_ylabel('H (bits)')
        ax.set_title(f'Entropía vs. desviación estándar ({LABELS.get(s, s)})',
                     loc='left', fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc='lower right')
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_compare(tab, tab2, signals, labels, path):
    fig, axes = plt.subplots(len(signals), 1, figsize=(11, 2.8 * len(signals)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, signals):
        ax.plot(tab['t_center'], tab[f'{s}_H'], color=C_DATA, linewidth=1.2, label=labels[0])
        ax.plot(tab2['t_center'], tab2[f'{s}_H'], color=C_ALT, linewidth=1.2,
                linestyle='--', label=labels[1])
        ax.set_ylabel('H (bits)')
        ax.set_title(f'Entropía de {LABELS.get(s, s)}: {labels[0]} vs. {labels[1]}',
                     loc='left', fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc='upper right')
        style_axes(ax)
    axes[-1].set_xlabel('tiempo (s)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='Entropía de Shannon por ventana deslizante')
    p.add_argument('--csv', required=True, help='CSV re-muestreado (t_sec, v, w)')
    p.add_argument('--signals', nargs='+', default=['v', 'w'])
    p.add_argument('--n', type=int, default=100, help='tamaño de ventana (muestras)')
    p.add_argument('--stride', type=int, default=None, help='paso (muestras); por defecto n/2')
    p.add_argument('--nbins', type=int, default=15, help='número de bins de la grilla fija')
    p.add_argument('--range', nargs='*', metavar='SEÑAL=MIN,MAX',
                   help='rango de la grilla; por defecto los límites físicos del burger')
    p.add_argument('--max-nan', type=float, default=0.0,
                   help='fracción máxima de NaN tolerada por ventana (por defecto 0)')
    p.add_argument('--miller-madow', action='store_true', help='aplicar corrección de sesgo')
    p.add_argument('--compare', default=None,
                   help='segundo CSV re-muestreado (p. ej. ZOH) para comparar')
    p.add_argument('--labels', nargs=2, default=['principal', 'comparación'],
                   help='etiquetas para --csv y --compare')
    p.add_argument('--out', default='outputs_u3a1/entropy_windows.csv')
    p.add_argument('--outdir', default=None, help='carpeta de figuras (por defecto, la de --out)')
    args = p.parse_args()

    args.stride = args.stride or max(1, args.n // 2)
    if args.nbins < 2:
        p.error('--nbins debe ser >= 2')
    ranges = parse_ranges(args.range)
    for s in args.signals:
        if s not in ranges:
            p.error(f'no hay rango para "{s}"; agregá --range {s}=MIN,MAX')
    edges = {s: make_edges(*ranges[s], args.nbins) for s in args.signals}

    outdir = args.outdir or (os.path.dirname(args.out) or '.')
    os.makedirs(outdir, exist_ok=True)
    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = load(args.csv, args.signals)
    for s in args.signals:
        x = df[s].to_numpy(dtype=float)
        out = np.sum((x < ranges[s][0]) | (x > ranges[s][1]))
        if out:
            print(f'[aviso] {s}: {out} valores fuera de [{ranges[s][0]}, {ranges[s][1]}]; '
                  'se asignan al bin extremo.')

    tab = window_table(df, args.signals, edges, args.n, args.stride, args.max_nan,
                       args.miller_madow)
    tab.to_csv(args.out, index=False, float_format='%.6f')

    text = summarize(tab, df, args.signals, edges, args)

    plot_series(df, tab, args.signals, os.path.join(outdir, 'entropy_series.png'))
    plot_vs_std(df, tab, args.signals, os.path.join(outdir, 'entropy_vs_std.png'))

    if args.compare:
        df2 = load(args.compare, args.signals)
        tab2 = window_table(df2, args.signals, edges, args.n, args.stride, args.max_nan,
                            args.miller_madow)
        out2 = os.path.splitext(args.out)[0] + '_compare.csv'
        tab2.to_csv(out2, index=False, float_format='%.6f')
        plot_compare(tab, tab2, args.signals, args.labels,
                     os.path.join(outdir, 'entropy_compare.png'))
        k = min(len(tab), len(tab2))
        lines = ['', f'Comparación {args.labels[0]} vs. {args.labels[1]}:']
        for s in args.signals:
            d = tab[f'{s}_H'].iloc[:k].to_numpy() - tab2[f'{s}_H'].iloc[:k].to_numpy()
            lines.append(f'  [{s}] H media: {tab[f"{s}_H"].mean():.3f} vs. '
                         f'{tab2[f"{s}_H"].mean():.3f} bits | diferencia media por ventana '
                         f'{np.nanmean(d):+.3f} | ventanas con |dif| > 0.1 bits: '
                         f'{int(np.nansum(np.abs(d) > 0.1))}')
        text += '\n'.join(lines) + '\n'

    with open(os.path.join(outdir, 'entropy_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)
    print(f'Tabla: {args.out} | figuras y resumen: {outdir}/')


if __name__ == '__main__':
    main()
