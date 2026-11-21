#!/usr/bin/env python3
"""
Re-muestreo de /cmd_vel a una grilla uniforme (Unidad 2).

Entrada: CSV del logger de la Unidad 1 (columnas t_sec, v, w; dt se ignora).
Salida:
    --out                     CSV re-muestreado: t_sec, v, w, dt, gap
                              (gap = 1 si el punto cae dentro de un hueco; v y w quedan vacíos)
    <outdir>/dt_before_after.png        Δt original vs. Δt de la grilla
    <outdir>/signal_compare_<met>.png   señal original vs. re-muestreada en un tramo
    <outdir>/resample_report_<met>.txt  resumen y resultado de la verificación

Uso:
    python3 resample_cmdvel_20hz.py --csv data/cmdvel_u1.csv --out data/cmdvel_20hz.csv
    python3 resample_cmdvel_20hz.py --csv data/cmdvel_u1.csv --method previous \
        --out data/cmdvel_20hz_zoh.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from u2_tools import prepare_series, clip_physical, resample_to_grid, verify_resampling

# Límites físicos del TurtleBot3 burger
V_MAX = 0.22    # m/s
W_MAX = 2.84    # rad/s

C_DATA = '#2a6fb0'
C_ALT = '#7f7f7f'
C_GAP = '#c0392b'


def style_axes(ax):
    ax.grid(True, color='#e5e5e5', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#999999')
    ax.tick_params(colors='#444444', labelsize=9)


def load(path):
    df = pd.read_csv(path)
    missing = [c for c in ('t_sec', 'v', 'w') if c not in df.columns]
    if missing:
        sys.exit(f'[error] faltan columnas en el CSV: {missing}')
    df = df[['t_sec', 'v', 'w']].dropna().reset_index(drop=True)
    if len(df) < 3:
        sys.exit('[error] el CSV tiene menos de 3 muestras')
    return df


def plot_dt_before_after(t_orig, t_grid, freq, path):
    dt_o = np.diff(t_orig)
    dt_o = dt_o[dt_o > 0]
    dt_g = np.diff(t_grid)
    # Misma escala en ambos paneles para que la comparación sea honesta
    hi = max(np.percentile(dt_o, 99.9), np.median(dt_o) * 1.5, dt_g.max() * 1.2)
    bins = np.linspace(0, hi, 81)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)

    axes[0].hist(dt_o[dt_o <= hi], bins=bins, color=C_DATA, edgecolor='white', linewidth=0.5)
    axes[0].axvline(np.median(dt_o), color='#555555', linewidth=1.2,
                    label=f'mediana = {np.median(dt_o):.4f} s')
    n_out = int(np.sum(dt_o > hi))
    t0 = 'Δt original'
    if n_out:
        t0 += f'  ({n_out} valores > {hi:.3f} s no se muestran)'
    axes[0].set_title(t0, loc='left', fontsize=10)
    axes[0].set_ylabel('frecuencia')
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].hist(dt_g, bins=bins, color=C_ALT, edgecolor='white', linewidth=0.5)
    axes[1].set_title(f'Δt de la grilla: {1 / freq:.4f} s siempre (por construcción)',
                      loc='left', fontsize=10)
    for ax in axes:
        ax.set_xlabel('Δt (s)')
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_compare(df, t_grid, grids, method, t_from, t_len, path):
    t = df['t_sec'].to_numpy()
    t_rel0 = t[0]
    a, b = t_from, t_from + t_len
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    for ax, col, label in ((axes[0], 'v', 'v (m/s)'), (axes[1], 'w', 'ω (rad/s)')):
        sel_o = (t - t_rel0 >= a) & (t - t_rel0 <= b)
        sel_g = (t_grid - t_rel0 >= a) & (t_grid - t_rel0 <= b)
        ax.plot(t_grid[sel_g] - t_rel0, grids[col][sel_g], color=C_ALT, linewidth=1.2,
                marker='.', markersize=4, label=f're-muestreada ({method})')
        ax.plot(t[sel_o] - t_rel0, df[col].to_numpy()[sel_o], linestyle='none',
                marker='o', markersize=5, markerfacecolor='none', markeredgecolor=C_DATA,
                markeredgewidth=1.2, label='original')
        gap = sel_g & np.isnan(grids[col])
        if gap.any():
            ax.scatter(t_grid[gap] - t_rel0, np.zeros(gap.sum()), marker='|', s=60,
                       color=C_GAP, label='hueco (NaN)')
        ax.set_ylabel(label)
        style_axes(ax)
    axes[0].set_title(f'Señal original vs. re-muestreada ({method}), tramo {a:g}–{b:g} s',
                      loc='left', fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    for h, l in zip(h1, l1):
        if l not in labels:
            handles.append(h)
            labels.append(l)
    fig.legend(handles, labels, fontsize=8, frameon=False, loc='upper right', ncol=len(labels))
    axes[1].set_xlabel('tiempo desde la primera muestra (s)')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='Re-muestrea /cmd_vel a una grilla uniforme')
    p.add_argument('--csv', required=True, help='CSV de entrada (t_sec, v, w)')
    p.add_argument('--out', default='data/cmdvel_20hz.csv', help='CSV de salida')
    p.add_argument('--freq', type=float, default=20.0, help='frecuencia de la grilla (Hz)')
    p.add_argument('--method', choices=['linear', 'nearest', 'previous'], default='linear',
                   help="interpolación: linear (defecto), nearest o previous (ZOH)")
    p.add_argument('--max-gap', type=float, default=None,
                   help='hueco máximo tolerable (s); por defecto 3 × mediana(Δt)')
    p.add_argument('--no-clip', action='store_true',
                   help='no descartar valores fuera de los límites físicos del burger')
    p.add_argument('--outdir', default='outputs_u2', help='carpeta para figuras y reporte')
    p.add_argument('--plot-from', type=float, default=None,
                   help='inicio del tramo a graficar (s desde el comienzo); por defecto, '
                        'el primer cambio de v')
    p.add_argument('--plot-len', type=float, default=15.0, help='largo del tramo a graficar (s)')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df = load(args.csv)
    n_in = len(df)

    # 1. Límites físicos: se descartan filas imposibles (en ambas señales a la vez,
    #    para que v y w compartan exactamente la misma grilla)
    n_clip = 0
    if not args.no_clip:
        v_c = clip_physical(df['v'], V_MAX)
        w_c = clip_physical(df['w'], W_MAX)
        bad = np.isnan(v_c) | np.isnan(w_c)
        n_clip = int(bad.sum())
        df = df[~bad].reset_index(drop=True)

    # 2. Timestamps: repetidos y retrocesos (comunes a ambas señales)
    t_raw = df['t_sec'].to_numpy(dtype=float)
    try:
        t_ok, idx_v = prepare_series(t_raw, np.arange(len(t_raw)))
    except ValueError as e:
        sys.exit(f'[error] {e}')
    idx = idx_v.astype(int)
    n_dup = len(t_raw) - len(idx)
    df = df.iloc[idx].reset_index(drop=True)
    t = df['t_sec'].to_numpy(dtype=float)

    max_gap = args.max_gap if args.max_gap is not None else 3.0 * np.median(np.diff(t))

    # 3. Re-muestreo
    grids = {}
    t_grid = None
    for col in ('v', 'w'):
        tg, xg = resample_to_grid(t, df[col].to_numpy(), args.freq, args.method, max_gap)
        t_grid = tg
        grids[col] = xg
    gap = np.isnan(grids['v'])

    # 4. Verificación
    report_ver = []
    for col in ('v', 'w'):
        try:
            verify_resampling(t, df[col].to_numpy(), t_grid, grids[col], max_gap)
            report_ver.append(f'  {col}: OK')
        except AssertionError as e:
            report_ver.append(f'  {col}: FALLÓ -> {e}')

    # 5. CSV de salida (mismo formato que la Unidad 1, más la columna gap)
    out = pd.DataFrame({
        't_sec': t_grid,
        'v': grids['v'],
        'w': grids['w'],
    })
    out['dt'] = out['t_sec'].diff()
    out['gap'] = gap.astype(int)
    out.to_csv(args.out, index=False, float_format='%.9f', na_rep='')

    # 6. Figuras
    plot_dt_before_after(t, t_grid, args.freq, os.path.join(args.outdir, 'dt_before_after.png'))
    t_from = args.plot_from
    if t_from is None:
        ch = np.flatnonzero(np.diff(df['v'].to_numpy()) != 0)
        t_from = max(0.0, (t[ch[0]] - t[0]) - 2.0) if len(ch) else 0.0
    plot_compare(df, t_grid, grids, args.method, t_from, args.plot_len,
                 os.path.join(args.outdir, f'signal_compare_{args.method}.png'))

    # 7. Reporte
    n_gap_runs = int(np.sum(np.diff(np.r_[0, gap.astype(int)]) == 1))
    lines = [
        f'Entrada: {args.csv}',
        f'Salida:  {args.out}',
        '',
        f'Muestras originales: {n_in}',
        f'  descartadas por límites físicos: {n_clip}' + ('' if not args.no_clip else ' (desactivado)'),
        f'  timestamps repetidos resueltos:  {n_dup}',
        f'Mediana Δt original: {np.median(np.diff(t)):.4f} s',
        '',
        f'Grilla: {args.freq:g} Hz (Δ = {1/args.freq:.4f} s), método = {args.method}',
        f'Puntos de la grilla: {len(t_grid)}',
        f'Hueco máximo tolerado: {max_gap:.4f} s',
        f'Puntos en huecos (NaN): {int(gap.sum())} ({100*gap.mean():.2f}%) en {n_gap_runs} hueco(s)',
        '',
        'Verificación:',
        *report_ver,
        '',
        'Recordá: que la grilla tenga Δt constante es una propiedad de la construcción,',
        'no una mejora del muestreo. El jitter original se traslada a la amplitud.',
    ]
    text = '\n'.join(lines) + '\n'
    with open(os.path.join(args.outdir, f'resample_report_{args.method}.txt'), 'w',
              encoding='utf-8') as f:
        f.write(text)
    print(text)
    print(f'Figuras y reporte en: {args.outdir}/')


if __name__ == '__main__':
    main()
