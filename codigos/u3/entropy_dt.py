#!/usr/bin/env python3
"""
Entropía de los intervalos entre mensajes Δt (Unidad 3, sección "La entropía de Δt").

IMPORTANTE: se usa el CSV ORIGINAL de la Unidad 1 (cmdvel_u1.csv), no el
re-muestreado: en la grilla uniforme todos los Δt valen lo mismo y H = 0.

Ventanas por TIEMPO (todas cubren la misma duración). Grilla fija de bins sobre
Δt entre 0 y --dt-max; los Δt mayores (pausas) caen en el último bin.

Como el número de mensajes por ventana varía, el sesgo del estimador también varía
(apunte, "El estimador está sesgado"). Por eso se reporta n_msgs y se ofrece
--miller-madow.

Salidas:
    --out                         CSV con una fila por ventana
    <outdir>/entropy_dt.png       Δt, H(Δt) y mensajes por ventana

Uso:
    python3 entropy_dt.py --csv data/cmdvel_u1.csv
    python3 entropy_dt.py --csv data/cmdvel_u1.csv --tw 10 --step 5 --nbins 20 --miller-madow
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

from u3_tools import make_edges, entropy

warnings.filterwarnings('ignore', category=RuntimeWarning)

C_DATA = '#2a6fb0'
C_REF = '#7f7f7f'
C_FLAG = '#c0392b'


def style_axes(ax):
    ax.grid(True, color='#e5e5e5', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#999999')
    ax.tick_params(colors='#444444', labelsize=9)


def main():
    p = argparse.ArgumentParser(description='Entropía de Δt por ventanas de tiempo')
    p.add_argument('--csv', required=True, help='CSV ORIGINAL de la Unidad 1 (t_sec, ...)')
    p.add_argument('--tw', type=float, default=10.0, help='ancho de ventana (s)')
    p.add_argument('--step', type=float, default=None, help='paso (s); por defecto tw/2')
    p.add_argument('--nbins', type=int, default=20)
    p.add_argument('--dt-max', type=float, default=None,
                   help='borde superior de la grilla (s); por defecto 3 × mediana(Δt)')
    p.add_argument('--miller-madow', action='store_true')
    p.add_argument('--out', default='outputs_u3a1/entropy_dt.csv')
    p.add_argument('--outdir', default=None)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if 't_sec' not in df.columns:
        sys.exit('[error] el CSV no tiene columna t_sec')
    t = df['t_sec'].dropna().to_numpy(dtype=float)
    dt = np.diff(t)
    if np.any(dt < 0):
        sys.exit('[error] hay timestamps que retroceden; revisá el registro')
    med = float(np.median(dt))
    if np.ptp(dt) < 1e-9:
        sys.exit('[error] todos los Δt son iguales: ¿es el CSV re-muestreado? '
                 'Usá el original de la Unidad 1.')

    dt_max = args.dt_max or 3.0 * med
    edges = make_edges(0.0, dt_max, args.nbins)
    step = args.step or args.tw / 2
    outdir = args.outdir or (os.path.dirname(args.out) or '.')
    os.makedirs(outdir, exist_ok=True)
    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

    t_end = t[1:]               # cada Δt se asigna al instante en que termina
    rows = []
    t0 = t[0]
    while t0 + args.tw <= t[-1]:
        sel = (t_end >= t0) & (t_end < t0 + args.tw)
        d = dt[sel]
        rows.append({
            't_start': t0 - t[0],
            't_center': t0 - t[0] + args.tw / 2,
            'n_msgs': int(sel.sum()),
            'H_dt': entropy(d, edges, args.miller_madow) if len(d) else np.nan,
            'median_dt': float(np.median(d)) if len(d) else np.nan,
            'n_long': int(np.sum(d > dt_max)),
        })
        t0 += step
    tab = pd.DataFrame(rows)
    if tab.empty:
        sys.exit(f'[error] el registro dura menos que una ventana de {args.tw} s')
    tab.to_csv(args.out, index=False, float_format='%.6f')

    H = tab['H_dt']
    hmed = np.nanmedian(H)
    mad = np.nanmedian(np.abs(H - hmed))
    thr = hmed + 3 * 1.4826 * mad if mad > 0 else np.inf
    flagged = H > thr

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    axes[0].plot(t_end - t[0], dt, color=C_REF, linewidth=0.7)
    axes[0].axhline(dt_max, color=C_FLAG, linewidth=0.8, linestyle='--',
                    label=f'borde de la grilla ({dt_max:.3f} s)')
    axes[0].set_ylabel('Δt (s)')
    axes[0].set_title('Intervalos entre mensajes', loc='left', fontsize=10)
    axes[0].legend(fontsize=8, frameon=False, loc='upper right')
    axes[1].plot(tab['t_center'], H, color=C_DATA, linewidth=1.2, marker='.', markersize=3)
    if flagged.any():
        axes[1].scatter(tab['t_center'][flagged], H[flagged], s=30, color=C_FLAG, zorder=3,
                        edgecolor='white', linewidth=0.8, label='H alta (Hampel, c = 3)')
        axes[1].legend(fontsize=8, frameon=False, loc='upper right')
    axes[1].set_ylabel('H(Δt) (bits)')
    axes[1].set_title(f'Entropía de Δt — ventanas de {args.tw:g} s, {args.nbins} bins '
                      f'en [0, {dt_max:.3f}] s', loc='left', fontsize=10)
    axes[2].plot(tab['t_center'], tab['n_msgs'], color=C_DATA, linewidth=1,
                 drawstyle='steps-mid')
    axes[2].set_ylabel('mensajes')
    axes[2].set_title('Mensajes por ventana (el sesgo de H depende de este número)',
                      loc='left', fontsize=10)
    axes[2].set_xlabel('tiempo (s)')
    for ax in axes:
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'entropy_dt.png'), dpi=150)
    plt.close(fig)

    print(f'Entrada: {args.csv} | mediana Δt = {med:.4f} s')
    print(f'Grilla: {args.nbins} bins en [0, {dt_max:.4f}] s | ventanas de {args.tw:g} s, '
          f'paso {step:g} s | total {len(tab)}')
    print(f'H(Δt): mediana {hmed:.3f} | min {np.nanmin(H):.3f} | max {np.nanmax(H):.3f} bits')
    print(f'Mensajes por ventana: mediana {tab["n_msgs"].median():.0f} | '
          f'min {tab["n_msgs"].min()} | max {tab["n_msgs"].max()}')
    print(f'Ventanas con H alta (Hampel, c = 3): {int(flagged.sum())}')
    for _, r in tab[flagged].head(10).iterrows():
        print(f'  t = {r.t_start:7.1f} s  H = {r.H_dt:.3f} bits  mensajes = {int(r.n_msgs)}  '
              f'Δt > borde: {int(r.n_long)}')
    print(f'\nTabla: {args.out} | figura: {outdir}/entropy_dt.png')


if __name__ == '__main__':
    main()
