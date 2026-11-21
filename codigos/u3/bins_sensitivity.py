#!/usr/bin/env python3
"""
Sensibilidad de la entropía al número de bins (Unidad 3, Actividad 1, paso 5).

Para cada N_bins de la lista, construye una grilla FIJA sobre el mismo rango,
calcula la entropía por ventana y compara:
    - el valor absoluto (media de H y de H normalizada);
    - el orden relativo de las ventanas (correlación de Spearman entre series).

Salidas:
    <outdir>/bins_sensitivity.csv   una fila por N_bins
    <outdir>/bins_spearman.csv      matriz de correlación de Spearman
    <outdir>/bins_sensitivity.png   figura de dos paneles

Uso:
    python3 bins_sensitivity.py --csv data/cmdvel_20hz.csv --signal v
    python3 bins_sensitivity.py --csv data/cmdvel_20hz.csv --signal w --nbins 5 10 20 40
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

from u3_tools import make_edges, entropy_sliding_windows, spearman

warnings.filterwarnings('ignore', category=RuntimeWarning)

DEFAULT_RANGES = {'v': (-0.22, 0.22), 'w': (-2.84, 2.84)}
C_DATA = '#2a6fb0'
C_REF = '#7f7f7f'


def style_axes(ax):
    ax.grid(True, color='#e5e5e5', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#999999')
    ax.tick_params(colors='#444444', labelsize=9)


def main():
    p = argparse.ArgumentParser(description='Sensibilidad de la entropía al número de bins')
    p.add_argument('--csv', required=True, help='CSV re-muestreado (t_sec, v, w)')
    p.add_argument('--signal', default='v')
    p.add_argument('--nbins', nargs='+', type=int, default=[5, 10, 15, 20, 25])
    p.add_argument('--range', nargs=2, type=float, metavar=('MIN', 'MAX'), default=None,
                   help='rango de la grilla; por defecto los límites físicos del burger')
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--stride', type=int, default=None)
    p.add_argument('--ref', type=int, default=None,
                   help='N_bins de referencia para el panel de Spearman (por defecto, el del medio)')
    p.add_argument('--outdir', default='outputs_u3a1')
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if args.signal not in df.columns:
        sys.exit(f'[error] no existe la columna "{args.signal}"')
    lo, hi = args.range if args.range else DEFAULT_RANGES.get(args.signal, (None, None))
    if lo is None:
        sys.exit(f'[error] indicá --range MIN MAX para la señal "{args.signal}"')
    stride = args.stride or max(1, args.n // 2)
    nbins = sorted(set(args.nbins))
    ref = args.ref if args.ref in nbins else nbins[len(nbins) // 2]
    os.makedirs(args.outdir, exist_ok=True)

    x = df[args.signal].to_numpy(dtype=float)
    series = {}
    rows = []
    for nb in nbins:
        H, _ = entropy_sliding_windows(x, args.n, stride, make_edges(lo, hi, nb))
        series[nb] = H
        rows.append({
            'n_bins': nb,
            'H_mean': np.nanmean(H),
            'H_std': np.nanstd(H),
            'H_max_posible': np.log2(nb),
            'Hnorm_mean': np.nanmean(H) / np.log2(nb),
        })
    res = pd.DataFrame(rows)
    rho = pd.DataFrame([[spearman(series[a], series[b]) for b in nbins] for a in nbins],
                       index=nbins, columns=nbins)
    res['spearman_vs_ref'] = [rho.loc[ref, nb] for nb in nbins]

    res.to_csv(os.path.join(args.outdir, 'bins_sensitivity.csv'), index=False,
               float_format='%.4f')
    rho.to_csv(os.path.join(args.outdir, 'bins_spearman.csv'), float_format='%.4f')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].errorbar(res['n_bins'], res['H_mean'], yerr=res['H_std'], color=C_DATA,
                     marker='o', markersize=5, linewidth=1.2, capsize=3,
                     label='H media ± std entre ventanas')
    axes[0].plot(res['n_bins'], res['H_max_posible'], color=C_REF, linestyle='--',
                 linewidth=1, label='máximo posible, log2(N_bins)')
    axes[0].set_xlabel('número de bins')
    axes[0].set_ylabel('entropía (bits)')
    axes[0].set_title('Valor absoluto', loc='left', fontsize=10)
    axes[0].legend(fontsize=8, frameon=False, loc='upper left')

    axes[1].plot(res['n_bins'], res['spearman_vs_ref'], color=C_DATA, marker='o',
                 markersize=5, linewidth=1.2)
    axes[1].axhline(1.0, color=C_REF, linewidth=0.8, linestyle=':')
    axes[1].set_ylim(min(0.0, np.nanmin(res['spearman_vs_ref']) - 0.05), 1.05)
    axes[1].set_xlabel('número de bins')
    axes[1].set_ylabel(f'Spearman con N_bins = {ref}')
    axes[1].set_title('Orden relativo de las ventanas', loc='left', fontsize=10)
    for ax in axes:
        style_axes(ax)
    fig.suptitle(f'Sensibilidad a la grilla — señal {args.signal}, rango [{lo:g}, {hi:g}]',
                 x=0.01, ha='left', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, 'bins_sensitivity.png'), dpi=150)
    plt.close(fig)

    print(f'Señal: {args.signal} | rango [{lo:g}, {hi:g}] | ventanas de {args.n}, paso {stride}')
    print(res.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('\nMatriz de Spearman entre configuraciones:')
    print(rho.to_string(float_format=lambda v: f'{v:.3f}'))
    print(f'\nArchivos en {args.outdir}/')


if __name__ == '__main__':
    main()
