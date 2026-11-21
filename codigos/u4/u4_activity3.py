#!/usr/bin/env python3
"""
Actividad 3: entropía y predictibilidad (Unidad 4).

Para Structured, Shuffled, IID, Periodic y Slow calcula:
    H          entropía de Shannon, grilla fija de --nbins bins en [0, 0.2] m/s
    Hcond      entropía condicional H(X_t+1 | X_t) con la misma grilla
    RMSE de persistencia y del mejor AR (orden en --orders, entrenado en la 1.ª mitad)
    NRMSE = RMSE / desvío de la señal (~1: no mejor que predecir la media)

Cuadrantes: entropía "alta" si H >= la mitad del máximo log2(n_bins);
error "alto" si NRMSE >= 0.5.

Salidas en outputs_u4a3/:
    comparison_table.csv        una fila por señal
    entropy_vs_prediction.png   H vs. NRMSE: dos puntos por señal (persistencia y mejor AR)
    signal_examples.png         primeras 120 muestras de cada señal
    correlation_matrix.png      correlaciones entre métricas (con solo 5 señales: orientativo)

Uso:
    python3 u4_activity3.py
"""
import argparse
import os

import numpy as np
import pandas as pd

from u4_tools import (predict_persistence, fit_ar, predict_ar, make_edges, entropy,
                      conditional_entropy)
from generate_signals import make_signals
from u4_plotstyle import plt, style_axes, save, SIGNAL_COLORS, C_REF


def best_ar(x, orders):
    half = len(x) // 2
    _, e_p = predict_persistence(x[half:])
    sd = float(np.std(x[half:], ddof=1))
    best = None
    for p in orders:
        _, e = predict_ar(x[half - p:], fit_ar(x[:half], p))
        e = e[-len(e_p):]
        rmse = float(np.sqrt(np.mean(e ** 2)))
        if best is None or rmse < best[1] - 1e-12:
            best = (p, rmse)
    return float(np.sqrt(np.mean(e_p ** 2))), best, sd


def main():
    ap = argparse.ArgumentParser(description='Actividad 3 de la Unidad 4')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--nbins', type=int, default=10)
    ap.add_argument('--orders', nargs='+', type=int, default=[1, 2, 4, 8])
    ap.add_argument('--outdir', default='outputs_u4a3')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print('=' * 70)
    print('ACTIVIDAD 3: Entropía y predictibilidad')
    print('=' * 70)

    signals = make_signals(args.n, include_slow=True)
    edges = make_edges(0.0, 0.2, args.nbins)
    rows = []
    for name, x in signals.items():
        rmse_p, (p_best, rmse_ar), sd = best_ar(x, args.orders)
        rows.append({
            'señal': name,
            'H_bits': entropy(x, edges),
            'H_max_bits': np.log2(args.nbins),
            'Hcond_bits': conditional_entropy(x, edges),
            'rmse_persistencia': rmse_p,
            'mejor_AR': f'AR({p_best})',
            'rmse_mejor_AR': rmse_ar,
            'nrmse_persistencia': rmse_p / sd,
            'nrmse_mejor_AR': rmse_ar / sd,
            'std': sd,
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(args.outdir, 'comparison_table.csv'), index=False,
               float_format='%.6f')

    # Cuadrantes: H respecto de la mitad del máximo posible; error normalizado vs. 0.5
    h_cut = np.log2(args.nbins) / 2
    e_cut = 0.5
    print(f'\nGrilla: {args.nbins} bins en [0, 0.2] m/s (máximo {np.log2(args.nbins):.2f} bits)\n')
    print(f'  {"señal":<11} {"H":>6} {"Hcond":>6} {"RMSE pers.":>11} {"mejor AR":>9} '
          f'{"RMSE AR":>8}   cuadrante (pers. -> mejor AR)')
    print(f'  (H alta si >= {h_cut:.2f} bits; error alto si NRMSE >= {e_cut})')
    for _, r in tab.iterrows():
        q = lambda h, e: f'{"alta" if h > h_cut else "baja"} H / {"alto" if e > e_cut else "bajo"} error'
        print(f'  {r["señal"]:<11} {r["H_bits"]:6.3f} {r["Hcond_bits"]:6.3f} '
              f'{r["rmse_persistencia"]:11.4f} {r["mejor_AR"]:>9} {r["rmse_mejor_AR"]:8.4f}   '
              f'{q(r["H_bits"], r["nrmse_persistencia"])} -> {q(r["H_bits"], r["nrmse_mejor_AR"])}')

    # Scatter
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axvline(h_cut, color=C_REF, linewidth=0.8, linestyle=':')
    ax.axhline(e_cut, color=C_REF, linewidth=0.8, linestyle=':')
    for _, r in tab.iterrows():
        c = SIGNAL_COLORS[r['señal']]
        ax.annotate('', xy=(r['H_bits'], r['nrmse_mejor_AR']),
                    xytext=(r['H_bits'], r['nrmse_persistencia']),
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.1))
        ax.scatter(r['H_bits'], r['nrmse_persistencia'], s=60, color=c, edgecolor='white',
                   linewidth=1, zorder=3)
        ax.scatter(r['H_bits'], r['nrmse_mejor_AR'], s=60, facecolor='white', edgecolor=c,
                   linewidth=1.5, zorder=3)
        ax.annotate(r['señal'], (r['H_bits'], r['nrmse_persistencia']),
                    textcoords='offset points', xytext=(7, 4), fontsize=9, color='#333333')
    ax.scatter([], [], s=50, color='#555555', label='persistencia')
    ax.scatter([], [], s=50, facecolor='white', edgecolor='#555555', label='mejor AR')
    ax.set_xlabel(f'entropía de Shannon (bits, {args.nbins} bins)')
    ax.set_ylabel('error normalizado, NRMSE = RMSE / desvío de la señal')
    ax.set_title('Entropía vs. error de predicción (flecha: de persistencia al mejor AR)',
                 loc='left', fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc='upper center')
    ax.set_ylim(-0.05, max(1.6, tab['nrmse_persistencia'].max() + 0.1))
    ax.text(h_cut, ax.get_ylim()[1], ' H máx / 2', fontsize=8, color='#555555', va='top')
    ax.text(ax.get_xlim()[0], e_cut, ' NRMSE = 0.5', fontsize=8, color='#555555', va='bottom')
    style_axes(ax)
    save(fig, os.path.join(args.outdir, 'entropy_vs_prediction.png'))

    # Ejemplos de señales
    fig, axes = plt.subplots(len(signals), 1, figsize=(11, 1.8 * len(signals)), sharex=True)
    for ax, (name, x) in zip(axes, signals.items()):
        r = tab[tab['señal'] == name].iloc[0]
        ax.plot(np.arange(120), x[:120], color=SIGNAL_COLORS[name], linewidth=1.1)
        ax.set_title(f'{name}   H = {r["H_bits"]:.2f} bits · RMSE pers. = '
                     f'{r["rmse_persistencia"]:.3f} · RMSE {r["mejor_AR"]} = '
                     f'{r["rmse_mejor_AR"]:.3f} m/s', loc='left', fontsize=9)
        ax.set_ylabel('m/s')
        style_axes(ax)
    axes[-1].set_xlabel('muestra')
    save(fig, os.path.join(args.outdir, 'signal_examples.png'))

    # Matriz de correlación (orientativa: 5 señales)
    cols = ['H_bits', 'Hcond_bits', 'nrmse_persistencia', 'nrmse_mejor_AR', 'std']
    corr = tab[cols].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    labels = ['H', 'H cond.', 'NRMSE pers.', 'NRMSE AR', 'std']
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if abs(corr.values[i, j]) > 0.6 else '#222222')
    ax.set_title('Correlación entre métricas (solo 5 señales: orientativo)',
                 loc='left', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, os.path.join(args.outdir, 'correlation_matrix.png'))

    print(f'\nTabla: {args.outdir}/comparison_table.csv | figuras: {args.outdir}/')


if __name__ == '__main__':
    main()
