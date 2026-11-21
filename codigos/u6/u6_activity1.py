#!/usr/bin/env python3
"""
Actividad 1: el autoencoder PUNTUAL como control (apunte, sección "Qué ve un modelo").

Entrena un autoencoder 2 -> 16 -> 1 -> 16 -> 2 sobre los puntos (v, w) estandarizados de
structured y mide su error en la prueba nominal, en shuffle_blocks y en iid. Calcula además
H(v) en ventanas de 5 s para comparar.

Lo que deberías ver: el error en shuffle_blocks es (numéricamente) igual al de structured;
el de iid es varios órdenes mayor. Igual que H(v): el modelo puntual ve valores, no el orden.

Salidas (en --out-dir):
    act1_resumen.csv      mediana, p99 y media del error, y H(v) medio, por modo
    act1_error.png        error contra el tiempo (escala log) y boxplot por modo

Uso:
    python3 u6_activity1.py                         # datos de generate_datasets.py
    python3 u6_activity1.py --data-dir mis_datos --seed 1
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from u6_data import (MODE_COLORS, Standardizer, build_features, entropy_windows,
                     load_cmdvel, make_windows, resample_zoh, split_blocks, style_axes)
from u6_models import WindowAE, recon_errors, set_seed, train_model

EPS = 1e-12     # para graficar el cero en escala logarítmica


def grid_of(path):
    return resample_zoh(*load_cmdvel(path))


def main():
    ap = argparse.ArgumentParser(description='Actividad 1: autoencoder puntual')
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--out-dir', default='outputs_u6/act1')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    tr_grid = grid_of(f'{a.data_dir}/train_structured.csv')
    Xtr, Xva = split_blocks(build_features(tr_grid))
    std = Standardizer().fit(Xtr)
    Wtr, _ = make_windows(std.transform(Xtr), 1)
    Wva, _ = make_windows(std.transform(Xva), 1)
    set_seed(a.seed)
    model = WindowAE(T=1, d=2, hidden=16, latent=1)
    model, hist = train_model(model, Wtr, Wtr, Wva, Wva, epochs=150, seed=a.seed)
    print(f'Autoencoder puntual entrenado: {len(hist)} épocas, '
          f'pérdida de validación {hist[-1][2]:.2e}')

    edges = np.linspace(-0.22, 0.22, 16)            # misma grilla fija que la Unidad 5
    files = {'structured': 'test_structured.csv', 'shuffle_blocks': 'shuffle_blocks.csv',
             'iid': 'iid.csv'}
    rows, curves = [], {}
    for mode, fn in files.items():
        g = grid_of(f'{a.data_dir}/{fn}')
        W, idx = make_windows(std.transform(build_features(g)), 1)
        e, _, _ = recon_errors(model, W)
        Hv = entropy_windows(g['v'].to_numpy(), edges, win=50)
        curves[mode] = (g['t'].to_numpy()[idx], e)
        rows.append({'modo': mode, 'error_mediana': np.median(e),
                     'error_p99': np.quantile(e, 0.99), 'error_media': e.mean(),
                     'H_v_medio_bits': np.nanmean(Hv)})
    res = pd.DataFrame(rows)
    res.to_csv(f'{a.out_dir}/act1_resumen.csv', index=False, float_format='%.3e')
    with pd.option_context('display.float_format', '{:.3e}'.format):
        print(res.to_string(index=False))

    # Pares (v, w) de structured: el error de cada par es el mismo en cualquier modo
    pares = np.array([[0.20, 0.0], [0.0, 0.0], [0.0, 0.80]])
    ep, _, _ = recon_errors(model, std.transform(pares)[:, None, :])
    print('Error por par (v, w):',
          ', '.join(f'({v:.2f}, {w:.2f}) -> {e:.1e}' for (v, w), e in zip(pares, ep)))

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'width_ratios': [2.2, 1]})
    for mode, (t, e) in curves.items():
        axs[0].plot(t, e + EPS, lw=1.0, color=MODE_COLORS[mode], label=mode)
    axs[0].set_yscale('log')
    axs[0].set_xlabel('tiempo (s)')
    axs[0].set_ylabel('error de reconstrucción + 1e-12')
    axs[0].set_title('Autoencoder puntual: error contra el tiempo', fontsize=10)
    axs[0].legend(fontsize=8, frameon=False, loc='center right')
    data = [curves[m][1] + EPS for m in files]
    bp = axs[1].boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    for patch, m in zip(bp['boxes'], files):
        patch.set_facecolor(MODE_COLORS[m])
        patch.set_alpha(0.6)
    axs[1].set_xticks([1, 2, 3], list(files), fontsize=8)
    axs[1].set_yscale('log')
    axs[1].set_title('Distribución por modo', fontsize=10)
    for ax in axs:
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(f'{a.out_dir}/act1_error.png', dpi=130)
    print(f'Figuras y tabla en {a.out_dir}/')


if __name__ == '__main__':
    main()
