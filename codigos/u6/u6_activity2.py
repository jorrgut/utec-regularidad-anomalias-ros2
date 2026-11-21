#!/usr/bin/env python3
"""
Actividad 2: autoencoder de VENTANA y elección de T (apunte, sección "Autoencoder de ventana").

Para cada T y cada semilla:
  1. entrena un WindowAE (2T -> 64 -> L -> 64 -> 2T) con el primer 80 % de train_structured;
  2. fija tau en el cuantil 0.99 del error sobre el último 20 % (validación);
  3. mide, en la prueba nominal, shuffle_blocks e iid, la fracción de ventanas con error > tau
     y la mediana del error.
Repite lo mismo con PCA (autoencoder lineal, misma L) como línea base, y para T = 40 barre
el tamaño del latente L.

Lo que deberías ver: en la prueba nominal, fracción ~0.01 para todo T; en shuffle_blocks,
una fracción que crece claramente con T.

Salidas (en --out-dir):
    act2_resultados.csv   una fila por (modelo, T, L, semilla, modo)
    act2_resumen.csv      promedio y desvío entre semillas
    act2_fraccion_T.png   fracción sobre tau contra T, por modo (AE y PCA)
    act2_error_T40.png    error contra el tiempo con T = 40 en los tres modos

Uso:
    python3 u6_activity2.py                          # T = 10 40 80, semillas 0 1 2
    python3 u6_activity2.py --T 10 40 --seeds 0      # versión rápida
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from u6_data import (MODE_COLORS, Standardizer, build_features, load_cmdvel, make_windows,
                     resample_zoh, split_blocks, style_axes)
from u6_models import PCAWindow, WindowAE, n_params, recon_errors, set_seed, train_model

TESTS = {'structured': 'test_structured.csv', 'shuffle_blocks': 'shuffle_blocks.csv',
         'iid': 'iid.csv'}


def load_all(data_dir):
    Xtr, Xva = split_blocks(build_features(resample_zoh(*load_cmdvel(
        f'{data_dir}/train_structured.csv'))))
    std = Standardizer().fit(Xtr)
    tests = {}
    for mode, fn in TESTS.items():
        g = resample_zoh(*load_cmdvel(f'{data_dir}/{fn}'))
        tests[mode] = (g['t'].to_numpy(), std.transform(build_features(g)))
    return std.transform(Xtr), std.transform(Xva), tests


def evaluate(err_fn, Wva, tests, T):
    tau = float(np.quantile(err_fn(Wva), 0.99))
    out, curves = [], {}
    for mode, (t, X) in tests.items():
        W, idx = make_windows(X, T)
        e = err_fn(W)
        out.append({'modo': mode, 'tau': tau, 'frac_sobre_tau': float(np.mean(e > tau)),
                    'error_mediana': float(np.median(e))})
        curves[mode] = (t[idx], e)
    return out, curves, tau


def main():
    ap = argparse.ArgumentParser(description='Actividad 2: autoencoder de ventana')
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--out-dir', default='outputs_u6/act2')
    ap.add_argument('--T', type=int, nargs='+', default=[10, 40, 80])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--latent', type=int, default=4)
    ap.add_argument('--latent-sweep', type=int, nargs='*', default=[2, 4, 16],
                    help='valores de L a probar con T = 40 (vacío para omitir)')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    Xtr, Xva, tests = load_all(a.data_dir)
    rows, keep_curves = [], None

    configs = [(T, a.latent) for T in a.T]
    if 40 in a.T or a.latent_sweep:
        configs += [(40, L) for L in a.latent_sweep if (40, L) not in configs]

    for T, L in configs:
        Wtr, _ = make_windows(Xtr, T)
        Wva, _ = make_windows(Xva, T)
        # Línea base lineal (determinística: una sola vez)
        pca = PCAWindow(L).fit(Wtr)
        res_pca, _, _ = evaluate(pca.errors, Wva, tests, T)
        for r in res_pca:
            rows.append({'modelo': 'PCA', 'T': T, 'L': L, 'semilla': 0, **r})
        for seed in a.seeds:
            set_seed(seed)
            model = WindowAE(T, d=2, hidden=64, latent=L)
            model, hist = train_model(model, Wtr, Wtr, Wva, Wva, seed=seed)
            fn = lambda W: recon_errors(model, W)[0]
            res, curves, tau = evaluate(fn, Wva, tests, T)
            for r in res:
                rows.append({'modelo': 'AE', 'T': T, 'L': L, 'semilla': seed, **r})
            msg = ', '.join(f"{r['modo']} {r['frac_sobre_tau']:.2f}" for r in res)
            print(f'AE  T={T:3d} L={L:2d} semilla {seed}: {n_params(model):6d} parámetros, '
                  f'{len(hist):3d} épocas | fracción > tau: {msg}')
            if T == 40 and L == a.latent and seed == a.seeds[0]:
                keep_curves = (curves, tau)
        msg = ', '.join(f"{r['modo']} {r['frac_sobre_tau']:.2f}" for r in res_pca)
        print(f'PCA T={T:3d} L={L:2d}: varianza explicada {pca.explained:.2f}'
              f'{" " * 17}| fracción > tau: {msg}')

    df = pd.DataFrame(rows)
    df.to_csv(f'{a.out_dir}/act2_resultados.csv', index=False, float_format='%.5f')
    summ = (df.groupby(['modelo', 'T', 'L', 'modo'], sort=False)
              .agg(frac_media=('frac_sobre_tau', 'mean'), frac_desvio=('frac_sobre_tau', 'std'),
                   error_mediana=('error_mediana', 'mean'))
              .reset_index())
    summ.to_csv(f'{a.out_dir}/act2_resumen.csv', index=False, float_format='%.4f')
    print('\nFracción de ventanas con error > tau (promedio entre semillas):')
    piv = summ.pivot_table(index=['modelo', 'T', 'L'], columns='modo', values='frac_media',
                           sort=False)[list(TESTS)]
    print(piv.round(2).to_string())

    # Figura 1: fracción sobre tau contra T (L fijo)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    sub = summ[summ['L'] == a.latent]
    for mode in TESTS:
        for model_name, ls, mk in (('AE', '-', 'o'), ('PCA', '--', 's')):
            s = sub[(sub['modo'] == mode) & (sub['modelo'] == model_name)].sort_values('T')
            if s.empty:
                continue
            ax.errorbar(s['T'], s['frac_media'], yerr=s['frac_desvio'].fillna(0),
                        color=MODE_COLORS[mode], ls=ls, marker=mk, ms=6, lw=2, capsize=3,
                        label=f'{mode} ({model_name})')
    ax.axhline(0.01, color='#999999', lw=1, ls=':')
    ax.text(ax.get_xlim()[1], 0.03, 'esperado en nominal: 0.01', ha='right', fontsize=8,
            color='#666666')
    ax.set_xlabel('longitud de ventana T (muestras a 10 Hz)')
    ax.set_ylabel('fracción de ventanas con error > tau')
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(f'Autoencoder de ventana (L = {a.latent}) y PCA', fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc='center right')
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(f'{a.out_dir}/act2_fraccion_T.png', dpi=130)

    # Figura 2: el error con T = 40 en los tres modos
    if keep_curves is not None:
        curves, tau = keep_curves
        fig, axs = plt.subplots(3, 1, figsize=(10, 6.2), sharex=True, sharey=True)
        for ax, mode in zip(axs, TESTS):
            t, e = curves[mode]
            ax.plot(t, e, lw=0.9, color=MODE_COLORS[mode])
            ax.axhline(tau, color='#c0392b', lw=1, ls='--')
            ax.set_ylabel('error', fontsize=9)
            ax.set_title(f'{mode}: {np.mean(e > tau):.0%} de las ventanas sobre tau',
                         fontsize=9, loc='left')
            ax.set_yscale('symlog', linthresh=max(tau, 1e-6))
            style_axes(ax)
        axs[-1].set_xlabel('tiempo (s)')
        axs[0].text(0.99, 0.85, 'tau (cuantil 0.99 de validación)', color='#c0392b',
                    transform=axs[0].transAxes, ha='right', fontsize=8)
        fig.suptitle(f'Autoencoder de ventana, T = 40, L = {a.latent}', fontsize=10)
        fig.tight_layout()
        fig.savefig(f'{a.out_dir}/act2_error_T40.png', dpi=130)
    print(f'\nTablas y figuras en {a.out_dir}/')


if __name__ == '__main__':
    main()
