#!/usr/bin/env python3
"""
Actividad 1: predicción a un paso y error de predicción (Unidad 4).

Para cada señal sintética (Structured, Shuffled, IID, Periodic):
  - predictor de persistencia;
  - AR(p) con p en --orders, ENTRENADO en la primera mitad y EVALUADO en la segunda;
  - MSE, RMSE, MAE y skill respecto de la persistencia (siempre sobre la mitad de prueba).
También verifica MSE_persistencia = V^2 * r para las señales de dos niveles.

Salidas en outputs_u4a1/:
    metrics_comparison.csv    una fila por señal y predictor
    signals_comparison.png    las cuatro señales (primeras 160 muestras)
    error_series.png          errores de persistencia y del mejor AR en la mitad de prueba
    rmse_by_order.png         RMSE de prueba según el orden del AR
    error_autocorr.png        autocorrelación de los errores de persistencia

Uso:
    python3 u4_activity1.py
    python3 u4_activity1.py --orders 1 2 4 8 16 --n 1000
"""
import argparse
import os

import numpy as np
import pandas as pd

from u4_tools import (predict_persistence, fit_ar, predict_ar, error_metrics, skill,
                      autocorr)
from generate_signals import make_signals, V
from u4_plotstyle import plt, style_axes, save, SIGNAL_COLORS, C_REF, C_ALT


def evaluate(x, orders):
    """Métricas de persistencia y AR(p) sobre la segunda mitad de x."""
    half = len(x) // 2
    test = x[half:]
    _, e_pers = predict_persistence(test)
    sd = float(np.std(test, ddof=1))
    rows = [{'predictor': 'persistencia', 'p': 1, **error_metrics(e_pers), 'skill': 0.0}]
    errors = {'persistencia': e_pers}
    for p in orders:
        coef = fit_ar(x[:half], p)
        # Para comparar sobre las mismas muestras que la persistencia, se predice la
        # mitad de prueba usando como contexto las p muestras anteriores a ella.
        _, e_ar = predict_ar(x[half - p:], coef)
        e_ar = e_ar[-len(e_pers):]
        rows.append({'predictor': f'AR({p})', 'p': p, **error_metrics(e_ar),
                     'skill': skill(e_ar, e_pers)})
        errors[f'AR({p})'] = e_ar
    for r in rows:
        r['nrmse'] = r['rmse'] / sd if sd > 0 else np.nan     # RMSE / desvío de la señal
    return rows, errors


def main():
    ap = argparse.ArgumentParser(description='Actividad 1 de la Unidad 4')
    ap.add_argument('--n', type=int, default=500, help='muestras por señal')
    ap.add_argument('--orders', nargs='+', type=int, default=[1, 2, 4, 8])
    ap.add_argument('--outdir', default='outputs_u4a1')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print('=' * 70)
    print('ACTIVIDAD 1: Predicción a un paso y error de predicción')
    print('=' * 70)

    signals = make_signals(args.n)
    all_rows, all_err = [], {}
    for name, x in signals.items():
        rows, errs = evaluate(x, args.orders)
        for r in rows:
            r['señal'] = name
        all_rows += rows
        all_err[name] = errs

    tab = pd.DataFrame(all_rows)[['señal', 'predictor', 'p', 'bias', 'mse', 'rmse', 'nrmse', 'mae',
                                'skill']]
    tab.to_csv(os.path.join(args.outdir, 'metrics_comparison.csv'), index=False,
               float_format='%.6f')

    # Resumen en la terminal
    print(f'\nSeñales de {args.n} muestras; AR entrenado en la 1.ª mitad, evaluado en la 2.ª.\n')
    best = {}
    for name in signals:
        sub = tab[tab['señal'] == name]
        ar = sub[sub['predictor'] != 'persistencia']
        # el menor orden cuyo RMSE está a menos de 0.1% del mínimo (evita empates numéricos)
        tol = ar['rmse'].min() * 1.001 + 1e-9
        b = ar[ar['rmse'] <= tol].sort_values('p').iloc[0]
        best[name] = b['predictor']
        pers = sub[sub['predictor'] == 'persistencia'].iloc[0]
        print(f'  {name:<11} persistencia RMSE {pers["rmse"]:.4f} (NRMSE {pers["nrmse"]:.2f}) | '
              f'mejor AR: {b["predictor"]:<6} RMSE {b["rmse"]:.4f} (NRMSE {b["nrmse"]:.2f}) | '
              f'skill {b["skill"]:+.3f}')

    print('\nNRMSE = RMSE / desvío estándar de la señal: ~1 significa "no mejor que predecir'
          ' la media".')
    print('\nVerificación de MSE_persistencia = V^2 * r (señales de dos niveles):')
    for name in ('Structured', 'Shuffled'):
        x = signals[name][len(signals[name]) // 2:]
        e = np.diff(x)
        r = np.mean(e != 0)
        print(f'  {name:<11} tasa de cambios r = {r:.4f} | V^2 r = {V**2 * r:.6f} | '
              f'MSE medido = {np.mean(e**2):.6f}')

    # Figura 1: señales
    fig, axes = plt.subplots(len(signals), 1, figsize=(11, 1.9 * len(signals)), sharex=True)
    for ax, (name, x) in zip(axes, signals.items()):
        ax.plot(np.arange(160), x[:160], color=SIGNAL_COLORS[name], linewidth=1.1,
                drawstyle='steps-post' if name in ('Structured', 'Shuffled') else 'default')
        ax.set_ylabel('m/s')
        ax.set_title(name, loc='left', fontsize=10)
        style_axes(ax)
    axes[-1].set_xlabel('muestra')
    save(fig, os.path.join(args.outdir, 'signals_comparison.png'))

    # Figura 2: series de error (prueba)
    fig, axes = plt.subplots(len(signals), 1, figsize=(11, 1.9 * len(signals)), sharex=True)
    for ax, name in zip(axes, signals):
        e_p = all_err[name]['persistencia']
        e_b = all_err[name][best[name]]
        k = np.arange(len(e_p)) + args.n // 2 + 1
        ax.plot(k, e_p, color=SIGNAL_COLORS[name], linewidth=0.9, label='persistencia')
        ax.plot(k, e_b, color=C_REF, linewidth=0.9, linestyle='--', label=best[name])
        ax.set_ylabel('error (m/s)')
        ax.set_title(f'{name}: error en la mitad de prueba', loc='left', fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc='upper right', ncol=2)
        style_axes(ax)
    axes[-1].set_xlabel('muestra')
    save(fig, os.path.join(args.outdir, 'error_series.png'))

    # Figura 3: RMSE según el orden
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for name in signals:
        sub = tab[(tab['señal'] == name) & (tab['predictor'] != 'persistencia')]
        pers = tab[(tab['señal'] == name) & (tab['predictor'] == 'persistencia')]['rmse'].iloc[0]
        ax.plot(sub['p'], sub['rmse'], marker='o', markersize=5, linewidth=1.3,
                color=SIGNAL_COLORS[name], label=name)
        ax.axhline(pers, color=SIGNAL_COLORS[name], linewidth=0.8, linestyle=':')
    ax.set_xscale('log', base=2)
    ax.set_xticks(args.orders)
    ax.set_xticklabels([str(o) for o in args.orders])
    ax.set_xlabel('orden del AR (p)')
    ax.set_ylabel('RMSE de prueba (m/s)')
    ax.set_title('RMSE según el orden (punteado: persistencia de cada señal)',
                 loc='left', fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))
    style_axes(ax)
    save(fig, os.path.join(args.outdir, 'rmse_by_order.png'))

    # Figura 4: autocorrelación del error de persistencia
    fig, ax = plt.subplots(figsize=(8, 4.2))
    lags = 60
    for name in signals:
        r = autocorr(all_err[name]['persistencia'], lags)
        ax.plot(np.arange(lags + 1), r, color=SIGNAL_COLORS[name], linewidth=1.2, label=name)
    ax.axhline(0, color=C_REF, linewidth=0.8)
    ax.set_xlabel('retardo (muestras)')
    ax.set_ylabel('autocorrelación')
    ax.set_title('Autocorrelación del error de persistencia', loc='left', fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    style_axes(ax)
    save(fig, os.path.join(args.outdir, 'error_autocorr.png'))

    print(f'\nTabla completa: {args.outdir}/metrics_comparison.csv')
    print(f'Figuras: {args.outdir}/')


if __name__ == '__main__':
    main()
