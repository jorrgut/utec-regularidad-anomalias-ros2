#!/usr/bin/env python3
"""
Error de predicción y NLL sobre /cmd_vel re-muestreado (Unidad 4, Actividad 4).

Entrada: CSV re-muestreado de la Unidad 2 (t_sec, v, w; NaN en los huecos).

Pasos:
  1. Período de referencia: la primera fracción --train-frac del registro.
  2. Predictores por señal: persistencia y AR(--ar-order), ajustado en la referencia.
  3. sigma de referencia de cada señal (con el predictor elegido en --predictor).
  4. NLL por señal y NLL conjunto (suma de v y w: cada uno normalizado por su sigma).
  5. Rolling NLL conjunto causal (ventana --window), umbral mu + k*sigma calibrado en la
     referencia y alarma con persistencia de --consec ventanas.

Salidas (prefijo --out):
    <out>_series.csv     una fila por muestra: errores, NLL, rolling y alarma
    <out>_summary.txt    métricas, sigma, umbral e intervalos de alarma
    <out>.png            señales, errores y Rolling NLL conjunto

Uso:
    python3 predict_cmdvel.py --csv data/cmdvel_20hz_zoh.csv --out outputs_u4a4/zoh
    python3 predict_cmdvel.py --csv data/cmdvel_20hz.csv --out outputs_u4a4/linear
    python3 predict_cmdvel.py --csv data/replay_20hz_zoh.csv --out outputs_u4a4/replay \
        --truth 200 210
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

from u4_tools import (estimate_sigma, nll_gauss, rolling_mean_causal_nan, threshold,
                      persistent_alarm, alarm_intervals, error_metrics)
from u4_plotstyle import plt, style_axes, save, C_DATA, C_ALT, C_REF, C_FLAG, C_SHADE

warnings.filterwarnings('ignore', category=RuntimeWarning)
UNITS = {'v': 'm/s', 'w': 'rad/s'}


def lag_rows(x, p):
    """Regresores [x_{t-1}..x_{t-p}, 1] y objetivo x_t para t = p..T-1 (NaN donde falte)."""
    T = len(x)
    X = np.column_stack([x[p - k:T - k] for k in range(1, p + 1)] + [np.ones(T - p)])
    return X, x[p:]


def ar_errors(x, p, train_end):
    """AR(p) ajustado con las filas completas de la referencia; error para cada t (NaN si no hay)."""
    X, y = lag_rows(x, p)
    ok = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    tr = ok & (np.arange(p, len(x)) < train_end)
    if tr.sum() < 5 * (p + 1):
        sys.exit(f'[error] muy pocas filas completas para ajustar AR({p}) en la referencia')
    coef, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
    e = np.full(len(x), np.nan)
    e[p:] = y - X @ coef
    return e, coef


def persistence_errors(x):
    e = np.full(len(x), np.nan)
    e[1:] = x[1:] - x[:-1]
    return e


def main():
    ap = argparse.ArgumentParser(description='Error de predicción y NLL sobre /cmd_vel')
    ap.add_argument('--csv', required=True, help='CSV re-muestreado (t_sec, v, w)')
    ap.add_argument('--signals', nargs='+', default=['v', 'w'])
    ap.add_argument('--train-frac', type=float, default=1 / 3,
                    help='fracción inicial usada como referencia (entrenamiento, sigma, umbral)')
    ap.add_argument('--ar-order', type=int, default=4)
    ap.add_argument('--predictor', choices=['persistencia', 'ar'], default='persistencia',
                    help='predictor usado para el NLL y la detección')
    ap.add_argument('--window', type=int, default=100, help='ventana del Rolling NLL (muestras)')
    ap.add_argument('--k', type=float, default=3.0)
    ap.add_argument('--consec', type=int, default=3)
    ap.add_argument('--sigma-floor', type=float, default=1e-3,
                    help='piso de sigma (en unidades de la señal)')
    ap.add_argument('--truth', nargs=2, type=float, metavar=('INICIO', 'FIN'), default=None,
                    help='intervalo anómalo conocido (s desde el inicio), para evaluar')
    ap.add_argument('--out', default='outputs_u4a4/cmdvel')
    args = ap.parse_args()

    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.read_csv(args.csv)
    missing = [c for c in ['t_sec', *args.signals] if c not in df.columns]
    if missing:
        sys.exit(f'[error] faltan columnas: {missing}')
    t = df['t_sec'].to_numpy(dtype=float)
    t = t - t[0]
    dt = np.diff(t)
    if np.ptp(dt) > 1e-6:
        print('[aviso] el paso temporal no es constante: usá el CSV re-muestreado de la Unidad 2.')
    step = float(np.median(dt))
    n = len(t)
    train_end = int(n * args.train_frac)

    res = {'t': t}
    lines = [f'Entrada: {args.csv}', f'Muestras: {n} (paso {step:.3f} s) | referencia: 0-'
             f'{t[train_end]:.1f} s ({100 * args.train_frac:.0f}%)',
             f'Predictor para NLL y detección: {args.predictor}'
             + (f' (AR({args.ar_order}))' if args.predictor == 'ar' else ''), '']
    nll_joint = np.zeros(n)
    for s in args.signals:
        x = df[s].to_numpy(dtype=float)
        e_p = persistence_errors(x)
        e_a, _ = ar_errors(x, args.ar_order, train_end)
        test = slice(train_end, n)
        mp = error_metrics(e_p[test][~np.isnan(e_p[test])])
        ma = error_metrics(e_a[test][~np.isnan(e_a[test])])
        sk = 1 - ma['mse'] / mp['mse'] if mp['mse'] > 0 else np.nan
        n_chg = int(np.nansum(np.abs(e_p) > 1e-9))
        lines.append(f'[{s}] prueba: RMSE persistencia {mp["rmse"]:.4f} | RMSE AR({args.ar_order}) '
                     f'{ma["rmse"]:.4f} {UNITS.get(s, "")} | skill {sk:+.3f}')
        lines.append(f'     muestras con error de persistencia distinto de 0: {n_chg} de '
                     f'{int(np.sum(~np.isnan(e_p)))}')
        e = e_a if args.predictor == 'ar' else e_p
        sig = estimate_sigma(e[:train_end][~np.isnan(e[:train_end])], floor=args.sigma_floor)
        lines.append(f'     sigma de referencia: {sig:.4f} {UNITS.get(s, "")}')
        nll = nll_gauss(e, sig)
        nll_joint = nll_joint + nll          # NaN se propaga: el conjunto exige ambas señales
        res.update({f'e_pers_{s}': e_p, f'e_ar_{s}': e_a, f'nll_{s}': nll})

    roll = rolling_mean_causal_nan(nll_joint, args.window)
    ref = roll[:train_end]
    thr = threshold(ref, k=args.k)
    above = np.nan_to_num(roll, nan=-np.inf) > thr
    alarm = persistent_alarm(above, args.consec)
    res.update({'nll_joint': nll_joint, 'rolling_nll_joint': roll, 'alarm': alarm.astype(int)})
    pd.DataFrame(res).to_csv(f'{args.out}_series.csv', index=False, float_format='%.6f')

    ints = alarm_intervals(alarm)
    lines += ['', f'Rolling NLL conjunto: ventana {args.window} muestras '
              f'({args.window * step:.1f} s), umbral mu + {args.k:g} sigma = {thr:.3f} nats, '
              f'persistencia {args.consec}',
              f'Intervalos de alarma: {len(ints)}']
    for a, b in ints[:15]:
        lines.append(f'  {t[a]:7.1f} s - {t[min(b, n - 1)]:7.1f} s')
    if args.truth:
        a0, a1 = args.truth
        in_truth = (t >= a0) & (t < a1)
        hits = np.flatnonzero(alarm & (t >= a0))
        delay = (t[hits[0]] - a0) if len(hits) else None
        fa = sum(1 for a, b in ints if t[min(b, n - 1)] < a0 or t[a] > a1 + args.window * step)
        lines += ['', f'Verdad conocida: {a0:g}-{a1:g} s',
                  f'  detectada: {"sí" if delay is not None and delay <= (a1 - a0) + args.window * step else "no"}'
                  + (f' | retardo {delay:.2f} s' if delay is not None else ''),
                  f'  intervalos de alarma ajenos a la anomalía: {fa}']
    text = '\n'.join(lines) + '\n'
    with open(f'{args.out}_summary.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)

    # Figura
    ns = len(args.signals)
    fig, axes = plt.subplots(2 * ns + 1, 1, figsize=(11, 2.0 * (2 * ns + 1)), sharex=True)
    for i, s in enumerate(args.signals):
        x = df[s].to_numpy(dtype=float)
        axes[2 * i].plot(t, x, color=C_DATA, linewidth=0.9)
        axes[2 * i].set_ylabel(UNITS.get(s, ''))
        axes[2 * i].set_title(f'Señal {s}', loc='left', fontsize=10)
        e = res[f'e_ar_{s}'] if args.predictor == 'ar' else res[f'e_pers_{s}']
        axes[2 * i + 1].plot(t, e, color=C_DATA, linewidth=0.8)
        axes[2 * i + 1].set_ylabel(UNITS.get(s, ''))
        axes[2 * i + 1].set_title(f'Error de predicción de {s} ({args.predictor})',
                                  loc='left', fontsize=10)
    ax = axes[-1]
    ax.plot(t, roll, color=C_DATA, linewidth=1.2, label='Rolling NLL conjunto')
    ax.axhline(thr, color=C_REF, linestyle='--', linewidth=1, label=f'umbral μ + {args.k:g}σ')
    if alarm.any():
        ax.scatter(t[alarm], roll[alarm], s=6, color=C_FLAG, zorder=3, label='alarma')
    ax.set_ylabel('NLL (nats)')
    ax.set_title(f'Rolling NLL conjunto ({" + ".join(args.signals)}), ventana de '
                 f'{args.window * step:.1f} s', loc='left', fontsize=10)
    ax.set_xlabel('tiempo (s)')
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    for a in axes:
        a.axvline(t[train_end], color=C_REF, linewidth=0.8, linestyle=':')
        if args.truth:
            a.axvspan(args.truth[0], args.truth[1], color=C_SHADE, zorder=0)
        style_axes(a)
    save(fig, f'{args.out}.png')
    print(f'Serie: {args.out}_series.csv | figura: {args.out}.png')


if __name__ == '__main__':
    main()
