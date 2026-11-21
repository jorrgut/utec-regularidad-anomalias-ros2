#!/usr/bin/env python3
"""
Actividad 2: Rolling NLL y detección (Unidad 4).

Señal de bloques regulares (ciclo de 40 muestras) con un tramo anómalo conocido
(muestras 500 a 700). Predictor: persistencia.

  - sigma se estima en el PERÍODO DE REFERENCIA (muestras 0 a --ref-end).
  - Rolling MSE y Rolling NLL causales para cada tamaño de ventana.
  - Umbral mu + k*sigma (y, para comparar, el robusto mediana + k*1.4826*MAD)
    calibrado en la referencia, con persistencia de --consec ventanas.
  - Retardo, falsas alarmas y fin de la alarma, contra la verdad conocida.
  - Comparación con el NLL calculado estimando sigma EN CADA VENTANA (el error común).

Salidas en outputs_u4a2/:
    detection_stats.csv          una fila por ventana y tipo de umbral
    rolling_nll_analysis.png     señal, error, Rolling MSE y Rolling NLL (ventana --main)
    window_size_comparison.png   Rolling NLL para cada tamaño de ventana
    sigma_comparison.png         NLL con sigma de referencia vs. sigma de la propia ventana

Uso:
    python3 u4_activity2.py
    python3 u4_activity2.py --windows 25 40 50 100 --k 3 --consec 3
"""
import argparse
import os

import numpy as np
import pandas as pd

from u4_tools import (predict_persistence, estimate_sigma, nll_gauss, rolling_mean_causal,
                      threshold, persistent_alarm, evaluate_detection, alarm_intervals)
from generate_signals import make_anomaly_signal
from u4_plotstyle import plt, style_axes, save, C_DATA, C_REF, C_FLAG, C_SHADE, C_ALT

A_START, A_END = 500, 700


def shade(ax):
    ax.axvspan(A_START, A_END, color=C_SHADE, zorder=0, label='anomalía real')


def self_sigma_nll(e, window, floor=1e-10):
    """El ERROR COMÚN, a propósito: sigma estimada en cada ventana."""
    out = np.full(len(e), np.nan)
    for t in range(window - 1, len(e)):
        w = e[t - window + 1:t + 1]
        s = max(np.std(w), floor)
        out[t] = np.mean(nll_gauss(w, s))
    return out


def main():
    ap = argparse.ArgumentParser(description='Actividad 2 de la Unidad 4')
    ap.add_argument('--windows', nargs='+', type=int, default=[25, 40, 50, 100])
    ap.add_argument('--main', type=int, default=50, help='ventana de la figura principal')
    ap.add_argument('--ref-end', type=int, default=300, help='fin del período de referencia')
    ap.add_argument('--k', type=float, default=3.0)
    ap.add_argument('--consec', type=int, default=3, help='ventanas consecutivas para alarmar')
    ap.add_argument('--outdir', default='outputs_u4a2')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print('=' * 70)
    print('ACTIVIDAD 2: Rolling NLL y detección')
    print('=' * 70)

    x, truth = make_anomaly_signal()
    n = len(x)
    _, e = predict_persistence(x)
    e = np.r_[0.0, e]            # e[t] = error al predecir x[t]; e[0] no existe -> 0
    sigma = estimate_sigma(e[1:args.ref_end])
    print(f'\nSeñal: {n} muestras | anomalía real: {A_START}-{A_END}')
    print(f'Período de referencia: 0-{args.ref_end} | sigma de referencia = {sigma:.4f} m/s')

    nll = nll_gauss(e, sigma)
    rows, series = [], {}
    for N in args.windows:
        r_mse = rolling_mean_causal(e ** 2, N)
        r_nll = rolling_mean_causal(nll, N)
        series[N] = (r_mse, r_nll)
        ref = r_nll[N - 1:args.ref_end]
        for kind, robust in (('mu+k*sigma', False), ('robusto (MAD)', True)):
            thr = threshold(ref, k=args.k, robust=robust)
            above = np.nan_to_num(r_nll, nan=-np.inf) > thr
            alarm = persistent_alarm(above, args.consec)
            ev = evaluate_detection(alarm, A_START, A_END)
            ints = alarm_intervals(alarm)
            last = max((b for a, b in ints), default=None)
            mad = np.median(np.abs(ref - np.median(ref)))
            rows.append({
                'ventana': N, 'umbral': kind, 'valor_umbral': thr,
                'mad_referencia': mad,
                'detectada': ev['detected'], 'retardo': ev['delay'],
                'muestras_alarma_antes': ev['false_alarms_before'],
                'fin_alarma': last,
                'retorno_tras_anomalia': (last - A_END) if last is not None else None,
                'n_intervalos_alarma': len(ints),
            })
            if N == args.main and not robust:
                main_thr, main_alarm = thr, alarm
    stats = pd.DataFrame(rows)
    stats.to_csv(os.path.join(args.outdir, 'detection_stats.csv'), index=False,
                 float_format='%.6f')

    print(f'\nUmbral calibrado en la referencia (k = {args.k:g}), persistencia de '
          f'{args.consec} ventanas:\n')
    print(f'  {"N":>4} {"umbral":<15} {"retardo":>8} {"muestras en alarma antes":>25} '
          f'{"vuelve a normal":>16}')
    for _, r in stats.iterrows():
        d = '—' if pd.isna(r['retardo']) else f'{int(r["retardo"])}'
        back = '—' if pd.isna(r['retorno_tras_anomalia']) else f'+{int(r["retorno_tras_anomalia"])}'
        print(f'  {r["ventana"]:>4} {r["umbral"]:<15} {d:>8} {int(r["muestras_alarma_antes"]):>25} '
              f'{back:>16}')
        if r['umbral'].startswith('robusto') and r['mad_referencia'] < 1e-12:
            print(f'       (MAD de referencia = 0: el indicador tiene muchos empates)')

    # Figura principal
    r_mse, r_nll = series[args.main]
    k = np.arange(n)
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(k, x, color=C_DATA, linewidth=0.8, drawstyle='steps-post')
    axes[0].set_ylabel('v (m/s)')
    axes[0].set_title('Señal', loc='left', fontsize=10)
    axes[1].plot(k, e, color=C_DATA, linewidth=0.8)
    axes[1].set_ylabel('error (m/s)')
    axes[1].set_title('Error de persistencia', loc='left', fontsize=10)
    axes[2].plot(k, r_mse, color=C_DATA, linewidth=1.2)
    axes[2].set_ylabel('MSE (m²/s²)')
    axes[2].set_title(f'Rolling MSE (N = {args.main}, causal)', loc='left', fontsize=10)
    axes[3].plot(k, r_nll, color=C_DATA, linewidth=1.2, label='Rolling NLL')
    axes[3].axhline(main_thr, color=C_REF, linestyle='--', linewidth=1,
                    label=f'umbral μ + {args.k:g}σ')
    axes[3].scatter(k[main_alarm], r_nll[main_alarm], s=8, color=C_FLAG, zorder=3,
                    label=f'alarma ({args.consec} ventanas seguidas)')
    axes[3].set_ylabel('NLL (nats)')
    axes[3].set_title(f'Rolling NLL (N = {args.main}, σ de referencia = {sigma:.3f} m/s)',
                      loc='left', fontsize=10)
    axes[3].set_xlabel('muestra (cada ventana se asigna a su instante final)')
    for ax in axes:
        shade(ax)
        ax.axvline(args.ref_end, color=C_REF, linewidth=0.8, linestyle=':')
        style_axes(ax)
    axes[0].text(args.ref_end / 2, 0.21, 'referencia', ha='center', fontsize=8, color='#444444')
    axes[3].legend(fontsize=8, frameon=False, loc='upper left')
    save(fig, os.path.join(args.outdir, 'rolling_nll_analysis.png'))

    # Comparación de ventanas
    fig, axes = plt.subplots(len(args.windows), 1, figsize=(11, 2.2 * len(args.windows)),
                             sharex=True)
    for ax, N in zip(np.atleast_1d(axes), args.windows):
        _, r_n = series[N]
        thr = stats[(stats['ventana'] == N) & (stats['umbral'] == 'mu+k*sigma')]['valor_umbral'].iloc[0]
        shade(ax)
        ax.plot(k, r_n, color=C_DATA, linewidth=1.1)
        ax.axhline(thr, color=C_REF, linestyle='--', linewidth=1)
        ax.set_ylabel('NLL (nats)')
        ax.set_title(f'N = {N}', loc='left', fontsize=10)
        style_axes(ax)
    np.atleast_1d(axes)[-1].set_xlabel('muestra')
    save(fig, os.path.join(args.outdir, 'window_size_comparison.png'))

    # sigma de referencia vs. sigma de la ventana
    self_nll = self_sigma_nll(e, args.main)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    shade(axes[0])
    axes[0].plot(k, r_nll, color=C_DATA, linewidth=1.1)
    axes[0].set_title('σ fija, estimada en la referencia (correcto)', loc='left', fontsize=10)
    shade(axes[1])
    axes[1].plot(k, self_nll, color=C_ALT, linewidth=1.1)
    axes[1].set_title('σ estimada en cada ventana (error común): ≈ ½ + ½·ln(2πσ²_ventana)',
                      loc='left', fontsize=10)
    for ax in axes:
        ax.set_ylabel('NLL (nats)')
        style_axes(ax)
    axes[1].set_xlabel('muestra')
    save(fig, os.path.join(args.outdir, 'sigma_comparison.png'))

    c = np.corrcoef(*[s[~np.isnan(r_mse)] for s in (r_mse, r_nll)])[0, 1]
    print(f'\nCorrelación Rolling MSE vs. Rolling NLL (σ fija, N = {args.main}): {c:.4f}')
    print(f'NLL con σ de la ventana: mínimo {np.nanmin(self_nll):.1f} nats '
          '(ventanas sin errores: σ -> piso numérico)')
    print(f'\nTabla: {args.outdir}/detection_stats.csv | figuras: {args.outdir}/')


if __name__ == '__main__':
    main()
