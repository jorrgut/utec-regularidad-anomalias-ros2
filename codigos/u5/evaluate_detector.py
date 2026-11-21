#!/usr/bin/env python3
"""
Evaluación fuera de línea del detector (Unidad 5, Actividades 1 y 3).

Recalcula el score de un registro con la calibración dada y lo evalúa POR EVENTOS
contra las anomalías conocidas, para:
  - la configuración de la calibración (tau, --K);
  - una grilla de umbrales (cuantiles del score nominal) x valores de K;
  - una grilla de CUSUM (k, h).

Salidas en --outdir:
    grid_consecutive.csv   una fila por (tau, K): detectados, retardo, falsas/hora
    grid_cusum.csv         una fila por (k, h)
    run_score.png          indicadores, contribuciones al score, score con tau y alarmas
    tradeoff.png           falsas alarmas por hora vs. retardo medio, por K

Uso:
    # validación nominal (sin eventos): solo falsas alarmas
    python3 evaluate_detector.py --log-dir outputs_u5/validation --calib outputs_u5/calib.yaml
    # corrida con eventos (pares inicio fin, en s desde el comienzo del registro)
    python3 evaluate_detector.py --log-dir outputs_u5/run1 --calib outputs_u5/calib.yaml \
        --events 120 140 300 320
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from u5_core import (INDICATORS, load_calibration, score, k_consecutive, cusum,
                     evaluate_events, intervals)
from u5_offline import (load_indicators, read_events, style_axes, IND_COLORS, C_DATA, C_REF,
                        C_FLAG, C_SHADE)


def main():
    ap = argparse.ArgumentParser(description='Evaluación por eventos del detector')
    ap.add_argument('--log-dir', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--events', nargs='*', type=float, default=[],
                    help='pares inicio fin, en el MISMO reloj que la columna t de los CSV '
                         '(sim time con ROS; s desde el inicio con metrics_from_csv.py)')
    ap.add_argument('--events-file', default=None,
                    help='archivo con una línea "inicio fin" por evento (el que escribe '
                         'cmdvel_attacker); por defecto, <log-dir>/events.txt si existe')
    ap.add_argument('--K', type=int, default=3, help='persistencia de la configuración principal')
    ap.add_argument('--K-grid', nargs='+', type=int, default=[1, 3, 5, 10])
    ap.add_argument('--tolerance', type=float, default=None,
                    help='margen tras cada evento (s); por defecto, la ventana de 5 s')
    ap.add_argument('--warmup', type=float, default=None,
                    help='segundos iniciales sin decidir; por defecto, el de la calibración')
    ap.add_argument('--outdir', default=None, help='por defecto, <log-dir>/eval')
    args = ap.parse_args()

    if len(args.events) % 2:
        sys.exit('[error] --events necesita pares inicio fin')
    events = [(args.events[i], args.events[i + 1]) for i in range(0, len(args.events), 2)]
    ef = args.events_file or os.path.join(args.log_dir, 'events.txt')
    if not events and os.path.exists(ef):
        events = read_events(ef)
        print(f'Eventos leídos de {ef}')
    cfg, centers, weights, direction = load_calibration(args.calib)
    tol = args.tolerance if args.tolerance is not None else 5.0
    warm = args.warmup if args.warmup is not None else cfg.get('warmup_sec', 10.0)
    outdir = args.outdir or os.path.join(args.log_dir, 'eval')
    os.makedirs(outdir, exist_ok=True)

    df = load_indicators(args.log_dir)
    t = df['t'].to_numpy()
    S, contrib = score({k: df[k].to_numpy() for k in INDICATORS}, centers, weights, direction,
                       cfg.get('z_max'))
    t_rel = df['t_rel'].to_numpy()
    S = np.where(t_rel < warm, 0.0, S)    # calentamiento: el detector no decide
    dur_h = (t[-1] - t[0]) / 3600

    tau = cfg['tau']
    alarm = k_consecutive(S >= tau, args.K)
    main_ev = evaluate_events(t, alarm, events, tol)

    print(f'Registro: {args.log_dir} | t de {t[0]:.1f} a {t[-1]:.1f} s | calibración: {args.calib} '
          f'(p = {cfg["p"]:g}, tau = {tau:.3f})')
    print(f'Eventos conocidos: {events if events else "ninguno (corrida nominal)"}\n')
    print(f'Configuración principal (tau = {tau:.3f}, K = {args.K}):')
    if events:
        d = main_ev['retardo_medio']
        print(f'  eventos detectados: {main_ev["detectados"]} de {main_ev["eventos"]}'
              + (f' | retardo medio {d:.2f} s' if d is not None else ''))
    print(f'  falsas alarmas: {main_ev["falsas_alarmas"]} '
          f'({main_ev["falsas_por_hora"]:.1f} por hora) | tiempo nominal en alarma: '
          f'{100 * main_ev["frac_alarma_nominal"]:.1f}%')
    for a0, a1 in main_ev['intervalos_falsos'][:10]:
        i0 = int(np.searchsorted(t, a0))
        top = max(contrib, key=lambda k: contrib[k][i0])
        print(f'    falsa alarma {a0:7.1f}-{a1:7.1f} s (indicador dominante: {top})')
    for (e0, e1) in events:
        sel = (t >= e0) & (t <= e1 + tol) & alarm
        if sel.any():
            i0 = np.flatnonzero(sel)[0]
            top = max(contrib, key=lambda k: contrib[k][i0])
            print(f'    evento {e0:g}-{e1:g} s: primera alarma en {t[i0]:.1f} s '
                  f'(indicador dominante: {top})')
        else:
            print(f'    evento {e0:g}-{e1:g} s: NO detectado')

    # Grilla tau x K
    rows = []
    for q, tq in cfg['tau_quantiles'].items():
        for K in args.K_grid:
            ev = evaluate_events(t, k_consecutive(S >= tq, K), events, tol)
            rows.append({'cuantil': float(q), 'tau': tq, 'K': K,
                         'detectados': ev['detectados'], 'eventos': ev['eventos'],
                         'retardo_medio_s': ev['retardo_medio'],
                         'falsas_alarmas': ev['falsas_alarmas'],
                         'falsas_por_hora': ev['falsas_por_hora'],
                         'pct_tiempo_nominal_en_alarma': 100 * ev['frac_alarma_nominal']})
    grid = pd.DataFrame(rows)
    grid.to_csv(os.path.join(outdir, 'grid_consecutive.csv'), index=False, float_format='%.3f')

    # CUSUM sobre el score CENTRADO en su mediana nominal, en unidades de su escala nominal
    S_c = (S - cfg.get('score_median', 0.0)) / cfg.get('score_scale', 1.0)
    S_c = np.where(t_rel < warm, 0.0, S_c)
    rows = []
    for k in (0.5, 1.0, 2.0):
        for h in (5.0, 10.0, 20.0, 40.0):
            ev = evaluate_events(t, cusum(S_c, k, h), events, tol)
            rows.append({'k': k, 'h': h, 'detectados': ev['detectados'], 'eventos': ev['eventos'],
                         'retardo_medio_s': ev['retardo_medio'],
                         'falsas_alarmas': ev['falsas_alarmas'],
                         'falsas_por_hora': ev['falsas_por_hora'],
                         'pct_tiempo_nominal_en_alarma': 100 * ev['frac_alarma_nominal']})
    gcus = pd.DataFrame(rows)
    gcus.to_csv(os.path.join(outdir, 'grid_cusum.csv'), index=False, float_format='%.3f')

    print('\nGrilla tau x K (falsas alarmas por hora | eventos detectados | retardo medio):')
    piv = grid.pivot(index='cuantil', columns='K', values='falsas_por_hora')
    print(piv.to_string(float_format=lambda v: f'{v:6.1f}'))
    if events:
        print('\nEventos detectados:')
        print(grid.pivot(index='cuantil', columns='K', values='detectados').to_string())
        print('\nRetardo medio (s):')
        print(grid.pivot(index='cuantil', columns='K', values='retardo_medio_s')
              .to_string(float_format=lambda v: f'{v:6.2f}', na_rep='     —'))
    print('\nCUSUM sobre el score centrado (falsas/h | % tiempo nominal en alarma | '
          'detectados | retardo):')
    for _, r in gcus.iterrows():
        d = '—' if pd.isna(r['retardo_medio_s']) else f'{r["retardo_medio_s"]:.2f} s'
        print(f'  k = {r["k"]:<4g} h = {r["h"]:<5g}  {r["falsas_por_hora"]:6.1f}  '
              f'{r["pct_tiempo_nominal_en_alarma"]:5.1f}%  '
              f'{int(r["detectados"])}/{int(r["eventos"])}  {d}')

    # Figuras
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for name in INDICATORS:
        c, s = centers[name]
        z = (df[name] - c) / s
        zm = cfg.get('z_max')
        if zm:
            z = z.clip(-zm, zm)
        axes[0].plot(t, z, color=IND_COLORS[name], linewidth=0.9, label=name)
    axes[0].set_ylabel('z robusto' + (f' (saturado en ±{cfg["z_max"]:g})' if cfg.get('z_max') else ''))
    axes[0].set_title('Indicadores normalizados', loc='left', fontsize=10)
    axes[0].legend(fontsize=8, frameon=False, loc='upper left', ncol=5)
    bottom = np.zeros(len(t))
    for name in contrib:
        axes[1].fill_between(t, bottom, bottom + contrib[name], color=IND_COLORS[name],
                             linewidth=0, label=name)
        bottom = bottom + contrib[name]
    axes[1].set_ylabel('contribución')
    axes[1].set_title('Contribución de cada indicador al score (apilado)', loc='left', fontsize=10)
    axes[1].legend(fontsize=8, frameon=False, loc='upper left', ncol=5)
    axes[2].plot(t, S, color=C_DATA, linewidth=1, label='score')
    axes[2].axhline(tau, color=C_REF, linestyle='--', linewidth=1, label=f'τ = {tau:.2f}')
    if alarm.any():
        axes[2].scatter(t[alarm], S[alarm], s=6, color=C_FLAG, zorder=3,
                        label=f'alarma (K = {args.K})')
    axes[2].set_ylabel('score')
    axes[2].set_title('Score, umbral y alarmas', loc='left', fontsize=10)
    axes[2].legend(fontsize=8, frameon=False, loc='upper left')
    axes[2].set_xlabel('tiempo (s)')
    for ax in axes:
        for (e0, e1) in events:
            ax.axvspan(e0, e1, color=C_SHADE, zorder=0)
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'run_score.png'), dpi=150)
    plt.close(fig)

    if events:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = ['#2a6fb0', '#2f9e6e', '#e07b39', '#7a5195']
        for c, K in zip(colors, args.K_grid):
            g = grid[grid['K'] == K].sort_values('cuantil')
            ax.plot(g['retardo_medio_s'], g['falsas_por_hora'], marker='o', markersize=5,
                    color=c, linewidth=1.2, label=f'K = {K}')
            for _, r in g.iterrows():
                if r['cuantil'] in (0.9, 0.99, 0.999) and not pd.isna(r['retardo_medio_s']):
                    ax.annotate(f'Q{r["cuantil"]:g}', (r['retardo_medio_s'], r['falsas_por_hora']),
                                textcoords='offset points', xytext=(5, 3), fontsize=7,
                                color='#555555')
        ax.set_xlabel('retardo medio de detección (s)')
        ax.set_ylabel('falsas alarmas por hora')
        ax.set_title('Compromiso: cada punto es un umbral (cuantil del score nominal)',
                     loc='left', fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc='upper right')
        style_axes(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, 'tradeoff.png'), dpi=150)
        plt.close(fig)

    print(f'\nTablas y figuras en {outdir}/')


if __name__ == '__main__':
    main()
