#!/usr/bin/env python3
"""
Actividad 4: el error del modelo como indicador del detector de la Unidad 5, evaluado
por eventos (apunte, secciones "Integración con el detector" y "Límites").

Pasos:
  1. Indicadores de la Unidad 5 (H_v, Hcond_v, H_dt, mse_v, mse_w) con OnlineIndicators,
     a 2 Hz y ventanas de 5 s, sobre la corrida de calibración y la de eventos.
  2. Error de cada modelo (guardado con train_model.py) sobre la grilla de 10 Hz, agregado
     en la misma ventana causal de 5 s que los indicadores de la Unidad 5 (promedio, como el
     rolling MSE de mse_v). El error por muestra es muy "picudo" (casi cero, con saltos en
     cada transición): sin agregar, su MAD es diminuto y el z satura en cada transición
     nominal. Con --agg max --agg-sec 0.5 podés ver ese efecto.
  3. Para cada configuración (solo Unidad 5, cada modelo solo, Unidad 5 + cada modelo):
     calibración robusta y tau = cuantil 0.99 del score en la corrida NOMINAL de calibración,
     persistencia K consecutivos, y evaluación por eventos en la corrida con eventos.

La corrida de calibración tiene que ser nominal y distinta de la de entrenamiento del
modelo (por defecto, test_structured.csv).

Salidas (en --out-dir):
    act4_eventos.csv       por configuración y evento: detectado y retardo (si hay eventos)
    act4_alarmas.csv       intervalos de alarma, con el indicador que más aporta en cada uno
    act4_resumen.csv       por configuración: detectados, falsas alarmas por hora, tau
    act4_score.png         score, tau, alarma y eventos para cada configuración

Uso:
    python3 train_model.py --kind ae --T 40 --out models/ae_T40
    python3 train_model.py --kind ae --T 40 --counts --out models/ae_T40_nk
    python3 u6_activity4.py --models models/ae_T40 models/ae_T40_nk
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from u5_core import (INDICATORS, OnlineIndicators, evaluate_events, k_consecutive,
                     quantile_threshold, robust_center_scale, score)
from u6_data import (Standardizer, build_features, load_cmdvel, read_events, resample_zoh,
                     style_axes)
from u6_models import load_model
from train_model import model_errors

PUBLISH_HZ, WINDOW_SEC, WARMUP = 2.0, 5.0, 10.0


def u5_indicators(t, v, w):
    """Mismo cálculo que metrics_from_csv.py de la Unidad 5."""
    ind = OnlineIndicators(window_sec=WINDOW_SEC, min_samples=30)
    rows, i, n = [], 0, len(t)
    tick = 1.0 / PUBLISH_HZ
    while tick <= t[-1]:
        while i < n and t[i] <= tick:
            ind.add(t[i], v[i], w[i])
            i += 1
        rows.append({'t': tick, **ind.compute(tick)})
        tick += 1.0 / PUBLISH_HZ
    return pd.DataFrame(rows)


def model_indicator(prefix, t, v, w, ticks, agg='mean', agg_sec=WINDOW_SEC):
    model, meta = load_model(prefix)
    grid = resample_zoh(t, v, w, rate=meta['rate_hz'])
    X = Standardizer.from_dict(meta['standardizer']).transform(
        build_features(grid, counts='n_k' in meta['channels']))
    e, idx = model_errors(model, meta, X)
    tg = grid['t'].to_numpy()[idx]
    out = np.full(len(ticks), np.nan)
    f = np.mean if agg == 'mean' else np.max
    for k, tk in enumerate(ticks):            # agregado causal en (tk - agg_sec, tk]
        sel = (tg > tk - agg_sec) & (tg <= tk)
        if sel.any():
            out[k] = f(e[sel])
    return out


def main():
    ap = argparse.ArgumentParser(description='Actividad 4: modelo + detector de la Unidad 5')
    ap.add_argument('--models', nargs='+', required=True, help='prefijos de train_model.py')
    ap.add_argument('--calib-csv', default='data/test_structured.csv',
                    help='corrida NOMINAL para calibrar el score (distinta de la de entrenamiento)')
    ap.add_argument('--events-csv', default='data/eventos.csv')
    ap.add_argument('--events', default='data/eventos.txt',
                    help='intervalos conocidos; "" si no se conocen (caso a auditar)')
    ap.add_argument('--K', type=int, default=3, help='persistencia: superaciones consecutivas')
    ap.add_argument('--q', type=float, default=0.99, help='cuantil para tau')
    ap.add_argument('--z-max', type=float, default=10.0)
    ap.add_argument('--agg', choices=['mean', 'max'], default='mean',
                    help='cómo agregar el error del modelo en cada instante del detector')
    ap.add_argument('--agg-sec', type=float, default=WINDOW_SEC,
                    help='ventana de agregación del error del modelo (s)')
    ap.add_argument('--tolerance', type=float, default=5.0,
                    help='segundos después del fin de un evento en que una alarma todavía cuenta')
    ap.add_argument('--out-dir', default='outputs_u6/act4')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    events = read_events(a.events) if a.events and os.path.exists(a.events) else []
    if not events:
        print('[aviso] sin eventos conocidos: se listan los intervalos de alarma '
              '(act4_alarmas.csv), sin evaluación por eventos.')

    runs = {}
    for name, path in (('calib', a.calib_csv), ('eventos', a.events_csv)):
        t, v, w = load_cmdvel(path)
        df = u5_indicators(t, v, w)
        for prefix in a.models:
            df[os.path.basename(prefix)] = model_indicator(prefix, t, v, w, df['t'].to_numpy(),
                                                               a.agg, a.agg_sec)
        df.loc[df['t'] < WARMUP, df.columns[1:]] = np.nan      # calentamiento: no decide
        runs[name] = df

    u5_names = list(INDICATORS)
    model_names = [os.path.basename(p) for p in a.models]
    configs = {'Unidad 5': u5_names}
    for m in model_names:
        configs[f'{m} solo'] = [m]
        configs[f'Unidad 5 + {m}'] = u5_names + [m]

    weights = {k: v['weight'] for k, v in INDICATORS.items()}
    direction = {k: v['direction'] for k, v in INDICATORS.items()}
    for m in model_names:
        weights[m], direction[m] = 1.0, 'up'

    ev_rows, sum_rows, alarm_rows, traces = [], [], [], {}
    cal, evr = runs['calib'], runs['eventos']
    t_ev = evr['t'].to_numpy()
    for cname, names in configs.items():
        calib = {k: robust_center_scale(cal[k])[:2] for k in names}
        S_cal, _ = score({k: cal[k] for k in names}, calib, weights, direction, a.z_max)
        tau = quantile_threshold(S_cal[cal['t'] >= WARMUP], a.q)
        S, contrib = score({k: evr[k] for k in names}, calib, weights, direction, a.z_max)
        alarm = k_consecutive(S >= tau, a.K)
        # intervalos de alarma, con el indicador que más aporta al empezar cada uno
        on = np.flatnonzero(np.diff(np.r_[0, alarm.astype(int), 0]) == 1)
        off = np.flatnonzero(np.diff(np.r_[0, alarm.astype(int), 0]) == -1) - 1
        for i0, i1 in zip(on, off):
            alarm_rows.append({'configuracion': cname, 'inicio_s': t_ev[i0], 'fin_s': t_ev[i1],
                               'indicador_principal': max(contrib, key=lambda k: contrib[k][i0])})
        if not events:
            sum_rows.append({'configuracion': cname, 'tau': tau,
                             'intervalos_de_alarma': int(len(on))})
            traces[cname] = (S, tau, alarm)
            continue
        r = evaluate_events(t_ev, alarm, events, a.tolerance)
        sum_rows.append({'configuracion': cname, 'tau': tau, 'detectados': r['detectados'],
                         'eventos': r['eventos'], 'falsas_alarmas': r['falsas_alarmas'],
                         'falsas_por_hora': r['falsas_por_hora'],
                         'frac_alarma_nominal': r['frac_alarma_nominal']})
        for (e0, e1) in events:
            hits = t_ev[alarm & (t_ev >= e0) & (t_ev <= e1 + a.tolerance)]
            # indicador que más aporta en el primer instante de alarma del evento
            top = ''
            if len(hits):
                j = int(np.searchsorted(t_ev, hits[0]))
                top = max(contrib, key=lambda k: contrib[k][j])
            ev_rows.append({'configuracion': cname, 'evento': f'{e0:g}-{e1:g}',
                            'detectado': bool(len(hits)),
                            'retardo_s': (hits[0] - e0) if len(hits) else np.nan,
                            'indicador_principal': top})
        traces[cname] = (S, tau, alarm)

    summ = pd.DataFrame(sum_rows)
    summ.to_csv(f'{a.out_dir}/act4_resumen.csv', index=False, float_format='%.3f')
    pd.DataFrame(alarm_rows).to_csv(f'{a.out_dir}/act4_alarmas.csv', index=False,
                                    float_format='%.1f')
    print('Resumen por configuración:')
    print(summ.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    if events:
        evdf = pd.DataFrame(ev_rows)
        evdf.to_csv(f'{a.out_dir}/act4_eventos.csv', index=False, float_format='%.2f')
        print('\nPor evento (retardo en s; vacío = no detectado):')
        piv = evdf.pivot_table(index='configuracion', columns='evento', values='retardo_s',
                               aggfunc='first', sort=False)
        print(piv.round(1).to_string(na_rep='---'))
    print(f'\nIntervalos de alarma de cada configuración: {a.out_dir}/act4_alarmas.csv')

    n = len(traces)
    fig, axs = plt.subplots(n, 1, figsize=(10.5, 1.9 * n + 0.6), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, (cname, (S, tau, alarm)) in zip(axs, traces.items()):
        for e0, e1 in events:
            ax.axvspan(e0, e1, color='#f6e3c8', lw=0)
        ax.plot(t_ev, S, color='#2a6fb0', lw=0.9)
        ax.axhline(tau, color='#c0392b', lw=1, ls='--')
        top = max(np.nanmax(S), tau) * 1.12
        # alarma: franja roja en la parte de abajo del panel
        ax.fill_between(t_ev, -0.12 * top, -0.03 * top, where=alarm, color='#c0392b', lw=0,
                        step='mid')
        ax.set_ylim(-0.14 * top, top)
        ax.set_title(cname, fontsize=9, loc='left')
        ax.set_ylabel('score', fontsize=8)
        style_axes(ax)
    axs[-1].set_xlabel('tiempo (s)')
    fig.text(0.99, 0.995, 'franja naranja: eventos   ·   barra roja: alarma   ·   línea a trazos: tau',
             ha='right', va='top', fontsize=8, color='#555555')
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(f'{a.out_dir}/act4_score.png', dpi=130)
    print(f'\nTablas y figura en {a.out_dir}/')


if __name__ == '__main__':
    main()
