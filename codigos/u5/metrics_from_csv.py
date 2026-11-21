#!/usr/bin/env python3
"""
Genera, a partir de un CSV de /cmd_vel, los mismos registros que producen los
nodos entropy_metrics y cmdvel_predictor (Unidad 5).

Sirve para:
  - practicar la calibración y la evaluación SIN ROS ni Gazebo;
  - analizar un registro de la Unidad 1 (o del simulador) con el pipeline de la Unidad 5.

Simula la operación en línea: recorre los mensajes en orden de llegada y, cada
1/publish_hz segundos, calcula los indicadores con OnlineIndicators (el mismo código
que usan los nodos).

Entrada: CSV con t_sec, v, w (formato del logger de la Unidad 1, sin re-muestrear).
Salida (en --log-dir, mismo formato que csv_logger):
    entropy_metrics.csv        t, n_msgs, H_v, Hcond_v, H_dt
    cmd_vel_pred_metrics.csv   t, n_msgs, mse_v, mse_w

Uso:
    python3 metrics_from_csv.py --csv data/cmdvel_u1.csv --log-dir outputs_u5/baseline
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from u5_core import OnlineIndicators


def main():
    ap = argparse.ArgumentParser(description='Indicadores en línea a partir de un CSV de /cmd_vel')
    ap.add_argument('--csv', required=True, help='CSV ORIGINAL (t_sec, v, w)')
    ap.add_argument('--log-dir', required=True)
    ap.add_argument('--window-sec', type=float, default=5.0)
    ap.add_argument('--publish-hz', type=float, default=2.0)
    ap.add_argument('--min-samples', type=int, default=30)
    ap.add_argument('--dt-max', type=float, default=0.3)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    for c in ('t_sec', 'v', 'w'):
        if c not in df.columns:
            sys.exit(f'[error] falta la columna {c}')
    df = df[['t_sec', 'v', 'w']].dropna()
    t = df['t_sec'].to_numpy(dtype=float)
    if 'gap' in pd.read_csv(args.csv, nrows=1).columns or np.ptp(np.diff(t)) < 1e-9:
        print('[aviso] el CSV parece re-muestreado (Δt constante): la entropía de Δt no va a '
              'tener información. Usá el CSV original del logger.')
    t = t - t[0]
    v, w = df['v'].to_numpy(dtype=float), df['w'].to_numpy(dtype=float)

    ind = OnlineIndicators(window_sec=args.window_sec, min_samples=args.min_samples,
                           dt_max=args.dt_max)
    period = 1.0 / args.publish_hz
    rows_e, rows_p = [], []
    i, n = 0, len(t)
    tick = period
    while tick <= t[-1]:
        while i < n and t[i] <= tick:
            ind.add(t[i], v[i], w[i])
            i += 1
        m = ind.compute(tick)
        rows_e.append({'t': tick, 'n_msgs': m['n_msgs'], 'H_v': m['H_v'],
                       'Hcond_v': m['Hcond_v'], 'H_dt': m['H_dt']})
        rows_p.append({'t': tick, 'n_msgs': m['n_msgs'], 'mse_v': m['mse_v'],
                       'mse_w': m['mse_w']})
        tick += period

    os.makedirs(args.log_dir, exist_ok=True)
    pd.DataFrame(rows_e).to_csv(os.path.join(args.log_dir, 'entropy_metrics.csv'),
                                index=False, float_format='%.6f')
    pd.DataFrame(rows_p).to_csv(os.path.join(args.log_dir, 'cmd_vel_pred_metrics.csv'),
                                index=False, float_format='%.6f')
    print(f'{len(rows_e)} registros ({args.publish_hz:g} Hz, ventanas de {args.window_sec:g} s, '
          f'{t[-1]:.0f} s de datos) en {args.log_dir}/')


if __name__ == '__main__':
    main()
