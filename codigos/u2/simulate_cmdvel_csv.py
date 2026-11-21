#!/usr/bin/env python3
"""
Genera un CSV sintético con el mismo formato que cmd_vel_logger.py (Unidad 1).

Sirve para practicar la auditoría sin ROS 2, o para preparar ejercicios con
irregularidades conocidas. Implementa el modelo de llegada de la unidad:

    t_k = t_0 + k*T + eps_k + L_k

con jitter del temporizador (eps), latencia variable (L) y eventos opcionales:
mensajes perdidos, una pausa, retrasos aislados (par largo-corto) e
inyección de mensajes extra (flooding).

Ejemplos:
    python3 simulate_cmdvel_csv.py --out data/sintetico_limpio.csv
    python3 simulate_cmdvel_csv.py --out data/sintetico_eventos.csv \
        --drop-prob 0.01 --pause-at 150 --pause-len 2.0 --n-delays 5
    python3 simulate_cmdvel_csv.py --out data/sintetico_flood.csv \
        --flood-at 200 --flood-len 10 --flood-rate 40
"""
import argparse
import csv
import os

import numpy as np


def pattern(t, v_fwd, w_turn, t_fwd, t_turn, t_stop):
    cycle = t_fwd + t_stop + t_turn + t_stop
    tau = t % cycle
    if tau < t_fwd:
        return v_fwd, 0.0
    if tau < t_fwd + t_stop:
        return 0.0, 0.0
    if tau < t_fwd + t_stop + t_turn:
        return 0.0, w_turn
    return 0.0, 0.0


def main():
    p = argparse.ArgumentParser(description='CSV sintético de /cmd_vel')
    p.add_argument('--out', default='data/cmdvel_sintetico.csv')
    p.add_argument('--rate', type=float, default=10.0, help='frecuencia nominal (Hz)')
    p.add_argument('--duration', type=float, default=300.0, help='duración (s)')
    p.add_argument('--timer-jitter', type=float, default=0.002,
                   help='desvío estándar del error del temporizador (s)')
    p.add_argument('--lat-mean', type=float, default=0.005, help='latencia media (s)')
    p.add_argument('--lat-jitter', type=float, default=0.001,
                   help='desvío estándar de la latencia (s)')
    p.add_argument('--drop-prob', type=float, default=0.0,
                   help='probabilidad de perder cada mensaje')
    p.add_argument('--pause-at', type=float, default=None, help='instante de una pausa (s)')
    p.add_argument('--pause-len', type=float, default=0.0, help='duración de la pausa (s)')
    p.add_argument('--n-delays', type=int, default=0,
                   help='cantidad de mensajes con retraso aislado')
    p.add_argument('--delay-len', type=float, default=0.06, help='retraso aislado (s)')
    p.add_argument('--quantum', type=float, default=0.0,
                   help='cuantización del reloj (s), p. ej. 0.001 para imitar sim time')
    p.add_argument('--flood-at', type=float, default=None,
                   help='inicio de una inyección de mensajes extra (s)')
    p.add_argument('--flood-len', type=float, default=0.0, help='duración de la inyección (s)')
    p.add_argument('--flood-rate', type=float, default=40.0,
                   help='mensajes extra por segundo durante la inyección')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--v-fwd', type=float, default=0.20)
    p.add_argument('--w-turn', type=float, default=0.80)
    p.add_argument('--t-fwd', type=float, default=6.0)
    p.add_argument('--t-turn', type=float, default=3.0)
    p.add_argument('--t-stop', type=float, default=2.0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    T = 1.0 / args.rate
    n = int(args.duration * args.rate)
    k = np.arange(n)
    t_plan = k * T

    eps = rng.normal(0.0, args.timer_jitter, n)
    lat = np.clip(rng.normal(args.lat_mean, args.lat_jitter, n), 0.0, None)

    # Retrasos aislados: afectan solo a la latencia de un mensaje
    if args.n_delays > 0:
        idx = rng.choice(np.arange(1, n - 1), size=min(args.n_delays, n - 2), replace=False)
        lat[idx] += args.delay_len

    t_rx = t_plan + eps + lat

    # Pausa: nada se publica dentro de [pause_at, pause_at + pause_len)
    keep = np.ones(n, dtype=bool)
    if args.pause_at is not None and args.pause_len > 0:
        keep &= ~((t_plan >= args.pause_at) & (t_plan < args.pause_at + args.pause_len))
    if args.drop_prob > 0:
        keep &= rng.random(n) >= args.drop_prob

    t_plan, t_rx = t_plan[keep], t_rx[keep]

    # Inyección: mensajes extra en instantes al azar, con valores legítimos
    # (los que el patrón tendría en ese instante), como haría un atacante
    if args.flood_at is not None and args.flood_len > 0:
        n_f = int(args.flood_len * args.flood_rate)
        t_f = np.sort(rng.uniform(args.flood_at, args.flood_at + args.flood_len, n_f))
        t_plan = np.r_[t_plan, t_f]
        t_rx = np.r_[t_rx, t_f + lat.mean()]
    order = np.argsort(t_rx, kind='stable')     # un mensaje retrasado puede llegar después
    t_plan, t_rx = t_plan[order], t_rx[order]
    if args.quantum > 0:
        t_rx = np.floor(t_rx / args.quantum) * args.quantum
    t_rx = t_rx - t_rx[0]

    folder = os.path.dirname(args.out)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_sec', 'v', 'w', 'dt'])
        prev = None
        for tp, tr in zip(t_plan, t_rx):
            v, om = pattern(tp, args.v_fwd, args.w_turn, args.t_fwd, args.t_turn, args.t_stop)
            dt = '' if prev is None else f'{tr - prev:.9f}'
            w.writerow([f'{tr:.9f}', v, om, dt])
            prev = tr

    print(f'Generadas {len(t_rx)} muestras en {args.out}')


if __name__ == '__main__':
    main()
