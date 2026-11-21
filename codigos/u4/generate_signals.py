#!/usr/bin/env python3
"""
Señales sintéticas de la Unidad 4 (Actividades 1 a 3).

Todas en m/s, con valores entre 0 y 0.2 y semilla fija (reproducibles):
    structured  bloques regulares: 20 muestras en 0.2, 20 en 0 (ciclo de 40)
    shuffled    los mismos valores de structured, en orden aleatorio
    iid         valores independientes, uniformes en [0, 0.2]
    periodic    oscilación rápida 0.1 + 0.1 sin(2 pi t / 8): cambia en todos los pasos,
                recorre 5 valores (entropía alta) y la predice exactamente un AR(2)
    slow        senoide lenta entre 0 y 0.2 (período de 100 muestras)
    anomaly     structured de 1000 muestras con valores al azar entre 500 y 700

Uso como módulo:
    from generate_signals import make_signals, make_anomaly_signal
Uso como script (guarda las señales en CSV para inspeccionarlas):
    python3 generate_signals.py --out data/u4_signals.csv
"""
import argparse
import os

import numpy as np

V = 0.2            # nivel alto de los bloques (m/s)
BLOCK = 20         # muestras por bloque
SEED = 42


def structured(n=500, block=BLOCK, level=V):
    cycle = [level] * block + [0.0] * block
    reps = n // len(cycle) + 1
    return np.array((cycle * reps)[:n], dtype=float)


def make_signals(n=500, seed=SEED, include_slow=False):
    """Diccionario {nombre: señal} con las señales de las Actividades 1 y 3."""
    rng = np.random.default_rng(seed)
    s = structured(n)
    sh = s.copy()
    rng.shuffle(sh)
    sig = {
        'Structured': s,
        'Shuffled': sh,
        'IID': rng.uniform(0.0, V, n),
        'Periodic': 0.1 + 0.1 * np.sin(2 * np.pi * np.arange(n) / 8),
    }
    if include_slow:
        t = np.arange(n)
        sig['Slow'] = 0.1 + 0.1 * np.sin(2 * np.pi * t / 100)
    return sig


def make_anomaly_signal(n=1000, start=500, end=700, seed=SEED):
    """Señal de la Actividad 2 y máscara booleana de la anomalía (verdad conocida)."""
    rng = np.random.default_rng(seed)
    x = structured(n)
    x[start:end] = rng.uniform(0.0, V, end - start)
    truth = np.zeros(n, dtype=bool)
    truth[start:end] = True
    return x, truth


def main():
    p = argparse.ArgumentParser(description='Señales sintéticas de la Unidad 4')
    p.add_argument('--out', default='data/u4_signals.csv')
    p.add_argument('--n', type=int, default=500)
    args = p.parse_args()
    import pandas as pd
    sig = make_signals(args.n, include_slow=True)
    x, truth = make_anomaly_signal()
    df = pd.DataFrame(sig)
    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index_label='k', float_format='%.6f')
    pd.DataFrame({'x': x, 'anomaly': truth.astype(int)}).to_csv(
        os.path.splitext(args.out)[0] + '_anomaly.csv', index_label='k', float_format='%.6f')
    print(f'Señales guardadas en {args.out} y {os.path.splitext(args.out)[0]}_anomaly.csv')


if __name__ == '__main__':
    main()
