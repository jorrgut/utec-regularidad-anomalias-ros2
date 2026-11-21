#!/usr/bin/env python3
"""
Actividad 3 (avanzada): predictor LSTM frente a persistencia (apunte, sección "Redes
recurrentes LSTM").

Para cada T y cada semilla entrena un LSTMPredictor que estima x_t a partir de
(x_{t-T}, ..., x_{t-1}) y compara su MSE con el de persistencia (repetir el último valor)
y, opcionalmente, con un AR(p) por mínimos cuadrados (Unidad 4):

    skill = 1 - MSE_modelo / MSE_persistencia

Lo que deberías ver: con T = 10, skill ~ 0 en structured (no puede saber cuánto lleva la
fase); con T = 70, skill ~ 1 en structured y muy negativo en shuffle_blocks.

Salidas (en --out-dir):
    act3_resultados.csv     una fila por (modelo, T, semilla, modo)
    act3_resumen.csv        promedio entre semillas
    act3_prediccion.png     30 s de structured: señal, LSTM (cada T) y persistencia

Uso:
    python3 u6_activity3.py                        # T = 10 70, semillas 0 1 2, AR(p) incluido
    python3 u6_activity3.py --T 10 70 --seeds 0    # versión rápida
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from u6_data import MODE_COLORS, make_windows, style_axes
from u6_activity2 import TESTS, load_all
from u6_models import (LSTMPredictor, n_params, predict, predictor_pairs, set_seed,
                       train_model)


def ar_fit(P, Y, p):
    """AR(p) multicanal por mínimos cuadrados: Y ≈ [últimas p muestras de P, 1] @ B."""
    A = np.c_[P[:, -p:, :].reshape(len(P), -1), np.ones(len(P))]
    B, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return lambda P2: np.c_[P2[:, -p:, :].reshape(len(P2), -1), np.ones(len(P2))] @ B


def main():
    ap = argparse.ArgumentParser(description='Actividad 3: LSTM frente a persistencia')
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--out-dir', default='outputs_u6/act3')
    ap.add_argument('--T', type=int, nargs='+', default=[10, 70])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--ar-orders', type=int, nargs='*', default=[1, 2, 5, 10, 20, 40, 60],
                    help='órdenes de AR(p) a probar (se elige por validación; vacío = omitir)')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    Xtr, Xva, tests = load_all(a.data_dir)
    rows, preds = [], {}
    for T in a.T:
        Ptr, Ytr = predictor_pairs(make_windows(Xtr, T + 1)[0])
        Pva, Yva = predictor_pairs(make_windows(Xva, T + 1)[0])
        test_pairs = {}
        for mode, (t, X) in tests.items():
            W, idx = make_windows(X, T + 1)
            test_pairs[mode] = (t[idx],) + predictor_pairs(W)

        def record(model_name, seed, pred):
            for mode, (t, P, Y) in test_pairs.items():
                mse = float(((pred(P) - Y) ** 2).mean())
                per = float(((P[:, -1] - Y) ** 2).mean())
                rows.append({'modelo': model_name, 'T': T, 'semilla': seed, 'modo': mode,
                             'mse': mse, 'mse_persistencia': per, 'skill': 1 - mse / per})

        for seed in a.seeds:
            set_seed(seed)
            model = LSTMPredictor(d=2, hidden=a.hidden)
            model, hist = train_model(model, Ptr, Ytr, Pva, Yva, seed=seed)
            pred_fn = lambda P, m=model: predict(m, P)
            record('LSTM', seed, pred_fn)
            msg = ', '.join(f"{r['modo']} {r['skill']:+.2f}" for r in rows[-3:])
            print(f'LSTM T={T:3d} semilla {seed}: {n_params(model)} parámetros, '
                  f'{len(hist):3d} épocas | skill: {msg}')
            if seed == a.seeds[0]:
                t, P, Y = test_pairs['structured']
                preds[T] = (t, Y, pred_fn(P), P[:, -1])

        # AR(p) con p < T, elegido por validación
        orders = [p for p in a.ar_orders if p <= T]
        if orders:
            best = min(orders, key=lambda p: float(((ar_fit(Ptr, Ytr, p)(Pva) - Yva) ** 2).mean()))
            record(f'AR({best})', 0, ar_fit(Ptr, Ytr, best))
            msg = ', '.join(f"{r['modo']} {r['skill']:+.2f}" for r in rows[-3:])
            print(f'AR({best}) T={T:3d}: skill: {msg}')

    df = pd.DataFrame(rows)
    df.to_csv(f'{a.out_dir}/act3_resultados.csv', index=False, float_format='%.5f')
    summ = (df.groupby(['modelo', 'T', 'modo'], sort=False)
              .agg(skill=('skill', 'mean'), skill_desvio=('skill', 'std'), mse=('mse', 'mean'),
                   mse_persistencia=('mse_persistencia', 'mean'))
              .reset_index())
    summ.to_csv(f'{a.out_dir}/act3_resumen.csv', index=False, float_format='%.4f')
    print('\nSkill respecto de persistencia (promedio entre semillas):')
    print(summ.pivot_table(index=['modelo', 'T'], columns='modo', values='skill', sort=False)
              [list(TESTS)].round(2).to_string())

    # Figura: 30 s de structured, canal v (estandarizado)
    fig, axs = plt.subplots(len(preds), 1, figsize=(10, 2.6 * len(preds)), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, (T, (t, Y, Yh, Yp)) in zip(axs, sorted(preds.items())):
        sel = (t >= 60) & (t < 90)
        ax.plot(t[sel], Y[sel, 0], color='#222222', lw=2.0, label='señal v')
        ax.plot(t[sel], Yp[sel, 0], color='#999999', lw=1.2, ls='--', label='persistencia')
        ax.plot(t[sel], Yh[sel, 0], color=MODE_COLORS['structured'], lw=1.4,
                label=f'LSTM, T = {T}')
        ax.set_ylabel('v (estandarizada)', fontsize=9)
        ax.set_ylim(-1.4, 1.9)
        ax.legend(fontsize=8, frameon=False, loc='upper left', ncol=3)
        style_axes(ax)
    axs[-1].set_xlabel('tiempo (s)')
    fig.suptitle('Predicción de un paso en structured (corrida de prueba)', fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{a.out_dir}/act3_prediccion.png', dpi=130)
    print(f'\nTablas y figura en {a.out_dir}/')


if __name__ == '__main__':
    main()
