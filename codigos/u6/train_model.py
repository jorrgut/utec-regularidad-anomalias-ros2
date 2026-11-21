#!/usr/bin/env python3
"""
Entrena UN modelo y lo guarda con todo lo necesario para usarlo en el detector
(actividad 4) o en un nodo en línea (apunte, sección "Qué guardar").

    <out>.pt     pesos
    <out>.json   tipo, T, canales, arquitectura, período de la grilla, estandarización,
                 y la mediana y escala robusta del error en VALIDACIÓN

Tipos:
    --kind ae     autoencoder de ventana; indicador = error medio de la ventana
                  (o el de la última muestra con --err last)
    --kind lstm   predictor LSTM de un paso; indicador = error de predicción

Con --counts la entrada agrega el canal n_k (mensajes por celda de la grilla).

Ejemplos:
    python3 train_model.py --kind ae --T 40 --out models/ae_T40
    python3 train_model.py --kind ae --T 40 --counts --out models/ae_T40_nk
    python3 train_model.py --kind lstm --T 70 --out models/lstm_T70
"""
import argparse

import numpy as np

from u5_core import robust_center_scale
from u6_data import (CHANNELS, CHANNELS_COUNT, Standardizer, build_features, load_cmdvel,
                     make_windows, resample_zoh, split_blocks)
from u6_models import (LSTMPredictor, WindowAE, n_params, pred_errors, predictor_pairs,
                       recon_errors, save_model, set_seed, train_model)


def model_errors(model, meta, X):
    """Error del modelo sobre la serie estandarizada X; devuelve (error, índice en la grilla)."""
    T = meta['T']
    if meta['kind'] == 'ae':
        W, idx = make_windows(X, T)
        e_mean, e_last, _ = recon_errors(model, W)
        return (e_last if meta.get('err', 'mean') == 'last' else e_mean), idx
    W, idx = make_windows(X, T + 1)
    P, Y = predictor_pairs(W)
    return pred_errors(model, P, Y)[0], idx


def main():
    ap = argparse.ArgumentParser(description='Entrenar y guardar un modelo de la Unidad 6')
    ap.add_argument('--csv', default='data/train_structured.csv', help='corrida NOMINAL')
    ap.add_argument('--kind', choices=['ae', 'lstm'], default='ae')
    ap.add_argument('--T', type=int, default=40)
    ap.add_argument('--latent', type=int, default=4, help='solo ae')
    ap.add_argument('--hidden', type=int, default=None, help='64 para ae, 32 para lstm')
    ap.add_argument('--counts', action='store_true', help='agregar el canal n_k')
    ap.add_argument('--err', choices=['mean', 'last'], default='mean', help='solo ae')
    ap.add_argument('--rate', type=float, default=10.0, help='frecuencia de la grilla (Hz)')
    ap.add_argument('--frac-train', type=float, default=0.8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True, help='prefijo de salida, p. ej. models/ae_T40')
    a = ap.parse_args()

    grid = resample_zoh(*load_cmdvel(a.csv), rate=a.rate)
    Xtr, Xva = split_blocks(build_features(grid, counts=a.counts), a.frac_train)
    std = Standardizer().fit(Xtr)
    Xtr, Xva = std.transform(Xtr), std.transform(Xva)
    d = Xtr.shape[1]
    hidden = a.hidden or (64 if a.kind == 'ae' else 32)
    set_seed(a.seed)

    if a.kind == 'ae':
        model = WindowAE(a.T, d, hidden, a.latent)
        Wtr, _ = make_windows(Xtr, a.T)
        Wva, _ = make_windows(Xva, a.T)
        model, hist = train_model(model, Wtr, Wtr, Wva, Wva, seed=a.seed)
    else:
        model = LSTMPredictor(d, hidden)
        Ptr, Ytr = predictor_pairs(make_windows(Xtr, a.T + 1)[0])
        Pva, Yva = predictor_pairs(make_windows(Xva, a.T + 1)[0])
        model, hist = train_model(model, Ptr, Ytr, Pva, Yva, seed=a.seed)

    meta = {'kind': a.kind, 'T': a.T, 'hidden': hidden, 'latent': a.latent,
            'channels': list(CHANNELS_COUNT if a.counts else CHANNELS), 'err': a.err,
            'rate_hz': a.rate, 'standardizer': std.to_dict(), 'seed': a.seed,
            'train_csv': a.csv, 'epochs': len(hist), 'val_loss': hist[-1][2]}
    e_va, _ = model_errors(model, meta, Xva)
    c, s, src = robust_center_scale(e_va)
    meta['val_error'] = {'median': c, 'scale': s, 'scale_source': src,
                         'q99': float(np.quantile(e_va, 0.99))}
    save_model(a.out, model, meta)
    print(f'{a.kind.upper()} T={a.T} canales={meta["channels"]}: {n_params(model)} parámetros, '
          f'{len(hist)} épocas')
    print(f'  error en validación: mediana {c:.3g}, escala {s:.3g} ({src}), '
          f'cuantil 0.99 {meta["val_error"]["q99"]:.3g}')
    print(f'  guardado en {a.out}.pt y {a.out}.json')


if __name__ == '__main__':
    main()
