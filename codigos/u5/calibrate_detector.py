#!/usr/bin/env python3
"""
Calibración del detector a partir de una línea base nominal (Unidad 5, Actividad 1).

Lee los CSV de indicadores de --log-dir (registrados por csv_logger o generados con
metrics_from_csv.py), descarta el período de calentamiento y calcula para cada
indicador:
    centro y escala robustos (1.4826*MAD, con respaldo std / piso; se informa cuál)
y con ellos el score de la línea base y el umbral tau = cuantil --p.

Escribe un archivo YAML que carga el nodo anomaly_detector y usa evaluate_detector.py.
Incluye también una tabla de cuantiles del score nominal, para explorar umbrales.

Salidas:
    --out                                  calibración (YAML)
    <dir de --out>/baseline_score.png      score nominal con tau
    <dir de --out>/baseline_correlation.csv  correlación entre indicadores

Uso:
    python3 calibrate_detector.py --log-dir outputs_u5/baseline --p 0.99 --out outputs_u5/calib.yaml
    python3 calibrate_detector.py --log-dir outputs_u5/baseline --weights H_v=0 Hcond_v=0.5
"""
import argparse
import datetime
import os
import sys

import numpy as np

from u5_core import INDICATORS, robust_center_scale, score, quantile_threshold, save_calibration
from u5_offline import load_indicators, style_axes, C_DATA, C_REF


def parse_kv(items, cast):
    out = {}
    for it in items or []:
        try:
            k, v = it.split('=')
            out[k] = cast(v)
        except ValueError:
            sys.exit(f'[error] formato inválido: "{it}" (esperado nombre=valor)')
        if k not in INDICATORS:
            sys.exit(f'[error] indicador desconocido: {k}. Opciones: {", ".join(INDICATORS)}')
    return out


def main():
    ap = argparse.ArgumentParser(description='Calibración del detector de la Unidad 5')
    ap.add_argument('--log-dir', required=True, help='carpeta con los CSV de la línea base')
    ap.add_argument('--p', type=float, default=0.99, help='cuantil para tau')
    ap.add_argument('--warmup', type=float, default=10.0,
                    help='segundos iniciales descartados (ventanas incompletas)')
    ap.add_argument('--weights', nargs='*', metavar='IND=PESO', help='cambiar pesos')
    ap.add_argument('--direction', nargs='*', metavar='IND=up|both', help='cambiar direcciones')
    ap.add_argument('--z-max', type=float, default=10.0,
                    help='saturación de cada término del score (|z| máximo); 0 = sin saturar')
    ap.add_argument('--out', default='outputs_u5/calib.yaml')
    args = ap.parse_args()

    df = load_indicators(args.log_dir)
    df = df[df['t_rel'] >= args.warmup].reset_index(drop=True)
    if len(df) < 50:
        sys.exit(f'[error] solo {len(df)} registros después del calentamiento: '
                 'la línea base es demasiado corta')
    duration = df['t'].iloc[-1] - df['t'].iloc[0]

    weights = {k: v['weight'] for k, v in INDICATORS.items()}
    weights.update(parse_kv(args.weights, float))
    direction = {k: v['direction'] for k, v in INDICATORS.items()}
    direction.update(parse_kv(args.direction, str))
    for k, d in direction.items():
        if d not in ('up', 'both'):
            sys.exit(f'[error] dirección inválida para {k}: {d} (up o both)')

    ind_cfg, centers = {}, {}
    print(f'Línea base: {args.log_dir} | {len(df)} registros | {duration:.0f} s '
          f'({duration / 60:.1f} min) después de {args.warmup:g} s de calentamiento\n')
    print(f'  {"indicador":<9} {"centro":>10} {"escala":>10} {"fuente":>8} {"peso":>5} {"dir.":>5}'
          f' {"NaN":>5}')
    for name in INDICATORS:
        c, s, src = robust_center_scale(df[name])
        centers[name] = (c, s)
        n_nan = int(df[name].isna().sum())
        ind_cfg[name] = {'center': float(c), 'scale': float(s), 'scale_source': src,
                         'weight': float(weights[name]), 'direction': direction[name]}
        print(f'  {name:<9} {c:10.4g} {s:10.4g} {src:>8} {weights[name]:5.2f} {direction[name]:>5}'
              f' {n_nan:5d}')
    if any(v['scale_source'] != 'mad' for v in ind_cfg.values()):
        print('\n  Nota: un indicador con escala "std" o "piso" tiene MAD = 0 en la línea base '
              '(pocos valores distintos). Revisalo antes de confiar en su z-score.')

    z_max = args.z_max if args.z_max > 0 else None
    S, _ = score({k: df[k].to_numpy() for k in INDICATORS}, centers, weights, direction, z_max)
    tau = quantile_threshold(S, args.p)
    qs = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]
    qtable = {f'{q:g}': quantile_threshold(S, q) for q in qs}

    # Muestras efectivas (correlación del score a 1 paso)
    s0 = S - np.mean(S)
    rho = float(np.sum(s0[1:] * s0[:-1]) / np.sum(s0 * s0)) if np.sum(s0 * s0) > 0 else 0.0
    n_eff = len(S) * (1 - rho) / (1 + rho) if rho < 1 else 1.0

    calib = {
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'baseline_dir': os.path.abspath(args.log_dir),
        'baseline_seconds': float(duration),
        'warmup_sec': args.warmup,
        'p': args.p,
        'z_max': z_max,
        'tau': tau,
        'tau_quantiles': qtable,
        'score_median': float(np.median(S)),
        'score_scale': float(robust_center_scale(S)[1]),
        'score_autocorr_lag1': rho,
        'indicators': ind_cfg,
    }
    out_dir = os.path.dirname(args.out) or '.'
    os.makedirs(out_dir, exist_ok=True)
    save_calibration(args.out, calib)

    corr = df[list(INDICATORS)].corr()
    corr.to_csv(os.path.join(out_dir, 'baseline_correlation.csv'), float_format='%.3f')

    print(f'\nScore nominal: mediana {np.median(S):.3f} | máximo {np.max(S):.3f}')
    print(f'tau = cuantil {args.p:g} = {tau:.3f}')
    print('Cuantiles del score nominal: ' + ', '.join(f'{k}: {v:.2f}' for k, v in qtable.items()))
    print(f'Correlación del score entre registros consecutivos: {rho:.2f} '
          f'-> unas {n_eff:.0f} muestras "efectivas" de {len(S)}')
    if n_eff < 10 / (1 - args.p):
        print(f'  [aviso] con ~{n_eff:.0f} muestras efectivas, el cuantil {args.p:g} es poco '
              'confiable. Registrá una línea base más larga o usá un p menor.')
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(INDICATORS)
             for b in list(INDICATORS)[i + 1:] if abs(corr.loc[a, b]) > 0.7]
    if pairs:
        print('Indicadores muy correlacionados en la línea base (|r| > 0.7): ' +
              ', '.join(f'{a}-{b} ({r:+.2f})' for a, b, r in pairs))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(df['t_rel'], S, color=C_DATA, linewidth=1)
    ax.axhline(tau, color=C_REF, linestyle='--', linewidth=1, label=f'τ = Q{args.p:g} = {tau:.2f}')
    ax.set_xlabel('tiempo desde el inicio del registro (s)')
    ax.set_ylabel('score')
    ax.set_title('Score en la línea base nominal', loc='left', fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'baseline_score.png'), dpi=150)
    plt.close(fig)
    print(f'\nCalibración: {args.out} | figura y correlaciones en {out_dir}/')


if __name__ == '__main__':
    main()
