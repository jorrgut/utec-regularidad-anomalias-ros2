#!/usr/bin/env python3
"""
Auditoría temporal rápida de un CSV de /cmd_vel (Unidad 1).

Entrada: CSV con columnas t_sec, v, w (la columna dt, si existe, se ignora y
se recalcula a partir de t_sec para no depender del logger).

Salidas en --outdir:
    velocity_signals.png   v(t) y w(t)
    dt_series.png          Δt_k vs. tiempo, con mediana y umbrales
    dt_histogram.png       histograma de Δt
    summary_stats.txt      estadísticas y lista de outliers
    dt_outliers.csv        filas con Δt atípico (si hay)

Uso:
    python3 quick_audit_dt.py --csv data/cmdvel_u1.csv --outdir outputs_u1
    python3 quick_audit_dt.py --csv data/cmdvel_u1.csv --outdir outputs_u1 --nominal-rate 10

No requiere ROS 2: solo numpy, pandas y matplotlib.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MAD_TO_SIGMA = 1.4826   # consistencia con σ bajo normalidad

# Estilo sobrio: una sola tinta para los datos, referencias en gris
C_DATA = '#2a6fb0'
C_REF = '#555555'
C_OUT = '#c0392b'


def style_axes(ax):
    ax.grid(True, color='#e5e5e5', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#999999')
    ax.tick_params(colors='#444444', labelsize=9)


def load_csv(path):
    df = pd.read_csv(path)
    missing = [c for c in ('t_sec', 'v', 'w') if c not in df.columns]
    if missing:
        sys.exit(f'[error] faltan columnas en el CSV: {missing}')
    df = df[['t_sec', 'v', 'w']].dropna().reset_index(drop=True)
    if len(df) < 3:
        sys.exit('[error] el CSV tiene menos de 3 muestras: no hay nada que auditar')
    return df


def compute_stats(df, k_ratio, hampel_c, nominal_rate):
    t = df['t_sec'].to_numpy(dtype=float)
    dt = np.diff(t)                      # dt[i] = t[i+1] - t[i]
    t_rel = t - t[0]

    n_neg = int(np.sum(dt < 0))
    n_zero = int(np.sum(dt == 0))
    dt_pos = dt[dt > 0]                  # para estimadores, solo intervalos válidos
    if len(dt_pos) == 0:
        sys.exit('[error] ningún Δt positivo: revisá el reloj usado en el registro')

    med = float(np.median(dt_pos))
    mad = float(np.median(np.abs(dt_pos - med)))
    s = {
        'N': len(t),
        'duracion': float(t[-1] - t[0]),
        'mediana': med,
        'media': float(np.mean(dt_pos)),
        'std': float(np.std(dt_pos, ddof=1)) if len(dt_pos) > 1 else float('nan'),
        'mad': mad,
        'sigma_mad': MAD_TO_SIGMA * mad,
        'min': float(np.min(dt_pos)),
        'max': float(np.max(dt_pos)),
        'p95': float(np.percentile(dt_pos, 95)),
        'p99': float(np.percentile(dt_pos, 99)),
        'p999': float(np.percentile(dt_pos, 99.9)),
        'f_hat': 1.0 / med,
        'f_media': 1.0 / float(np.mean(dt_pos)),
        'n_neg': n_neg,
        'n_zero': n_zero,
        'nominal_rate': nominal_rate,
    }

    thr_ratio = k_ratio * med
    half_band = hampel_c * MAD_TO_SIGMA * mad
    mask_ratio = dt > thr_ratio
    mask_hampel = np.abs(dt - med) > half_band if mad > 0 else np.zeros_like(dt, bool)
    mask_p99 = dt > s['p99']

    s.update({
        'k_ratio': k_ratio, 'thr_ratio': thr_ratio,
        'hampel_c': hampel_c, 'hampel_lo': med - half_band, 'hampel_hi': med + half_band,
        'n_ratio': int(mask_ratio.sum()), 'n_hampel': int(mask_hampel.sum()),
        'n_p99': int(mask_p99.sum()), 'n_dt': len(dt),
    })

    # Tabla de intervalos atípicos: el intervalo i termina en la muestra i+1
    out = pd.DataFrame({
        'k': np.arange(1, len(t)),
        't_rel_s': t_rel[1:],
        'dt_s': dt,
        'ratio_vs_mediana': dt / med,
        'regla_razon': mask_ratio,
        'hampel': mask_hampel,
        'negativo': dt < 0,
        'cero': dt == 0,
    })
    flagged = out[out['regla_razon'] | out['hampel'] | out['negativo']].copy()
    return t_rel, dt, s, flagged, mask_ratio


def pct(n, total):
    return 100.0 * n / total if total else 0.0


def write_summary(path, s, flagged, csv_path, max_rows=20):
    L = []
    L.append(f'Archivo: {csv_path}')
    L.append('')
    L.append(f'N muestras: {s["N"]}')
    L.append(f'Duración: {s["duracion"]:.2f} s')
    if s['nominal_rate']:
        L.append(f'Freq. nominal: {s["nominal_rate"]:.2f} Hz')
    L.append(f'Freq. efectiva: {s["f_hat"]:.2f} Hz   (1 / mediana dt)')
    L.append(f'Freq. con la media: {s["f_media"]:.2f} Hz   (solo para comparar; no usar)')
    if s['nominal_rate']:
        err = 100.0 * (s['f_hat'] - s['nominal_rate']) / s['nominal_rate']
        L.append(f'Desvío respecto de la nominal: {err:+.2f} %')
    L.append('')
    L.append(f'Mediana dt: {s["mediana"]:.4f} s')
    L.append(f'Media dt: {s["media"]:.4f} s')
    L.append(f'Std(dt): {s["std"]:.4f} s')
    L.append(f'MAD(dt): {s["mad"]:.4f} s   (1.4826*MAD = {s["sigma_mad"]:.4f} s)')
    L.append(f'Min / Max dt: {s["min"]:.4f} / {s["max"]:.4f} s')
    L.append(f'P95(dt): {s["p95"]:.4f} s')
    L.append(f'P99(dt): {s["p99"]:.4f} s')
    L.append(f'P99.9(dt): {s["p999"]:.4f} s')
    L.append('')
    L.append('Outliers:')
    L.append(f'  Regla de razón (dt > {s["k_ratio"]:g}*median = {s["thr_ratio"]:.4f} s): '
             f'{s["n_ratio"]} ({pct(s["n_ratio"], s["n_dt"]):.2f}%)')
    L.append(f'  Hampel (|dt - median| > {s["hampel_c"]:g}*1.4826*MAD, banda '
             f'[{s["hampel_lo"]:.4f}, {s["hampel_hi"]:.4f}] s): '
             f'{s["n_hampel"]} ({pct(s["n_hampel"], s["n_dt"]):.2f}%)')
    L.append(f'  Por encima de P99: {s["n_p99"]} ({pct(s["n_p99"], s["n_dt"]):.2f}%)')
    L.append('')
    L.append('Integridad del reloj:')
    L.append(f'  dt = 0: {s["n_zero"]}   (posible con tiempo simulado: mismo paso de simulación)')
    L.append(f'  dt < 0: {s["n_neg"]}   (reset de la simulación o ajuste del reloj del sistema)')
    if s['n_neg'] > 0:
        L.append('  [ATENCIÓN] hay tiempos que retroceden: revisá el experimento antes de analizar.')
    if s['mad'] == 0:
        L.append('  [nota] MAD = 0: dt cuantizado (típico en tiempo simulado); Hampel no es informativo.')
    L.append('')

    if len(flagged):
        L.append(f'Intervalos marcados (mostrando hasta {max_rows}, ordenados por dt):')
        L.append(f'  {"k":>7} {"t (s)":>10} {"dt (s)":>10} {"dt/med":>8}  reglas')
        top = flagged.sort_values('dt_s', ascending=False).head(max_rows)
        for _, r in top.iterrows():
            rules = [n for n, c in (('razón', r['regla_razon']), ('hampel', r['hampel']),
                                    ('negativo', r['negativo'])) if c]
            L.append(f'  {int(r["k"]):>7} {r["t_rel_s"]:>10.3f} {r["dt_s"]:>10.4f} '
                     f'{r["ratio_vs_mediana"]:>8.2f}  {", ".join(rules)}')
    else:
        L.append('No se marcaron intervalos atípicos.')

    text = '\n'.join(L) + '\n'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return text


def plot_velocity(df, t_rel, path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    for ax, col, label in ((axes[0], 'v', 'v (m/s)'), (axes[1], 'w', 'ω (rad/s)')):
        ax.plot(t_rel, df[col].to_numpy(), color=C_DATA, linewidth=1.2, drawstyle='steps-post')
        ax.set_ylabel(label)
        style_axes(ax)
    axes[0].set_title('Señales de control registradas', loc='left', fontsize=11)
    axes[1].set_xlabel('tiempo desde el primer mensaje (s)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_dt_series(t_rel, dt, s, mask_ratio, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    tt = t_rel[1:]
    ax.plot(tt, dt, color=C_DATA, linewidth=0.8, marker='.', markersize=2, label='Δt')
    ax.axhline(s['mediana'], color=C_REF, linewidth=1.2, label=f'mediana = {s["mediana"]:.4f} s')
    ax.axhline(s['thr_ratio'], color=C_REF, linewidth=1, linestyle='--',
               label=f'{s["k_ratio"]:g} × mediana')
    if s['mad'] > 0:
        ax.axhspan(s['hampel_lo'], s['hampel_hi'], color=C_REF, alpha=0.10,
                   label=f'banda de Hampel (c = {s["hampel_c"]:g})')
    if mask_ratio.any():
        ax.scatter(tt[mask_ratio], dt[mask_ratio], s=36, color=C_OUT, zorder=3,
                   edgecolor='white', linewidth=0.8, label='outlier (regla de razón)')
    ax.set_xlabel('tiempo desde el primer mensaje (s)')
    ax.set_ylabel('Δt (s)')
    ax.set_title('Serie temporal de Δt', loc='left', fontsize=11)
    style_axes(ax)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_dt_hist(dt, s, path, bins, log_y):
    dt_pos = dt[dt > 0]
    fig, ax = plt.subplots(figsize=(8, 4))
    # Rango centrado en el cuerpo de la distribución para que no lo aplasten los outliers
    hi = max(s['p999'], s['mediana'] * 1.5)
    body = dt_pos[dt_pos <= hi]
    ax.hist(body, bins=bins, color=C_DATA, edgecolor='white', linewidth=0.5)
    ax.axvline(s['mediana'], color=C_REF, linewidth=1.2, label=f'mediana = {s["mediana"]:.4f} s')
    if s['nominal_rate']:
        ax.axvline(1.0 / s['nominal_rate'], color=C_REF, linewidth=1, linestyle=':',
                   label=f'período nominal = {1.0 / s["nominal_rate"]:.4f} s')
    n_out = int(np.sum(dt_pos > hi))
    title = 'Histograma de Δt'
    if n_out:
        title += f'  ({n_out} valores > {hi:.3f} s fuera del rango graficado)'
    ax.set_title(title, loc='left', fontsize=11)
    ax.set_xlabel('Δt (s)')
    ax.set_ylabel('frecuencia' + (' (escala log)' if log_y else ''))
    if log_y:
        ax.set_yscale('log')
    style_axes(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='Auditoría temporal rápida de un CSV de /cmd_vel')
    p.add_argument('--csv', required=True, help='CSV con columnas t_sec, v, w')
    p.add_argument('--outdir', default='outputs_u1', help='carpeta de salida')
    p.add_argument('--nominal-rate', type=float, default=None,
                   help='frecuencia nominal (Hz), para comparar')
    p.add_argument('--k-ratio', type=float, default=3.0,
                   help='regla de razón: dt > k * mediana (por defecto 3)')
    p.add_argument('--hampel-c', type=float, default=3.0,
                   help='constante del identificador de Hampel (por defecto 3)')
    p.add_argument('--bins', type=int, default=60, help='bins del histograma')
    p.add_argument('--log-hist', action='store_true', help='histograma con eje y logarítmico')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_csv(args.csv)
    t_rel, dt, s, flagged, mask_ratio = compute_stats(
        df, args.k_ratio, args.hampel_c, args.nominal_rate)

    text = write_summary(os.path.join(args.outdir, 'summary_stats.txt'), s, flagged, args.csv)
    if len(flagged):
        flagged.to_csv(os.path.join(args.outdir, 'dt_outliers.csv'), index=False)

    plot_velocity(df, t_rel, os.path.join(args.outdir, 'velocity_signals.png'))
    plot_dt_series(t_rel, dt, s, mask_ratio, os.path.join(args.outdir, 'dt_series.png'))
    plot_dt_hist(dt, s, os.path.join(args.outdir, 'dt_histogram.png'), args.bins, args.log_hist)

    print(text)
    print(f'Figuras y resumen guardados en: {args.outdir}/')


if __name__ == '__main__':
    main()
