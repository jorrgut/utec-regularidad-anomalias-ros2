"""
u5_offline.py - Utilidades compartidas por los scripts fuera de línea de la Unidad 5:
lectura y alineación de los CSV de indicadores, y estilo de figuras.
"""
import os
import sys

import numpy as np
import pandas as pd

ENTROPY_FILE = 'entropy_metrics.csv'
PRED_FILE = 'cmd_vel_pred_metrics.csv'


def load_indicators(log_dir, tolerance=0.3):
    """
    Lee entropy_metrics.csv y cmd_vel_pred_metrics.csv de log_dir y los alinea en el
    tiempo (para cada registro de entropía, el de predicción más cercano dentro de
    'tolerance' segundos). Devuelve un DataFrame con t y los indicadores.
    """
    pe, pp = os.path.join(log_dir, ENTROPY_FILE), os.path.join(log_dir, PRED_FILE)
    for p in (pe, pp):
        if not os.path.exists(p):
            sys.exit(f'[error] no existe {p}. ¿Corrió el csv_logger (o metrics_from_csv.py)?')
    e = pd.read_csv(pe).sort_values('t')
    p = pd.read_csv(pp).sort_values('t').drop(columns=['n_msgs'], errors='ignore')
    df = pd.merge_asof(e, p, on='t', direction='nearest', tolerance=tolerance)
    # t se conserva ABSOLUTO (reloj del nodo, sim time): así coincide con los instantes que
    # imprime el nodo atacante. t_rel es el tiempo desde el primer registro.
    df['t_rel'] = df['t'] - df['t'].iloc[0]
    return df.reset_index(drop=True)


def read_events(path):
    """Lee un archivo de eventos: una línea 'inicio fin' por evento (se ignoran comentarios)."""
    ev = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if line:
                a, b = line.split()[:2]
                ev.append((float(a), float(b)))
    return ev


def style_axes(ax):
    ax.grid(True, color='#e5e5e5', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#999999')
    ax.tick_params(colors='#444444', labelsize=9)


# Una tinta fija por indicador (el color sigue al indicador, no a su posición)
IND_COLORS = {
    'mse_v': '#2a6fb0', 'mse_w': '#6a9fd4', 'Hcond_v': '#2f9e6e',
    'H_dt': '#e07b39', 'H_v': '#7a5195',
}
C_DATA, C_REF, C_FLAG, C_SHADE = '#2a6fb0', '#7f7f7f', '#c0392b', '#f2d7d5'
