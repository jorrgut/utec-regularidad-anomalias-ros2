"""Estilo común de las figuras de la Unidad 4 (sin pantalla: guarda archivos)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

C_DATA = '#2a6fb0'    # serie principal
C_ALT = '#e07b39'     # segunda serie (por ejemplo, AR frente a persistencia)
C_REF = '#7f7f7f'     # referencias, umbrales
C_FLAG = '#c0392b'    # alarmas / anomalía
C_SHADE = '#f2d7d5'   # sombreado de la anomalía real

# Una tinta fija por señal sintética (el color sigue a la señal, no a su posición)
SIGNAL_COLORS = {
    'Structured': '#2a6fb0',
    'Shuffled': '#e07b39',
    'IID': '#7a5195',
    'Periodic': '#2f9e6e',
    'Slow': '#b8860b',
}


def style_axes(ax):
    ax.grid(True, color='#e5e5e5', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#999999')
    ax.tick_params(colors='#444444', labelsize=9)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
