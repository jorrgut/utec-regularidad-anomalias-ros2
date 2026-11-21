"""
u6_models.py - Modelos de la Unidad 6 (PyTorch) y la línea base lineal (PCA, numpy).

Contenido:
  WindowAE          autoencoder denso de ventana: (B, T, d) -> (B, T, d)
                    con T = 1 es el autoencoder PUNTUAL de la actividad 1
  LSTMPredictor     predictor de un paso: (B, T, d) -> (B, d)
  set_seed          semilla (antes de construir el modelo: fija la inicialización)
  train_model       entrenamiento con Adam y parada temprana sobre validación
  recon_errors      error de reconstrucción por ventana: medio y de la última muestra
  predict           predicción de un paso por lotes
  pred_errors       error de predicción por ventana
  PCAWindow         autoencoder LINEAL (PCA) sobre ventanas aplanadas: línea base
  save_model / load_model   pesos + metadatos (T, canales, estandarización, error nominal)

Apunte: secciones "Autoencoders", "Redes recurrentes LSTM" y "Protocolo de entrenamiento".
"""
import copy
import json
import os
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:                                   # mensaje claro en lugar de un traceback
    sys.exit('[error] Falta PyTorch. Instalalo (versión CPU):\n'
             '    pip install torch --index-url https://download.pytorch.org/whl/cpu\n'
             'Ver README, sección "Instalación".')

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

class WindowAE(nn.Module):
    """Autoencoder denso sobre la ventana aplanada: 2T -> hidden -> L -> hidden -> 2T."""

    def __init__(self, T, d=2, hidden=64, latent=4):
        super().__init__()
        self.T, self.d = T, d
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(T * d, hidden), nn.Tanh(),
                                 nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.Tanh(),
                                 nn.Linear(hidden, T * d))

    def forward(self, x):                      # x: (B, T, d)
        return self.dec(self.enc(x)).view(-1, self.T, self.d)


class LSTMPredictor(nn.Module):
    """LSTM de una capa; el último estado oculto predice el próximo vector."""

    def __init__(self, d=2, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(d, hidden, batch_first=True)
        self.head = nn.Linear(hidden, d)

    def forward(self, x):                      # x: (B, T, d) -> (B, d)
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def set_seed(seed):
    """Fijar la semilla ANTES de construir el modelo: también define la inicialización."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def _tensor(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32)


def train_model(model, Xtr, Ytr, Xva, Yva, epochs=200, lr=1e-3, batch=64, patience=15,
                seed=0, verbose=False):
    """
    Minimiza el MSE entre model(X) e Y. Para un autoencoder, Y = X.
    Mezcla el orden SOLO dentro del entrenamiento (para armar los lotes); la validación
    viene de otro bloque de tiempo. Devuelve (modelo con los mejores pesos, historia).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xtr, ytr, xva, yva = _tensor(Xtr), _tensor(Ytr), _tensor(Xva), _tensor(Yva)
    g = torch.Generator().manual_seed(seed)
    best, best_state, wait, hist = float('inf'), None, 0, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), generator=g)
        tot = 0.0
        for i in range(0, len(xtr), batch):
            j = perm[i:i + batch]
            loss = ((model(xtr[j]) - ytr[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(j)
        model.eval()
        with torch.no_grad():
            val = ((model(xva) - yva) ** 2).mean().item()
        hist.append((ep, tot / len(xtr), val))
        if verbose and ep % 10 == 0:
            print(f'    época {ep:3d}  train {tot / len(xtr):.5f}  val {val:.5f}')
        if val < best - 1e-9:
            best, best_state, wait = val, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, hist


# ---------------------------------------------------------------------------
# Errores (indicadores)
# ---------------------------------------------------------------------------

def _batched(model, W, batch=4096):
    outs = []
    with torch.no_grad():
        for i in range(0, len(W), batch):
            outs.append(model(_tensor(W[i:i + batch])).numpy())
    return np.concatenate(outs) if outs else np.empty((0,))


def recon_errors(model, W):
    """
    W: (M, T, d) estandarizado. Devuelve:
      e_mean  error cuadrático medio sobre toda la ventana  (M,)
      e_last  error de la ÚLTIMA muestra de la ventana       (M,)  (reacciona antes)
      e_chan  error medio por canal                           (M, d) (interpretación)
    """
    model.eval()
    R = _batched(model, W).reshape(W.shape)
    E = (R - W) ** 2
    return E.mean(axis=(1, 2)), E[:, -1, :].mean(axis=1), E.mean(axis=1)


def predict(model, P):
    """Predicción del próximo valor para cada ventana del pasado P (M, T, d) -> (M, d)."""
    model.eval()
    return _batched(model, P)


def pred_errors(model, P, Y):
    """P: (M, T, d) pasado, Y: (M, d) próximo valor. Devuelve el error por ventana y por canal."""
    E = (predict(model, P) - Y) ** 2
    return E.mean(axis=1), E


def predictor_pairs(Wp1):
    """De ventanas de T+1 muestras: (pasado (M, T, d), próximo (M, d))."""
    return Wp1[:, :-1], Wp1[:, -1]


# ---------------------------------------------------------------------------
# Línea base lineal
# ---------------------------------------------------------------------------

class PCAWindow:
    """Autoencoder LINEAL óptimo: proyección sobre las L primeras componentes principales."""

    def __init__(self, latent=4):
        self.L = latent

    def fit(self, W):
        A = W.reshape(len(W), -1)
        self.mean = A.mean(axis=0)
        _, S, Vt = np.linalg.svd(A - self.mean, full_matrices=False)
        self.V = Vt[:self.L].T
        self.explained = float((S[:self.L] ** 2).sum() / (S ** 2).sum())
        return self

    def errors(self, W):
        A = W.reshape(len(W), -1) - self.mean
        R = A @ self.V @ self.V.T
        return ((R - A) ** 2).mean(axis=1)


# ---------------------------------------------------------------------------
# Guardar y cargar
# ---------------------------------------------------------------------------

def save_model(prefix, model, meta):
    """Guarda prefix.pt (pesos) y prefix.json (todo lo necesario para usarlo)."""
    os.makedirs(os.path.dirname(prefix) or '.', exist_ok=True)
    torch.save(model.state_dict(), prefix + '.pt')
    with open(prefix + '.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def build_from_meta(meta):
    d = len(meta['channels'])
    if meta['kind'] == 'ae':
        return WindowAE(meta['T'], d, meta['hidden'], meta['latent'])
    return LSTMPredictor(d, meta['hidden'])


def load_model(prefix):
    with open(prefix + '.json', encoding='utf-8') as f:
        meta = json.load(f)
    model = build_from_meta(meta)
    model.load_state_dict(torch.load(prefix + '.pt', map_location='cpu'))
    model.eval()
    return model, meta
