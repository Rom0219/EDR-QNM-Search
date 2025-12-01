"""
scripts/model_edr_full.py

Modelo EDR avanzado para ringdown:
 - Basado en GR (Berti) vía freq_tau()
 - Desviaciones independientes por modo:
      (22): δω22/ω22, δτ22/τ22
      (33): δω33/ω33, δτ33/τ33
      (21): δω21/ω21, δτ21/τ21
 - Amplitudes y fases independientes por modo
 - Inicio común de ringdown t0

Este módulo no usa aún la ecuación completa EDR-Field,
pero deja listo el espacio de parámetros para mapear luego:
   R_edr, lambda_edr, alpha_edr, etc.
"""

import numpy as np
from scripts.model_gr import freq_tau


def edr_shift_freq_tau(f_gr, tau_gr, d_om, d_tau):
    """
    Aplica desviación relativa EDR:
      f_EDR   = f_GR   * (1 + d_om)
      tau_EDR = tau_GR * (1 + d_tau)
    """
    f = f_gr * (1.0 + d_om)
    tau = tau_gr * (1.0 + d_tau)
    return f, tau


def damped_sine_mode(t, A, f0, tau, phi, t0):
    """
    Ringdown básico para un solo modo:
      h(t) = A exp(-(t-t0)/tau) sin(2π f0 (t-t0) + phi), t>=t0
    """
    s = np.zeros_like(t)
    mask = t >= t0
    if not np.any(mask):
        return s

    tt = t[mask] - t0
    s[mask] = A * np.exp(-tt / tau) * np.sin(2.0 * np.pi * f0 * tt + phi)
    return s


def edr_multimode_full(
    t,
    Mrem,
    chi,
    # amplitudes
    A22,
    A33,
    A21,
    # desviaciones relativas por modo
    d_om22,
    d_tau22,
    d_om33,
    d_tau33,
    d_om21,
    d_tau21,
    # fases
    phi22,
    phi33,
    phi21,
    # inicio común de ringdown
    t0,
):
    """
    Construye la señal total EDR multimodo con parámetros por modo.

    Parámetros físicos GR base se obtienen de freq_tau(Mrem, chi, mode).
    Luego cada modo se deforma con (d_omXX, d_tauXX).
    """

    # --- Modo 22 base GR ---
    f22_gr, tau22_gr = freq_tau(Mrem, chi, "22")
    f22, tau22 = edr_shift_freq_tau(f22_gr, tau22_gr, d_om22, d_tau22)
    h22 = damped_sine_mode(t, A22, f22, tau22, phi22, t0)

    # --- Modo 33 ---
    if A33 != 0.0:
        f33_gr, tau33_gr = freq_tau(Mrem, chi, "33")
        f33, tau33 = edr_shift_freq_tau(f33_gr, tau33_gr, d_om33, d_tau33)
        h33 = damped_sine_mode(t, A33, f33, tau33, phi33, t0)
    else:
        h33 = 0.0

    # --- Modo 21 ---
    if A21 != 0.0:
        f21_gr, tau21_gr = freq_tau(Mrem, chi, "21")
        f21, tau21 = edr_shift_freq_tau(f21_gr, tau21_gr, d_om21, d_tau21)
        h21 = damped_sine_mode(t, A21, f21, tau21, phi21, t0)
    else:
        h21 = 0.0

    h_total = h22 + h33 + h21

    return h_total
