"""
scripts/model_edr.py

Implementación de plantillas EDR:
 - Modificaciones paramétricas de la frecuencia y el tiempo de decaimiento
   respecto a GR.
 - Parámetros fundamentales:
       δω_over_ω   (delta_omega / omega)
       δtau_over_tau (delta_tau / tau)
 - Compatibles con los modos GR: (22), (33), (21)
 - Plantillas EDR multimodo opcionales
 - Ventana suave tipo Planck-taper usada en GR para iniciar el ringdown

Este archivo es el núcleo matemático de la teoría EDR dentro del pipeline.
"""

import numpy as np
from scripts.model_gr import freq_tau, planck_taper


# ============================================================
# Frecuencia EDR modificada
# ============================================================
def edr_f0(f0_gr, delta_omega_ratio):
    """
    Frecuencia EDR:
       f_EDR = f_GR * (1 + δω/ω)
    """
    return f0_gr * (1.0 + delta_omega_ratio)


# ============================================================
# Tiempo de decaimiento EDR modificado
# ============================================================
def edr_tau(tau_gr, delta_tau_ratio):
    """
    Tau EDR:
       tau_EDR = tau_GR * (1 + δtau/tau)
    """
    return tau_gr * (1.0 + delta_tau_ratio)


# ============================================================
# Señal EDR para un solo modo
# ============================================================
def edr_damped_sine(t, A, f0_gr, tau_gr, phi, t0, delta_omega_ratio, delta_tau_ratio):
    """
    Construye la señal EDR modificada de un modo específico.
    """

    # calcular parámetros EDR
    f0 = edr_f0(f0_gr, delta_omega_ratio)
    tau = edr_tau(tau_gr, delta_tau_ratio)

    s = np.zeros_like(t)

    mask = t >= t0
    if not np.any(mask):
        return s

    tt = t[mask] - t0
    s[mask] = A * np.exp(-tt/tau) * np.sin(2*np.pi*f0*tt + phi)

    # ventana suave
    w = planck_taper(t, t0)
    return s * w


# ============================================================
# EDR multimodo (22, 33, 21)
# ============================================================
def edr_multimode_template(fs, duration, M_solar, chi,
                           delta_omega_ratio=0.0,
                           delta_tau_ratio=0.0,
                           t0=0.01,
                           A22=1.0, A33=0.0, A21=0.0,
                           phi22=0.0, phi33=0.0, phi21=0.0):
    """
    Plantilla EDR con modificación paramétrica (frecuencia y tau)
    aplicada a cada modo.
    """

    t = np.arange(0, duration, 1.0/fs)

    # ====================
    # MODO 22
    # ====================
    f22_gr, tau22_gr = freq_tau(M_solar, chi, "22")
    s22 = edr_damped_sine(
        t, A22, f22_gr, tau22_gr, phi22, t0,
        delta_omega_ratio, delta_tau_ratio
    )

    # ====================
    # MODO 33
    # ====================
    if A33 != 0:
        f33_gr, tau33_gr = freq_tau(M_solar, chi, "33")
        s33 = edr_damped_sine(
            t, A33, f33_gr, tau33_gr, phi33, t0,
            delta_omega_ratio, delta_tau_ratio
        )
    else:
        s33 = 0.0

    # ====================
    # MODO 21
    # ====================
    if A21 != 0:
        f21_gr, tau21_gr = freq_tau(M_solar, chi, "21")
        s21 = edr_damped_sine(
            t, A21, f21_gr, tau21_gr, phi21, t0,
            delta_omega_ratio, delta_tau_ratio
        )
    else:
        s21 = 0.0

    return t, s22 + s33 + s21
