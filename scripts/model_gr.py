"""
scripts/model_gr.py

Módulo completo para generar plantillas GR de ringdown:
 - Cálculo de frecuencia y tiempo de decaimiento del modo (2,2,0)
   usando fits estándar de Berti et al.
 - Modos adicionales: (3,3,0) y (2,1,0)
 - Construcción de plantillas damped-sinusoid con:
        amplitud
        fase
        t0 (inicio del ringdown)
        ventana suave (planck-taper)
 - Sumatoria multi-modo opcional

Referencias principales usadas en toda la comunidad (ajustes numéricos):
 - E. Berti, V. Cardoso, C. Will, "Gravitational-wave spectroscopy…"
   (fits para QNM, frecuencia y calidad)
 - Nakano et al., resúmenes de parámetros QNM para Kerr.
"""

import numpy as np

# ============================================================
# Constantes físicas
# ============================================================
G = 6.67430e-11
c = 299792458.0
M_sun = 1.98847e30


# ============================================================
# Coeficientes de ajuste (Berti) para el modo (l=m=2,n=0)
# ============================================================
f1_22 = 1.5251
f2_22 = -1.1568
f3_22 = 0.1292

q1_22 = 0.7000
q2_22 = 1.4187
q3_22 = -0.4990


# ============================================================
# Coeficientes para modo (3,3,0)
# ============================================================
f1_33 = 1.8956
f2_33 = -1.3043
f3_33 = 0.1818

q1_33 = 1.0000
q2_33 = 1.4170
q3_33 = -0.4990


# ============================================================
# Coeficientes para modo (2,1,0)
# ============================================================
f1_21 = 0.6000
f2_21 = -0.2339
f3_21 = 0.4175

q1_21 = 0.5000
q2_21 = 0.7000
q3_21 = -0.3000


# ============================================================
# Función general de fits (frecuencia adimensional M omega_R)
# ============================================================
def MomegaR(chi, f1, f2, f3):
    return f1 + f2 * (1.0 - chi)**f3


def quality_factor(chi, q1, q2, q3):
    return q1 + q2 * (1.0 - chi)**q3


# ============================================================
# Conversión de MωR a frecuencia real y tau
# ============================================================
def freq_tau(M_solar, chi, mode="22"):
    """
    Retorna f0 (Hz) y tau (s) para el modo solicitado: "22", "33" o "21".
    """

    M_kg = M_solar * M_sun

    if mode == "22":
        f1, f2, f3 = f1_22, f2_22, f3_22
        q1, q2, q3 = q1_22, q2_22, q3_22
    elif mode == "33":
        f1, f2, f3 = f1_33, f2_33, f3_33
        q1, q2, q3 = q1_33, q2_33, q3_33
    elif mode == "21":
        f1, f2, f3 = f1_21, f2_21, f3_21
        q1, q2, q3 = q1_21, q2_21, q3_21
    else:
        raise ValueError("Modo no soportado. Usa '22', '33' o '21'.")

    # M omega_R en unidades geométricas
    Momega = MomegaR(chi, f1, f2, f3)

    # omega físico SI
    omega = Momega * c**3 / (G * M_kg)

    f0 = omega / (2.0 * np.pi)

    Q = quality_factor(chi, q1, q2, q3)
    tau = Q / (np.pi * f0)

    return f0, tau


# ============================================================
# Ventana de suavizado tipo Planck-taper
# ============================================================
def planck_taper(t, t0, eps=0.002):
    """
    Ventana suave para cambiar de 0 a 1 alrededor de t0.
    Controla que el ringdown inicie suavemente.
    """
    w = np.zeros_like(t)
    dt = eps * (t[-1] - t[0])

    mask = t >= t0
    if not np.any(mask):
        return w

    ti = t0
    tf = t0 + dt

    for i in range(len(t)):
        if t[i] < ti:
            w[i] = 0.0
        elif t[i] > tf:
            w[i] = 1.0
        else:
            x = (t[i] - ti) / (tf - ti)
            w[i] = 1.0 / (np.exp(1.0 / x + 1.0 / (1.0 - x)) + 1.0)

    return w


# ============================================================
# Plantilla damped-sine general
# ============================================================
def damped_sine(t, A, f0, tau, phi, t0):
    """
    Señal ringdown básica: A * exp(-(t-t0)/tau) * sin(2πf0(t-t0)+phi)
    Con ventana suave planck-taper.
    """
    s = np.zeros_like(t)

    # Aplicar solo para t >= t0
    mask = t >= t0
    if not np.any(mask):
        return s

    tt = t[mask] - t0
    s[mask] = A * np.exp(-tt/tau) * np.sin(2.0 * np.pi * f0 * tt + phi)

    # Aplicar ventana suave
    w = planck_taper(t, t0)
    return s * w


# ============================================================
# PLANTILLA MULTIMODO
# ============================================================
def gr_multimode_template(fs, duration, M_solar, chi, t0=0.01,
                          A22=1.0, A33=0.0, A21=0.0,
                          phi22=0.0, phi33=0.0, phi21=0.0):
    """
    Retorna tiempo t y señal total sumando modos GR:
        (22), (33) y (21)
    """
    t = np.arange(0, duration, 1.0/fs)

    # Modo 22
    f22, tau22 = freq_tau(M_solar, chi, "22")
    s22 = damped_sine(t, A22, f22, tau22, phi22, t0)

    # Modo 33
    if A33 != 0:
        f33, tau33 = freq_tau(M_solar, chi, "33")
        s33 = damped_sine(t, A33, f33, tau33, phi33, t0)
    else:
        s33 = 0.0

    # Modo 21
    if A21 != 0:
        f21, tau21 = freq_tau(M_solar, chi, "21")
        s21 = damped_sine(t, A21, f21, tau21, phi21, t0)
    else:
        s21 = 0.0

    total = s22 + s33 + s21

    return t, total
