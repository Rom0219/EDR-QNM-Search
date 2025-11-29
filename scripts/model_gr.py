"""
scripts/model_gr.py

Funciones para calcular la frecuencia (Hz) y tiempo de decaimiento (tau, s)
del modo fundamental QNM (l=m=2, n=0) en Relatividad General,
usando los fits estándar (Berti et al. style).

Uso:
    from scripts.model_gr import qnm_freq_and_tau
    f0, tau = qnm_freq_and_tau(M_rem_solar, chi_rem)

Donde:
 - M_rem_solar: masa del remanente en masas solares (M_sun)
 - chi_rem: spin adimensional del remanente (0 <= chi < 1)

Referencias:
 - E. Berti et al., review & fits (ver enlaces en repo/README).
"""

import numpy as np

# Constantes físicas
G = 6.67430e-11        # m^3 kg^-1 s^-2
c = 299792458.0        # m / s
M_sun = 1.98847e30     # kg

# Coeficientes de ajuste para l=m=2, n=0 (fitting)
# Fuente: literatura (Berti et al. fits / Nakano et al. summary).
f1 = 1.5251
f2 = -1.1568
f3 = 0.1292

q1 = 0.7000
q2 = 1.4187
q3 = -0.4990


def dimensionless_omegaR(chi):
    """
    Calcula el valor adimensional M * omega_R (geometric units) usando el fit:
      M * omega_R = f1 + f2 * (1 - chi)**f3

    Parámetro:
      chi: spin adimensional (0..1)

    Retorna:
      MomegaR (float)
    """
    return f1 + f2 * (1.0 - chi)**f3


def quality_factor_Q(chi):
    """
    Calcula el quality factor Q usando el fit:
      Q = q1 + q2 * (1 - chi)**q3

    Parámetro:
      chi: spin adimensional (0..1)

    Retorna:
      Q (float)
    """
    return q1 + q2 * (1.0 - chi)**q3


def qnm_freq_and_tau(M_rem_solar, chi_rem):
    """
    Devuelve la frecuencia f0 en Hz y el tiempo de decaimiento tau en segundos
    para el modo (l=m=2,n=0) en GR, dado remnant mass y spin.

    Entradas:
      M_rem_solar: masa remanente en masas solares (float)
      chi_rem: spin adimensional (float, entre 0 y <1)

    Salidas:
      f0_hz: frecuencia (Hz)
      tau_s: tiempo de decaimiento e-fold (s)
    """

    # Seguridad básica
    if not (0.0 <= chi_rem < 1.0):
        raise ValueError("chi_rem debe estar en [0, 1).")

    # Masa en unidades SI
    M_kg = M_rem_solar * M_sun

    # Valor adimensional M * omega_R (omega in geometric units, c=G=1)
    M_omegaR = dimensionless_omegaR(chi_rem)

    # Convertir M * omega (geometric) a frecuencia en Hz:
    # omega_geom = M_omegaR / M_geom, with M_geom = G*M/c^3
    # omega_SI = omega_geom * c^3 / (G*M) = M_omegaR * c^3 / (G * M_kg)
    omega_si = M_omegaR * c**3 / (G * M_kg)    # rad/s

    # frecuencia en Hz
    f0_hz = omega_si / (2.0 * np.pi)

    # calidad Q
    Q = quality_factor_Q(chi_rem)

    # relacion entre Q, f y tau: Q = pi * f * tau  -> tau = Q / (pi * f)
    tau_s = Q / (np.pi * f0_hz)

    return f0_hz, tau_s


# Ejemplo de uso:
if __name__ == "__main__":
    # ejemplo: remnant de 60 Msun y spin 0.7
    f0, tau = qnm_freq_and_tau(60.0, 0.7)
    print(f"f0 = {f0:.3f} Hz, tau = {tau:.4f} s")
