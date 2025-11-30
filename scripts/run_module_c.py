# scripts/run_module_c.py
"""
MÓDULO C – Análisis QNM refinado por evento y detector
------------------------------------------------------

- Usa TimeSeries.fetch_open_data (GWpy + GWOSC) para los 10 eventos.
- Preprocesa: detrend + bandpass [20, 1024] Hz.
- Ajusta un modo cuasinormal amortiguado:
    h(t) = A * exp(-(t - t0)/tau) * cos(2π f (t - t0) + phi)  para t >= t0
         = 0                                                  para t < t0
- Resume resultados por evento y detector (H1, L1).
- Guarda:
    results/qnm_moduleC_summary.json
    results/qnm_moduleC_summary.csv

Este módulo NO depende de los .hdf5 de data/, así no rompemos nada de A/B.
"""

import json
import os
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
from gwpy.timeseries import TimeSeries
from scipy.optimize import curve_fit
from scipy.signal import tukey


# -------------------------------------------------------------------
# 1. Configuración básica: eventos, GPS y detectores
# -------------------------------------------------------------------

# GPS centrales que ya confirmamos en el pipeline (Módulo A)
EVENT_GPS = {
    "GW150914": 1126259462.4,
    "GW151226": 1135136350.6,
    "GW170104": 1167559936.6,
    "GW170608": 1180922494.5,
    "GW170814": 1186741861.5,
    "GW170729": 1185389807.3,
    "GW170823": 1187529256.5,
    "GW190412": 1239082262.1,
    "GW190521": 1242442967.4,
    "GW190814": 1249852257.0,
}

DETECTORS = ["H1", "L1"]

# Ventana total alrededor del evento (en segundos)
DT = 8.0  # 4 s antes, 4 s después
# Ventana de ajuste de ringdown (relativa al GPS)
RING_T_START = 0.003   # 3 ms después del GPS
RING_T_END = 0.050     # 50 ms después del GPS (se puede ajustar)


# -------------------------------------------------------------------
# 2. Estructuras de datos
# -------------------------------------------------------------------

@dataclass
class QNMFitResult:
    event: str
    detector: str
    f_Hz: float
    tau_s: float
    A: float
    phi: float
    t0_rel: float
    success: bool
    message: str


# -------------------------------------------------------------------
# 3. Funciones de modelo y preprocesado
# -------------------------------------------------------------------

def damped_sinusoid(t: np.ndarray,
                    A: float,
                    f: float,
                    tau: float,
                    phi: float,
                    t0: float) -> np.ndarray:
    """
    Modo cuasinormal amortiguado activado en t >= t0.

    h(t) = A * e^{-(t - t0)/tau} * cos(2π f (t - t0) + phi)   (t >= t0)
         = 0                                                  (t < t0)
    """
    x = t - t0
    # Anulamos antes de t0 para que el ajuste se enfoque en el ringdown
    mask = x >= 0.0
    y = np.zeros_like(t)
    if tau <= 0:
        return y
    y[mask] = A * np.exp(-x[mask] / tau) * np.cos(2 * np.pi * f * x[mask] + phi)
    return y


def fetch_and_preprocess(event: str,
                         det: str,
                         dt: float = DT,
                         fmin: float = 20.0,
                         fmax: float = 1024.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Descarga y preprocesa el strain del detector para un evento:
    - TimeSeries.fetch_open_data
    - detrend(linear)
    - bandpass
    - ventana de Tukey suave en los bordes
    Devuelve:
        t_rel: tiempos en segundos relativos al GPS (0 en el GPS)
        h:     strain preprocesado
    """
    gps = EVENT_GPS[event]
    t0 = gps - dt / 2.0
    t1 = gps + dt / 2.0

    # GWpy ya está funcionando bien en tu entorno
    ts = TimeSeries.fetch_open_data(det, t0, t1, cache=True)
    ts = ts.detrend("linear").bandpass(fmin, fmax)

    # Ventana de Tukey para evitar bordes feos
    w = tukey(len(ts), alpha=0.1)
    h = ts.value * w

    # Convertimos tiempos a segundos relativos al GPS
    t_rel = ts.times.value - gps

    return t_rel, h


def fit_qnm_for_event_det(event: str, det: str) -> QNMFitResult:
    """
    Ajusta un modo QNM a la señal de un evento/detector.
    Usa una ventana corta después del GPS (ringdown).
    """
    try:
        t_rel, h = fetch_and_preprocess(event, det)
    except Exception as e:
        return QNMFitResult(
            event=event,
            detector=det,
            f_Hz=np.nan,
            tau_s=np.nan,
            A=np.nan,
            phi=np.nan,
            t0_rel=np.nan,
            success=False,
            message=f"Error descargando/preprocesando: {e}",
        )

    # Seleccionamos sólo la parte de ringdown
    mask = (t_rel >= RING_T_START) & (t_rel <= RING_T_END)
    t_fit = t_rel[mask]
    h_fit = h[mask]

    if len(t_fit) < 10:
        return QNMFitResult(
            event=event,
            detector=det,
            f_Hz=np.nan,
            tau_s=np.nan,
            A=np.nan,
            phi=np.nan,
            t0_rel=np.nan,
            success=False,
            message="Muy pocos puntos en ventana de ringdown",
        )

    # Estimaciones iniciales razonables
    A0 = float(np.max(np.abs(h_fit)) or 1e-22)
    f0 = 1500.0       # Hz, valor típico ~1–2 kHz
    tau0 = 0.01       # s
    phi0 = 0.0
    t00 = RING_T_START

    p0 = (A0, f0, tau0, phi0, t00)

    # Límites amplios pero físicos
    bounds = (
        [0.0, 100.0, 1e-4, -2 * np.pi, RING_T_START - 0.005],
        [np.inf, 4000.0, 1.0, 2 * np.pi, RING_T_START + 0.01],
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                damped_sinusoid,
                t_fit,
                h_fit,
                p0=p0,
                bounds=bounds,
                maxfev=20000,
            )
        A, f, tau, phi, t0 = [float(x) for x in popt]

        msg = "OK"
        success = True

    except Exception as e:
        A = f = tau = phi = t0 = np.nan
        msg = f"Error en ajuste: {e}"
        success = False

    return QNMFitResult(
        event=event,
        detector=det,
        f_Hz=f,
        tau_s=tau,
        A=A,
        phi=phi,
        t0_rel=t0,
        success=success,
        message=msg,
    )


# -------------------------------------------------------------------
# 4. Ejecución del Módulo C
# -------------------------------------------------------------------

def run_module_c() -> List[QNMFitResult]:
    """
    Ejecuta el Módulo C para todos los eventos y detectores.
    Imprime una tabla en consola y guarda JSON/CSV.
    """
    results: List[QNMFitResult] = []

    print("=== MÓDULO C – AJUSTE QNM DETALLADO ===\n")
    print("EVENTO     DET   f_QNM [Hz]      tau [s]      OK?  MENSAJE")
    print("---------------------------------------------------------------")

    for ev, gps in EVENT_GPS.items():
        print(f"\n>>> EVENTO: {ev} (GPS = {gps:.3f})")
        for det in DETECTORS:
            fit = fit_qnm_for_event_det(ev, det)
            results.append(fit)

            ok_str = "YES" if fit.success else " NO"
            f_str = f"{fit.f_Hz:10.2f}" if np.isfinite(fit.f_Hz) else "   nan"
            tau_str = f"{fit.tau_s:10.4f}" if np.isfinite(fit.tau_s) else "     nan"

            print(
                f"{ev:9s}  {det:2s}  {f_str}   {tau_str}   {ok_str}  {fit.message[:60]}"
            )

    # Crear carpeta de resultados si no existe
    os.makedirs("results", exist_ok=True)

    # Guardar JSON
    json_path = os.path.join("results", "qnm_moduleC_summary.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # Guardar CSV simple
    csv_path = os.path.join("results", "qnm_moduleC_summary.csv")
    with open(csv_path, "w") as f:
        f.write(
            "event,detector,f_Hz,tau_s,A,phi,t0_rel,success,message\n"
        )
        for r in results:
            f.write(
                f"{r.event},{r.detector},"
                f"{r.f_Hz if np.isfinite(r.f_Hz) else ''},"
                f"{r.tau_s if np.isfinite(r.tau_s) else ''},"
                f"{r.A if np.isfinite(r.A) else ''},"
                f"{r.phi if np.isfinite(r.phi) else ''},"
                f"{r.t0_rel if np.isfinite(r.t0_rel) else ''},"
                f"{int(r.success)},\"{r.message}\"\n"
            )

    print("\n=== MÓDULO C – COMPLETADO ===")
    print(f"Resumen JSON: {json_path}")
    print(f"Resumen CSV : {csv_path}")

    return results


if __name__ == "__main__":
    run_module_c()
