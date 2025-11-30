import os
import h5py
import numpy as np
from dataclasses import dataclass
from scipy.optimize import curve_fit

# ======================================================
# 1) Estructura para guardar resultados
# ======================================================

@dataclass
class QNMResult:
    event: str
    detector: str
    f_qnm: float        # frecuencia [Hz]
    tau: float          # tiempo de decaimiento [s]
    A: float            # amplitud
    phi: float          # fase [rad]
    t0: float           # inicio de ringdown [s]
    success: bool
    message: str


# ======================================================
# 2) Utilidades de carga
# ======================================================

def load_white_timeseries(event_name: str, detector: str,
                          base_dir: str = "data/white",
                          t_pre: float = 4.0):
    """
    Carga el strain blanqueado desde HDF5 y construye el eje de tiempo
    relativo al evento (t=0 en el GPS del catálogo).
    Asumimos que el archivo se generó con una ventana [gps-4, gps+4].
    """
    fname = os.path.join(base_dir, f"{event_name}_{detector}_white.hdf5")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"No existe archivo WHITE: {fname}")

    with h5py.File(fname, "r") as f:
        data = f["strain"][:]
        fs = float(f["strain"].attrs["fs"])

    n = len(data)
    dt = 1.0 / fs
    # Tiempo relativo: empieza en -t_pre y termina en -t_pre + n*dt
    t = np.arange(n) * dt - t_pre

    return t, data, fs


# ======================================================
# 3) Ringdown: recorte de ventana
# ======================================================

def select_ringdown_window(t, h,
                           t_start: float = 0.01,
                           t_end: float = 0.30):
    """
    Selecciona sólo el pedazo de señal donde esperamos el ringdown.
    Por defecto: entre 0.01 s y 0.30 s después del merger (t=0).
    """
    mask = (t >= t_start) & (t <= t_end)
    t_rd = t[mask]
    h_rd = h[mask]

    if len(t_rd) < 10:
        raise ValueError("Muy pocos puntos en la ventana de ringdown.")

    # Recentramos t para que el fit vea t=0 al inicio del ringdown
    t0 = t_rd[0]
    return t_rd - t0, h_rd, t0


# ======================================================
# 4) Modelo de seno amortiguado y ajustes
# ======================================================

def damped_sinusoid(t, A, f, tau, phi):
    """
    h(t) = A * exp(-t/tau) * cos(2π f t + phi)
    """
    return A * np.exp(-t / tau) * np.cos(2 * np.pi * f * t + phi)


def estimate_initial_frequency(t, h, fmin=50.0, fmax=1000.0):
    """
    Estimación grosera de la frecuencia dominante usando el máximo
    del módulo de la FFT en un rango [fmin, fmax].
    """
    dt = t[1] - t[0]
    n = len(t)

    # FFT real
    freqs = np.fft.rfftfreq(n, dt)
    Hf = np.fft.rfft(h)

    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        raise ValueError("Rango de frecuencias para QNM vacío.")

    freqs_sel = freqs[mask]
    Hf_sel = np.abs(Hf[mask])

    idx_max = np.argmax(Hf_sel)
    f_peak = freqs_sel[idx_max]
    return float(f_peak)


def fit_qnm(t_rd, h_rd):
    """
    Ajusta el modelo de seno amortiguado a los datos de ringdown.
    Devuelve parámetros (A, f, tau, phi).
    """
    # Estimaciones iniciales
    A0 = float(np.max(np.abs(h_rd)))
    f0 = estimate_initial_frequency(t_rd, h_rd)   # Hz
    tau0 = 0.05                                   # 50 ms: orden razonable
    phi0 = 0.0

    p0 = [A0, f0, tau0, phi0]

    # Pesos: opcional, aquí simples
    try:
        popt, pcov = curve_fit(
            damped_sinusoid,
            t_rd,
            h_rd,
            p0=p0,
            maxfev=10000
        )
        A_fit, f_fit, tau_fit, phi_fit = popt
        return A_fit, f_fit, tau_fit, phi_fit, True, "OK"
    except Exception as e:
        return A0, f0, tau0, phi0, False, f"Error en ajuste: {e}"


# ======================================================
# 5) Pipeline por evento / detector
# ======================================================

def analyze_qnm_for_event_detector(event_name: str,
                                   detector: str,
                                   t_pre: float = 4.0,
                                   t_start: float = 0.01,
                                   t_end: float = 0.30) -> QNMResult:
    """
    Pipeline completo:
    - Carga datos WHITE
    - Selecciona ringdown
    - Ajusta seno amortiguado
    """
    try:
        t, h_white, fs = load_white_timeseries(
            event_name=event_name,
            detector=detector,
            t_pre=t_pre
        )
    except Exception as e:
        return QNMResult(
            event=event_name,
            detector=detector,
            f_qnm=np.nan,
            tau=np.nan,
            A=np.nan,
            phi=np.nan,
            t0=np.nan,
            success=False,
            message=f"Error cargando: {e}"
        )

    try:
        t_rd, h_rd, t0 = select_ringdown_window(
            t, h_white,
            t_start=t_start,
            t_end=t_end
        )
    except Exception as e:
        return QNMResult(
            event=event_name,
            detector=detector,
            f_qnm=np.nan,
            tau=np.nan,
            A=np.nan,
            phi=np.nan,
            t0=np.nan,
            success=False,
            message=f"Error seleccionando ringdown: {e}"
        )

    A_fit, f_fit, tau_fit, phi_fit, ok, msg = fit_qnm(t_rd, h_rd)

    return QNMResult(
        event=event_name,
        detector=detector,
        f_qnm=float(f_fit),
        tau=float(tau_fit),
        A=float(A_fit),
        phi=float(phi_fit),
        t0=float(t0),
        success=ok,
        message=msg
    )
