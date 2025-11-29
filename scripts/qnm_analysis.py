import os
import json
import numpy as np
from scipy.optimize import curve_fit
from gwpy.timeseries import TimeSeries

WHITE_PATH = "data/white"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# misma lista de eventos que en run_pipeline.py
EVENTS = {
    "GW150914":   {"gps": 1126259462, "run": "O1"},
    "GW151226":   {"gps": 1135136350, "run": "O1"},
    "GW170104":   {"gps": 1167559936, "run": "O2"},
    "GW170608":   {"gps": 1180922494, "run": "O2"},
    "GW170814":   {"gps": 1186741861, "run": "O2"},
    "GW170729":   {"gps": 1185389807, "run": "O2"},
    "GW170823":   {"gps": 1187529256, "run": "O2"},
    "GW190412":   {"gps": 1239082262, "run": "O3"},
    "GW190521":   {"gps": 1242442967, "run": "O3"},
    "GW190814":   {"gps": 1249852257, "run": "O3"},
}

DET_LIST = ["H1", "L1"]


# -----------------------------
# Modelo: seno amortiguado
# -----------------------------
def damped_sine(t, A, f, tau, phi, C):
    return A * np.exp(-(t - t[0]) / tau) * np.sin(2 * np.pi * f * (t - t[0]) + phi) + C


def fit_qnm(ts, t0_offset=0.01, t_window=0.2):
    """
    t0_offset: segundos después del máximo para empezar el ringdown.
    t_window: duración de la ventana de ajuste (segundos).
    """
    # tomar datos como arrays
    t = ts.times.value
    y = ts.value

    # tomar máximo (amplitud) como referencia
    imax = np.argmax(np.abs(y))
    t0 = t[imax] + t0_offset
    t1 = t0 + t_window

    # recortar ventana
    mask = (t >= t0) & (t <= t1)
    t_win = t[mask]
    y_win = y[mask]

    if len(t_win) < 10:
        raise RuntimeError("Ventana de ringdown demasiado corta para ajuste")

    # estimaciones iniciales:
    dt = np.median(np.diff(t_win))
    fs = 1.0 / dt

    # frecuencia estimada vía FFT rápida
    yf = np.fft.rfft(y_win)
    xf = np.fft.rfftfreq(len(y_win), dt)
    f_guess = xf[np.argmax(np.abs(yf))]

    A_guess = np.max(np.abs(y_win))
    tau_guess = 0.05  # 50 ms
    phi_guess = 0.0
    C_guess = 0.0

    p0 = [A_guess, f_guess, tau_guess, phi_guess, C_guess]

    popt, pcov = curve_fit(damped_sine, t_win, y_win, p0=p0, maxfev=10000)

    A, f, tau, phi, C = popt
    # errores (desviaciones estándar aprox.)
    perr = np.sqrt(np.diag(pcov))
    A_err, f_err, tau_err, phi_err, C_err = perr

    Q = np.pi * f * tau  # factor de calidad aproximado

    return {
        "t0_ringdown": float(t0),
        "A": float(A),
        "A_err": float(A_err),
        "f_Hz": float(f),
        "f_err_Hz": float(f_err),
        "tau_s": float(tau),
        "tau_err_s": float(tau_err),
        "Q": float(Q),
        "phi_rad": float(phi),
        "phi_err_rad": float(phi_err),
        "C": float(C),
        "C_err": float(C_err),
        "n_points": int(len(t_win)),
        "dt_s": float(dt),
    }


def analyze_all_events():
    results = {}

    for ev_name in EVENTS.keys():
        for det in DET_LIST:
            key = f"{ev_name}_{det}"
            fname = os.path.join(WHITE_PATH, f"{ev_name}_{det}_white.hdf5")

            print(f"\n>>> Analizando QNM: {ev_name} / {det}")

            if not os.path.exists(fname):
                print("✖ No existe archivo blanqueado:", fname)
                continue

            try:
                ts = TimeSeries.read(fname, format="hdf5")
            except Exception as e:
                print("✖ Error leyendo:", e)
                continue

            try:
                qnm = fit_qnm(ts)
                print("✔ f_QNM ≈ {:.1f} Hz, tau ≈ {:.4f} s, Q ≈ {:.1f}".format(
                    qnm["f_Hz"], qnm["tau_s"], qnm["Q"]
                ))
                results[key] = qnm
            except Exception as e:
                print("✖ Error ajustando QNM:", e)
                continue

    # Guardar reporte JSON
    out_json = os.path.join(REPORTS_DIR, "EDR_QNM_fit_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n====================================")
    print("  REPORTE QNM GUARDADO EN:")
    print(" ", out_json)
    print("====================================")


if __name__ == "__main__":
    analyze_all_events()
