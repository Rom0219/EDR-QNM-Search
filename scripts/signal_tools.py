import os
import numpy as np
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
from gwpy.plot import Plot

# ======================================
#  WHITENING ESTABLE (fallback)
# ======================================
def whiten_stable(ts):
    """
    Whitening robusto:
    - Si falla ts.whiten(), se usa FFT manual.
    """
    try:
        return ts.whiten(2, 2)
    except Exception:
        print("⚠ Whitening alternativo (manual FFT)")

        data = ts.value
        fs = ts.sample_rate.value

        # FFT
        fdata = np.fft.rfft(data)
        psd = np.abs(fdata) ** 2

        # evitar ceros
        psd = np.where(psd == 0, 1e-20, psd)

        white = np.fft.irfft(fdata / np.sqrt(psd))

        return TimeSeries(white, sample_rate=fs)


# ======================================
#  FILTRO PROFUNDO (BANDA EDR)
# ======================================
def deep_filter(ts):
    """
    Filtro fuerte para buscar QNM EDR:
    - Banda 20–600 Hz
    - Notches en 60 Hz y armónicos
    """
    out = ts.bandpass(20, 600)
    for n in range(1, 6):
        out = out.notch(60 * n)
    return out


# ======================================
#  GRAFICAR FFT
# ======================================
def plot_fft(ts, out_path):
    fft = ts.aspectrum()
    p = Plot(fft, title="FFT Espectro")
    p.savefig(out_path)
    print("✔ Guardado FFT:", out_path)


# ======================================
#  GRAFICAR Q-TRANSFORM (modo EDR)
# ======================================
def plot_qtransform(ts, out_path):
    q = ts.q_transform(outseg=(ts.t0.value, ts.t1.value))
    p = Plot(q, interpolation="bicubic", title="Q-Transform (modo EDR)")
    p.colorbar(label="Amplitud")
    p.savefig(out_path)
    print("✔ Guardado Q-transform:", out_path)


# ======================================
#  GRAFICAR TIEMPO
# ======================================
def plot_timeseries(ts, out_path):
    p = Plot(ts, title="Señal en el tiempo")
    p.savefig(out_path)
    print("✔ Guardado TimeSeries:", out_path)


# ======================================
#  PROCESO COMPLETO PARA ANÁLISIS DE QNM
# ======================================
def analyze_event(event_name, det, ts_white):
    """
    Genera automáticamente:
    - FFT
    - Q-transform
    - Timeseries filtrada
    """

    base = f"outputs/{event_name}_{det}"
    os.makedirs(base, exist_ok=True)

    # filtrado profundo
    deep = deep_filter(ts_white)

    # Plot time
    plot_timeseries(deep, os.path.join(base, "time.png"))

    # FFT
    plot_fft(deep, os.path.join(base, "fft.png"))

    # Q-transform
    plot_qtransform(deep, os.path.join(base, "qtransform.png"))

    print(f"=== ANÁLISIS COMPLETO: {event_name}/{det} ===")
