import os
import numpy as np
import requests
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
from gwosc import datasets
from scipy.signal import butter, filtfilt

# =============================
# CONFIG DIRECTORIES
# =============================
RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"
PROC_DIR = "data/processed"

for d in [RAW_DIR, CLEAN_DIR, PROC_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================
# EVENTOS LIGO
# =============================
EVENTS = [
    "GW150914", "GW151226", "GW170104", "GW170608", "GW170814",
    "GW170729", "GW190412", "GW190521", "GW190814", "GW200129"
]

DETECTORS = ["H1", "L1"]

# =============================
# A1 — GPS estable
# =============================
def get_gps(event):
    try:
        url = f"https://www.gw-openscience.org/eventapi/json/event/{event}/"
        r = requests.get(url, timeout=10)
        data = r.json()
        gps = float(data["events"][event]["GPS"])
        print(f"✔ GPS de {event}: {gps}")
        return gps
    except Exception as e:
        print(f"✖ Error obteniendo GPS de {event}: {e}")
        return None

# =============================
# A2 — Descargar datos
# =============================
def download_event(event, det, gps):
    try:
        urls = datasets.get_event_urls(event, detector=det)
        if not urls:
            print(f"✖ No URLs for {event} [{det}]")
            return None

        url = urls[0]
        print(f"URL: {url}")
        ts = TimeSeries.read(url, format='hdf5')

        out = os.path.join(RAW_DIR, f"{event}_{det}_raw.hdf5")
        ts.name = f"{event}_{det}_raw"
        ts.write(out, path="/")
        print(f"✔ RAW saved: {out}")
        return out
    except Exception as e:
        print(f"✖ Error: {e}")
        return None

# =============================
# A3 — Highpass filter
# =============================
def butter_highpass(data, fs, cutoff=30, order=4):
    nyq = fs * 0.5
    norm = cutoff / nyq
    b, a = butter(order, norm, btype="high")
    return filtfilt(b, a, data)

# =============================
# A4 — Whitening manual
# =============================
def whiten_manual(ts, fftlength=4):
    psd = ts.psd(fftlength)
    psd_interp = psd.interpolate(ts.frequencies)
    white = ts / np.sqrt(psd_interp)
    return white

# =============================
# A5 — Procesamiento completo
# =============================
def process_event(event, det):
    print(f"\n=============== {event} — {det} ===============")

    gps = get_gps(event)
    if gps is None:
        return

    raw_path = download_event(event, det, gps)
    if raw_path is None:
        return

    ts_raw = TimeSeries.read(raw_path, format="hdf5")
    fs = ts_raw.sample_rate.value
    data = ts_raw.value
    t = ts_raw.times.value

    # Filtrado
    hp = butter_highpass(data, fs)
    ts_hp = TimeSeries(hp, times=t)
    ts_hp.name = f"{event}_{det}_clean"
    clean_path = os.path.join(CLEAN_DIR, f"{event}_{det}_clean.hdf5")
    ts_hp.write(clean_path, path="/")

    # Whitening manual
    ts_white = whiten_manual(ts_hp)
    ts_white.name = f"{event}_{det}_processed"
    proc_path = os.path.join(PROC_DIR, f"{event}_{det}_processed.hdf5")
    ts_white.write(proc_path, path="/")

    print(f"✔ CLEAN: {clean_path}")
    print(f"✔ PROCESSED: {proc_path}")

# =============================
# RUN
# =============================
def run_pipeline():
    print("\n==== PIPELINE MÓDULO A ====\n")
    for event in EVENTS:
        print(f"\n>>> EVENTO: {event} <<<")
        for det in DETECTORS:
            process_event(event, det)
    print("\n==== COMPLETADO ====\n")

if __name__ == "__main__":
    run_pipeline()
