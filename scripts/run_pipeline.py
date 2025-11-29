import os
import numpy as np
import requests
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
from gwosc import datasets
from scipy.signal import butter, filtfilt

# Directorios
RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"
PROC_DIR = "data/processed"
for d in [RAW_DIR, CLEAN_DIR, PROC_DIR]:
    os.makedirs(d, exist_ok=True)

# Eventos y detectores
EVENTS = [
    "GW150914", "GW151226", "GW170104", "GW170608", "GW170814",
    "GW170729", "GW190412", "GW190521", "GW190814", "GW200129"
]
DETECTORS = ["H1", "L1"]

# Tabla fija de tiempos GPS
GPS_TABLE = {
    "GW150914": 1126259462.4,
    "GW151226": 1135136350.6,
    "GW170104": 1167559936.6,
    "GW170608": 1180922494.5,
    "GW170814": 1186741861.5,
    "GW170729": 1185389807.3,
    "GW190412": 1239082262.2,
    "GW190521": 1242442967.4,
    "GW190814": 1249852257.0,
    "GW200129": 1264316115.4
}

def download_event(event, det):
    gps = GPS_TABLE.get(event)
    if gps is None:
        print(f"✖ Evento {event} no en la tabla GPS.")
        return None
    try:
        urls = datasets.get_event_urls(event, detector=det)
        if not urls:
            print(f"✖ No URLs for {event}[{det}]")
            return None
        url = urls[0]
        ts = TimeSeries.read(url, format='hdf5')
        out = os.path.join(RAW_DIR, f"{event}_{det}_raw.hdf5")
        ts.name = f"{event}_{det}_raw"
        ts.write(out, path="/")
        print(f"✔ RAW saved: {out}")
        return out
    except Exception as e:
        print(f"✖ Error download {event}/{det}: {e}")
        return None

def butter_highpass(data, fs, cutoff=30, order=4):
    nyq = fs * 0.5
    norm = cutoff / nyq
    b, a = butter(order, norm, btype='high')
    return filtfilt(b, a, data)

def whiten_manual(ts, fftlength=4):
    psd = ts.psd(fftlength)
    psd_interp = psd.interpolate(ts.frequencies)
    white = ts / np.sqrt(psd_interp)
    return white

def process_event(event, det):
    print(f"\n=== {event} — {det} ===")
    raw = download_event(event, det)
    if raw is None:
        return

    ts_raw = TimeSeries.read(raw, format='hdf5')
    fs = ts_raw.sample_rate.value
    t = ts_raw.times.value
    data = ts_raw.value

    hp = butter_highpass(data, fs)
    ts_hp = TimeSeries(hp, times=t)
    ts_hp.name = f"{event}_{det}_clean"
    clean = os.path.join(CLEAN_DIR, f"{event}_{det}_clean.hdf5")
    ts_hp.write(clean, path="/")
    print(f"✔ CLEAN: {clean}")

    ts_white = whiten_manual(ts_hp)
    ts_white.name = f"{event}_{det}_proc"
    proc = os.path.join(PROC_DIR, f"{event}_{det}_processed.hdf5")
    ts_white.write(proc, path="/")
    print(f"✔ PROCESSED: {proc}")

def run_pipeline():
    print("\n--- PIPELINE A ---\n")
    for ev in EVENTS:
        print(f"\n>> {ev}")
        for det in DETECTORS:
            process_event(ev, det)
    print("\n--- FIN PIPELINE A ---\n")

if __name__ == "__main__":
    run_pipeline()
