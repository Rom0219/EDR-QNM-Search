import os
import numpy as np
from gwpy.timeseries import TimeSeries
from scipy.signal import butter, filtfilt

# =============================
# DIRECTORIOS
# =============================
RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"
PROC_DIR = "data/processed"
for d in [RAW_DIR, CLEAN_DIR, PROC_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================
# EVENTOS Y GPS ESTABLES
# =============================
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

DETECTORS = ["H1", "L1"]

# =============================
# HIGH-PASS FILTER
# =============================
def butter_highpass(data, fs, cutoff=30, order=4):
    nyq = fs * 0.5
    norm = cutoff / nyq
    b, a = butter(order, norm, btype="high")
    return filtfilt(b, a, data)

# =============================
# WHITENING MANUAL
# =============================
def whiten_manual(ts, fftlength=4):
    psd = ts.psd(fftlength)
    psd_interp = psd.interpolate(ts.frequencies)
    white = ts / np.sqrt(psd_interp)
    return white

# =============================
# DESCARGA DIRECTA DESDE LOSC
# =============================
def fetch_event(event, det):
    gps = GPS_TABLE[event]
    try:
        print(f"Descargando {event} [{det}] usando TimeSeries.fetch()")
        ts = TimeSeries.fetch(det, gps - 8, gps + 8)  # ventana 16 s
        path = os.path.join(RAW_DIR, f"{event}_{det}_raw.hdf5")
        ts.name = f"{event}_{det}_raw"
        ts.write(path, path="/")
        print(f"✔ RAW guardado: {path}")
        return path
    except Exception as e:
        print(f"✖ Error en descarga LOSC: {e}")
        return None

# =============================
# PROCESAR UN EVENTO
# =============================
def process_event(event, det):
    print(f"\n==== {event} — {det} ====")

    raw = fetch_event(event, det)
    if raw is None:
        return

    ts_raw = TimeSeries.read(raw, format="hdf5")
    fs = ts_raw.sample_rate.value
    data = ts_raw.value
    t = ts_raw.times.value

    # High-pass
    hp = butter_highpass(data, fs)
    ts_hp = TimeSeries(hp, times=t)
    clean_path = os.path.join(CLEAN_DIR, f"{event}_{det}_clean.hdf5")
    ts_hp.name = f"{event}_{det}_clean"
    ts_hp.write(clean_path, path="/")
    print(f"✔ CLEAN generado: {clean_path}")

    # Whitening
    ts_white = whiten_manual(ts_hp)
    proc_path = os.path.join(PROC_DIR, f"{event}_{det}_processed.hdf5")
    ts_white.name = f"{event}_{det}_processed"
    ts_white.write(proc_path, path="/")
    print(f"✔ PROCESSED generado: {proc_path}")

# =============================
# PIPELINE COMPLETO
# =============================
def run_pipeline():
    print("\n=== MÓDULO A — INICIO ===\n")
    for event in GPS_TABLE.keys():
        print(f"\n>>> EVENTO: {event}")
        for det in DETECTORS:
            process_event(event, det)
    print("\n=== MÓDULO A — COMPLETADO ===\n")

if __name__ == "__main__":
    run_pipeline()
