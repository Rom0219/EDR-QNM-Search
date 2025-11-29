import os
import numpy as np
import requests
from gwpy.timeseries import TimeSeries
from gwpy.signal.filter_design import whiten
from gwosc import datasets
from gwosc.locate import get_urls
from scipy.signal import butter, filtfilt

# =============================
# CONFIGURACIÓN DE DIRECTORIOS
# =============================
RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"
PROC_DIR = "data/processed"
FIG_DIR = "figures"

for d in [RAW_DIR, CLEAN_DIR, PROC_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================
# EVENTOS Y DETECTORES
# =============================
EVENTS = [
    "GW150914", "GW151226", "GW170104", "GW170608", "GW170814",
    "GW170729", "GW190412", "GW190521", "GW190814", "GW200129"
]

DETECTORS = ["H1", "L1", "V1"]

# =============================
# A1: Obtener GPS confiable desde API estable
# =============================
def get_gps(event):
    """
    Obtiene el GPS desde la API estable de GWOSC.
    Esta API funciona SIEMPRE.
    """
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
# A2: Descargar datos reales del detector
# =============================
def download_event(event, det, gps):
    try:
        print(f"Descargando {event} ({det}) — GPS {gps}")

        urls = datasets.get_event_urls(event, detector=det)

        if len(urls) == 0:
            print(f"✖ No hay URLs disponibles para {event} [{det}]")
            return None

        url = urls[0]  
        print("URL:", url)

        ts = TimeSeries.read(url, format='hdf5')

        out = os.path.join(RAW_DIR, f"{event}_{det}_raw.hdf5")
        ts.name = f"{event}_{det}_raw"
        ts.write(out, path="/")

        print(f"✔ Guardado en {out}")
        return out

    except Exception as e:
        print(f"✖ Error descargando {event}/{det}: {e}")
        return None

# =============================
# A3: Highpass con Butterworth
# =============================
def butter_highpass(data, fs, cutoff=30, order=4):
    nyq = 0.5 * fs
    norm = cutoff / nyq
    b, a = butter(order, norm, btype='high', analog=False)
    return filtfilt(b, a, data)

# =============================
# A4 + A5: Preprocesamiento, whitening y guardado
# =============================
def process_event(event, det):
    print(f"\n===============================")
    print(f"   PROCESANDO {event} — {det}")
    print(f"===============================\n")

    gps = get_gps(event)
    if gps is None:
        print(f"✖ No se pudo obtener GPS de {event}")
        return

    raw_path = download_event(event, det, gps)
    if raw_path is None:
        return

    # Leer datos crudos
    ts_raw = TimeSeries.read(raw_path, format='hdf5')
    fs = int(ts_raw.sample_rate.value)
    t = ts_raw.times.value
    data = ts_raw.value

    # ---------- A3: filtro highpass ----------
    hp = butter_highpass(data, fs, cutoff=30)
    ts_hp = TimeSeries(hp, times=t)

    ts_hp.name = f"{event}_{det}_clean"
    clean_path = os.path.join(CLEAN_DIR, f"{event}_{det}_clean.hdf5")
    ts_hp.write(clean_path, path="/")
    print(f"✔ Guardado archivo CLEAN: {clean_path}")

    # ---------- A4: whitening ----------
    ts_white = whiten(ts_hp)
    ts_white.name = f"{event}_{det}_processed"
    proc_path = os.path.join(PROC_DIR, f"{event}_{det}_processed.hdf5")
    ts_white.write(proc_path, path="/")
    print(f"✔ Guardado archivo PROCESSED: {proc_path}")

# =============================
# EJECUCIÓN PRINCIPAL
# =============================
def run_pipeline():
    print("\n============================================")
    print("        EJECUTANDO PIPELINE COMPLETO")
    print("============================================\n")

    for event in EVENTS:
        print(f"\n========== EVENTO: {event} ==========\n")

        for det in DETECTORS:
            print("\n--- (A1, A2, A3, A4, A5) Procesando datos ---\n")
            process_event(event, det)

    print("\n============================================")
    print("   PIPELINE MÓDULO A COMPLETO")
    print("============================================\n")

if __name__ == "__main__":
    run_pipeline()
