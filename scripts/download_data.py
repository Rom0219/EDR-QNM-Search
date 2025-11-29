import os
import re
import requests
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

# --------------------------------------------
# Determinar el RUN por nombre del evento
# --------------------------------------------
def detect_run(event):
    if event.startswith("GW15"):
        return "O1"
    if event.startswith("GW17"):
        return "O2"
    if event.startswith("GW19") or event.startswith("GW20"):
        return "O3"
    return "O3"

# --------------------------------------------
# Obtener la URL real usando índice HTML
# --------------------------------------------
def find_losc_file(event, detector):
    gps = event_gps(event)
    run = detect_run(event)
    base_url = f"https://www.gw-openscience.org/archive/data/{run}/strain/"

    print(f"Buscando archivo REAL en: {base_url}")

    html = requests.get(base_url).text

    # Detector: H1 → "H-H1", L1 → "L-L1"
    ifo = detector[0]

    # Ejemplo filename:
    # H-H1_LOSC_4_V2-1126259446-32.hdf5
    pattern = rf"{ifo}-{detector}_LOSC_4_.*-{int(gps)}-\d+\.hdf5"

    match = re.search(pattern, html)
    if not match:
        raise ValueError(f"No se encontró archivo LOSC para {event}/{detector}")

    filename = match.group(0)
    return base_url + filename

# --------------------------------------------
# Descargar archivo
# --------------------------------------------
def download_strain(event, detector):
    out_path = f"{RAW_DIR}/{event}_{detector}.hdf5"

    if os.path.exists(out_path):
        print(f"✓ Ya existe {out_path}")
        return out_path

    try:
        url = find_losc_file(event, detector)
    except Exception as e:
        print(f"✖ Error buscando archivo: {e}")
        return None

    print(f"Descargando archivo:\n{url}")

    resp = requests.get(url, stream=True)
    if resp.status_code != 200:
        print(f"✖ HTTP {resp.status_code}")
        return None

    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(8192):
            if chunk:
                f.write(chunk)

    print(f"✓ Guardado en {out_path}")
    return out_path

# --------------------------------------------
# Cargar señal
# --------------------------------------------
def load_strain(path):
    try:
        return TimeSeries.read(path, format="hdf5.gwosc")
    except Exception as e:
        print(f"✖ Error leyendo {path}: {e}")
        return None
