import os
import requests
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

# -----------------------------------------
# Detectar el RUN al que pertenece el evento
# -----------------------------------------
def detect_run(event):
    if event.startswith("GW15"):
        return "O1"
    if event.startswith("GW16") or event.startswith("GW17"):
        return "O2"
    if event.startswith("GW19") or event.startswith("GW20"):
        return "O3"
    return "O3"


# -----------------------------------------
# Construir la URL del strain LOSC
# -----------------------------------------
def make_losc_url(event, detector):
    gps = event_gps(event)
    run = detect_run(event)

    if detector not in ["H1", "L1"]:
        raise ValueError("Detector inválido")

    # IFO = primera letra: H / L
    ifo = detector[0]

    url = (
        f"https://www.gw-openscience.org/archive/data/{run}/strain/"
        f"{ifo}-{detector}_LOSC_4_V1-{int(gps)}-4096.hdf5"
    )

    return url


# -----------------------------------------
# Descargar strain
# -----------------------------------------
def download_strain(event, detector):
    out_path = f"{RAW_DIR}/{event}_{detector}.hdf5"

    if os.path.exists(out_path):
        print(f"✓ Ya existe {out_path}")
        return out_path

    try:
        url = make_losc_url(event, detector)
    except Exception as e:
        print(f"✖ Error construyendo URL: {e}")
        return None

    print(f"Descargando LOSC:\n{url}")

    resp = requests.get(url, stream=True)
    if resp.status_code != 200:
        print(f"✖ Error HTTP {resp.status_code}")
        return None

    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"✓ Archivo guardado en {out_path}")
    return out_path


# -----------------------------------------
# Cargar strain en TimeSeries
# -----------------------------------------
def load_strain(path):
    try:
        ts = TimeSeries.read(path, format="hdf5.gwosc")
        return ts
    except Exception as e:
        print(f"✖ Error cargando {path}: {e}")
        return None
