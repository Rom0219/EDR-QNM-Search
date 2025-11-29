import os
import requests
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_dataset

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

# ---------------------------------------------------
# Obtener URL del strain LOSC real para un evento
# ---------------------------------------------------
def get_strain_url(event, detector):
    """
    Retorna la URL HDF5 oficial de strain para un detector.
    Ejemplo:
    detector = "H1" o "L1"
    """
    try:
        urls = event_dataset(event)
    except Exception as e:
        raise ValueError(f"No se pudo obtener dataset para {event}: {e}")

    # Buscar archivo LOSC del detector
    for url in urls:
        if f"-{detector}_LOSC" in url and url.endswith(".hdf5"):
            return url

    raise ValueError(f"No existe strain LOSC para {event}/{detector}")


# ---------------------------------------------------
# Descargar archivo HDF5
# ---------------------------------------------------
def download_strain(event, detector):
    out_path = f"{RAW_DIR}/{event}_{detector}.hdf5"

    if os.path.exists(out_path):
        print(f"✓ Ya existe {out_path}")
        return out_path

    try:
        url = get_strain_url(event, detector)
    except Exception as e:
        print(f"✖ Error: {e}")
        return None

    print(f"Descargando archivo LOSC:\n{url}")

    resp = requests.get(url, stream=True)
    if resp.status_code != 200:
        print(f"✖ Error HTTP {resp.status_code}")
        return None

    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"✓ Guardado en {out_path}")
    return out_path


# ---------------------------------------------------
# Cargar TimeSeries desde HDF5
# ---------------------------------------------------
def load_strain(path):
    try:
        ts = TimeSeries.read(path, format="hdf5.gwosc")
        print(f"✓ Señal cargada ({len(ts)} muestras)")
        return ts
    except Exception as e:
        print(f"✖ Error leyendo {path}: {e}")
        return None
