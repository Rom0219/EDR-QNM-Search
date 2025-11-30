# scripts/download_data.py
#
# Descarga y preprocesa strain real de GWOSC usando GWpy.
# No usa gwdatafind ni gwosc.api, solo la ruta moderna fetch_open_data.

import os
from typing import Tuple

from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps


# Carpetas de salida ---------------------------------------------------------

BASE_DIR = os.path.join("data")
RAW_DIR = os.path.join(BASE_DIR, "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "clean")
WHITE_DIR = os.path.join(BASE_DIR, "white")

for d in (BASE_DIR, RAW_DIR, CLEAN_DIR, WHITE_DIR):
    os.makedirs(d, exist_ok=True)


# Utilidades ----------------------------------------------------------------

def get_event_gps(event: str) -> float:
    """
    Devuelve el GPS central de un evento GWOSC.

    Soporta tanto un float como listas (algunos eventos devuelven lista).
    """
    gps = event_gps(event)
    if isinstance(gps, (list, tuple)):
        gps = gps[0]
    return float(gps)


def download_and_preprocess(
    event: str,
    det: str,
    window: float = 8.0,
    pad: float = 8.0,
    f_low: float = 20.0,
    f_high: float = 1024.0,
    fftlength: float = 4.0,
    overlap: float = 2.0,
) -> Tuple[str, str, str]:
    """
    Descarga y procesa strain para un evento y detector.

    Pasos:
      1. Usa gwosc.datasets.event_gps(event) para obtener el GPS.
      2. Descarga strain real con TimeSeries.fetch_open_data(det, t0, t1).
      3. Recorta, limpia (detrend + bandpass) y blanquea.
      4. Guarda tres archivos HDF5: raw, clean, white.

    Devuelve:
      (ruta_raw, ruta_clean, ruta_white)
    """
    gps = get_event_gps(event)
    # Ventana corta (por ejemplo 8 s alrededor del evento)
    t0 = int(gps) - int(window // 2)
    t1 = t0 + int(window)

    print(f"  GPS = {gps:.3f}")
    print(f"  Ventana de análisis: [{t0}, {t1}] s")

    # --- 1) Descargar datos abiertos desde GWOSC ---
    # fetch_open_data construye internamente la URL correcta y gestiona GWOSC
    print("  Descargando datos con TimeSeries.fetch_open_data(...)")
    ts = TimeSeries.fetch_open_data(det, t0 - int(pad), t1 + int(pad), cache=True)

    # --- 2) Recorte exacto a la ventana de interés ---
    ts_raw = ts.crop(t0, t1)

    # --- 3) Limpieza básica: detrend + bandpass ---
    print(f"  Limpieza: detrend + bandpass [{f_low}, {f_high}] Hz")
    ts_clean = ts_raw.detrend("constant").bandpass(f_low, f_high)

    # --- 4) Blanqueo usando método oficial de GWpy ---
    print(f"  Blanqueando: fftlength={fftlength}, overlap={overlap}")
    ts_white = ts_clean.whiten(fftlength=fftlength, overlap=overlap)

    # --- 5) Guardar en disco ---
    base = f"{event}_{det}"
    raw_path = os.path.join(RAW_DIR, f"{base}_raw.hdf5")
    clean_path = os.path.join(CLEAN_DIR, f"{base}_clean.hdf5")
    white_path = os.path.join(WHITE_DIR, f"{base}_white.hdf5")

    print(f"  Guardando RAW   -> {raw_path}")
    ts_raw.write(raw_path, format="hdf5")

    print(f"  Guardando CLEAN -> {clean_path}")
    ts_clean.write(clean_path, format="hdf5")

    print(f"  Guardando WHITE -> {white_path}")
    ts_white.write(white_path, format="hdf5")

    return raw_path, clean_path, white_path
