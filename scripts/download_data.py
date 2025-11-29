"""
scripts/download_data.py

Descarga datos de GWOSC usando GPS ya conocidos (tabla fija).
"""

import os
from gwosc.api import fetch_event_json
from gwosc.datasets import fetch_open_data
from gwpy.timeseries import TimeSeries

# Tabla oficial de tiempos GPS
GPS_TIMES = {
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

def download_event(event, det, outdir="data/raw"):

    # Validar evento
    if event not in GPS_TIMES:
        print(f"✖ El evento {event} no está en la tabla GPS.")
        return None

    gps = GPS_TIMES[event]
    duration = 8  # segundos de datos alrededor del evento
    start = int(gps - duration/2)
    end   = int(gps + duration/2)

    print(f"\nDescargando {event} ({det}) — GPS {gps}")

    try:
        out = fetch_open_data(
            detector=det,
            start=start,
            end=end,
            outdir=outdir,
            verbose=True
        )
        return out

    except Exception as e:
        print(f"✖ Error descargando {event} / {det}")
        print(e)
        return None
