"""
Download GWOSC strain data with Data Quality (DQ) flags.
Module A1 + A2 complete.

This script:
 - Obtains GPS time of the event
 - Downloads strain from H1, L1, V1
 - Applies Data Quality CAT2 flags (LIGO standard)
 - Saves cleaned + raw versions
"""

import os
import json
from gwosc import datasets
from gwpy.timeseries import TimeSeries
from gwpy.segments import DataQualityFlag

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

def download_raw_strain(det, start, end, outname):
    """Download raw open data from GWOSC using GWpy."""
    try:
        ts = TimeSeries.fetch_open_data(det, start, end)
        ts.write(outname)
        print(f"✔ Guardado RAW: {outname}")
        return True
    except Exception as e:
        print(f"✖ No se pudo descargar RAW de {det}: {e}")
        return False


def load_cat2_segments(det, start, end):
    """Fetch CAT2 Data Quality flag for detector."""
    try:
        dq = DataQualityFlag.fetch(f"{det}_DATA", start, end)
        return dq.active  # Active segments of CAT2
    except Exception as e:
        print(f"✖ No hay DQ flags para {det}: {e}")
        return None


def download_clean_strain(det, start, end, outname):
    """Download CAT2-cleaned strain by stitching good segments."""
    segments = load_cat2_segments(det, start, end)
    if segments is None or len(segments) == 0:
        print(f"✖ No hay segmentos CAT2 para {det}")
        return False

    combined = None
    for seg in segments:
        try:
            ts = TimeSeries.fetch_open_data(det, seg[0], seg[1])
            combined = ts if combined is None else combined.append(ts)
        except:
            pass

    if combined is None:
        print(f"✖ No se pudo construir señal limpia para {det}")
        return False

    combined.write(outname)
    print(f"✔ Guardado CLEAN (CAT2): {outname}")
    return True


def download_event(event, duration=16):
    """Main download routine for one event (RAW + CLEAN)."""

    print(f"\n===============================")
    print(f"   DESCARGANDO EVENTO: {event}")
    print(f"===============================")

    # 1. Obtener tiempo GPS del evento
    try:
        gps = datasets.event_gps(event)[event]
    except:
        print(f"✖ No se pudo obtener GPS de {event}")
        return

    t0 = gps
    start = t0 - duration
    end   = t0 + duration

    for det in ["H1", "L1", "V1"]:
        raw_out = os.path.join(RAW_DIR, f"{event}_{det}_raw.hdf5")
        clean_out = os.path.join(CLEAN_DIR, f"{event}_{det}_clean.hdf5")

        # Descargar RAW
        download_raw_strain(det, start, end, raw_out)
        
        # Descargar CLEAN (CAT2)
        download_clean_strain(det, start, end, clean_out)


if __name__ == "__main__":
    with open("events.json") as f:
        events = json.load(f)["events"]

    for ev in events:
        download_event(ev)
