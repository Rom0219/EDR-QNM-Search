import os
import sys
import h5py
import numpy as np
from gwpy.timeseries import TimeSeries

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

EVENTS = {
    "GW150914", "GW151226", "GW170104", "GW170608",
    "GW170814", "GW170729", "GW170823", "GW190412",
    "GW190521", "GW190814"
}
DETS = ["H1", "L1"]


def preprocess(det, event):
    print(f"Procesando {event} {det} ...")

    infile = f"{RAW_DIR}/{event}_{det}_raw.hdf5"
    outfile = f"{OUT_DIR}/{event}_{det}_processed.hdf5"

    if not os.path.exists(infile):
        print(f"✖ No existe: {infile}")
        return

    # leer strain y usar su metadata interna
    with h5py.File(infile, "r") as f:
        data = f["strain"][:]                     # señal
        fs = f["strain"].attrs.get("fs", None)    # sample_rate si existe
        t0 = f["strain"].attrs.get("t0", 0)       # tiempo inicial si existe

    # si el RAW no incluye fs, usamos un default seguro: 4096 Hz
    if fs is None:
        fs = 4096.0

    ts = TimeSeries(data, sample_rate=fs, t0=t0)

    white = ts.whiten()

    if os.path.exists(outfile):
        os.remove(outfile)

    white.write(outfile, path="strain")

    print(f"✔ Guardado procesado: {outfile}")


def preprocess_all():
    for event in EVENTS:
        for det in DETS:
            preprocess(det, event)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "all":
        preprocess_all()
    elif len(sys.argv) == 3:
        preprocess(sys.argv[1], sys.argv[2])
    else:
        print("Uso:")
        print("  python3 -m scripts.preprocess all")
        print("  python3 -m scripts.preprocess H1 GW150914")
