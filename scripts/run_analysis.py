import json, glob
import numpy as np
from preprocess import preprocess
from ringdown_templates import template
from matched_filter import compute_snr
from compare_GR_EDR import GR_f0, GR_tau, EDR_f0, EDR_tau

FS = 4096
RINGDOWN_DUR = 0.3

def analyze_event(event):

    print(f"\n=== Analizando {event} ===")

    files = glob.glob(f"data/{event}_*.hdf5")
    results = {}

    # masa efectiva: usar aproximada según GWOSC
    approx_mass = 60

    for f in files:
        det = f.split("_")[-1].replace(".hdf5","")

        white = preprocess(f).value
        white = white[int(len(white)/2):]   # ringdown approx

        # GR template
        t, gr_temp = template(FS, RINGDOWN_DUR, GR_f0(approx_mass), GR_tau(approx_mass))

        # EDR template
        t, edr_temp = template(FS, RINGDOWN_DUR, EDR_f0(approx_mass, 0.01), EDR_tau(approx_mass, 0.01))

        snr_gr = compute_snr(white[:len(gr_temp)], gr_temp, FS)
        snr_edr = compute_snr(white[:len(edr_temp)], edr_temp, FS)

        results[det] = {"GR": snr_gr, "EDR": snr_edr}

    return results


if __name__ == "__main__":
    with open("events.json") as f:
        events = json.load(f)["events"]

    for ev in events:
        res = analyze_event(ev)
        print(ev, res)

