import os
from gwpy.timeseries import TimeSeries
from gwosc.api import find_urls

EVENTS = {
    "GW150914":   {"gps": 1126259462, "run": "O1"},
    "GW151226":   {"gps": 1135136350, "run": "O1"},
    "GW170104":   {"gps": 1167559936, "run": "O2"},
    "GW170608":   {"gps": 1180922494, "run": "O2"},
    "GW170729":   {"gps": 1185389807, "run": "O2"},
    "GW170814":   {"gps": 1186741861, "run": "O2"},
    "GW170823":   {"gps": 1187529256, "run": "O2"},
    "GW190412":   {"gps": 1239082262, "run": "O3a"},
    "GW190521":   {"gps": 1242442967, "run": "O3a"},
    "GW190814":   {"gps": 1249852257, "run": "O3a"},
}

DETECTORS = ["H1", "L1"]

RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"
WHITE_DIR = "data/white"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(WHITE_DIR, exist_ok=True)


def run_pipeline():
    print("\n=== PIPELINE COMPLETO GWOSC ===\n")

    for ev, meta in EVENTS.items():
        gps = meta["gps"]
        gps_end = gps + 4  # <— NECESARIO PARA find_urls()

        print(f"\n>>> EVENTO: {ev}\n")

        for det in DETECTORS:

            print(f"\n===== {ev} — {det} =====")
            print(f"Descargando {ev} ({det}) — GPS {gps}")

            try:
                urls = find_urls(det, gps, gps_end)
                if not urls:
                    raise RuntimeError("No URLs devueltos por la API")

                url = urls[0]
                print("URL encontrada:", url)

                out_file = os.path.join(RAW_DIR, f"{ev}_{det}.gwf")
                ts = TimeSeries.fetch(url)
                ts.write(out_file)

                print("✔ Guardado:", out_file)
            except Exception as e:
                print("✖ Error descargando:", e)
                print("✖ No se pudo descargar.")
                continue

    print("\n=== PIPELINE TERMINADO ===")


if __name__ == "__main__":
    run_pipeline()
