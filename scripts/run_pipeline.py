from scripts.download_data import download_and_preprocess

# Lista de eventos y GPS
EVENTS = {
    "GW150914": 1126259462.4,
    "GW151226": 1135136350.6,
    "GW170104": 1167559936.6,
    "GW170608": 1180922494.5,
    "GW170814": 1186741861.5,
    "GW170729": 1185389807.3,
    "GW170823": 1187529256.5,
    "GW190412": 1239082262.1,
    "GW190521": 1242442967.4,
    "GW190814": 1249852257.0
}

DETECTORS = ["H1", "L1"]


print("=== PIPELINE GWOSC + GWpy (fetch_open_data) ===\n")


for event_name, gps in EVENTS.items():

    print(f">>> EVENTO: {event_name}\n")

    for det in DETECTORS:

        print(f"===== {event_name} — {det} =====")

        download_and_preprocess(
            event_name=event_name,
            detector=det,
            gps=gps,
            t_pre=4,
            t_post=4,
            fmin=20,
            fmax=1024
        )

    print("")  # Línea en blanco entre eventos

print("=== PIPELINE COMPLETO ===")
