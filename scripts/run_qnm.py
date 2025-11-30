from scripts.qnm_analysis import analyze_qnm_for_event_detector

# Misma lista de eventos que en run_pipeline.py
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


def main():
    print("=== ANÁLISIS QNM (EDR TOOLKIT) ===\n")

    header = f"{'EVENTO':<10} {'DET':<3} {'f_QNM [Hz]':>12} {'tau [s]':>10} {'OK?':>6}  MENSAJE"
    print(header)
    print("-" * len(header))

    for ev in EVENTS.keys():
        for det in DETECTORS:
            res = analyze_qnm_for_event_detector(ev, det)

            ok_str = "YES" if res.success else "NO"
            f_str = f"{res.f_qnm:>12.2f}" if res.success else f"{'nan':>12}"
            tau_str = f"{res.tau:>10.4f}" if res.success else f"{'nan':>10}"

            print(f"{ev:<10} {det:<3} {f_str} {tau_str} {ok_str:>6}  {res.message}")

    print("\n=== FIN ANÁLISIS QNM ===")


if __name__ == "__main__":
    main()
