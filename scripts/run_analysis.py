"""
run_analysis.py

Pipeline maestro:
 - Descarga los datos (run_pipeline.py) si es necesario
 - Ejecuta GR fitting
 - Ejecuta EDR fitting
 - Compara GR vs EDR
 - Guarda todos los resultados

Este archivo automatiza TODO el análisis de la teoría EDR.
"""

import os
import subprocess
from scripts.compare_gr_edr import compare_GR_EDR

# =============================================
# CONFIGURACIÓN: lista de eventos
# =============================================
EVENTS = {
    "GW150914":  {"Mrem": 67.0, "chi": 0.67},
    "GW170729":  {"Mrem": 80.0, "chi": 0.58},
    "GW190521":  {"Mrem": 142.0, "chi": 0.72},
    "GW170814":  {"Mrem": 56.0, "chi": 0.70},
    "GW190412":  {"Mrem": 33.0, "chi": 0.44},
    # Puedes añadir más
}

DETECTORS = ["H1", "L1"]  # Agrega "V1" si quieres


# =============================================
# 1) Verifica si existen datos procesados
#    Si no existen, ejecuta el módulo de descarga
# =============================================
def ensure_data():
    processed_dir = "data/processed"
    if not os.path.exists(processed_dir) or len(os.listdir(processed_dir)) == 0:
        print("\n⚠ No hay datos procesados. Ejecutando run_pipeline.py...")
        subprocess.run(["python3", "scripts/run_pipeline.py"], check=True)
    else:
        print("\n✔ Datos procesados encontrados. Continuamos.")


# =============================================
# 2) Ejecutar el análisis GR / EDR / Comparación
# =============================================
def run_full_analysis():
    ensure_data()

    for event, params in EVENTS.items():
        Mrem = params["Mrem"]
        chi  = params["chi"]

        for det in DETECTORS:
            print("\n==============================================")
            print(f" EVENTO {event} — DETECTOR {det}")
            print("==============================================")

            try:
                result = compare_GR_EDR(det, event, Mrem, chi)
                print("\n>>> Resultados completos:")
                print(result)

            except Exception as e:
                print("\n❌ ERROR procesando", event, det)
                print(e)


# =============================================
# EJECUTAR
# =============================================
if __name__ == "__main__":
    run_full_analysis()
