# scripts/run_edr_field.py
import json
import os
from scripts.fit_edr_full import fit_edr_full
from scripts.edr_field_params import infer_edr_field_params

OUTDIR = "results/edr_field"
os.makedirs(OUTDIR, exist_ok=True)

def run_edr_field(det, event, Mrem, chi):
    print(f"\n===== EDR-FIELD — {event} — {det} =====")

    # 1) Ajuste FULL (22, 33, 21)
    params, h_best, resid = fit_edr_full(det, event, Mrem, chi)

    # 2) Convertir a parámetros físicos EDR-Field
    phys = infer_edr_field_params(params)

    # 3) Guardar JSON
    out = {
        "event": event,
        "detector": det,
        "params_edr_full": {
            "A22": params[0],
            "A33": params[1],
            "A21": params[2],
            "d_om22": params[3],
            "d_tau22": params[4],
            "d_om33": params[5],
            "d_tau33": params[6],
            "d_om21": params[7],
            "d_tau21": params[8],
            "phi22": params[9],
            "phi33": params[10],
            "phi21": params[11],
            "t0": params[12]
        },
        "params_edr_field": {
            "spiral_intensity":       phys.spiral_intensity,
            "radial_scale":           phys.radial_scale,
            "effective_viscosity":    phys.effective_viscosity,
            "multipole_anisotropy":   phys.multipole_anisotropy,
            "mode_coupling":          phys.mode_coupling
        }
    }

    fout = f"{OUTDIR}/{event}_{det}_edr_field.json"
    with open(fout, "w") as f:
        json.dump(out, f, indent=2)

    print(f"✔ Guardado: {fout}")
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("USO: python3 -m scripts.run_edr_field DET EVENT Mrem chi")
        print("EJEMPLO: python3 -m scripts.run_edr_field H1 GW150914 68 0.67")
        sys.exit(1)

    det  = sys.argv[1]
    event = sys.argv[2]
    Mrem = float(sys.argv[3])
    chi  = float(sys.argv[4])

    run_edr_field(det, event, Mrem, chi)
