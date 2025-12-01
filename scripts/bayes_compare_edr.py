"""
scripts/bayes_compare_edr.py

Análisis Bayesiano GR vs EDR-FULL usando nested sampling (dynesty):

 - Modelo GR: ringdown (2,2) tipo damped-sine
 - Modelo EDR-FULL: modos (22,33,21) con shifts δω/ω y δτ/τ independientes
 - Likelihood Gaussiana sobre datos whitened
 - Evidencias logZ_GR y logZ_EDR
 - Factor de Bayes BF = exp(logZ_EDR - logZ_GR)

Uso:
  python3 -m scripts.bayes_compare_edr H1 GW150914 68 0.67
  python3 -m scripts.bayes_compare_edr all

Requiere:
  pip install dynesty
"""

import os
import sys
import json
import numpy as np
from gwpy.timeseries import TimeSeries

from scripts.model_gr import freq_tau, damped_sine

try:
    import dynesty
except ImportError:
    raise ImportError("Necesitas instalar dynesty:  pip install dynesty")

DATA_DIR = "data/processed"
OUT_DIR = "results/bayes_compare"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Cargar dato whitened
# ============================================================
def load_processed(det, event):
    fname = os.path.join(DATA_DIR, f"{event}_{det}_processed.hdf5")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"No existe procesado: {fname}")
    ts = TimeSeries.read(fname)
    return ts.value, ts.times.value


# ============================================================
# MODELO GR: single damped-sine (22)
# params_GR = [A22, f0, tau, phi, t0]
# ============================================================
def make_gr_model(t, Mrem, chi):
    f_th, tau_th = freq_tau(Mrem, chi, "22")

    # rangos de priors
    A_min, A_max = 0.0, 2.0
    f_min, f_max = 0.5 * f_th, 1.5 * f_th
    tau_min, tau_max = 0.5 * tau_th, 1.5 * tau_th
    t0_min, t0_max = 0.0, 0.05  # 0–50 ms

    def prior_transform(u):
        """
        u in [0,1]^5  ->  theta física
        """
        A = A_min + u[0] * (A_max - A_min)
        f0 = f_min + u[1] * (f_max - f_min)
        tau = tau_min + u[2] * (tau_max - tau_min)
        phi = -np.pi + u[3] * (2.0 * np.pi)
        t0 = t0_min + u[4] * (t0_max - t0_min)
        return np.array([A, f0, tau, phi, t0])

    def loglike(theta):
        A, f0, tau, phi, t0 = theta
        h = damped_sine(t, A, f0, tau, phi, t0)
        resid = data - h
        return -0.5 * np.sum(resid * resid)

    return prior_transform, loglike


# ============================================================
# MODELO EDR-FULL: modos (22,33,21) con shifts
# params_EDR = [
#   A22, A33, A21,
#   d_om22, d_tau22,
#   d_om33, d_tau33,
#   d_om21, d_tau21,
#   phi22, phi33, phi21,
#   t0
# ]
# ============================================================
def make_edr_model(t, Mrem, chi):
    # frecuencias y taus GR de base
    f22_gr, tau22_gr = freq_tau(Mrem, chi, "22")
    f33_gr, tau33_gr = freq_tau(Mrem, chi, "33")
    f21_gr, tau21_gr = freq_tau(Mrem, chi, "21")

    # rangos de priors
    A_min, A_max = 0.0, 1.0
    d_min, d_max = -0.5, 0.5       # para δω/ω y δτ/τ
    t0_min, t0_max = 0.0, 0.05

    def prior_transform(u):
        """
        u in [0,1]^13  ->  theta física
        """
        A22 = A_min + u[0] * (A_max - A_min)
        A33 = A_min + u[1] * (A_max - A_min)
        A21 = A_min + u[2] * (A_max - A_min)

        d_om22 = d_min + u[3] * (d_max - d_min)
        d_tau22 = d_min + u[4] * (d_max - d_min)

        d_om33 = d_min + u[5] * (d_max - d_min)
        d_tau33 = d_min + u[6] * (d_max - d_min)

        d_om21 = d_min + u[7] * (d_max - d_min)
        d_tau21 = d_min + u[8] * (d_max - d_min)

        phi22 = -np.pi + u[9] * (2.0 * np.pi)
        phi33 = -np.pi + u[10] * (2.0 * np.pi)
        phi21 = -np.pi + u[11] * (2.0 * np.pi)

        t0 = t0_min + u[12] * (t0_max - t0_min)

        return np.array([
            A22, A33, A21,
            d_om22, d_tau22,
            d_om33, d_tau33,
            d_om21, d_tau21,
            phi22, phi33, phi21,
            t0
        ])

    def loglike(theta):
        (
            A22, A33, A21,
            d_om22, d_tau22,
            d_om33, d_tau33,
            d_om21, d_tau21,
            phi22, phi33, phi21,
            t0
        ) = theta

        # modos modificados
        f22 = f22_gr * (1.0 + d_om22)
        tau22 = tau22_gr * (1.0 + d_tau22)

        f33 = f33_gr * (1.0 + d_om33)
        tau33 = tau33_gr * (1.0 + d_tau33)

        f21 = f21_gr * (1.0 + d_om21)
        tau21 = tau21_gr * (1.0 + d_tau21)

        h22 = damped_sine(t, A22, f22, tau22, phi22, t0)
        h33 = damped_sine(t, A33, f33, tau33, phi33, t0)
        h21 = damped_sine(t, A21, f21, tau21, phi21, t0)

        h = h22 + h33 + h21

        resid = data - h
        return -0.5 * np.sum(resid * resid)

    return prior_transform, loglike


# ============================================================
# Run nested sampling para un modelo
# ============================================================
def run_nested(prior_transform, loglike, ndim, nlive=400):
    sampler = dynesty.NestedSampler(
        loglike,
        prior_transform,
        ndim,
        nlive=nlive,
        bound="multi",
        sample="rwalk",
    )
    sampler.run_nested(print_progress=False)
    res = sampler.results
    logZ = res.logz[-1]
    logZ_err = res.logzerr[-1]
    return logZ, logZ_err, res


# ============================================================
# Wrapper principal para un evento/detector
# ============================================================
def bayes_compare(det, event, Mrem, chi, nlive=400):
    global data  # para las closures de loglike
    data, t = load_processed(det, event)

    # --------- Modelo GR ----------
    pt_gr, ll_gr = make_gr_model(t, Mrem, chi)
    logZ_gr, logZerr_gr, res_gr = run_nested(pt_gr, ll_gr, ndim=5, nlive=nlive)

    # --------- Modelo EDR-FULL ----------
    pt_edr, ll_edr = make_edr_model(t, Mrem, chi)
    logZ_edr, logZerr_edr, res_edr = run_nested(pt_edr, ll_edr, ndim=13, nlive=nlive)

    # factor de Bayes
    dlogZ = logZ_edr - logZ_gr
    BF = np.exp(dlogZ)

    favored = "EDR" if BF > 1 else "GR"

    print("\n==============================")
    print(f" BAYES GR vs EDR — {event} — {det}")
    print("==============================")
    print(f"logZ_GR   = {logZ_gr:.3f} ± {logZerr_gr:.3f}")
    print(f"logZ_EDR  = {logZ_edr:.3f} ± {logZerr_edr:.3f}")
    print(f"ΔlogZ     = logZ_EDR - logZ_GR = {dlogZ:.3f}")
    print(f"Bayes F.  = {BF:.3f}")
    print(f"Modelo fav.: {favored}")

    out = {
        "event": event,
        "detector": det,
        "Mrem": Mrem,
        "chi": chi,
        "logZ_GR": float(logZ_gr),
        "logZerr_GR": float(logZerr_gr),
        "logZ_EDR": float(logZ_edr),
        "logZerr_EDR": float(logZerr_edr),
        "dlogZ": float(dlogZ),
        "BayesFactor": float(BF),
        "favored": favored,
    }

    outname = os.path.join(OUT_DIR, f"{event}_{det}_bayes.json")
    with open(outname, "w") as f:
        json.dump(out, f, indent=2)

    print(f"✔ Guardado: {outname}")

    return out


# ============================================================
# Runner múltiple (como pipeline)
# ============================================================
DEFAULT_EVENTS = {
    "GW150914":  {"Mrem": 68,  "chi": 0.67},
    "GW151226":  {"Mrem": 20.5,"chi": 0.74},
    "GW170104":  {"Mrem": 49,  "chi": 0.66},
    "GW170608":  {"Mrem": 19,  "chi": 0.74},
    "GW170814":  {"Mrem": 54.5,"chi": 0.74},
    "GW170729":  {"Mrem": 80,  "chi": 0.81},
    "GW170823":  {"Mrem": 60,  "chi": 0.72},
    "GW190412":  {"Mrem": 34,  "chi": 0.67},
    "GW190521":  {"Mrem": 142, "chi": 0.72},
    "GW190814":  {"Mrem": 25,  "chi": 0.91},
}

DETECTORS = ["H1", "L1"]


def main():
    if len(sys.argv) == 5:
        # python3 -m scripts.bayes_compare_edr H1 GW150914 68 0.67
        det = sys.argv[1]
        event = sys.argv[2]
        Mrem = float(sys.argv[3])
        chi = float(sys.argv[4])
        bayes_compare(det, event, Mrem, chi)
    elif len(sys.argv) == 2 and sys.argv[1] == "all":
        for ev, pars in DEFAULT_EVENTS.items():
            for det in DETECTORS:
                try:
                    bayes_compare(det, ev, pars["Mrem"], pars["chi"])
                except Exception as e:
                    print(f"\n❌ ERROR en {ev} {det}: {e}")
    else:
        print("Uso:")
        print("  python3 -m scripts.bayes_compare_edr H1 GW150914 68 0.67")
        print("  python3 -m scripts.bayes_compare_edr all")


if __name__ == "__main__":
    main()
