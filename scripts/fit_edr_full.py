"""
scripts/fit_edr_full.py

Ajuste EDR avanzado:
 - Usa datos whitened en data/processed/{event}_{det}_processed.hdf5
 - Ajusta multimodo EDR con parámetros independientes:
      A22, A33, A21
      d_om22, d_tau22
      d_om33, d_tau33
      d_om21, d_tau21
      phi22, phi33, phi21
      t0
 - Likelihood gaussiana en el dominio tiempo

Uso desde terminal:
  python3 -m scripts.fit_edr_full H1 GW150914 68 0.67
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from scipy.optimize import minimize

from scripts.model_edr_full import edr_multimode_full


DATA_DIR = "data/processed"
PLOT_DIR = "plots/fit_edr_full"
os.makedirs(PLOT_DIR, exist_ok=True)


def load_processed(det, event):
    fname = os.path.join(DATA_DIR, f"{event}_{det}_processed.hdf5")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"No existe archivo procesado: {fname}")
    ts = TimeSeries.read(fname, path="strain")
    return ts.value, ts.times.value


def neg_log_like(params, t, data, Mrem, chi):
    """
    params =
      [A22, A33, A21,
       d_om22, d_tau22,
       d_om33, d_tau33,
       d_om21, d_tau21,
       phi22, phi33, phi21,
       t0]
    """
    (
        A22, A33, A21,
        d_om22, d_tau22,
        d_om33, d_tau33,
        d_om21, d_tau21,
        phi22, phi33, phi21,
        t0
    ) = params

    h = edr_multimode_full(
        t,
        Mrem,
        chi,
        A22, A33, A21,
        d_om22, d_tau22,
        d_om33, d_tau33,
        d_om21, d_tau21,
        phi22, phi33, phi21,
        t0,
    )

    resid = data - h
    return 0.5 * np.sum(resid * resid)


def fit_edr_full(det, event, Mrem, chi):

    print(f"\n===== Ajuste EDR FULL — {event} — {det} =====")

    data, t = load_processed(det, event)
    dt = t[1] - t[0]

    # Valores iniciales razonables
    params0 = [
        1.0,   # A22
        0.1,   # A33
        0.1,   # A21
        0.0,   # d_om22
        0.0,   # d_tau22
        0.0,   # d_om33
        0.0,   # d_tau33
        0.0,   # d_om21
        0.0,   # d_tau21
        0.0,   # phi22
        0.0,   # phi33
        0.0,   # phi21
        0.01,  # t0
    ]

    bounds = [
        (0.0, None),   # A22
        (0.0, None),   # A33
        (0.0, None),   # A21
        (-0.5, 0.5),   # d_om22
        (-0.5, 0.5),   # d_tau22
        (-0.5, 0.5),   # d_om33
        (-0.5, 0.5),   # d_tau33
        (-0.5, 0.5),   # d_om21
        (-0.5, 0.5),   # d_tau21
        (-2*np.pi, 2*np.pi),  # phi22
        (-2*np.pi, 2*np.pi),  # phi33
        (-2*np.pi, 2*np.pi),  # phi21
        (0.0, 0.05),   # t0
    ]

    res = minimize(
        neg_log_like,
        params0,
        args=(t, data, Mrem, chi),
        bounds=bounds,
        method="L-BFGS-B",
    )

    best = res.x
    (
        A22, A33, A21,
        d_om22, d_tau22,
        d_om33, d_tau33,
        d_om21, d_tau21,
        phi22, phi33, phi21,
        t0
    ) = best

    print("\n✔ Parámetros EDR FULL:")
    print(f"A22      = {A22:.4f}")
    print(f"A33      = {A33:.4f}")
    print(f"A21      = {A21:.4f}")
    print(f"d_om22   = {d_om22:.5f}")
    print(f"d_tau22  = {d_tau22:.5f}")
    print(f"d_om33   = {d_om33:.5f}")
    print(f"d_tau33  = {d_tau33:.5f}")
    print(f"d_om21   = {d_om21:.5f}")
    print(f"d_tau21  = {d_tau21:.5f}")
    print(f"phi22    = {phi22:.3f}")
    print(f"phi33    = {phi33:.3f}")
    print(f"phi21    = {phi21:.3f}")
    print(f"t0       = {t0*1e3:.2f} ms")

    h_best = edr_multimode_full(
        t,
        Mrem,
        chi,
        A22, A33, A21,
        d_om22, d_tau22,
        d_om33, d_tau33,
        d_om21, d_tau21,
        phi22, phi33, phi21,
        t0,
    )
    resid = data - h_best

    # Gráficas
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax[0].plot(t, data, label="Datos whitened")
    ax[0].plot(t, h_best, label="EDR FULL", alpha=0.8)
    ax[0].set_title(f"{event} — {det} : EDR FULL multimodo")
    ax[0].legend()

    ax[1].plot(t, resid)
    ax[1].set_title("Residuo (dato - EDR FULL)")

    freqs = np.fft.rfftfreq(len(resid), dt)
    spec = np.abs(np.fft.rfft(resid))
    ax[2].plot(freqs, spec)
    ax[2].set_xlim(0, 1000)
    ax[2].set_title("Espectro del residuo")

    outfig = os.path.join(PLOT_DIR, f"{event}_{det}_edr_full.png")
    plt.tight_layout()
    plt.savefig(outfig)
    plt.close()

    print(f"✔ Figura guardada en: {outfig}")

    return best, h_best, resid


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Uso:")
        print("  python3 -m scripts.fit_edr_full H1 GW150914 68 0.67")
        sys.exit(1)

    det = sys.argv[1]
    event = sys.argv[2]
    Mrem = float(sys.argv[3])
    chi = float(sys.argv[4])

    fit_edr_full(det, event, Mrem, chi)
