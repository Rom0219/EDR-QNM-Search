"""
build_paper.py

Script maestro que:
 - Lee todos los resultados bayesianos en results/edr_full_pipeline/
 - Construye tabla CSV + JSON + LaTeX
 - Genera 3 gráficas: dlogZ, BayesFactor, Heatmap
 - Genera paper/latex/paper.tex listo para compilar

Ejecutar con:
    python3 -m scripts.build_paper
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================
# 1) Cargar resultados JSON
# =====================================
def load_results():
    files = glob.glob("results/bayes_compare/*.json")
    entries = []

    for f in files:
        with open(f, "r") as fp:
            data = json.load(fp)

        # Verificación de que sea JSON con resultados bayesianos reales
        required_keys = ["logZ_GR", "logZ_EDR", "dlogZ", "BayesFactor", "favored"]

        if not all(k in data for k in required_keys):
            print(f"⚠ Saltado (no es resultado bayesiano): {f}")
            continue

        entry = {
            "event": data["event"],
            "det": data["detector"],
            "logZ_GR": data["logZ_GR"],
            "logZ_EDR": data["logZ_EDR"],
            "dlogZ": data["dlogZ"],
            "BayesFactor": data["BayesFactor"],
            "favored": data["favored"]
        }
        entries.append(entry)

    if len(entries) == 0:
        raise RuntimeError("❌ No se encontró ningún resultado bayesiano válido en results/edr_full_pipeline/")

    return pd.DataFrame(entries)


# =====================================
# 2) Guardar tabla CSV, JSON, LaTeX
# =====================================
def save_table(df):

    os.makedirs("paper/tables", exist_ok=True)

    df.to_csv("paper/tables/results_table.csv", index=False)
    df.to_json("paper/tables/results_table.json", orient="records", indent=2)

    # LaTeX table
    latex_table = df.to_latex(index=False, float_format="%.6f")
    with open("paper/tables/results_table.tex", "w") as f:
        f.write(latex_table)


# =====================================
# 3) Gráficas
# =====================================
def plot_all(df):
    os.makedirs("paper/figures", exist_ok=True)

    # --- A) dlogZ ---
    plt.figure(figsize=(10,5))
    plt.bar(df["event"] + "_" + df["det"], df["dlogZ"])
    plt.xticks(rotation=90)
    plt.ylabel("ΔlogZ (EDR - GR)")
    plt.title("ΔlogZ por evento y detector")
    plt.tight_layout()
    plt.savefig("paper/figures/dlogZ.png", dpi=300)
    plt.close()

    # --- B) Bayes Factor ---
    plt.figure(figsize=(10,5))
    plt.bar(df["event"] + "_" + df["det"], df["BayesFactor"])
    plt.xticks(rotation=90)
    plt.ylabel("Bayes Factor")
    plt.title("Bayes Factor por evento y detector")
    plt.tight_layout()
    plt.savefig("paper/figures/bayes_factor.png", dpi=300)
    plt.close()

    # --- C) Heatmap ---
    pivot = df.pivot(index="event", columns="det", values="dlogZ")

    plt.figure(figsize=(6,6))
    plt.imshow(pivot, cmap="coolwarm", aspect="auto")
    plt.colorbar(label="ΔlogZ")
    plt.xticks(ticks=range(len(pivot.columns)), labels=pivot.columns)
    plt.yticks(ticks=range(len(pivot.index)), labels=pivot.index)
    plt.title("Mapa de calor de ΔlogZ")
    plt.tight_layout()
    plt.savefig("paper/figures/heatmap_dlogZ.png", dpi=300)
    plt.close()


# =====================================
# 4) Generar archivo LaTeX final
# =====================================
def write_latex():
    os.makedirs("paper/latex", exist_ok=True)

    template = r"""
\documentclass[12pt]{article}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{geometry}
\geometry{margin=1in}

\title{Comparación Bayesiana entre GR y la Teoría EDR en Modos QNM}
\author{Camilo Robinson Martínez}
\date{}

\begin{document}
\maketitle

\section{Resumen}
En este artículo se comparan las evidencias bayesianas (logZ) entre la Relatividad General (GR) y la Teoría del Espacio Dinámico Rotacional (EDR) utilizando los modos de ringdown (QNM) de múltiples eventos de ondas gravitacionales.

\section{Tabla de resultados}
\input{../tables/results_table.tex}

\section{Gráficos}

\subsection{Delta logZ}
\begin{center}
\includegraphics[width=0.95\textwidth]{../figures/dlogZ.png}
\end{center}

\subsection{Bayes Factor}
\begin{center}
\includegraphics[width=0.95\textwidth]{../figures/bayes_factor.png}
\end{center}

\subsection{Mapa de calor ΔlogZ}
\begin{center}
\includegraphics[width=0.7\textwidth]{../figures/heatmap_dlogZ.png}
\end{center}

\end{document}
"""

    with open("paper/latex/paper.tex", "w") as f:
        f.write(template)


# =====================================
# MAIN
# =====================================
def build_paper():
    print("📄 Cargando resultados...")
    df = load_results()

    print("📊 Guardando tabla...")
    save_table(df)

    print("📈 Generando gráficas...")
    plot_all(df)

    print("📝 Generando archivo LaTeX...")
    write_latex()

    print("\n✅ PAPER COMPLETO GENERADO EXITOSAMENTE")
    print("   - Tablas en: paper/tables/")
    print("   - Figuras en: paper/figures/")
    print("   - LaTeX en:  paper/latex/paper.tex")


if __name__ == "__main__":
    build_paper()
