"""Genera los conjuntos de datos de ejemplo (semilla fija → reproducibles).

Ejecutar:  python data/generate_examples.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "ejemplos"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)


def hedonic(mu, n, sd=1.3, lo=1, hi=9):
    return np.clip(np.round(rng.normal(mu, sd, n)), lo, hi).astype(int)


# 1. Afectiva · 3 niveles de azúcar · hedónica 9 puntos (formato largo)
judges = [f"C{i:02d}" for i in range(1, 37)]
bias = rng.normal(0, 0.7, len(judges))
rows = []
for j, b in zip(judges, bias):
    for m, mu in (("Azucar_bajo", 5.6), ("Azucar_medio", 7.0), ("Azucar_alto", 7.3)):
        rows.append({"juez": j, "muestra": m, "agrado": int(np.clip(round(mu + b + rng.normal(0, 1.1)), 1, 9))})
df1 = pd.DataFrame(rows)
df1.to_csv(OUT / "afectiva_hedonica_3_muestras.csv", index=False)

# 1b. Misma prueba CON ERRORES para demostrar las alertas
err = df1.copy().astype({"agrado": object})
err.loc[3, "agrado"] = "siete"            # texto en escala numérica
err.loc[10, "agrado"] = 77                # fuera de escala
err.loc[20, "agrado"] = None              # dato faltante
err.loc[31, "muestra"] = "azucar_medio "  # etiqueta mal escrita
err = err.drop(index=[45])                # juez que no evaluó una muestra
err = pd.concat([err, err.iloc[[5]]])     # registro duplicado
err.to_csv(OUT / "afectiva_con_errores.csv", index=False, sep=";")

# 2. Afectiva · 2 muestras (formato ancho, jueces = filas)
n = 40
b = rng.normal(0, 0.8, n)
df2 = pd.DataFrame({"consumidor": range(1, n + 1),
                    "DNP": np.clip(np.round(7.1 + b + rng.normal(0, 1, n)), 1, 9).astype(int),
                    "Comercial": np.clip(np.round(6.3 + b + rng.normal(0, 1, n)), 1, 9).astype(int)})
df2.to_csv(OUT / "afectiva_2_muestras_ancho.csv", index=False)

# 3. Afectiva · preferencia pareada (28 de 40 prefieren DNP, como en la guía)
pref = np.array(["DNP"] * 28 + ["Comercial"] * 12)
rng.shuffle(pref)
pd.DataFrame({"juez": range(1, 41), "preferencia": pref}).to_csv(OUT / "afectiva_preferencia_pareada.csv", index=False)

# 4. Afectiva · Q de Cochran (¿lo compraría? sí/no, 3 prototipos)
n = 32
pd.DataFrame({"juez": range(1, n + 1),
              "Prototipo_A": rng.choice(["si", "no"], n, p=[0.75, 0.25]),
              "Prototipo_B": rng.choice(["si", "no"], n, p=[0.45, 0.55]),
              "Prototipo_C": rng.choice(["si", "no"], n, p=[0.70, 0.30])}).to_csv(
    OUT / "afectiva_compra_si_no_cochran.csv", index=False)

# 5. Discriminativa · triangular (¿se nota la reducción de azúcar?)
n = 30
acierto = np.array([1] * 16 + [0] * 14)
rng.shuffle(acierto)
pd.DataFrame({"juez": [f"J{i:02d}" for i in range(1, n + 1)], "acierto_triangular": acierto}).to_csv(
    OUT / "discriminativa_triangular.csv", index=False)

# 6. Discriminativa · A – no A
rows = []
for i in range(1, 51):
    presented = "A" if i <= 25 else "no A"
    p_say_a = 0.8 if presented == "A" else 0.35
    rows.append({"juez": i, "muestra_presentada": presented,
                 "respuesta": "A" if rng.random() < p_say_a else "no A"})
pd.DataFrame(rows).to_csv(OUT / "discriminativa_a_no_a.csv", index=False)

# 7. Ordenamiento (Friedman) · 4 galletas por dulzor, 1 = más dulce
rows = []
true = {"G1": 1.5, "G2": 2.2, "G3": 2.9, "G4": 3.4}
for j in range(1, 21):
    noisy = {k: v + rng.normal(0, 0.8) for k, v in true.items()}
    order = sorted(noisy, key=noisy.get)
    rows.append({"juez": j, **{k: order.index(k) + 1 for k in true}})
pd.DataFrame(rows).to_csv(OUT / "ordenamiento_friedman.csv", index=False)

# 8. Descriptiva · QDA de 4 galletas, 10 panelistas, escala 0–15
attrs = {"dulzor": [6, 9, 9.5, 12], "dureza": [10, 7, 8, 6], "crujencia": [11, 8, 10, 7],
         "aroma_tostado": [9, 5, 7, 4], "color": [8, 8.3, 7.8, 8.1]}
rows = []
for j in range(1, 11):
    jb = rng.normal(0, 0.6)
    for i, s in enumerate(["DNP", "Comercial_1", "Comercial_2", "Comercial_3"]):
        rows.append({"panelista": f"P{j}", "muestra": s,
                     **{a: round(float(np.clip(v[i] + jb + rng.normal(0, 1.2), 0, 15)), 1) for a, v in attrs.items()}})
pd.DataFrame(rows).to_csv(OUT / "descriptiva_qda.csv", index=False)

print("Ejemplos generados en", OUT)
