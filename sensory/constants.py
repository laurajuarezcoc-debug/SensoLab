"""Constantes compartidas: tipos de prueba, escalas, protocolos y paleta."""
from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# Tipos de prueba sensorial
# --------------------------------------------------------------------------
AFECTIVA = "afectiva"
DISCRIMINATIVA = "discriminativa"
DESCRIPTIVA = "descriptiva"

TEST_TYPES = {
    AFECTIVA: "Afectiva · ¿me gusta? (consumidores)",
    DISCRIMINATIVA: "Discriminativa · ¿son diferentes? (jueces seleccionados)",
    DESCRIPTIVA: "Descriptiva · ¿cómo son? (panel entrenado)",
}

# --------------------------------------------------------------------------
# Tipos de respuesta detectables en los datos
# --------------------------------------------------------------------------
ESCALA = "escala"          # notas numéricas (hedónica, intensidad, QDA)
RANGO = "rango"            # orden 1..k por juez
BINARIO = "binario"        # acierto/fallo, sí/no, acepta/rechaza
CATEGORICO = "categorico"  # elección de una muestra (preferencia) o A / no A

RESPONSE_LABELS = {
    ESCALA: "Escala numérica (notas)",
    RANGO: "Orden / ranking",
    BINARIO: "Conteo binario (acierto/fallo, sí/no)",
    CATEGORICO: "Elección categórica (preferencia, A / no A)",
}

# Palabras reconocidas como respuestas binarias (minúsculas, sin tildes)
BINARY_TRUE = {"1", "si", "s", "yes", "y", "true", "verdadero", "v", "correcto",
               "correcta", "acierto", "acerto", "ok", "x", "acepta", "aceptado",
               "aceptada", "acepto", "compraria"}
BINARY_FALSE = {"0", "no", "n", "false", "falso", "f", "incorrecto", "incorrecta",
                "fallo", "error", "rechaza", "rechazado", "rechazada",
                "rechazo", "no compraria"}

# Valores reconocidos para A / no A
A_VALUES = {"a"}
NOT_A_VALUES = {"no a", "noa", "no_a", "no-a", "not a"}

# --------------------------------------------------------------------------
# Escala hedónica de 9 puntos (Peryam & Pilgrim)
# --------------------------------------------------------------------------
HEDONIC_9 = {
    1: "me disgusta extremadamente",
    2: "me disgusta mucho",
    3: "me disgusta moderadamente",
    4: "me disgusta ligeramente",
    5: "ni me gusta ni me disgusta",
    6: "me gusta ligeramente",
    7: "me gusta moderadamente",
    8: "me gusta mucho",
    9: "me gusta extremadamente",
}

# Escalas típicas (para sugerir la escala declarada)
COMMON_SCALES = [(1, 5), (1, 9), (1, 7), (0, 10), (1, 10), (0, 15), (0, 100)]


# --------------------------------------------------------------------------
# Protocolos discriminativos (probabilidad de acertar por azar)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Protocol:
    key: str
    name: str
    p0: float
    alternative: str  # "greater" (unilateral) o "two-sided"
    description: str


PROTOCOLS = {
    "triangular": Protocol("triangular", "Triangular", 1 / 3, "greater",
                           "3 muestras (2 iguales, 1 distinta); el juez señala la distinta."),
    "duo_trio": Protocol("duo_trio", "Dúo-trío", 1 / 2, "greater",
                         "Referencia + 2 muestras; el juez señala cuál es igual a la referencia."),
    "pareada_dir": Protocol("pareada_dir", "Comparación pareada direccional", 1 / 2, "greater",
                            "2 muestras; el juez señala cuál tiene más intensidad (respuesta correcta conocida)."),
    "dos_de_cinco": Protocol("dos_de_cinco", "2 de 5", 1 / 10, "greater",
                             "5 muestras (2 de un tipo, 3 de otro); el juez agrupa correctamente."),
    "tetrada": Protocol("tetrada", "Tétrada", 1 / 3, "greater",
                        "4 muestras (2 + 2); el juez forma los dos grupos."),
    "preferencia": Protocol("preferencia", "Preferencia pareada (afectiva)", 1 / 2, "two-sided",
                            "2 muestras; el consumidor elige la que prefiere (bilateral)."),
}

# Jueces mínimos recomendados (guía ISO/ASTM y bibliografía clásica)
MIN_JUDGES = {
    AFECTIVA: 30,        # consumidores (ideal 50-100)
    DISCRIMINATIVA: 18,  # 18-24 jueces seleccionados para triangular/dúo-trío
    DESCRIPTIVA: 6,      # panel entrenado 6-12
}

# --------------------------------------------------------------------------
# Paleta (validada para daltonismo; orden fijo, nunca ciclado)
# --------------------------------------------------------------------------
SERIES_COLORS = ["#d14996", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
COLOR_PRIMARY = "#d14996"
COLOR_MUTED = "#a8a7a2"
COLOR_GOOD = "#1a7f37"
COLOR_BAD = "#c4312f"
COLOR_TEXT = "#0b0b0b"
COLOR_TEXT_2 = "#52514e"
COLOR_GRID = "#e4e3df"
