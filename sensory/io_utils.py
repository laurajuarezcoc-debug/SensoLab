"""Carga robusta de archivos CSV / Excel (boletas individuales o consolidado)."""
from __future__ import annotations

import csv
import io
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ENCODINGS = ("utf-8-sig", "utf-8", "latin-1", "cp1252")
MAX_BYTES = 20 * 1024 * 1024  # 20 MB


class DataLoadError(Exception):
    """Error legible para el usuario al cargar datos."""


@dataclass
class LoadResult:
    df: pd.DataFrame
    notes: list[str] = field(default_factory=list)


def normalize_text(value: str) -> str:
    """Minúsculas, sin tildes y sin espacios extra (para comparar etiquetas)."""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", value).strip().lower()


def is_text(series: pd.Series) -> bool:
    """True para columnas de texto (object o str, compatible con pandas 2 y 3)."""
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _decode(raw: bytes) -> tuple[str, str]:
    for enc in ENCODINGS:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise DataLoadError("No se pudo leer el texto del archivo (codificación desconocida). "
                        "Guárdalo como 'CSV UTF-8' desde Excel.")


def _sniff_separator(text: str) -> str:
    sample = "\n".join(text.splitlines()[:20])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        counts = {sep: sample.count(sep) for sep in (";", ",", "\t", "|")}
        return max(counts, key=counts.get) if any(counts.values()) else ","


def _fix_decimal_commas(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Convierte columnas tipo '7,5' a números (Excel en español)."""
    notes = []
    for col in df.columns:
        if not is_text(df[col]):
            continue
        s = df[col].dropna().astype(str).str.strip()
        if s.empty:
            continue
        if s.str.fullmatch(r"-?\d+,\d+|-?\d+").mean() > 0.8 and s.str.contains(",").any():
            df[col] = pd.to_numeric(df[col].astype(str).str.strip().str.replace(",", ".", regex=False),
                                    errors="coerce").where(df[col].notna())
            notes.append(f"Columna «{col}»: se convirtió la coma decimal a punto.")
    return df, notes


def _clean_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    df = df.copy()
    # Encabezados limpios y únicos
    cols = [str(c).strip() for c in df.columns]
    seen: dict[str, int] = {}
    clean = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            new = f"{c}_{seen[c]}"
            notes.append(f"Encabezado duplicado «{c}» renombrado a «{new}».")
            clean.append(new)
        else:
            seen[c] = 0
            clean.append(c)
    df.columns = clean
    # Filas y columnas completamente vacías
    n0 = len(df)
    df = df.dropna(how="all")
    if len(df) < n0:
        notes.append(f"Se ignoraron {n0 - len(df)} filas completamente vacías.")
    # Solo se eliminan columnas vacías sin nombre (las vacías con nombre se reportan como faltantes)
    empty_unnamed = [c for c in df.columns
                     if (c.lower().startswith("unnamed") or c == "") and df[c].isna().all()]
    if empty_unnamed:
        df = df.drop(columns=empty_unnamed)
    # Espacios en celdas de texto
    for col in df.columns:
        if is_text(df[col]):
            df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
            df[col] = df[col].replace({"": pd.NA, "-": pd.NA, "NA": pd.NA, "N/A": pd.NA, "n/a": pd.NA})
    df, dec_notes = _fix_decimal_commas(df)
    notes.extend(dec_notes)
    return df.reset_index(drop=True), notes


def read_table(raw: bytes, filename: str) -> LoadResult:
    """Lee un CSV o Excel desde bytes y devuelve un DataFrame limpio."""
    if not raw:
        raise DataLoadError(f"El archivo «{filename}» está vacío.")
    if len(raw) > MAX_BYTES:
        raise DataLoadError(f"El archivo «{filename}» supera 20 MB.")
    suffix = Path(filename).suffix.lower()
    notes: list[str] = []
    try:
        if suffix in (".xlsx", ".xlsm", ".xls"):
            df = pd.read_excel(io.BytesIO(raw), dtype=object)
            notes.append("Excel: se leyó la primera hoja.")
        else:
            text, enc = _decode(raw)
            sep = _sniff_separator(text)
            df = pd.read_csv(io.StringIO(text), sep=sep, dtype=object,
                             skipinitialspace=True, on_bad_lines="error")
            if enc not in ("utf-8", "utf-8-sig"):
                notes.append(f"Archivo leído con codificación {enc}.")
            if sep != ",":
                notes.append(f"Separador detectado: «{'TAB' if sep == chr(9) else sep}».")
    except DataLoadError:
        raise
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"«{filename}» tiene filas con distinto número de columnas: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - mostramos cualquier fallo de lectura
        raise DataLoadError(f"No se pudo leer «{filename}»: {exc}") from exc

    # Convertir columnas numéricas (los strings numéricos pasan a float)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if df[col].notna().sum() and converted.notna().sum() == df[col].notna().sum():
            df[col] = converted
    df, clean_notes = _clean_frame(df)
    notes.extend(clean_notes)
    if df.empty or df.shape[1] == 0:
        raise DataLoadError(f"«{filename}» no contiene datos después de la limpieza.")
    if df.shape[1] < 2:
        raise DataLoadError(f"«{filename}» tiene una sola columna. ¿Usaste el separador correcto? "
                            "Se esperan columnas como juez, muestra y calificación.")
    return LoadResult(df, notes)


def combine_ballots(results: list[tuple[str, LoadResult]], judge_col_hint: str | None = None) -> LoadResult:
    """Une varias boletas individuales en un consolidado del panel.

    Si una boleta no trae columna de juez, se usa el nombre del archivo.
    """
    if len(results) == 1:
        return results[0][1]
    frames, notes = [], []
    col_sets = [set(r.df.columns) for _, r in results]
    common = set.intersection(*col_sets)
    if len(common) < 1:
        raise DataLoadError("Las boletas no comparten columnas; revisa que usen el mismo formato.")
    differing = set.union(*col_sets) - common
    if differing:
        notes.append("Columnas que no están en todas las boletas (quedarán vacías donde falten): "
                      + ", ".join(sorted(differing)))
    for name, res in results:
        df = res.df.copy()
        has_judge = judge_col_hint and judge_col_hint in df.columns
        if not has_judge and "juez" not in [normalize_text(c) for c in df.columns]:
            df.insert(0, "juez", Path(name).stem)
        df["_archivo"] = name
        frames.append(df)
        notes.extend(f"[{name}] {n}" for n in res.notes)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    notes.append(f"Se consolidaron {len(results)} boletas en una tabla de {len(combined)} filas.")
    return LoadResult(combined, notes)
