# SensoLab · App de análisis sensorial

App en **Streamlit** que va de las boletas sensoriales al reporte PDF:

1. **Carga de datos**: CSV/Excel (consolidado o varias boletas individuales), ejemplos o ingreso manual.
2. **Tipo de prueba**: afectiva, discriminativa o descriptiva (la app sugiere una según los datos).
3. **Revisión automática**: datos incompletos, jueces faltantes, escalas mal codificadas (texto, valores fuera de escala, decimales), rangos repetidos, respuestas binarias no reconocidas, duplicados, etiquetas mal escritas, panel pequeño, jueces "planos" y jueces poco alineados (descriptiva).
4. **Recomendador**: aplica la regla de la guía (conteo → binomial/χ² · notas → t/ANOVA+Tukey · orden → Friedman) y explica por qué; se pueden elegir uno o varios métodos.
5. **Interpretación en lenguaje claro** y decisión para el producto.
6. **Gráficos y reporte PDF**, más descarga de datos depurados para validar en Excel.

Métodos: Media + IA %, t de Student (pareada/independiente/Welch), ANOVA (juez como bloque) + Tukey con letras, Friedman (+ DMS de rangos), binomial (triangular, dúo-trío, pareada, 2 de 5, tétrada, preferencia), Ji² bondad de ajuste, Ji² 2×2 A–no A (con Yates y Fisher), Q de Cochran (+ McNemar), PCA (mapa sensorial).

Los valores críticos (mínimo de tabla binomial, F, t, χ², q de Tukey) se calculan con las distribuciones exactas; las pruebas en `tests/` los comparan con las tablas clásicas.

## Ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

Pruebas: `pytest -q`

## Estructura

```
app.py                     interfaz Streamlit (pasos 1–6)
sensory/io_utils.py        lectura robusta CSV/Excel (codificación, separador, coma decimal)
sensory/schema.py          detección de columnas, formato largo, tipo de respuesta
sensory/validation.py      alertas y limpieza de datos
sensory/recommender.py     recomendador de métodos
sensory/methods/core.py    cálculos estadísticos puros
sensory/methods/catalog.py métodos + interpretación + decisión
sensory/plots.py           gráficos
sensory/report.py          reporte PDF
data/ejemplos/             9 conjuntos de ejemplo (uno con errores intencionales)
```

## Formatos de archivo

| Prueba | Columnas |
|---|---|
| Notas (hedónica / intensidad) | `juez, muestra, agrado` o ancho `juez, A, B, C` |
| Descriptiva (QDA) | `juez, muestra, dulzor, dureza, …` |
| Discriminativa | `juez, acierto` (1/0, sí/no, correcto/incorrecto) |
| A – no A | `juez, muestra_presentada, respuesta` (A / no A) |
| Preferencia | `juez, preferencia` (muestra elegida) |
| Ordenamiento | `juez, A, B, C` con lugares 1…k |
| Aceptación sí/no (Cochran) | `juez, A, B, C` con sí/no |
