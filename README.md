# ¿Qué hace que una vivienda sea más cara? Análisis y predicción del precio de la vivienda en México

Proyecto de análisis y modelado predictivo del monto de financiamiento de vivienda en México, usando datos abiertos de la API del **SNIIV (Sistema Nacional de Información e Indicadores de Vivienda, SEDATU)**.

## Objetivo

Identificar qué características explican el monto de financiamiento de una vivienda en México, mediante análisis estadístico y un modelo predictor del monto unitario de financiamiento (proxy del precio de la vivienda).

## Cumplimiento de requisitos

| # | Requisito | Dónde se cumple | Salida |
|---|-----------|-----------------|--------|
| 1 | Obtener los datos | `proyecto.py` → Requisito 1 (API SNIIV) | `data/datos_crudos.csv` |
| 2 | Limpieza con Python + Pandas | `proyecto.py` → Requisito 2 | `data/datos_limpios.csv` |
| 3 | Estadística descriptiva | `analisis.py` → Requisito 3 | `data/estadistica_descriptiva.csv`, `data/estadistica_global.csv` |
| 4 | Correlaciones | `analisis.py` → Requisito 4 | `data/correlaciones.csv`, `data/correlaciones_con_precio.csv`, `data/viz_6_correlaciones.png` |
| 5 | Regresión lineal | `analisis.py` → Requisito 5 | `data/regresion_lineal_coeficientes.csv`, `data/viz_7_regresion_lineal.png`, `data/viz_8_residuos.png` |
| 6 | Conclusiones | `analisis.py` → Requisito 6 | `data/conclusiones.csv` (y sección *Conclusiones* abajo) |

## Fuente de datos

- **API SNIIV (SEDATU):** https://sniiv.sedatu.gob.mx/api/CuboAPI/
- Endpoints usados:
  - `GetFinanciamiento` — acciones y montos de financiamiento por segmento, modalidad, género
  - `GetInfonavit` — créditos INFONAVIT por línea, esquema, rango salarial
  - `GetCNBV` — créditos de la banca comercial
  - `GetFOVISSSTE` — créditos FOVISSSTE
- Cobertura: nacional, 2024–2025. Desagregación estatal y municipal.

**Limitante:** los datos son agregados (montos y acciones por categoría), no transacciones individuales. Por ello no se dispone de m², número de habitaciones/baños, antigüedad o estacionamiento a nivel de vivienda individual.

## Metodología

1. **Obtener los datos** — descarga desde la API SNIIV (`proyecto.py`).
2. **Limpieza (Pandas)** — normalización de categorías, conversión de montos, eliminación de "No disponible" y valores no válidos.
3. **Variable objetivo** — `monto_unitario = monto / acciones` (monto promedio por financiamiento).
4. **Estadística descriptiva** — media, mediana, desviación, cuartiles, asimetría y curtosis global y por segmento (`analisis.py`).
5. **Correlaciones** — codificación *one-hot* y correlación de cada categoría con el precio (`analisis.py`).
6. **Regresión lineal** — modelo sobre variables codificadas, con coeficientes, métricas y análisis de residuos (`analisis.py`).
7. **Modelos complementarios** — Random Forest y Gradient Boosting, con validación cruzada (`proyecto.py`).
8. **Conclusiones** — interpretación de resultados (`analisis.py`).
9. **Análisis geográfico** — monto promedio por entidad federativa (`analisis_geografico.py`).

## Resultados

Mejor modelo: **Random Forest**

| Modelo             | RMSE      | MAE       | R²    |
|--------------------|-----------|-----------|-------|
| Regresión Lineal   | 1,513,441 | 943,605   | 0.097 |
| **Random Forest**  | **835,172** | **377,370** | **0.725** |
| Gradient Boosting  | 876,300   | 421,472   | 0.697 |

### Validación cruzada (5-fold)

| Modelo             | R² medio      | RMSE medio    |
|--------------------|---------------|---------------|
| Regresión Lineal   | 0.104 ± 0.011 | $1,482,308    |
| **Random Forest**  | **0.724 ± 0.009** | **$823,270** |
| Gradient Boosting  | 0.699 ± 0.012 | $859,101      |

La baja desviación estándar de Random Forest (±0.009) confirma que el modelo generaliza de forma estable y no está sobreajustado.

### Importancia de características

| Característica  | Importancia |
|-----------------|-------------|
| segmento        | 0.807 |
| modalidad       | 0.102 |
| rango_salarial  | 0.064 |
| linea_credito   | 0.012 |
| genero          | 0.005 |
| vivienda        | 0.003 |
| esquema         | 0.003 |
| destino_credito | 0.003 |

### Monto promedio por segmento

| Segmento         | Monto promedio |
|------------------|----------------|
| Residencial plus | $4,338,307 |
| Residencial      | $2,097,132 |
| Media            | $1,246,872 |
| Tradicional      | $710,934   |
| Popular          | $463,974   |
| Económica        | $181,042   |

### Estadística descriptiva (monto unitario, MXN)

| Estadístico | Valor |
|-------------|-------|
| N | 27,361 |
| Media | $1,730,197 |
| Mediana | $1,298,162 |
| Desv. estándar | $1,567,026 |
| Mínimo | $2,249 |
| Q1 | $770,000 |
| Q3 | $2,160,000 |
| Máximo | $29,970,493 |
| Asimetría | 2.951 |
| Curtosis | 16.267 |

La distribución presenta **asimetría positiva** (skew=2.95): hay viviendas de lujo que elevan la media por encima de la mediana.

### Correlaciones

Correlación de cada categoría con el monto unitario (top 5):

| Categoría | r |
|-----------|---|
| segmento = Residencial plus | **0.687** |
| segmento = Tradicional | −0.312 |
| rango_salarial = Más de 12 | 0.233 |
| segmento = Media | −0.214 |
| segmento = Económica | −0.194 |

### Regresión lineal

Modelo sobre variables codificadas (one-hot):

| Métrica | Valor |
|---------|-------|
| R² | 0.644 |
| RMSE | $950,393 |
| MAE | $498,111 |

Coeficientes más influyentes: `segmento=Residencial plus` (+899,647), `segmento=Tradicional` (−392,476), `segmento=Económica` (−355,173).

## Conclusiones

1. La distribución del monto unitario tiene **asimetría positiva** (skew=2.95): la media ($1.73M) supera a la mediana ($1.30M) por la presencia de viviendas de lujo.
2. La variable con mayor correlación con el precio es el **segmento "Residencial plus"** (r=0.69).
3. La característica más explicativa es el **segmento de vivienda** (80.7% de importancia en Random Forest): Residencial plus ($4.34M) > Residencial ($2.10M) > Media ($1.25M) > Tradicional ($0.71M) > Popular ($0.46M) > Económica ($0.18M).
4. Le siguen la **modalidad** (nueva/existente) y el **rango salarial** del acreditado.
5. La **regresión lineal** alcanza un R²=0.644 y el **Random Forest** un R²=0.725, confirmando que las características modeladas explican ~70% del precio; el resto es variabilidad no capturada (ubicación fina, m², acabados).

A nivel geográfico, los estados con mayor monto promedio fueron **Quintana Roo** ($1.69M), **Sinaloa** ($1.65M) y **Ciudad de México** ($1.65M); los menores, **Tlaxcala** ($1.14M), **Campeche** ($1.18M) y **Chiapas** ($1.22M).

## Estructura del proyecto

```
practica-06-precio-vivienda/
├── proyecto.py               # Req 1-2: ingesta, limpieza + modelos y CV
├── analisis.py               # Req 3-6: descriptiva, correlaciones, regresión, conclusiones
├── visualizaciones.py        # Generación de 5 gráficos
├── analisis_geografico.py    # Análisis por entidad federativa
├── dashboard.py              # Dashboard interactivo (Streamlit)
├── test_pipeline.py          # 10 pruebas de validación del pipeline
├── requirements.txt
├── .gitignore
├── data/
│   ├── datos_crudos.csv              # salida req 1
│   ├── datos_limpios.csv             # salida req 2
│   ├── estadistica_descriptiva.csv   # salida req 3
│   ├── estadistica_global.csv        # salida req 3
│   ├── correlaciones.csv             # salida req 4
│   ├── correlaciones_con_precio.csv  # salida req 4
│   ├── regresion_lineal_coeficientes.csv  # salida req 5
│   ├── conclusiones.csv              # salida req 6
│   ├── resultados_modelos.csv
│   ├── cv_resultados.csv
│   ├── importancia_caracteristicas.csv
│   ├── analisis_estados.csv
│   ├── datos_por_estado.csv
│   └── viz_*.png
└── README.md
```

## Reproducción

```bash
pip install -r requirements.txt
python proyecto.py            # req 1-2: datos + limpieza + modelos + CV
python analisis.py            # req 3-6: descriptiva, correlaciones, regresión, conclusiones
python visualizaciones.py     # 5 gráficos
python analisis_geografico.py # análisis estatal
python test_pipeline.py       # pruebas de validación
streamlit run dashboard.py    # dashboard interactivo
```

## Requisitos

Python 3.12+, pandas, numpy, scikit-learn, matplotlib, seaborn, requests, streamlit, plotly.

## Fuente

SEDATU — Sistema Nacional de Información e Indicadores de Vivienda (SNIIV). Datos abiertos de acceso público. Se permite su uso mencionando la fuente.
