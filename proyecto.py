import requests
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("¿Qué hace que una vivienda sea más cara?")
print("Análisis y predicción del precio de la vivienda en México")
print("Fuente: API SNIIV (SEDATU)")
print("=" * 60)

# ============================================================
# REQUISITO 1: OBTENER LOS DATOS (API SNIIV)
# ============================================================
print("\n[REQUISITO 1] Obteniendo datos de la API SNIIV...")

base_url = "https://sniiv.sedatu.gob.mx/api/CuboAPI"

datasets = {}

# Financiamientos
print("  - Financiamientos...")
url = f"{base_url}/GetFinanciamiento/2024,2025/00/000/valor_vivienda,modalidad,genero,organismo,destino_credito"
resp = requests.get(url, timeout=60)
if resp.status_code == 200:
    df_fin = pd.DataFrame(resp.json())
    datasets['financiamientos'] = df_fin
    print(f"    {len(df_fin)} registros")

# INFONAVIT
print("  - INFONAVIT...")
url = f"{base_url}/GetInfonavit/2024,2025/00/000/valor_vivienda,modalidad,genero,vivienda,linea_credito,rango_salarial,rango_edad"
resp = requests.get(url, timeout=60)
if resp.status_code == 200:
    df_inf = pd.DataFrame(resp.json())
    datasets['infonavit'] = df_inf
    print(f"    {len(df_inf)} registros")

# CNBV
print("  - CNBV...")
url = f"{base_url}/GetCNBV/2024,2025/00/000/valor_vivienda,modalidad,genero,tipo_credito,rango_salarial"
resp = requests.get(url, timeout=60)
if resp.status_code == 200:
    df_cnbv = pd.DataFrame(resp.json())
    datasets['cnbv'] = df_cnbv
    print(f"    {len(df_cnbv)} registros")

# FOVISSSTE
print("  - FOVISSSTE...")
url = f"{base_url}/GetFOVISSSTE/2024,2025/00/000/valor_vivienda,modalidad,genero,esquema,rango_salarial"
resp = requests.get(url, timeout=60)
if resp.status_code == 200:
    df_fov = pd.DataFrame(resp.json())
    datasets['fovissste'] = df_fov
    print(f"    {len(df_fov)} registros")

# Consolidar y guardar datos crudos (Requisito 1)
df_all = pd.concat(datasets, names=['fuente']).reset_index(level=0)
df_all.to_csv('data/datos_crudos.csv', index=False, encoding='utf-8-sig')
print(f"  TOTAL: {len(df_all)} registros -> data/datos_crudos.csv")

# ============================================================
# REQUISITO 2: LIMPIEZA CON PYTHON + PANDAS
# ============================================================
print("\n[REQUISITO 2] Limpieza y preprocesamiento con Pandas...")

# Renombrar columnas
column_renames = {
    'año': 'anio',
    'sexo': 'genero',
    'valor_vivienda': 'segmento',
    'monto': 'monto_financiamiento',
    'acciones': 'num_acciones'
}
df_all.rename(columns=column_renames, inplace=True)

# Limpiar valores nulos y "No disponible"
df_all.replace('No disponible', np.nan, inplace=True)
df_all.replace('', np.nan, inplace=True)

# Convertir monto a numérico
df_all['monto_financiamiento'] = pd.to_numeric(df_all['monto_financiamiento'], errors='coerce')
df_all['num_acciones'] = pd.to_numeric(df_all['num_acciones'], errors='coerce')

# Calcular monto por acción (precio unitario)
df_all['monto_unitario'] = df_all['monto_financiamiento'] / df_all['num_acciones']
df_all['monto_unitario'] = df_all['monto_unitario'].replace([np.inf, -np.inf], np.nan)

# Filtrar registros válidos
df_clean = df_all.dropna(subset=['monto_unitario', 'segmento', 'modalidad', 'genero']).copy()
df_clean = df_clean[df_clean['monto_unitario'] > 0]

print(f"  Registros totales: {len(df_all)}")
print(f"  Registros válidos: {len(df_clean)}")

# ============================================================
# PASO 3: ANÁLISIS EXPLORATORIO (EDA)
# ============================================================
print("\n[PASO 3] Análisis exploratorio...")

print("\n--- Distribución de monto unitario por segmento ---")
print(df_clean.groupby('segmento')['monto_unitario'].agg(['count', 'mean', 'median', 'std']).round(0))

print("\n--- Distribución por modalidad ---")
print(df_clean.groupby('modalidad')['monto_unitario'].agg(['count', 'mean', 'median']).round(0))

print("\n--- Distribución por género ---")
print(df_clean.groupby('genero')['monto_unitario'].agg(['count', 'mean', 'median']).round(0))

# ============================================================
# PASO 4: INGENIERÍA DE CARACTERÍSTICAS
# ============================================================
print("\n[PASO 4] Ingeniería de características...")

# Variables categóricas
cat_cols = ['segmento', 'modalidad', 'genero']
if 'destino_credito' in df_clean.columns:
    cat_cols.append('destino_credito')
if 'vivienda' in df_clean.columns:
    cat_cols.append('vivienda')
if 'linea_credito' in df_clean.columns:
    cat_cols.append('linea_credito')
if 'rango_salarial' in df_clean.columns:
    cat_cols.append('rango_salarial')
if 'rango_edad' in df_clean.columns:
    cat_cols.append('rango_edad')
if 'tipo_credito' in df_clean.columns:
    cat_cols.append('tipo_credito')
if 'esquema' in df_clean.columns:
    cat_cols.append('esquema')

# Label encoding
label_encoders = {}
df_model = df_clean.copy()

for col in cat_cols:
    if col in df_model.columns:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].fillna('No disponible').astype(str))
        label_encoders[col] = le

# Variable objetivo
y = df_model['monto_unitario'].values

# Características
feature_cols = [c for c in cat_cols if c in df_model.columns]
X = df_model[feature_cols].values

print(f"  Características: {feature_cols}")
print(f"  Variable objetivo: monto_unitario")
print(f"  Total muestras: {len(X)}")

# ============================================================
# PASO 5: ENTRENAMIENTO DE MODELOS
# ============================================================
print("\n[PASO 5] Entrenando modelos...")

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelos
models = {
    'Regresión Lineal': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"  - Entrenando {name}...")
    if name == 'Regresión Lineal':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'model': model,
        'y_pred': y_pred
    }

    print(f"    RMSE: {rmse:,.0f}")
    print(f"    MAE:  {mae:,.0f}")
    print(f"    R²:   {r2:.4f}")

# ============================================================
# PASO 6: EVALUACIÓN Y COMPARACIÓN
# ============================================================
print("\n[PASO 6] Evaluación y comparación de modelos...")

print("\n" + "=" * 60)
print("RESULTADOS COMPARATIVOS")
print("=" * 60)
print(f"{'Modelo':<20} {'RMSE':>15} {'MAE':>15} {'R²':>10}")
print("-" * 60)
for name, res in results.items():
    print(f"{name:<20} {res['RMSE']:>15,.0f} {res['MAE']:>15,.0f} {res['R2']:>10.4f}")

# Mejor modelo
best_model_name = min(results, key=lambda x: results[x]['RMSE'])
best_model = results[best_model_name]['model']
print(f"\nMejor modelo: {best_model_name}")

# ============================================================
# PASO 6.5: VALIDACIÓN CRUZADA (5-fold)
# ============================================================
print("\n[PASO 6.5] Validación cruzada (5-fold)...")

cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_models = {
    'Regresión Lineal': make_pipeline(StandardScaler(), LinearRegression()),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

cv_summary = []
for name, model in cv_models.items():
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'))
    cv_summary.append({
        'Modelo': name,
        'R2_media': r2_scores.mean(),
        'R2_std': r2_scores.std(),
        'RMSE_media': rmse_scores.mean(),
        'RMSE_std': rmse_scores.std()
    })
    print(f"  {name:<20} R²={r2_scores.mean():.4f} (±{r2_scores.std():.4f})  RMSE={rmse_scores.mean():,.0f} (±{rmse_scores.std():,.0f})")

cv_df = pd.DataFrame(cv_summary)

# ============================================================
# PASO 7: INTERPRETACIÓN DE RESULTADOS
# ============================================================
print("\n[PASO 7] Interpretación de resultados...")

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feat_imp = pd.DataFrame({
        'Característica': feature_cols,
        'Importancia': importances
    }).sort_values('Importancia', ascending=False)

    print("\n--- Importancia de características ---")
    for _, row in feat_imp.iterrows():
        print(f"  {row['Característica']:<20}: {row['Importancia']:.4f}")

# Análisis por segmento
print("\n--- Monto promedio por segmento ---")
segmento_analysis = df_clean.groupby('segmento')['monto_unitario'].agg(['count', 'mean', 'median']).round(0)
segmento_analysis = segmento_analysis.sort_values('mean', ascending=False)
print(segmento_analysis)

# Análisis por modalidad
print("\n--- Monto promedio por modalidad ---")
modalidad_analysis = df_clean.groupby('modalidad')['monto_unitario'].agg(['count', 'mean', 'median']).round(0)
modalidad_analysis = modalidad_analysis.sort_values('mean', ascending=False)
print(modalidad_analysis)

# ============================================================
# PASO 8: GUARDAR RESULTADOS
# ============================================================
print("\n[PASO 8] Guardando resultados...")

# Guardar datos limpios
df_clean.to_csv('data/datos_limpios.csv', index=False, encoding='utf-8-sig')

# Guardar resultados de modelos
results_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'RMSE': [results[k]['RMSE'] for k in results.keys()],
    'MAE': [results[k]['MAE'] for k in results.keys()],
    'R2': [results[k]['R2'] for k in results.keys()]
})
results_df.to_csv('data/resultados_modelos.csv', index=False, encoding='utf-8-sig')

# Guardar importancia de características
if hasattr(best_model, 'feature_importances_'):
    feat_imp.to_csv('data/importancia_caracteristicas.csv', index=False, encoding='utf-8-sig')

# Guardar resultados de validación cruzada
cv_df.to_csv('data/cv_resultados.csv', index=False, encoding='utf-8-sig')

print("\n" + "=" * 60)
print("PROYECTO COMPLETADO")
print("=" * 60)
print("\nArchivos generados:")
print("  - data/datos_limpios.csv")
print("  - data/resultados_modelos.csv")
print("  - data/importancia_caracteristicas.csv")
print("  - data/cv_resultados.csv")
