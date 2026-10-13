"""
Informe de análisis: estadística descriptiva, correlaciones y regresión lineal.

Cumple los requisitos 3, 4, 5 y 6 del proyecto sobre el precio de la vivienda
en México. Toma como entrada data/datos_limpios.csv generado por proyecto.py
(que cubre los requisitos 1 y 2).

Ejecucion:
    python analisis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("¿Qué hace que una vivienda sea más cara?")
print("Análisis y predicción del precio de la vivienda en México")
print("Análisis: estadística descriptiva, correlaciones y regresión lineal")
print("Fuente: API SNIIV (SEDATU) - México")
print("=" * 70)

df = pd.read_csv('data/datos_limpios.csv', encoding='utf-8-sig')
print(f"\nRegistros cargados: {len(df):,}")

cat_cols = ['segmento', 'modalidad', 'genero', 'destino_credito', 'vivienda',
            'linea_credito', 'rango_salarial', 'esquema']
cat_cols = [c for c in cat_cols if c in df.columns and df[c].notna().any()]

# =====================================================================
# REQUISITO 3: ESTADÍSTICA DESCRIPTIVA
# =====================================================================
print("\n" + "=" * 70)
print("REQUISITO 3: ESTADÍSTICA DESCRIPTIVA")
print("=" * 70)

precio = df['monto_unitario']

print("\n-- Medidas de tendencia central y dispersión (monto unitario, MXN) --")
print(f"  N            : {precio.count():,}")
print(f"  Media        : ${precio.mean():,.0f}")
print(f"  Mediana      : ${precio.median():,.0f}")
print(f"  Desv. estándar: ${precio.std():,.0f}")
print(f"  Mínimo       : ${precio.min():,.0f}")
print(f"  Q1 (25%)     : ${precio.quantile(0.25):,.0f}")
print(f"  Q3 (75%)     : ${precio.quantile(0.75):,.0f}")
print(f"  Máximo       : ${precio.max():,.0f}")
print(f"  Asimetría    : {precio.skew():.3f}  (cola hacia {'la derecha' if precio.skew()>0 else 'la izquierda'})")
print(f"  Curtosis     : {precio.kurt():.3f}")

print("\n-- Monto unitario por segmento de vivienda --")
desc_segmento = df.groupby('segmento')['monto_unitario'].agg(
    n='count', media='mean', mediana='median', desv_std='std',
    minimo='min', maximo='max'
).round(0).sort_values('media', ascending=False)
print(desc_segmento.to_string())

print("\n-- Monto unitario por modalidad (top) --")
desc_modalidad = df.groupby('modalidad')['monto_unitario'].agg(
    n='count', media='mean', mediana='median'
).round(0).sort_values('media', ascending=False)
print(desc_modalidad.head(8).to_string())

desc_segmento.round(2).to_csv('data/estadistica_descriptiva.csv', encoding='utf-8-sig')

# Guardar resumen global
resumen = pd.DataFrame({
    'estadistico': ['N', 'Media', 'Mediana', 'Desv_std', 'Minimo', 'Q1', 'Q3', 'Maximo', 'Asimetria', 'Curtosis'],
    'valor': [precio.count(), precio.mean(), precio.median(), precio.std(),
              precio.min(), precio.quantile(0.25), precio.quantile(0.75),
              precio.max(), precio.skew(), precio.kurt()]
})
resumen.round(2).to_csv('data/estadistica_global.csv', index=False, encoding='utf-8-sig')
print("\n  -> data/estadistica_descriptiva.csv")
print("  -> data/estadistica_global.csv")

# =====================================================================
# REQUISITO 4: CORRELACIONES
# =====================================================================
print("\n" + "=" * 70)
print("REQUISITO 4: CORRELACIONES")
print("=" * 70)

# Codificación one-hot: permite medir la correlación de cada categoría con el precio
dummies = pd.get_dummies(df[cat_cols].fillna('No disponible').astype(str), prefix_sep='=')
dummies['monto_unitario'] = df['monto_unitario'].values

corr_con_precio = (dummies.corr()['monto_unitario']
                   .drop('monto_unitario')
                   .sort_values(key=np.abs, ascending=False))

print("\n-- Correlación de cada categoría con el monto unitario (top 12) --")
print(corr_con_precio.head(12).round(3).to_string())

corr_con_precio.round(4).to_csv('data/correlaciones_con_precio.csv', encoding='utf-8-sig')

# Matriz de correlación (variables codificadas)
corr_matrix = dummies.corr().round(3)
corr_matrix.to_csv('data/correlaciones.csv', encoding='utf-8-sig')

plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False, cbar_kws={'shrink': .7})
plt.title('Matriz de Correlaciones (variables codificadas)', fontweight='bold')
plt.tight_layout()
plt.savefig('data/viz_6_correlaciones.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n  Matriz de correlaciones: data/correlaciones.csv")
print("  Heatmap: data/viz_6_correlaciones.png")

# =====================================================================
# REQUISITO 5: REGRESIÓN LINEAL
# =====================================================================
print("\n" + "=" * 70)
print("REQUISITO 5: REGRESIÓN LINEAL")
print("=" * 70)

X = dummies.drop(columns='monto_unitario').values
y = dummies['monto_unitario'].values
feature_names = dummies.drop(columns='monto_unitario').columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = make_pipeline(StandardScaler(), LinearRegression())
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"\n  Ecuación: monto_unitario = b0 + sum(bi * xi)")
print(f"  R²   : {r2:.4f}  (explica el {r2*100:.1f}% de la variabilidad)")
print(f"  RMSE : ${rmse:,.0f}")
print(f"  MAE  : ${mae:,.0f}")

coefs = pd.DataFrame({
    'variable': feature_names,
    'coeficiente': modelo.named_steps['linearregression'].coef_
}).sort_values('coeficiente', key=np.abs, ascending=False)

print("\n  -- Coeficientes más influyentes (sobre variables estandarizadas) --")
print(coefs.head(12).round(0).to_string(index=False))
coefs.to_csv('data/regresion_lineal_coeficientes.csv', index=False, encoding='utf-8-sig')

# Grafico predicho vs real
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.3, s=12, color='steelblue')
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, 'r--', linewidth=2, label='Predicción perfecta')
plt.xlabel('Monto real (MXN)')
plt.ylabel('Monto predicho (MXN)')
plt.title(f'Regresión Lineal: Predicho vs Real (R²={r2:.3f})', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('data/viz_7_regresion_lineal.png', dpi=150, bbox_inches='tight')
plt.close()

# Grafico de residuos
residuos = y_test - y_pred
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuos, alpha=0.3, s=12, color='darkorange')
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Monto predicho (MXN)')
plt.ylabel('Residuo (real - predicho)')
plt.title('Análisis de Residuos', fontweight='bold')
plt.tight_layout()
plt.savefig('data/viz_8_residuos.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n  Graficos: data/viz_7_regresion_lineal.png, data/viz_8_residuos.png")
print("  Coeficientes: data/regresion_lineal_coeficientes.csv")

# =====================================================================
# REQUISITO 6: CONCLUSIONES
# =====================================================================
print("\n" + "=" * 70)
print("REQUISITO 6: CONCLUSIONES")
print("=" * 70)

top_seg = desc_segmento.index[0]
top_corr = corr_con_precio.index[0]
print(f"""
  1. La distribución del monto unitario presenta asimetría positiva
     (skew={precio.skew():.2f}): hay viviendas muy caras que elevan la media
     (${precio.mean():,.0f}) por encima de la mediana (${precio.median():,.0f}).

  2. La característica con mayor correlación con el precio es "{top_corr}"
     (r={corr_con_precio.iloc[0]:.2f}).

  3. El segmento de vivienda es el factor decisivo: "{top_seg}" registra el
     mayor monto promedio (${desc_segmento.loc[top_seg, 'media']:,.0f}),
     frente a la vivienda económica (~${desc_segmento['media'].min():,.0f}).

  4. La regresión lineal alcanza un R²={r2:.3f}, confirmando que las
     características categorizadas explican una parte importante del precio,
     aunque existe variabilidad no capturada (RMSE=${rmse:,.0f}).

  Conclusión: el precio de la vivienda en México se explica principalmente
  por el SEGMENTO (económica < popular < tradicional < media < residencial
  < residencial plus), seguido de la modalidad (nueva/existente) y el nivel
  salarial del acreditado.
""")

resumen_conclusiones = pd.DataFrame({
    'hallazgo': ['Asimetría', 'Correlación más fuerte', 'Segmento más caro',
                 'Segmento más barato', 'R² regresión lineal', 'RMSE regresión lineal'],
    'valor': [f"{precio.skew():.3f}", f"{top_corr} (r={corr_con_precio.iloc[0]:.2f})",
              f"{top_seg} (${desc_segmento.loc[top_seg, 'media']:,.0f})",
              f"{desc_segmento.index[-1]} (${desc_segmento['media'].min():,.0f})",
              f"{r2:.4f}", f"${rmse:,.0f}"]
})
resumen_conclusiones.to_csv('data/conclusiones.csv', index=False, encoding='utf-8-sig')

print("=" * 70)
print("ANÁLISIS COMPLETADO")
print("=" * 70)
