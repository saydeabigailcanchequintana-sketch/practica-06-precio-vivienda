import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

df = pd.read_csv('data/datos_limpios.csv', encoding='utf-8-sig')

print("Generando visualizaciones...")

# 1. Monto promedio por segmento
fig, ax = plt.subplots(figsize=(10, 6))
segmento_order = df.groupby('segmento')['monto_unitario'].mean().sort_values(ascending=False).index
sns.boxplot(data=df, x='segmento', y='monto_unitario', order=segmento_order, ax=ax)
ax.set_title('Distribución de Monto Unitario por Segmento de Vivienda', fontsize=14, fontweight='bold')
ax.set_xlabel('Segmento de Vivienda')
ax.set_ylabel('Monto Unitario (MXN)')
ax.ticklabel_format(style='plain', axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/viz_1_segmento.png', dpi=150, bbox_inches='tight')
plt.close()
print("  1. viz_1_segmento.png")

# 2. Monto por modalidad
fig, ax = plt.subplots(figsize=(12, 6))
modalidad_order = df.groupby('modalidad')['monto_unitario'].mean().sort_values(ascending=False).index
sns.boxplot(data=df, x='modalidad', y='monto_unitario', order=modalidad_order, ax=ax)
ax.set_title('Distribución de Monto Unitario por Modalidad', fontsize=14, fontweight='bold')
ax.set_xlabel('Modalidad de Financiamiento')
ax.set_ylabel('Monto Unitario (MXN)')
ax.ticklabel_format(style='plain', axis='y')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('data/viz_2_modalidad.png', dpi=150, bbox_inches='tight')
plt.close()
print("  2. viz_2_modalidad.png")

# 3. Feature importance
feat_imp = pd.read_csv('data/importancia_caracteristicas.csv', encoding='utf-8-sig')
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=feat_imp, x='Importancia', y='Característica', palette='viridis', ax=ax)
ax.set_title('Importancia de Características (Random Forest)', fontsize=14, fontweight='bold')
ax.set_xlabel('Importancia')
ax.set_ylabel('Característica')
plt.tight_layout()
plt.savefig('data/viz_3_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  3. viz_3_feature_importance.png")

# 4. Monto por género
fig, ax = plt.subplots(figsize=(8, 5))
sns.violinplot(data=df, x='genero', y='monto_unitario', ax=ax)
ax.set_title('Distribución de Monto Unitario por Género', fontsize=14, fontweight='bold')
ax.set_xlabel('Género')
ax.set_ylabel('Monto Unitario (MXN)')
ax.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
plt.savefig('data/viz_4_genero.png', dpi=150, bbox_inches='tight')
plt.close()
print("  4. viz_4_genero.png")

# 5. Histograma de montos
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df['monto_unitario'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(df['monto_unitario'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: ${df["monto_unitario"].mean():,.0f}')
ax.axvline(df['monto_unitario'].median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: ${df["monto_unitario"].median():,.0f}')
ax.set_title('Distribución de Montos Unitarios de Financiamiento', fontsize=14, fontweight='bold')
ax.set_xlabel('Monto Unitario (MXN)')
ax.set_ylabel('Frecuencia')
ax.legend()
ax.ticklabel_format(style='plain', axis='x')
plt.tight_layout()
plt.savefig('data/viz_5_histograma.png', dpi=150, bbox_inches='tight')
plt.close()
print("  5. viz_5_histograma.png")

print("\nTodas las visualizaciones guardadas en data/")
