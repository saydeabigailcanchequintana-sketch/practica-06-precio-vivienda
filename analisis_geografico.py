import requests
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("ANÁLISIS GEOGRÁFICO: Monto de Financiamiento por Estado")
print("Fuente: API SNIIV (SEDATU)")
print("=" * 60)

# Claves de estados (INEGI)
estados = {
    '01': 'Aguascalientes', '02': 'Baja California', '03': 'Baja California Sur',
    '04': 'Campeche', '05': 'Coahuila', '06': 'Colima', '07': 'Chiapas',
    '08': 'Chihuahua', '09': 'Ciudad de México', '10': 'Durango',
    '11': 'Guanajuato', '12': 'Guerrero', '13': 'Hidalgo', '14': 'Jalisco',
    '15': 'México', '16': 'Michoacán', '17': 'Morelos', '18': 'Nayarit',
    '19': 'Nuevo León', '20': 'Oaxaca', '21': 'Puebla', '22': 'Querétaro',
    '23': 'Quintana Roo', '24': 'San Luis Potosí', '25': 'Sinaloa',
    '26': 'Sonora', '27': 'Tabasco', '28': 'Tamaulipas', '29': 'Tlaxcala',
    '30': 'Veracruz', '31': 'Yucatán', '32': 'Zacatecas'
}

base_url = "https://sniiv.sedatu.gob.mx/api/CuboAPI"

print("\nDescargando datos por estado...")

all_data = []

for clave, nombre in estados.items():
    try:
        url = f"{base_url}/GetFinanciamiento/2024,2025/{clave}/000/valor_vivienda,modalidad,genero"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                df_est = pd.DataFrame(data)
                df_est['estado_clave'] = clave
                df_est['estado_nombre'] = nombre
                all_data.append(df_est)
                print(f"  {nombre}: {len(data)} registros")
    except Exception as e:
        print(f"  {nombre}: ERROR - {e}")

if all_data:
    df_all = pd.concat(all_data, ignore_index=True)

    # Limpiar
    df_all.replace('No disponible', np.nan, inplace=True)
    df_all['monto'] = pd.to_numeric(df_all['monto'], errors='coerce')
    df_all['acciones'] = pd.to_numeric(df_all['acciones'], errors='coerce')
    df_all['monto_unitario'] = df_all['monto'] / df_all['acciones']
    df_all['monto_unitario'] = df_all['monto_unitario'].replace([np.inf, -np.inf], np.nan)

    df_clean = df_all.dropna(subset=['monto_unitario', 'estado_nombre'])
    df_clean = df_clean[df_clean['monto_unitario'] > 0]

    print(f"\nTotal registros válidos: {len(df_clean)}")

    # Análisis por estado
    print("\n" + "=" * 60)
    print("MONTO PROMEDIO DE FINANCIAMIENTO POR ESTADO")
    print("=" * 60)

    estado_stats = df_clean.groupby('estado_nombre')['monto_unitario'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std')
    ]).round(0).sort_values('mean', ascending=False)

    print(estado_stats)

    # Guardar resultados
    estado_stats.to_csv('data/analisis_estados.csv', encoding='utf-8-sig')
    df_clean.to_csv('data/datos_por_estado.csv', index=False, encoding='utf-8-sig')

    print("\nArchivos guardados:")
    print("  - data/analisis_estados.csv")
    print("  - data/datos_por_estado.csv")

    # Top 5 más caros y más baratos
    print("\n" + "=" * 60)
    print("TOP 5 ESTADOS CON MAYOR MONTO PROMEDIO")
    print("=" * 60)
    for i, (estado, row) in enumerate(estado_stats.head(5).iterrows(), 1):
        print(f"  {i}. {estado}: ${row['mean']:,.0f} (n={int(row['count'])})")

    print("\nTOP 5 ESTADOS CON MENOR MONTO PROMEDIO")
    print("=" * 60)
    for i, (estado, row) in enumerate(estado_stats.tail(5).iterrows(), 1):
        print(f"  {i}. {estado}: ${row['mean']:,.0f} (n={int(row['count'])})")
else:
    print("No se obtuvieron datos.")
