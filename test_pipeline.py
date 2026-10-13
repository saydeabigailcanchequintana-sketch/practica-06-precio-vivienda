"""
Pruebas de validación del pipeline del proyecto de precio de vivienda.

Ejecucion:
    python test_pipeline.py        # sin dependencias externas
    pytest test_pipeline.py        # si pytest esta instalado
"""

import os
import pandas as pd
import numpy as np

DATA = 'data'

REQUIRED_FILES = [
    'datos_limpios.csv',
    'resultados_modelos.csv',
    'importancia_caracteristicas.csv',
    'cv_resultados.csv',
    'analisis_estados.csv',
    'estadistica_descriptiva.csv',
    'correlaciones_con_precio.csv',
    'regresion_lineal_coeficientes.csv',
    'conclusiones.csv',
]


def test_archivos_generados():
    for f in REQUIRED_FILES:
        ruta = os.path.join(DATA, f)
        assert os.path.exists(ruta), f"Falta {ruta}"


def test_datos_limpios_esquema():
    df = pd.read_csv(os.path.join(DATA, 'datos_limpios.csv'), encoding='utf-8-sig')
    for col in ['segmento', 'modalidad', 'genero', 'monto_unitario']:
        assert col in df.columns, f"Falta columna {col}"
    assert len(df) > 1000, "Muy pocos registros"


def test_montos_positivos():
    df = pd.read_csv(os.path.join(DATA, 'datos_limpios.csv'), encoding='utf-8-sig')
    assert (df['monto_unitario'] > 0).all(), "Hay montos no positivos"


def test_modelos_resultados():
    df = pd.read_csv(os.path.join(DATA, 'resultados_modelos.csv'), encoding='utf-8-sig')
    assert len(df) == 3, "Se esperaban 3 modelos"
    assert df['R2'].max() > 0.6, "El mejor modelo deberia superar R2=0.6"
    assert df['R2'].min() >= 0, "R2 negativo inesperado"


def test_validacion_cruzada_estable():
    df = pd.read_csv(os.path.join(DATA, 'cv_resultados.csv'), encoding='utf-8-sig')
    assert len(df) == 3, "CV debe reportar 3 modelos"
    best = df.sort_values('R2_media', ascending=False).iloc[0]
    assert best['R2_media'] > 0.6, "R2 de CV bajo"
    assert best['R2_std'] < 0.1, "Demasiada varianza en CV"


def test_importancia_suma_uno():
    df = pd.read_csv(os.path.join(DATA, 'importancia_caracteristicas.csv'), encoding='utf-8-sig')
    total = df['Importancia'].sum()
    assert abs(total - 1.0) < 1e-3, f"Las importancias no suman 1 (sum={total})"
    top = df.sort_values('Importancia', ascending=False).iloc[0]['Característica']
    assert top == 'segmento', f"Se esperaba 'segmento' como principal, no {top}"


def test_analisis_estados_completo():
    df = pd.read_csv(os.path.join(DATA, 'analisis_estados.csv'), encoding='utf-8-sig')
    assert len(df) == 32, f"Se esperaban 32 entidades, hay {len(df)}"
    assert df['mean'].gt(0).all(), "Montos promedio no positivos"


def test_estadistica_descriptiva_coherente():
    df = pd.read_csv(os.path.join(DATA, 'datos_limpios.csv'), encoding='utf-8-sig')
    desc = pd.read_csv(os.path.join(DATA, 'estadistica_descriptiva.csv'), index_col=0, encoding='utf-8-sig')
    assert desc['media'].gt(0).all(), "Medias por segmento no positivas"
    assert abs(desc['n'].sum() - len(df)) < 1, "El conteo por segmento no cuadra"
    assert desc['media'].idxmax() == 'Residencial plus', "El segmento más caro debe ser Residencial plus"


def test_correlaciones_en_rango():
    df = pd.read_csv(os.path.join(DATA, 'correlaciones_con_precio.csv'), encoding='utf-8-sig')
    valores = df.iloc[:, 1]
    assert valores.abs().max() <= 1.0, "Correlación fuera de [-1, 1]"
    assert valores.abs().max() > 0.5, "La correlación más alta debería superar 0.5"


def test_regresion_lineal_buena():
    df = pd.read_csv(os.path.join(DATA, 'regresion_lineal_coeficientes.csv'), encoding='utf-8-sig')
    assert {'variable', 'coeficiente'} <= set(df.columns), "Faltan columnas de coeficientes"
    assert len(df) > 5, "Pocos coeficientes"
    assert df['coeficiente'].abs().max() > 0, "Coeficientes nulos"


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    ok, fail = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            ok += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            fail += 1
    print(f"\n{ok} pasaron, {fail} fallaron")
    raise SystemExit(1 if fail else 0)
