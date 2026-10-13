import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Precio de Vivienda México", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('data/datos_limpios.csv', encoding='utf-8-sig')
    return df

@st.cache_resource
def train_model(df):
    cat_cols = ['segmento', 'modalidad', 'genero', 'destino_credito', 'vivienda',
                'linea_credito', 'rango_salarial', 'esquema']
    cat_cols = [c for c in cat_cols if c in df.columns]
    encoders = {}
    df_model = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].fillna('No disponible').astype(str))
        encoders[col] = le
    X = df_model[cat_cols].values
    y = df_model['monto_unitario'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {'r2': r2_score(y_test, y_pred), 'mae': mean_absolute_error(y_test, y_pred)}
    return model, encoders, cat_cols, metrics

df = load_data()
model, encoders, cat_cols, metrics = train_model(df)

st.title("¿Qué características explican que una vivienda sea más cara?")
st.caption("Fuente: API SNIIV (SEDATU) · Datos 2024–2025")

col1, col2, col3 = st.columns(3)
col1.metric("Registros", f"{len(df):,}")
col2.metric("R² (Random Forest)", f"{metrics['r2']:.3f}")
col3.metric("MAE", f"${metrics['mae']:,.0f}")

st.sidebar.header("Filtros")
sel_segmento = st.sidebar.multiselect("Segmento", sorted(df['segmento'].dropna().unique()), default=None)
sel_modalidad = st.sidebar.multiselect("Modalidad", sorted(df['modalidad'].dropna().unique()), default=None)
sel_genero = st.sidebar.multiselect("Género", sorted(df['genero'].dropna().unique()), default=None)

dff = df.copy()
if sel_segmento:
    dff = dff[dff['segmento'].isin(sel_segmento)]
if sel_modalidad:
    dff = dff[dff['modalidad'].isin(sel_modalidad)]
if sel_genero:
    dff = dff[dff['genero'].isin(sel_genero)]

tab1, tab2, tab3, tab4 = st.tabs(["Análisis", "Importancia", "Predictor", "Datos"])

with tab1:
    st.subheader("Monto unitario por segmento")
    fig = px.box(dff, x='segmento', y='monto_unitario', color='segmento')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monto unitario por modalidad")
    fig2 = px.box(dff, x='modalidad', y='monto_unitario', color='modalidad')
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Importancia de características")
    importances = model.feature_importances_
    fi = pd.DataFrame({'Característica': cat_cols, 'Importancia': importances}).sort_values('Importancia', ascending=True)
    fig3 = px.bar(fi, x='Importancia', y='Característica', orientation='h')
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.subheader("Predictor de monto de financiamiento")
    inputs = {}
    for col in cat_cols:
        opts = sorted(str(x) for x in encoders[col].classes_.tolist())
        inputs[col] = st.selectbox(col, opts)
    if st.button("Predecir"):
        row = [encoders[c].transform([str(inputs[c])])[0] for c in cat_cols]
        pred = model.predict([row])[0]
        st.success(f"Monto estimado: ${pred:,.0f} MXN")

with tab4:
    st.subheader("Datos filtrados")
    st.dataframe(dff.head(1000))
    st.download_button(
        "Descargar CSV",
        dff.to_csv(index=False).encode('utf-8-sig'),
        "datos_filtrados.csv",
        "text/csv"
    )
