import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os

# 🛠️ Configurar la página
st.set_page_config(page_title="Tablero de Monitoreo", page_icon="📊", layout="wide")

# 📢 Título del tablero
st.title("📊 Tablero de Monitoreo del Sistema")

# 💞 Cargar Dataset
DATASET_URL = "dataset_procesado.csv"
if not os.path.exists(DATASET_URL):
    st.error("❌ Error: El dataset no se encuentra en la ruta especificada.")
    st.stop()

df = pd.read_csv(DATASET_URL)
df.columns = df.columns.str.strip()
df['Fecha'] = pd.to_datetime(df['Fecha'])

# 📌 Filtros
estados_seleccionados = st.multiselect(
    "Selecciona uno o más Estados:", 
    df["Estado del Sistema"].unique(), 
    default=df["Estado del Sistema"].unique()
)
df_filtrado = df[df["Estado del Sistema"].isin(estados_seleccionados)]

# 🔹 Sección 1: Estado Actual
st.header("📌 Estado Actual")

# 📊 Cálculo de Indicadores Claves
uso_promedio_cpu = df_filtrado["Uso CPU (%)"].mean()
temperatura_media = df_filtrado["Temperatura (°C)"].mean()
eficiencia_termica = uso_promedio_cpu / temperatura_media if temperatura_media != 0 else 0

# 🔹 Mostrar Indicadores
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Uso Promedio CPU (%)", f"{uso_promedio_cpu:.2f}")
kpi2.metric("Temperatura Media (°C)", f"{temperatura_media:.2f}")
kpi3.metric("Eficiencia Térmica", f"{eficiencia_termica:.2f}")

# 📊 Gráfico Boxplot: Detección de Outliers
st.subheader("📊 Distribución de Outliers")

# Reorganizar datos para visualización en boxplot
df_melted = df_filtrado.melt(value_vars=["Uso CPU (%)", "Temperatura (°C)"], var_name="variable", value_name="value")

# Crear gráfico
fig_boxplot = px.box(df_melted, x="variable", y="value", title="Distribución de Outliers")

# Mostrar gráfico en Streamlit
st.plotly_chart(fig_boxplot, use_container_width=True)
