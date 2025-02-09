import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
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
estados_seleccionados = st.multiselect("Selecciona uno o más Estados:", df["Estado del Sistema"].unique(), default=df["Estado del Sistema"].unique())
df_filtrado = df[df["Estado del Sistema"].isin(estados_seleccionados)]

# 📌 Cálculo de nuevas métricas
df_filtrado = df_filtrado.replace({"Temperatura (°C)": {0: np.nan}})
df_filtrado.dropna(subset=["Temperatura (°C)"], inplace=True)
df_filtrado["Eficiencia Térmica"] = df_filtrado["Uso CPU (%)"] / df_filtrado["Temperatura (°C)"]
df_filtrado["Eventos Críticos"] = df_filtrado["Estado del Sistema"].apply(lambda x: 1 if x == "Crítico" else 0)

# 🔹 Sección 1: KPIs
st.header("📌 Estado Actual")
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Uso Promedio CPU (%)", round(df_filtrado["Uso CPU (%)"].mean(), 2))
kpi2.metric("Temperatura Media (°C)", round(df_filtrado["Temperatura (°C)"].mean(), 2))
kpi3.metric("Eficiencia Térmica", round(df_filtrado["Eficiencia Térmica"].mean(), 2))

# 🔹 Sección 2: Gráficos de Monitoreo
st.header("📊 Análisis Visual del Sistema")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(px.scatter(df_filtrado, x="Uso CPU (%)", y="Temperatura (°C)", color="Estado del Sistema", title="📊 Relación entre Uso de CPU y Temperatura"), use_container_width=True)
    st.plotly_chart(px.box(df_filtrado, y=["Uso CPU (%)", "Temperatura (°C)"], title="📊 Distribución de Outliers"), use_container_width=True)

with col2:
    st.plotly_chart(px.pie(df_filtrado, values="Eventos Críticos", names="Estado del Sistema", title="📊 Distribución de Estados"), use_container_width=True)
    st.plotly_chart(px.bar(df_filtrado, x="Estado del Sistema", y=["Uso CPU (%)", "Memoria Utilizada (%)", "Carga de Red (MB/s)"], barmode="group", title="📊 Uso de Recursos"), use_container_width=True)

# 🔹 Sección 3: Modelado Predictivo
st.header("📈 Predicción de Estados del Sistema")

le = LabelEncoder()
df_filtrado["Estado Codificado"] = le.fit_transform(df_filtrado["Estado del Sistema"])
features = ["Uso CPU (%)", "Memoria Utilizada (%)", "Carga de Red (MB/s)"]
X = df_filtrado[features]
y = df_filtrado["Estado Codificado"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": features, "Importancia": importances}).sort_values(by="Importancia", ascending=False)

st.plotly_chart(px.bar(feature_importance_df, x="Feature", y="Importancia", title="📊 Importancia de Variables en la Predicción"), use_container_width=True)
st.write(f"Precisión del modelo: {model.score(X, y):.2f}")

# 🔹 Sección 4: Alertas del Sistema
st.header("⚠️ Alertas del Sistema")
umbral_cpu = st.slider("Selecciona umbral de CPU para alerta:", 50, 100, 85)

alta_carga = df_filtrado[df_filtrado["Uso CPU (%)"] > umbral_cpu]
if not alta_carga.empty:
    st.warning("⚠️ Se han detectado valores altos de uso de CPU.")
    st.dataframe(alta_carga[["Fecha", "Uso CPU (%)", "Estado del Sistema"]])
else:
    st.success("✅ No se detectaron anomalías en el uso de CPU.")

# 🔹 **Nueva Sección Adicional**: Análisis Avanzado
st.header("📊 Análisis Avanzado del Sistema")

col3, col4 = st.columns(2)
with col3:
    st.subheader("📈 Evolución de la Temperatura")
    st.plotly_chart(px.line(df_filtrado, x="Fecha", y="Temperatura (°C)", title="📈 Variación de Temperatura a lo Largo del Tiempo"), use_container_width=True)
    
    st.subheader("📉 Uso de CPU vs Carga de Red")
    st.plotly_chart(px.scatter(df_filtrado, x="Uso CPU (%)", y="Carga de Red (MB/s)", color="Estado del Sistema", title="📊 Relación entre CPU y Carga de Red"), use_container_width=True)

with col4:
    st.subheader("🔍 Análisis de Memoria Utilizada")
    st.plotly_chart(px.box(df_filtrado, y="Memoria Utilizada (%)", title="📊 Distribución de Memoria Utilizada"), use_container_width=True)

    st.subheader("📊 Comparación de Eficiencia Térmica")
    st.plotly_chart(px.violin(df_filtrado, y="Eficiencia Térmica", title="📊 Distribución de Eficiencia Térmica en Diferentes Estados"), use_container_width=True)

# Final del tablero
st.write("📌 Este tablero ha sido actualizado con nuevas visualizaciones para mejorar la toma de decisiones basada en datos.")
