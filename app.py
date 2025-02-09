import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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

# 💫 Generar Datos de Estado
total_counts = df_filtrado["Estado del Sistema"].value_counts().reset_index()
total_counts.columns = ["Estado", "Cantidad"]

df_grouped = df_filtrado.groupby(["Fecha", "Estado del Sistema"]).size().reset_index(name="Cantidad")
df_grouped["Cantidad_Suavizada"] = df_grouped.groupby("Estado del Sistema")["Cantidad"].transform(lambda x: x.rolling(7, min_periods=1).mean())

df_avg = df_filtrado.groupby("Estado del Sistema")[["Uso CPU (%)", "Memoria Utilizada (%)", "Carga de Red (MB/s)"]].mean().reset_index()

# 🔹 Sección 1: Estado Actual
st.header("📌 Estado Actual")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Crítico", total_counts.loc[total_counts["Estado"] == "Crítico", "Cantidad"].values[0] if "Crítico" in total_counts["Estado"].values else 0)
kpi2.metric("Advertencia", total_counts.loc[total_counts["Estado"] == "Advertencia", "Cantidad"].values[0] if "Advertencia" in total_counts["Estado"].values else 0)
kpi3.metric("Normal", total_counts.loc[total_counts["Estado"] == "Normal", "Cantidad"].values[0] if "Normal" in total_counts["Estado"].values else 0)
kpi4.metric("Inactivo", total_counts.loc[total_counts["Estado"] == "Inactivo", "Cantidad"].values[0] if "Inactivo" in total_counts["Estado"].values else 0)

# 🔹 Sección 2: Sección de Pronósticos
st.header("📈 Sección de Pronósticos")

# 📌 Predicción de Temperatura Crítica
st.subheader("🌡️ Predicción de Temperatura Crítica")
if "Uso CPU (%)" in df_filtrado.columns and "Temperatura (°C)" in df_filtrado.columns:
    df_temp = df_filtrado[["Fecha", "Uso CPU (%)", "Carga de Red (MB/s)", "Temperatura (°C)"]].dropna()
    X = df_temp[["Uso CPU (%)", "Carga de Red (MB/s)"]]
    y = df_temp["Temperatura (°C)"]
    model_temp = RandomForestRegressor(n_estimators=100, random_state=42)
    model_temp.fit(X, y)
    
    future_data = pd.DataFrame({"Uso CPU (%)": np.linspace(X["Uso CPU (%)"].min(), X["Uso CPU (%)"].max(), num=12), "Carga de Red (MB/s)": np.linspace(X["Carga de Red (MB/s)"].min(), X["Carga de Red (MB/s)"].max(), num=12)})
    future_temp_pred = model_temp.predict(future_data)
    df_future_temp = pd.DataFrame({"Fecha": pd.date_range(start=df_temp["Fecha"].max(), periods=12, freq="M"), "Temperatura Predicha (°C)": future_temp_pred})
    st.plotly_chart(px.line(df_future_temp, x="Fecha", y="Temperatura Predicha (°C)", title="📈 Predicción de Temperatura Crítica", markers=True), use_container_width=True)

# 🔹 Sección 3: Nuevos Indicadores
st.header("📊 Indicadores Adicionales")

# 📌 Eficiencia Térmica y Eventos Críticos
df_filtrado = df_filtrado.replace({"Temperatura (°C)": {0: np.nan}}).dropna(subset=["Temperatura (°C)"])
df_filtrado["Eficiencia Térmica"] = df_filtrado["Uso CPU (%)"] / df_filtrado["Temperatura (°C)"]
df_filtrado["Eventos Críticos"] = df_filtrado["Estado del Sistema"].apply(lambda x: 1 if x == "Crítico" else 0)

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Uso Promedio CPU (%)", round(df_filtrado["Uso CPU (%)"].mean(), 2))
kpi2.metric("Temperatura Media (°C)", round(df_filtrado["Temperatura (°C)"].mean(), 2))
kpi3.metric("Eficiencia Térmica", round(df_filtrado["Eficiencia Térmica"].mean(), 2))

# 📌 Visualizaciones adicionales
st.plotly_chart(px.scatter(df_filtrado, x="Uso CPU (%)", y="Temperatura (°C)", color="Estado del Sistema", title="📊 Relación entre Uso de CPU y Temperatura"), use_container_width=True)
st.plotly_chart(px.box(df_filtrado, y=["Uso CPU (%)", "Temperatura (°C)"], title="📊 Distribución de Outliers"), use_container_width=True)

# 📌 Predicción de Estados del Sistema con Random Forest
le = LabelEncoder()
df_filtrado["Estado Codificado"] = le.fit_transform(df_filtrado["Estado del Sistema"])
features = ["Uso CPU (%)", "Memoria Utilizada (%)", "Carga de Red (MB/s)"]
X = df_filtrado[features]
y = df_filtrado["Estado Codificado"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
feature_importance_df = pd.DataFrame({"Feature": features, "Importancia": model.feature_importances_}).sort_values(by="Importancia", ascending=False)
st.plotly_chart(px.bar(feature_importance_df, x="Feature", y="Importancia", title="📊 Importancia de Variables en la Predicción"), use_container_width=True)

# 📌 Alertas Dinámicas
umbral_cpu = st.slider("Selecciona umbral de CPU para alerta:", 50, 100, 85)
alta_carga = df_filtrado[df_filtrado["Uso CPU (%)"] > umbral_cpu]
st.dataframe(alta_carga[["Fecha", "Uso CPU (%)", "Estado del Sistema"]]) if not alta_carga.empty else st.success("✅ No se detectaron anomalías en el uso de CPU.")
