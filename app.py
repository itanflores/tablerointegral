import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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
# 🔹 Sección 1: Estado Actual
st.header("📌 Estado Actual")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Crítico", total_counts.loc[total_counts["Estado"] == "Crítico", "Cantidad"].values[0] if "Crítico" in total_counts["Estado"].values else 0)
kpi2.metric("Advertencia", total_counts.loc[total_counts["Estado"] == "Advertencia", "Cantidad"].values[0] if "Advertencia" in total_counts["Estado"].values else 0)
kpi3.metric("Normal", total_counts.loc[total_counts["Estado"] == "Normal", "Cantidad"].values[0] if "Normal" in total_counts["Estado"].values else 0)
kpi4.metric("Inactivo", total_counts.loc[total_counts["Estado"] == "Inactivo", "Cantidad"].values[0] if "Inactivo" in total_counts["Estado"].values else 0)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(px.pie(total_counts, values="Cantidad", names="Estado", title="📊 Distribución de Estados"), use_container_width=True)
    st.write("Este gráfico muestra la proporción de cada estado del sistema en el dataset. Útil para identificar tendencias y anomalías.")
    st.plotly_chart(px.bar(df_avg, x="Estado del Sistema", y=["Uso CPU (%)", "Memoria Utilizada (%)", "Carga de Red (MB/s)"], barmode="group", title="📊 Uso de Recursos"), use_container_width=True)
    st.write("Este gráfico compara el uso promedio de CPU, memoria y carga de red según el estado del sistema.")
with col2:
    st.plotly_chart(px.line(df_grouped, x="Fecha", y="Cantidad_Suavizada", color="Estado del Sistema", title="📈 Evolución en el Tiempo", markers=True), use_container_width=True)
    st.write("Este gráfico representa la evolución temporal de los estados del sistema, permitiendo visualizar patrones y tendencias a lo largo del tiempo.")
    
    # Gráfico de dispersión: Relación entre Uso de CPU y Temperatura
    st.plotly_chart(px.scatter(
        df_filtrado,
        x="Uso CPU (%)",
        y="Temperatura (°C)",
        color="Estado del Sistema",
        title="📊 Relación entre Uso de CPU y Temperatura",
        labels={"Uso CPU (%)": "Uso de CPU (%)", "Temperatura (°C)": "Temperatura (°C)"},
        hover_name="Estado del Sistema"
    ), use_container_width=True)
    st.write("Este gráfico muestra la relación entre el uso de CPU y la temperatura, permitiendo identificar patrones y anomalías.")
    
# 🔹 Sección 2: Sección de Pronósticos
st.header("📈 Sección de Pronósticos")

# 📌 Predicción de Estados del Sistema con Regresión Lineal
st.subheader("📈 Predicción de Estados del Sistema")
pred_horizonte = 12
predicciones = []

for estado in df_grouped["Estado del Sistema"].unique():
    df_estado = df_grouped[df_grouped["Estado del Sistema"] == estado].copy()
    df_estado = df_estado.dropna(subset=["Cantidad_Suavizada"])
    
    if len(df_estado) > 1:
        X = np.array(range(len(df_estado))).reshape(-1, 1)
        y = df_estado["Cantidad_Suavizada"].values
        model = LinearRegression()
        model.fit(X, y)
        
        future_dates = pd.date_range(start=df_estado["Fecha"].max(), periods=pred_horizonte, freq="M")
        X_future = np.array(range(len(df_estado), len(df_estado) + pred_horizonte)).reshape(-1, 1)
        y_pred = model.predict(X_future)
        
        df_pred = pd.DataFrame({"Fecha": future_dates, "Estado del Sistema": estado, "Cantidad_Suavizada": y_pred})
        predicciones.append(df_pred)

df_pred_final = pd.concat([df_grouped] + predicciones, ignore_index=True)
st.plotly_chart(px.line(df_pred_final, x="Fecha", y="Cantidad_Suavizada", color="Estado del Sistema", title="📈 Predicción de Estados del Sistema", markers=True), use_container_width=True)
st.write("Este gráfico presenta la predicción de la cantidad de eventos por estado del sistema en los próximos meses.")

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
    st.write("Este gráfico predice la temperatura crítica en función del uso de CPU y la carga de red.")

# 🔹 Sección 3: Análisis de Outliers y Eficiencia Térmica
st.header("📊 Análisis de Outliers y Eficiencia Térmica")

# Calcular métricas
uso_promedio_cpu = df_filtrado["Uso CPU (%)"].mean()
temperatura_media = df_filtrado["Temperatura (°C)"].mean()
eficiencia_termica = uso_promedio_cpu / temperatura_media if temperatura_media != 0 else 0

# Mostrar métricas en columnas
col1, col2, col3 = st.columns(3)
col1.metric("Uso Promedio de CPU (%)", f"{uso_promedio_cpu:.2f}")
col2.metric("Temperatura Media (°C)", f"{temperatura_media:.2f}")
col3.metric("Eficiencia Térmica", f"{eficiencia_termica:.2f}")

# Crear el Boxplot para Uso de CPU y Temperatura
st.subheader("📊 Distribución de Outliers (Boxplot)")
fig = px.box(df_filtrado, y=["Uso CPU (%)", "Temperatura (°C)"], title="Distribución de Uso de CPU y Temperatura")
st.plotly_chart(fig, use_container_width=True)
st.write("Este gráfico muestra la distribución de los valores de Uso de CPU y Temperatura, permitiendo identificar outliers y tendencias centrales.")

# Explicación de las métricas
st.write("""
- **Uso Promedio de CPU (%)**: Promedio del uso de CPU en el dataset filtrado.
- **Temperatura Media (°C)**: Promedio de la temperatura en el dataset filtrado.
- **Eficiencia Térmica**: Relación entre el uso de CPU y la temperatura. Un valor más alto indica mayor eficiencia.
""")

