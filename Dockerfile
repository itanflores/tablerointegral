# Usar una imagen base con Python
FROM python:3.9

# Configurar el directorio de trabajo
WORKDIR /app

# Copiar los archivos al contenedor
COPY . /app

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto 8080 (App Runner usa este puerto por defecto)
EXPOSE 8080

# Comando para ejecutar la app
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
