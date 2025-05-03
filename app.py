import json
from shapely.geometry import shape, Polygon, MultiPolygon
import streamlit as st
import datetime
from dotenv import load_dotenv
import os
from sentinelhub import (
    SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType
)
import numpy as np
import matplotlib.pyplot as plt

# Cargar .env
load_dotenv()

# Ahora SHConfig tomará las variables de entorno
config = SHConfig()
# (Opcional) Forzar lectura desde env si no se cargaron automáticamente
if not config.sh_client_id:
    config.sh_client_id = os.getenv("SH_CLIENT_ID")
if not config.sh_client_secret:
    config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")

# Mensaje en pantalla si faltan
if not config.sh_client_id or not config.sh_client_secret:
    st.error(
        "⚠️ No se han cargado las credenciales. "
        "Revisa tu archivo .env con SH_CLIENT_ID y SH_CLIENT_SECRET."
    )

def fetch_image(geom, date, producto="RGB", resolution=10):
    """
    Descarga una imagen Sentinel-2 para la geometría y fecha indicadas.
    - geom: shapely Polygon/MultiPolygon (WGS84)
    - date: datetime.date o 'YYYY-MM-DD'
    - producto: "RGB" o "NDVI"
    - resolution: metros por píxel
    Devuelve:
      - RGB: array (H, W, 3)
      - NDVI: array (H, W)
    """
    # 1. Bounding box de la geometría
    minx, miny, maxx, maxy = geom.bounds
    bbox = BBox(bbox=(minx, miny, maxx, maxy), crs=CRS.WGS84)

    # 2. Evalscript según producto
    if producto == "RGB":
        evalscript = """
        //VERSION=3
        function setup() {
          return { input: ["B04","B03","B02"], output: { bands: 3 } };
        }
        function evaluatePixel(sample) {
          return [sample.B04, sample.B03, sample.B02];
        }
        """
    else:  # NDVI
        evalscript = """
        //VERSION=3
        function setup() {
          return { input: ["B08","B04"], output: { bands: 1 } };
        }
        function evaluatePixel(sample) {
          let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
          return [ndvi];
        }
        """

    # 3. Cálculo de tamaño en píxeles aproximado
    width = int((maxx - minx) * 111320 / resolution)
    height = int((maxy - miny) * 111320 / resolution)

    # 4. Crear la petición a Sentinel Hub
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                DataCollection.SENTINEL2_L2A,
                time_interval=(str(date), str(date))
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=(width, height),
        config=config
    )

    # 5. Obtener datos y convertir a NumPy
    data = request.get_data()[0]
    array = np.squeeze(data)
    return array

def detect_change(img1, img2, umbral, producto="RGB"):
    """
    Dado img1 y img2 como arrays NumPy:
    - Si RGB: forma (H, W, 3)
    - Si NDVI: forma (H, W)
    umbral: valor de diferencia para detección.
    Devuelve:
      - mask: array booleano (H, W) con True donde hay cambio
      - pct_change: porcentaje de píxeles cambiados
    """
    # 1. Asegurar misma forma
    assert img1.shape == img2.shape, "Las dos imágenes deben tener la misma forma"
    
    if producto == "RGB":
        # Euclídea en cada píxel entre los 3 canales
        diff = np.linalg.norm(img1.astype(float) - img2.astype(float), axis=-1)
    else:
        # NDVI: simplemente resta y abs
        diff = np.abs(img1 - img2)
    
    # 2. Máscara de cambio
    mask = diff >= umbral
    
    # 3. Cálculo de porcentaje
    pct_change = 100.0 * np.count_nonzero(mask) / mask.size
    
    return mask, pct_change


st.sidebar.header("1. Zona de estudio")

# Variables iniciales
geojson_obj = None
geom = None  # ¡Asegúrate de inicializar geom aquí!

# Opción A: subir un archivo .geojson
uploaded_file = st.sidebar.file_uploader(
    "Sube un archivo GeoJSON",
    type=["geojson", "json"]
)

# Opción B: pegar GeoJSON
text_geojson = st.sidebar.text_area(
    "O pega aquí tu GeoJSON",
    height=150
)

st.sidebar.header("2. Selección de fechas")

# Fecha inicial
fecha_inicio = st.sidebar.date_input(
    "Fecha de inicio",
    value=datetime.date.today() - datetime.timedelta(days=7),
    max_value=datetime.date.today()
)

# Fecha final
fecha_fin = st.sidebar.date_input(
    "Fecha de fin",
    value=datetime.date.today(),
    min_value=fecha_inicio,
    max_value=datetime.date.today()
)

# Validación básica
if fecha_fin <= fecha_inicio:
    st.sidebar.error("La fecha de fin debe ser posterior a la fecha de inicio.")
else:
    st.session_state["fecha_inicio"] = fecha_inicio
    st.session_state["fecha_fin"] = fecha_fin
    st.sidebar.success(f"Periodo: {fecha_inicio} ➔ {fecha_fin}")

st.sidebar.header("3. Parámetros de análisis")

# Tipo de producto
producto = st.sidebar.selectbox(
    "Tipo de producto",
    options=["RGB", "NDVI"]
)

# Umbral de detección de cambio
if producto == "RGB":
    umbral = st.sidebar.slider(
        "Umbral de diferencia (RGB)",
        min_value=0, max_value=255, value=30
    )
else:  # NDVI
    umbral = st.sidebar.slider(
        "Umbral de diferencia (NDVI)",
        min_value=0.0, max_value=1.0, value=0.1, step=0.01
    )

# Guardamos en session_state
st.session_state["producto"] = producto
st.session_state["umbral"] = umbral
st.sidebar.success(f"Configurado: {producto} con umbral = {umbral}")

# Botón para descargar imágenes
if st.sidebar.button("4. Descargar imágenes"):
    geom = st.session_state.get("geometry")
    fi = st.session_state.get("fecha_inicio")
    ff = st.session_state.get("fecha_fin")
    prod = st.session_state.get("producto")

    if not geom or not fi or not ff:
        st.sidebar.error("❗ Debes configurar primero zona, fechas y parámetros.")
    else:
        with st.spinner("Descargando imágenes…"):
            img1 = fetch_image(geom, fi, producto=prod)
            img2 = fetch_image(geom, ff, producto=prod)
        st.session_state["img1"] = img1
        st.session_state["img2"] = img2
        st.sidebar.success("✅ Imágenes descargadas correctamente.")


# Botón para calcular el cambio
if st.sidebar.button("5. Calcular cambio"):
    img1 = st.session_state.get("img1")
    img2 = st.session_state.get("img2")
    umbral = st.session_state.get("umbral")
    producto = st.session_state.get("producto")

    if img1 is None or img2 is None:
        st.sidebar.error("❗ Primero descarga las imágenes (paso 4).")
    else:
        with st.spinner("Calculando diferencias…"):
            mask, pct = detect_change(img1, img2, umbral, producto=producto)
        
        # Mostrar porcentaje
        st.markdown(f"**Porcentaje de píxeles cambiados:** {pct:.2f}%")
        if pct == 0:
            st.success("No se detectan cambios significativos.")
        else:
            st.warning("¡Se detectaron cambios!")
        
        # Visualización “antes / después / cambio”
        cols = st.columns(3)
        with cols[0]:
            st.header("Antes")
            if producto == "RGB":
                st.image(img1, use_column_width=True)
            else:
                plt.imshow(img1, vmin=-1, vmax=1)
                plt.axis("off")
                st.pyplot(plt)
        with cols[1]:
            st.header("Después")
            if producto == "RGB":
                st.image(img2, use_column_width=True)
            else:
                plt.imshow(img2, vmin=-1, vmax=1)
                plt.axis("off")
                st.pyplot(plt)
        with cols[2]:
            st.header("Cambio")
            # Mostrar máscara en gris/negro
            plt.imshow(mask, cmap="gray")
            plt.axis("off")
            st.pyplot(plt)
        
        # Botón de descarga de resultados
        resultado = {
            "porcentaje_cambio": pct,
            "producto": producto,
            "umbral": umbral,
            "fecha_inicio": str(st.session_state["fecha_inicio"]),
            "fecha_fin": str(st.session_state["fecha_fin"])
        }
        st.download_button(
            "Descargar resumen (JSON)",
            data=json.dumps(resultado, ensure_ascii=False, indent=2),
            file_name="resultado_cambio.json",
            mime="application/json"
        )


if uploaded_file:
    try:
        geojson_obj = json.load(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Error al leer el archivo: {e}")
elif text_geojson:
    try:
        geojson_obj = json.loads(text_geojson)
    except Exception as e:
        st.sidebar.error(f"Error al parsear el texto: {e}")

if geojson_obj:
    try:
        geom_candidate = shape(geojson_obj["features"][0]["geometry"])
        if not isinstance(geom_candidate, (Polygon, MultiPolygon)):
            raise ValueError("La geometría debe ser (Multi)Polygon")
        geom = geom_candidate
        st.sidebar.success("✅ GeoJSON válido.")
    except Exception as e:
        st.sidebar.error(f"GeoJSON inválido: {e}")
        geom = None

if geom:
    st.session_state["geometry"] = geom


