# app_dash.py

import warnings
warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL"
)

import os
import json
import datetime
import base64
from io import BytesIO

import numpy as np
from PIL import Image
from shapely.geometry import shape

from dotenv import load_dotenv
load_dotenv()

from sentinelhub import (
    SHConfig, BBox, CRS, SentinelHubRequest,
    DataCollection, MimeType
)

import dash
from dash import html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash_leaflet import EditControl

# --- Config Sentinel Hub from .env ---
config = SHConfig()
config.sh_client_id = os.getenv("SH_CLIENT_ID")
config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")
if not config.sh_client_id or not config.sh_client_secret:
    raise RuntimeError("Faltan credenciales de Sentinel Hub en el .env")

# --- Funciones de backend ---
def fetch_image(geom, date, producto="RGB", resolution=10):
    minx, miny, maxx, maxy = geom.bounds
    bbox = BBox(bbox=(minx, miny, maxx, maxy), crs=CRS.WGS84)

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

    width = int((maxx - minx) * 111320 / resolution)
    height = int((maxy - miny) * 111320 / resolution)

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(
            DataCollection.SENTINEL2_L2A,
            time_interval=(str(date), str(date))
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=(width, height),
        config=config
    )
    data = request.get_data()[0]
    return np.squeeze(data)


def detect_change(img1, img2, umbral, producto="RGB"):
    assert img1.shape == img2.shape, "Las imágenes no tienen la misma forma"
    if producto == "RGB":
        diff = np.linalg.norm(img1.astype(float) - img2.astype(float), axis=-1)
    else:
        diff = np.abs(img1 - img2)
    mask = diff >= umbral
    pct = 100.0 * np.count_nonzero(mask) / mask.size
    return mask, pct


def encode_image(arr, is_mask=False):
    """Convierte un array NumPy a URI base64 PNG para <img>."""
    if arr.ndim == 3:
        img = arr.astype(np.uint8)
        pil = Image.fromarray(img)
    else:
        if is_mask:
            arr2 = (arr.astype(np.uint8) * 255)
        else:
            arr2 = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
        pil = Image.fromarray(arr2)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{data}"


# --- App Setup ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
server = app.server

# Layout
app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col(width=4, children=[
            html.H4("1. Zona de estudio"),
            dl.Map(
                center=[40.4104693, -3.7874826],
                zoom=14,
                style={"width": "100%", "height": "400px"},
                children=[
                    dl.TileLayer(),
                    dl.FeatureGroup([
                        EditControl(
                            id="draw-control",
                            draw={
                                "rectangle": True,
                                "polyline": False,
                                "polygon": False,
                                "circle": False,
                                "circlemarker": False,
                                "marker": False
                            },
                            edit=False
                        )
                    ])
                ]
            ),

            html.Hr(),
            html.H4("2. Selección de fechas"),
            dcc.DatePickerRange(
                id="date-picker",
                start_date=(datetime.date.today() - datetime.timedelta(days=7)),
                end_date=datetime.date.today(),
                min_date_allowed=datetime.date(2025, 1, 1),
                max_date_allowed=datetime.date.today(),
                display_format="YYYY-MM-DD"
            ),

            html.Hr(),
            html.H4("3. Parámetros"),
            dcc.Dropdown(
                id="dropdown-producto",
                options=[
                    {"label": "RGB", "value": "RGB"},
                    {"label": "NDVI", "value": "NDVI"}
                ],
                value="RGB",
                clearable=False
            ),
            html.Div(id="threshold-container"),

            html.Hr(),
            dbc.Button("Detectar cambio", id="btn-detect", color="primary", className="w-100"),
        ]),

        dbc.Col(width=8, children=[
            html.Div(id="result-content", children=[
                html.P("Define zona, fechas y parámetros y pulsa 'Detectar cambio'.")
            ])
        ])
    ])
])


# ---- Callbacks ----

# 1) Slider dinámico según producto
@app.callback(
    Output("threshold-container", "children"),
    Input("dropdown-producto", "value")
)
def update_threshold_slider(producto):
    if producto == "RGB":
        return dcc.Slider(
            id="threshold-slider",
            min=0, max=255, step=1, value=30,
            marks={i: str(i) for i in range(0, 256, 51)},
            tooltip={"placement": "bottom"}
        )
    else:
        return dcc.Slider(
            id="threshold-slider",
            min=0, max=1, step=0.01, value=0.1,
            marks={round(i * 0.1, 1): str(round(i * 0.1, 1)) for i in range(0, 11)},
            tooltip={"placement": "bottom"}
        )

# 2) Detección de cambio
@app.callback(
    Output("result-content", "children"),
    Input("btn-detect", "n_clicks"),
    State("draw-control", "geojson"),
    State("date-picker", "start_date"),
    State("date-picker", "end_date"),
    State("dropdown-producto", "value"),
    State("threshold-slider", "value"),
)
def on_detect(n_clicks, drawn_geojson, start_date, end_date, producto, umbral):
    if not n_clicks:
        return html.P("Define zona, fechas y parámetros y pulsa 'Detectar cambio'.")
    if not drawn_geojson:
        return dbc.Alert("Por favor, dibuja un rectángulo en el mapa.", color="warning")

    # Convertir a Shapely
    try:
        geom = shape(drawn_geojson["geometry"])
    except Exception as e:
        return dbc.Alert(f"GeoJSON inválido: {e}", color="danger")

    # Descargar imágenes
    try:
        img1 = fetch_image(geom, start_date, producto)
        img2 = fetch_image(geom, end_date, producto)
    except Exception as e:
        return dbc.Alert(f"Error descargando imágenes: {e}", color="danger")

    # Detectar cambio
    mask, pct = detect_change(img1, img2, umbral, producto)

    # Codificar para <img>
    before_src = encode_image(img1)
    after_src  = encode_image(img2)
    mask_src   = encode_image(mask, is_mask=True)

    # Construir resultado
    return [
        html.H4(f"Cambio detectado: {pct:.2f}%"),
        dbc.Row([
            dbc.Col(html.Div([html.H5("Antes"),  html.Img(src=before_src, style={"width":"100%"})]), width=4),
            dbc.Col(html.Div([html.H5("Después"),html.Img(src=after_src,  style={"width":"100%"})]), width=4),
            dbc.Col(html.Div([html.H5("Cambio"), html.Img(src=mask_src,   style={"width":"100%"})]), width=4),
        ], className="mt-4"),
        html.Br(),
        html.A("Realizar otro análisis", href="/", className="btn btn-secondary")
    ]

if __name__ == "__main__":
    app.run(debug=True)

