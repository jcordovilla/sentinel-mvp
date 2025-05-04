# app-dash.py

import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import os
import json
from datetime import date, datetime, timedelta
import base64
from io import BytesIO

import numpy as np
from PIL import Image
from shapely.geometry import shape

from dotenv import load_dotenv
load_dotenv()

from sentinelhub import (
    SHConfig, BBox, CRS, SentinelHubRequest,
    DataCollection, MimeType, MosaickingOrder
)

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash_leaflet import EditControl

# --- Configuración de Sentinel Hub ---
config = SHConfig()
config.sh_client_id     = os.getenv("SH_CLIENT_ID")
config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")
if not config.sh_client_id or not config.sh_client_secret:
    raise RuntimeError("Faltan credenciales en .env")

# --- Funciones de backend ---
def fetch_image(geom, date_input, producto="RGB", resolution=10, window=3):
    if isinstance(date_input, str):
        date_input = datetime.fromisoformat(date_input).date()

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
    else:
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

    width  = int((maxx - minx) * 111320 / resolution)
    height = int((maxy - miny) * 111320 / resolution)

    start = (date_input - timedelta(days=window)).isoformat()
    end   = (date_input + timedelta(days=window)).isoformat()

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(
            DataCollection.SENTINEL2_L2A,
            time_interval=(start, end),
            mosaicking_order=MosaickingOrder.LEAST_CC
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
    pct  = 100.0 * np.count_nonzero(mask) / mask.size
    return mask, pct

def encode_image(arr, is_mask=False):
    """
    Convierte array NumPy a base64 PNG.
    - Si arr.ndim==3 (RGB): aplica estiramiento 2–98% por canal.
    - Si is_mask: bool → 0/255.
    - Si NDVI: escala lineal de mínimo a máximo.
    """
    if arr.ndim == 3:
        # Estiramiento por percentiles 2–98%
        channels = []
        for i in range(3):
            band = arr[:, :, i].astype(float)
            p2  = np.nanpercentile(band, 2)
            p98 = np.nanpercentile(band, 98)
            stretch = np.clip((band - p2) / (p98 - p2), 0, 1)
            channels.append(stretch)
        img_arr = np.stack(channels, axis=-1)
        img_uint8 = (img_arr * 255).astype(np.uint8)
        pil = Image.fromarray(img_uint8)
    else:
        if is_mask:
            img_uint8 = (arr.astype(np.uint8) * 255)
            pil = Image.fromarray(img_uint8)
        else:
            # NDVI: escala lineal
            norm = (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))
            img_uint8 = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
            pil = Image.fromarray(img_uint8)

    buf = BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

# --- App Setup ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col(width=4, children=[
            html.H4("1. Zona de estudio"),
            dl.Map(
                center=[40.418891, -3.7911866], zoom=16,
                style={"width":"100%","height":"400px"},
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
                start_date=(date.today() - timedelta(days=7)),
                end_date=date.today(),
                min_date_allowed=date(2025,1,1),
                max_date_allowed=date.today(),
                display_format="YYYY-MM-DD"
            ),
            html.Hr(),
            html.H4("3. Parámetros"),
            dcc.Dropdown(
                id="dropdown-producto",
                options=[
                    {"label":"RGB","value":"RGB"},
                    {"label":"NDVI","value":"NDVI"}
                ],
                value="RGB", clearable=False
            ),
            html.Div(id="threshold-container", children=[
                html.Div(
                    dcc.Slider(
                        id="threshold-slider-rgb",
                        min=0, max=255, step=1, value=30,
                        marks={i: str(i) for i in range(0,256,51)},
                        tooltip={"placement":"bottom"}, updatemode="drag"
                    ),
                    id="threshold-slider-rgb-container",
                    style={"display":"block"}
                ),
                html.Div(
                    dcc.Slider(
                        id="threshold-slider-ndvi",
                        min=0.0, max=1.0, step=0.01, value=0.1,
                        marks={round(i*0.1,1):str(round(i*0.1,1)) for i in range(0,11)},
                        tooltip={"placement":"bottom"}, updatemode="drag"
                    ),
                    id="threshold-slider-ndvi-container",
                    style={"display":"none"}
                )
            ]),
            html.Hr(),
            dbc.Button("Detectar cambio", id="btn-detect", color="primary", className="w-100")
        ]),
        dbc.Col(width=8, children=[
            html.Div(id="result-content", children=[
                html.P("Define zona, fechas y parámetros y pulsa 'Detectar cambio'.")
            ])
        ])
    ])
])

# --- Callbacks ---
@app.callback(
    Output("threshold-slider-rgb-container", "style"),
    Output("threshold-slider-ndvi-container", "style"),
    Input("dropdown-producto", "value")
)
def toggle_sliders(prod):
    if prod == "RGB":
        return {"display":"block"}, {"display":"none"}
    return {"display":"none"}, {"display":"block"}

@app.callback(
    Output("result-content", "children"),
    Input("btn-detect",            "n_clicks"),
    State("draw-control",          "geojson"),
    State("date-picker",           "start_date"),
    State("date-picker",           "end_date"),
    State("dropdown-producto",     "value"),
    State("threshold-slider-rgb",  "value"),
    State("threshold-slider-ndvi", "value"),
)
def on_detect(n_clicks, drawn_geojson, start_date, end_date,
              producto, umbral_rgb, umbral_ndvi):
    if not n_clicks:
        return html.P("Define zona, fechas y parámetros y pulsa 'Detectar cambio'.")
    if not drawn_geojson:
        return dbc.Alert("Dibuja primero un rectángulo en el mapa.", color="warning")

    if isinstance(drawn_geojson, str):
        try:
            drawn_geojson = json.loads(drawn_geojson)
        except:
            return dbc.Alert("GeoJSON mal formado.", color="danger")

    geom_json = (
        drawn_geojson.get("geometry")
        or (drawn_geojson.get("features") or [{}])[0].get("geometry")
    )
    if not geom_json:
        return dbc.Alert("No se encontró la geometría.", color="danger")

    try:
        geom = shape(geom_json)
    except Exception as e:
        return dbc.Alert(f"GeoJSON inválido: {e}", color="danger")

    umbral = umbral_rgb if producto == "RGB" else umbral_ndvi

    try:
        img1 = fetch_image(geom, start_date, producto)
        img2 = fetch_image(geom, end_date,   producto)
    except Exception as e:
        return dbc.Alert(f"Error descargando imágenes: {e}", color="danger")

    mask, pct = detect_change(img1, img2, umbral, producto)

    before_src = encode_image(img1)
    after_src  = encode_image(img2)
    mask_src   = encode_image(mask, is_mask=True)

    return [
        html.H4(f"Cambio detectado: {pct:.2f}%"),
        dbc.Row([
            dbc.Col(html.Div([html.H5("Antes"),   html.Img(src=before_src, style={"width":"100%"})]), width=4),
            dbc.Col(html.Div([html.H5("Después"), html.Img(src=after_src,  style={"width":"100%"})]), width=4),
            dbc.Col(html.Div([html.H5("Cambio"),  html.Img(src=mask_src,   style={"width":"100%"})]), width=4),
        ], className="mt-4"),
        html.Br(),
        html.A("Realizar otro análisis", href="/", className="btn btn-secondary")
    ]

if __name__ == "__main__":
    app.run(debug=True)
