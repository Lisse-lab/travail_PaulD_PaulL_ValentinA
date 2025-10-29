import dash
from dash import dcc, html, Input, Output

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import hydrophone_placement_scripts.optimisation.class_mod as cls_mod

import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

l_n_tetrahedras = os.listdir(os.path.join(os.path.dirname(__file__), "results"))
dic_models_versions = {}
dic_hist = {}
dic_maps = {}
dic_values = {}
for n_tetrahedras in l_n_tetrahedras:
    n = n_tetrahedras.split("_")[0]
    l_versions = os.listdir(os.path.join(os.path.dirname(__file__), "results", n_tetrahedras))
    dic_models_versions[n] = [version[1:] for version in l_versions]
    for version in l_versions:
        vers = version[1:]
        mod = cls_mod.load(int(n), vers)
        dic_hist[(n, vers)] = mod.create_hist_areas_according_error() 
        dic_maps[(n, vers)] = mod.create_error_map().get_root().render()
        dic_values[(n, vers)] = mod.best_value()


# Initialisation de l'application Dash
app = dash.Dash(__name__)


app.layout = html.Div([
    html.H1("Models Comparison", style={"textAlign": "center"}),
    html.Div([
        # Première colonne
        html.Div([
            # Encadré pour la sélection du modèle et de la version
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        options=[{"label" : f"Model with {n} tetrahedras", "value" : n} for n in dic_models_versions.keys()],
                        value=list(dic_models_versions.keys())[0],
                        id="model-dropdown-1",
                        style={"width": "45%"}
                    ),
                    dcc.Dropdown(id="version-dropdown-1", style={"width": "45%"})
                ], style={"display": "flex", "justifyContent": "space-between"}),
                html.Div(id="value-display-1", style={"marginTop": 20})
            ], style={"border": "1px solid #ccc", "padding": 20, "marginBottom": 20}),

            # Graphique (histogramme)
            dcc.Graph(id="histogram-1"),

            # Carte (exemple avec un placeholder)
            html.Div(
                html.Iframe(id="map-1", width="100%", height="100%"), style={"width": "100%", "height": "90vh"})
        ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top", "padding": 10}),

        # Deuxième colonne
        html.Div([
            # Encadré pour la sélection du modèle et de la version
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        options=[{"label" : f"Model with {n} tetrahedras", "value" : n} for n in dic_models_versions.keys()],
                        value=list(dic_models_versions.keys())[0],
                        id="model-dropdown-2",
                        style={"width": "45%"}
                    ),
                    dcc.Dropdown(id="version-dropdown-2", style={"width": "45%"})
                ], style={"display": "flex", "justifyContent": "space-between"}),
                html.Div(id="value-display-2", style={"marginTop": 20})
            ], style={"border": "1px solid #ccc", "padding": 20, "marginBottom": 20}),

            # Graphique (histogramme)
            dcc.Graph(id="histogram-2"),

            # Carte (exemple avec un placeholder)
            html.Div(
                html.Iframe(id="map-2", width="100%", height="100%"), style={"width": "100%", "height": "90vh"})
        ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top", "padding": 10}),
    ]),
])


# Callback pour mettre à jour les versions disponibles (colonne 1)
@app.callback(
    Output("version-dropdown-1", "options"),
    Output("version-dropdown-1", "value"),
    Input("model-dropdown-1", "value")
)
def update_versions_1(model):
    options = [{"label": f"(Version {vers})", "value": vers} for vers in dic_models_versions[model]]
    return options, options[0]["value"]

# Callback pour afficher la valeur (colonne 1)
@app.callback(
    Output("value-display-1", "children"),
    Output("histogram-1", "figure"),
    Output("map-1", "srcDoc"),
    Input("model-dropdown-1", "value"),
    Input("version-dropdown-1", "value")
)
def update_column_1(model, version):
    return f"Valeur : {dic_values[(model, version)]}", dic_hist[(model, version)], dic_maps[(model, version)]


# Callback pour mettre à jour les versions disponibles (colonne 2)
@app.callback(
    Output("version-dropdown-2", "options"),
    Output("version-dropdown-2", "value"),
    Input("model-dropdown-2", "value")
)
def update_versions_2(model):
    options = [{"label": f"(Version {vers})", "value": vers} for vers in dic_models_versions[model]]
    return options, options[0]["value"]

# Callback pour afficher la valeur (colonne 2)
@app.callback(
    Output("value-display-2", "children"),
    Output("histogram-2", "figure"),
    Output("map-2", "srcDoc"),
    Input("model-dropdown-2", "value"),
    Input("version-dropdown-2", "value")
)
def update_column_2(model, version):
    return f"Valeur : {dic_values[(model, version)]}", dic_hist[(model, version)], dic_maps[(model, version)]



if __name__ == "__main__":
    app.run(debug=True)
