import dash
from dash import dcc, html, Input, Output
import folium
from shapely.geometry import Polygon, mapping
import geopandas as gpd
import branca.colormap as cm
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import hydrophone_placement_scripts.optimisation.genetic_algo as ga

def create_weights_map(iteration, mod, plot_weights, df):
    map = folium.Map(location=[(mod.converter.lat_min + mod.converter.lat_max)/2, (mod.converter.lon_min + mod.converter.lon_max)/2], zoom_start=10)
    if plot_weights:
        colormap = cm.LinearColormap(colors=['blue', 'red'], vmin=df["density"].min(), vmax=df["density"].max(), caption="Density")
        colormap.add_to(map)
        for row in df.itertuples():
            couleur = colormap(row.density)
            folium.GeoJson(
                row.polygone,
                style_function=lambda _, couleur=couleur, alpha=0.7: {
                    'fillColor': couleur,
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': alpha,
                }
            ).add_to(map)
        
        gdf = gpd.read_file(mod.calculator.traversier_path)
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        folium.GeoJson(
                mapping(gdf_wgs84.loc[0, "geometry"]),
                style_function=lambda _, c="yellow": {"color": c, "weight": 6}
            ).add_to(map)
        folium.GeoJson(
                mapping(gdf_wgs84.loc[1, "geometry"]),
                style_function=lambda _, c="yellow": {"color": c, "weight": 6}
            ).add_to(map)
    
    for coords in mod.bayesian_process.set_of_npointsbayesian.set_of_npoints[iteration].coords():
        lat, lon = mod.converter.utm2lla(coords[0], coords[1])
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color='black',
            fill=True,
            fill_color='black',
            fill_opacity=1,
            ).add_to(map)
    map_html = map.get_root().render()
    return map_html

def create_app(mod):
    plotly_figure = mod.create_optimisation_history()
    app = dash.Dash(__name__)
    df = mod.calculator.df_areas.groupby(['x', 'y'], as_index=False).agg({'w': 'sum'})  #, 'error': 'mean'})
    df["polygone"] = df.apply(lambda row : Polygon(mod.converter.area2perim_lla((row.x, row.y))), axis=1)
    df["density"] = df.apply(lambda row : mod.calculator.calc_mu.h(mod.converter.area2utm((row.x,row.y))[0], mod.converter.area2utm((row.x,row.y))[1]), axis = 1)
    
    values = mod.bayesian_process.set_of_npointsbayesian.values
    expected_improvements = mod.bayesian_process.expected_improvements
    max_expected_improvement = max(expected_improvements)
    n_first_points = len(values) - len(expected_improvements)
    n_iter = len(values)
    best_iter = mod.bayesian_process.best_iteration()

    app.layout = html.Div([
        html.H1("Optimisation Visualisation", style={"textAlign": "center"}),
        html.Div([
            html.Div([
                html.Iframe(
                    id="map",
                    width="100%",
                    height="100%",
                    style={"border": "none"}
                ),
            ], style={
                "width": "50%",
                "display": "inline-block",
                "verticalAlign": "top",
                "height": "calc(100vh - 100px)",
                "overflow": "hidden",
                "padding": "0",
                "margin": "0"
            }),
            html.Div([
                html.Div([
                    html.Div([
                        html.H2("Iteration", style={
                            "fontSize": 36,
                            "display": "inline-block",
                            "margin": "0 10px 0 0"
                        }),
                        dcc.Input(
                            id="iteration-input",
                            type="number",
                            value=best_iter,
                            min=1,
                            max=n_iter,
                            step=1,
                            style={
                                "fontSize": 24,
                                "width": "80px",
                                "display": "inline-block",
                                "marginRight": "20px"
                            }
                        ),
                        html.Div([
                            dcc.Checklist(
                                id="plot-weights",
                                options=[{"label": "Plot Weights", "value": "plot"}],
                                value=[],
                                style={"fontSize": 18}
                            )
                        ], style={
                            "display": "inline-block",
                            "marginLeft": "auto",
                            "verticalAlign": "middle"
                        }),
                        html.Div([
                            dcc.RadioItems(
                                id="plot-type",
                                options=[
                                    {"label": "Global Optimisation", "value": "global"},
                                    {"label": "Runs Optimisation", "value": "runs"}
                                ],
                                value="global",
                                labelStyle={"display": "block", "fontSize": 18, "margin": "5px 0"}
                            )
                        ], style={"margin": "10px 0"}),
                    ], style={
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "space-between",
                        "width": "100%"
                    }),
                    html.Div(id="value-display", style={"fontSize": 18, "margin": "10px"}),
                    html.Div(id="expected-improvement-display", style={"fontSize": 18, "margin": "10px"}),

                ], style={
                    "border": "1px solid #ccc",
                    "borderRadius": "8px",
                    "padding": "20px",
                    "marginBottom": "20px",
                    "backgroundColor": "#fafafa"
                }),
                html.Div(id="plot-container", style={
                    "border": "1px solid #ccc",
                    "borderRadius": "8px",
                    "padding": "10px",
                    "backgroundColor": "#fafafa",
                    "height": "calc(100vh - 300px)",
                    "overflowY": "auto"
                }),
            ], style={
                "width": "48%",
                "display": "inline-block",
                "verticalAlign": "top",
                "padding": "0 10px"
            }),
        ]),
    ])

    # Callbacks pour mettre à jour les valeurs affichées
    @app.callback(
        [Output("value-display", "children"),
        Output("expected-improvement-display", "children")],
        [Input("iteration-input", "value")]
    )
    def update_values(iteration):
        value = values[iteration-1]
        if iteration <= n_first_points:
            return [f"Value : {value:.2f}", ""]
        else:
            expected_improvement = expected_improvements[iteration-n_first_points-1]
            return [
                f"Value : {value:.2f}",
                f"Expected Improvement : {expected_improvement:.2f}"
            ]

    @app.callback(
        Output("plot-container", "children"),
        [Input("plot-type", "value"),
        Input("iteration-input", "value")]
    )
    def update_plot_container(plot_type, iteration):
        if plot_type == "global":
            return dcc.Graph(id="plot", figure=plotly_figure)
        elif plot_type == "runs":
            l_runs = ga.load_runs(mod.path, iteration) 
            #return dcc.Graph(figure=l_runs[0].display_plotly(), style={"marginBottom": "20px"})
            return [dcc.Graph(figure=run.display_plotly(max_expected_improvement), style={"marginBottom": "20px"}) for run in l_runs]

    # Callback pour mettre à jour la carte
    @app.callback(
        Output("map", "srcDoc"),
        [Input("iteration-input", "value"),
         Input("plot-weights", "value")]
    )
    def update_map(iteration, plot_weights_value):
        plot_weights = "plot" in plot_weights_value
        return create_weights_map(iteration - 1, mod, plot_weights, df)
    
    return app

# Exécution de l'application
if __name__ == "__main__":
    import hydrophone_placement_scripts.optimisation.class_mod as cls_mod
    
    n_sensors = 5

    mod =cls_mod.load(n_sensors)
    app = create_app(mod)
    app.run()
