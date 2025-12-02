"""
La classe Model est une surcouche de celle de calculator avec différentes fonctions d'affichage et de gestion des paths, sachant que lorsque les arguments sont différents, les différentes zones ne sont plus les mêmes il faut donc recréer les dictionnaires de profoneur, de substrat et le df_areas
The Model class is a wrapper around the Calculator class, adding various display functions and path-management features. Since changing the arguments alters the zone definitions, the depth dictionary, the substrate dictionary, and the df_areas must be regenerated accordingly
"""


import os
import matplotlib.pyplot as plt
import pickle
import folium
from shapely.geometry import Polygon, mapping
import geopandas as gpd
import branca.colormap as cm

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import hydrophone_placement_scripts.utils_scripts.class_points as cls_points
import hydrophone_placement_scripts.optimisation.bayesian_process as bp
import hydrophone_placement_scripts.optimisation.genetic_algo as ga

import hydrophone_placement_scripts.utils_scripts.conversions_coordinates as conv
import hydrophone_placement_scripts.to_optimise.func_to_optimise as f


class Model:
    def __init__(self, lat_min : float, lat_max : float, lon_min : float, lon_max : float, width_area : float, depth_area : float, n_tetrahedras : int, load:bool=False, version:bool=None, new:bool=False, **kwargs):
        self.path = determine_path(n_tetrahedras, load, version)
        os.makedirs(os.path.join(os.path.dirname(__file__), "../datas/for_model"), exist_ok=True)
        
        if not new:
            new = self.compare_args(lat_min, lat_max, lon_min, lon_max, width_area, depth_area)
        self.converter = conv.Conv(lat_min, lat_max, lon_min, lon_max, width_area, depth_area)
        cls_points.Point.update_point(self.converter.n_areas_x, self.converter.n_areas_y, self.converter.width_area)
        for attr, _ in kwargs.items():
            if not hasattr(f.Calculator, attr):
                raise AttributeError(f"Calculator has no attribute '{attr}'.")
        if "n_processes" in kwargs.keys():
            ga.Genetic_Algo.n_processes = min(kwargs["n_processes"], ga.Genetic_Algo.n_processes)
        self.calculator = f.Calculator(self.converter, new_dics=new, **kwargs)
        self.save_args(lat_min, lat_max, lon_min, lon_max, width_area, depth_area, kwargs)
        cls_points.NPointBayesian.set_value(self.calculator.main_func)
        cls_points.NPoint.set_n_tetrahedras(n_tetrahedras)
        if load:
            self.bayesian_process = bp.load(self.path)
        else:
            self.bayesian_process = bp.Bayesian_Process(self.path)

    def compare_args(self, lat_min : float, lat_max : float, lon_min : float, lon_max : float, width_area : float, depth_area : float):
        news = [lat_min, lat_max, lon_min, lon_max, width_area, depth_area]
        if not "last_args.pkl" in os.listdir(os.path.join(os.path.dirname(__file__), "../datas/for_model")):
            new = True
        else:
            with open(os.path.join(os.path.dirname(__file__), "../datas/for_model/last_args.pkl"), "rb") as f:
                lasts = pickle.load(f)
            new = (lasts != news)
        return new

    def save_args(self, lat_min : float, lat_max : float, lon_min : float, lon_max : float, width_area : float, depth_area : float, kwargs):
        with open(os.path.join(os.path.dirname(__file__), "../datas/for_model/last_args.pkl"), "wb") as f:
            pickle.dump([lat_min, lat_max, lon_min, lon_max, width_area, depth_area], f)
        with open(os.path.join(self.path, "last_args.pkl"), "wb") as f:
            pickle.dump((lat_min, lat_max, lon_min, lon_max, width_area, depth_area, kwargs), f)
            
    def find_max(self, **kwargs):
        self.bayesian_process.modify(**kwargs)     
        npoint = self.bayesian_process.find_max(self.converter, self.calculator,)
        npoint.value()
        self.calculator.save_error(self.path)
        coords = [point.coords for point in npoint.points]
        return [self.converter.utm2lla(c[0], c[1]) for c in coords]

    def max_values(self):
        l = []
        m = self.bayesian_process.set_of_npointsbayesian.values[0]
        for i in range(len(self.bayesian_process.set_of_npointsbayesian.values)):
            if m < self.bayesian_process.set_of_npointsbayesian.values[i]:
                m = self.bayesian_process.set_of_npointsbayesian.values[i]
            l.append(m)
        return l

    def best_value(self):
        return self.bayesian_process.best_value()

    def display(self):
        _, ax1 = plt.subplots()
        n1 = len(self.bayesian_process.expected_improvements)
        n2 = len(self.bayesian_process.set_of_npointsbayesian.values)
        x1 = list(range(n2-n1+1, 1+n2))
        x2 = list(range(1, 1+n2))
        ax2 = ax1.twinx()
        #line1 = plt.axvline(x=n2-n1+0.5, color='black', linestyle='--', label = "Beginning of the bayesian process")
        line2, = ax2.plot(x1, self.bayesian_process.expected_improvements, c="b", marker='o', label="Expected improvements")
        
        line3, = ax1.plot(x2, self.bayesian_process.set_of_npointsbayesian.values, c="r", marker='o', label="Values")
        line4, = ax1.plot(x2, self.bayesian_process.max_values(), c="g", marker='o', label="Best Value")
        ax1.set_ylabel("Values / Best Value")
        ax2.set_ylabel("Expected Improvements")
        #lines = [line1, line2, line3, line4]
        lines = [line2, line3, line4]
        plt.legend(handles = lines)
        plt.title("Optimisation History Plot")
        plt.show()
        return None

    def create_optimisation_history(self, show : bool = False):
        # Récupère les données
        expected_improvements = self.bayesian_process.expected_improvements
        values = self.bayesian_process.set_of_npointsbayesian.values
        best_values = self.bayesian_process.max_values()  # Assure-toi que cette méthode existe

        n1 = len(expected_improvements)
        n2 = len(values)

        x1 = list(range(n2-n1+1, 1+n2))  # Abscisses pour les "Expected Improvements"
        x2 = list(range(1,1+n2))           # Abscisses pour les "Values" et "Best Value"

        # Crée une figure avec deux axes Y
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Ajoute la ligne verticale noire (début du processus bayésien)
        # fig.add_shape(
        #     type="line",
        #     x0=n2 - n1 + 0.5,
        #     y0=0,
        #     x1=n2 - n1 + 0.5,
        #     y1=1,  # Ajuste cette valeur en fonction de tes données
        #     yref="paper",  # Utilise l'échelle du papier pour couvrir toute la hauteur
        #     line=dict(color="black", dash="dash"),
        #     name="Beginning of the (new) bayesian process"
        # )

        # Ajoute la courbe des "Expected Improvements" (axe Y principal)
        fig.add_trace(
            go.Scatter(
                x=x1,
                y=expected_improvements,
                mode="lines+markers",
                marker=dict(color="blue", size=8),
                line=dict(color="blue"),
                name="Expected improvements"
            ),
            secondary_y=True
        )

        # Ajoute la courbe des "Values" (axe Y secondaire)
        fig.add_trace(
            go.Scatter(
                x=x2,
                y=values,
                mode="lines+markers",
                marker=dict(color="red", size=8),
                line=dict(color="red"),
                name="Values"
            ),
            secondary_y=False
        )

        # Ajoute la courbe des "Best Values" (axe Y secondaire)
        fig.add_trace(
            go.Scatter(
                x=x2,
                y=best_values,
                mode="lines+markers",
                marker=dict(color="green", size=8),
                line=dict(color="green"),
                name="Best Value"
            ),
            secondary_y=False
        )

        # Met à jour le titre et la légende
        fig.update_layout(
            title="Optimisation History Plot",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Met à jour les labels des axes
        fig.update_yaxes(title_text="Expected Improvements", secondary_y=True)
        fig.update_yaxes(title_text="Values / Best Value", secondary_y=False)

        # Affiche la figure
        if show:
            fig.show()
        return fig

    def best_npoint(self):
        return self.bayesian_process.best_npoint()

    def create_maps(self):
        self.calculator.load_error(self.path)
        df = self.calculator.df_areas.groupby(['x', 'y'], as_index=False).agg({'w': 'sum', 'error': 'mean'})
        df["polygone"] = df.apply(lambda row : Polygon(self.converter.area2perim_lla((row.x, row.y))), axis=1)

        map = folium.Map(location=[(self.converter.lat_min + self.converter.lat_max)/2, (self.converter.lon_min + self.converter.lon_max)/2], zoom_start=11)
        df["density"] = df.apply(lambda row : self.calculator.calc_mu.h(self.converter.area2utm((row.x,row.y))[0], self.converter.area2utm((row.x,row.y))[1]), axis = 1)
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
        
        gdf = gpd.read_file(self.calculator.traversier_path)
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        folium.GeoJson(
                mapping(gdf_wgs84.loc[0, "geometry"]),
                style_function=lambda _, c="yellow": {"color": c, "weight": 6}
            ).add_to(map)
        folium.GeoJson(
                mapping(gdf_wgs84.loc[1, "geometry"]),
                style_function=lambda _, c="yellow": {"color": c, "weight": 6}
            ).add_to(map)
        for coords in self.best_npoint().coords():
            lat, lon = self.converter.utm2lla(coords[0], coords[1])
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color='black',
                fill=True,
                fill_color='black',
                fill_opacity=1,
                ).add_to(map)
        map.save(os.path.join(self.path, f"weights_map_{cls_points.NPoint.n_tetrahedras}.html"))

        df = df[~df["error"].isna()]
        
        min_alpha, max_alpha = 0.5, 1.0
        df["alpha"] = min_alpha + (max_alpha - min_alpha) * ((df["w"] - df["w"].min()) / (df["w"].max() - df["w"].min()))
        colormap = cm.LinearColormap(colors=['blue', 'red'], vmin=df["error"].min(), vmax=df["error"].max(), caption="Error (m)")
        map = folium.Map(location=[(self.converter.lat_min + self.converter.lat_max)/2, (self.converter.lon_min + self.converter.lon_max)/2], zoom_start=11)
        colormap.add_to(map)
        for row in df.itertuples():
            couleur = colormap(row.error)
            folium.GeoJson(
                row.polygone,
                style_function=lambda _, couleur=couleur, alpha=row.alpha: {
                    'fillColor': couleur,
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': alpha,
                }
            ).add_to(map)
        for coords in self.best_npoint().coords():
            lat, lon = self.converter.utm2lla(coords[0], coords[1])
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color='black',
                fill=True,
                fill_color='black',
                fill_opacity=1,
                ).add_to(map)
        map.save(os.path.join(self.path, f"errors_map_{cls_points.NPoint.n_tetrahedras}.html")) 
        return None

    def load_error(self):
        if "error.csv" in os.listdir(self.path):
            self.calculator.load_error(self.path)
        else :
            self.best_npoint().value()
            self.calculator.save_error(self.path)
        return None

    def create_hist_areas_according_error(self, n_bins : int = 100, max_y : float = None, show : bool = False):
        self.load_error()
        fig = go.Figure(data=[
            go.Histogram(x=self.calculator.df_areas["error"].dropna(), nbinsx=n_bins)
        ])
        fig.update_layout(
            title="Number of areas according to their error",
            xaxis_title="Error (m)",
            yaxis_title="Number of areas",
            yaxis=dict(range=[0, max_y])
        )
        if show:
            fig.show()
        return fig

    def create_error_map(self):
        self.load_error()
        df = self.calculator.df_areas.groupby(['x', 'y'], as_index=False).agg({'w': 'sum', 'error': 'mean'})
        df["polygone"] = df.apply(lambda row : Polygon(self.converter.area2perim_lla((row.x, row.y))), axis=1)
        df.dropna(inplace = True)
        
        min_alpha, max_alpha = 0.5, 1.0
        df["alpha"] = min_alpha + (max_alpha - min_alpha) * ((df["w"] - df["w"].min()) / (df["w"].max() - df["w"].min()))
        colormap = cm.LinearColormap(colors=['blue', 'red'], vmin=df["error"].min(), vmax=df["error"].max(), caption="Error (m)")
        map = folium.Map(location=[(self.converter.lat_min + self.converter.lat_max)/2, (self.converter.lon_min + self.converter.lon_max)/2], zoom_start=11)
        colormap.add_to(map)
        for row in df.itertuples():
            couleur = colormap(row.error)
            folium.GeoJson(
                row.polygone,
                style_function=lambda _, couleur=couleur, alpha=row.alpha: {
                    'fillColor': couleur,
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': alpha,
                }
            ).add_to(map)
        for coords in self.best_npoint().coords():
            lat, lon = self.converter.utm2lla(coords[0], coords[1])
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color='black',
                fill=True,
                fill_color='black',
                fill_opacity=1,
                ).add_to(map)
        return map

def determine_path(n_tetrahedras : int, load : bool, version : int | None = None):
    path_n = os.path.join(os.path.dirname(__file__), f"../results/{n_tetrahedras}_tetrahedras/")
    if version is None:
        os.makedirs(path_n, exist_ok=True)
        l_vers=[]
        for folder in os.listdir(path_n):
            l_vers.append(int(folder[1:]))
        if load :
            path = os.path.join(path_n, "V" + str(max(l_vers)))
        else:
            if len(l_vers) == 0:
                path = os.path.join(path_n, "V1")
            else:
                path = os.path.join(path_n, "V" + str(1 + max(l_vers)))
            os.makedirs(path, exist_ok=True)
    else:
        path = os.path.join(path_n, "V" + str(version))
    return path

def load(n_tetrahedras : int, version : int | None = None, **kwargs):
    path = determine_path(n_tetrahedras, True, version)
    with open(os.path.join(path, "last_args.pkl"), "rb") as f:
        lat_min, lat_max, lon_min, lon_max, width_area, depth_area, old_kwargs = pickle.load(f)
    for key in kwargs.keys():
        old_kwargs[key] = kwargs[key]
    return Model(lat_min, lat_max, lon_min, lon_max, width_area, depth_area, n_tetrahedras, True, version, **old_kwargs)