# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import sys
import os
sys.path.append("../..")

# %%
import hydrophone_placement_scripts.utils_scripts.conversions_coordinates as conv
from hydrophone_placement_scripts.to_optimise import topo
import hydrophone_placement_scripts.coords_belugas.calc_mu as clc_mu
import hydrophone_placement_scripts.coords_belugas.ornstein_uhlenbeck as ornstein_uhlenbeck

# %%
import importlib
importlib.reload(conv)
importlib.reload(topo)
importlib.reload(clc_mu)
importlib.reload(ornstein_uhlenbeck)

# %%
import pandas as pd
import numpy as np
import random

# %%
args = {
    "lat_min" : 47.65,
    "lat_max" : 48.07,
    "lon_min" : -70.04,
    "lon_max" : -69.28,
    "width_area" : 1000,
    "depth_area" : 20,
    "accuracy" : 1,
}
geotiff_path = "../datas/BelugaRelativeDens/BelugaRelativeDens.tif"
step = 100

converter = conv.Conv(**args)
#os.path.join(os.path.dirname(__file__), "../datas/")
#obj_topo = topo.Topo(converter, save_path="datas/", substrat=False)
calc_mu = clc_mu.Calc_mu(geotiff_path, step)

# %%
df_trajs = pd.read_csv("../datas/coords_belugas/cleaned_coords.csv", sep=";").drop(columns=["Unnamed: 0"])
df_trajs

# %%
serie_in_area = converter.in_area(df_trajs["x"], df_trajs["y"])
serie_in_map = df_trajs.apply(lambda row : calc_mu.in_map(row.x, row.y), axis=1)
for nct in df_trajs["NCT"].unique():
    l = df_trajs[df_trajs["NCT"] == nct].index
    for i, e in enumerate(l[:-1]):
        if (serie_in_area.loc[e] != serie_in_area.loc[e+1]) | (serie_in_map.loc[e] != serie_in_map.loc[e+1]):
            df_trajs.loc[l[i+1:], "NCT"] = max(df_trajs["NCT"][df_trajs["Année"] == df_trajs.loc[e, "Année"]]) + 1

# %%
df_trajs_in = df_trajs[serie_in_map & serie_in_area].reset_index(drop=True)
df_trajs_out = df_trajs[serie_in_map & ~serie_in_area].reset_index(drop=True)

# %%
ncts_train = random.sample(list(df_trajs_out["NCT"].unique()), int(len(df_trajs_out["NCT"].unique()) / 2))
serie = df_trajs_out.apply(lambda row : row.NCT in ncts_train, axis=1)
df_trajs_train1 = df_trajs_out[serie]
df_trajs_train1.reset_index(inplace = True, drop = True)
df_trajs_train2 = df_trajs_out[~serie].reset_index(drop = True)
df_trajs_train2.reset_index(inplace = True, drop = True)

# %%
delta2 = 200 #(d/np.sqrt(2))^2

# %%
sigma2 = ornstein_uhlenbeck.optimise_sigma2(df_trajs_train1, delta2)
sigma2

# %%
calc_mu.set_sigma2(sigma2)

# %%
inv_tau0 = ornstein_uhlenbeck.get_inv_tau0(df_trajs_train1, calc_mu.mu)
inv_tau0

# %%
deltav2 = ornstein_uhlenbeck.get_deltav2(df_trajs_train1)
deltav2

# %%
importlib.reload(ornstein_uhlenbeck)
importlib.reload(clc_mu)
calc_mu = clc_mu.Calc_mu(geotiff_path, step)
calc_mu.set_sigma2(sigma2)

# %%
sigmav2 = ornstein_uhlenbeck.get_sigmav2(df_trajs_train2, inv_tau0, calc_mu)
sigmav2

# %%
import regression
import importlib
importlib.reload(regression)

# %%
step_reg = 2 # in min
step_err = 5 # * step_reg must be an int

df_reg_trajs = regression.create_df_newtrajs(step_reg, step_err, df_trajs_in, converter)

# %%
n = converter.n_areas_x * converter.n_areas_y
x_values = np.arange(converter.n_areas_x)
y_values = np.arange(converter.n_areas_y)
xy_pairs = [(x, y) for x in x_values for y in y_values]

df_areas = pd.DataFrame(xy_pairs, columns=['x', 'y'])
df_areas["error"] = 0.
df_areas["n_error"] = 0

# %%
for nct in df_reg_trajs["NCT"].unique():
    l = df_reg_trajs[df_reg_trajs["NCT"] == nct].index
    traj = ornstein_uhlenbeck.forecast_trajectory(df_reg_trajs["x"].loc[l[0]], df_reg_trajs["y"].loc[l[0]], 0, 0, inv_tau0, 0, calc_mu, lambda x : None, False)
    for i in range (l[0], l[-1]):
        deltat = df_reg_trajs["Time"].loc[i+1] - df_reg_trajs["Time"].loc[i]
        err_x, err_v = traj.err_set(deltat, np.array([df_reg_trajs["x"].loc[i+1], df_reg_trajs["y"].loc[i+1]]))          
        area = converter.utm2area(df_reg_trajs["x"].loc[i], df_reg_trajs["y"].loc[i])
        df_areas.loc[(df_areas["x"] == area[0]) & (df_areas["y"] == area[1]), "error"] += err_x
        df_areas.loc[(df_areas["x"] == area[0]) & (df_areas["y"] == area[1]), "n_error"] += 1

# %%
df_areas_with_error = df_areas[df_areas["n_error"] > 0].copy()
df_areas_with_error["mean_error"] = np.sqrt(df_areas_with_error["error"] / df_areas_with_error["n_error"])

# %%
df_areas_with_error["mean_error"] = df_areas_with_error["error"] / df_areas_with_error["n_error"]

# %%
from shapely.geometry import Polygon, mapping
import geopandas as gpd
import branca.colormap as cm
import folium

# %%
df_areas_with_error["rmse"] = np.sqrt(df_areas_with_error["mean_error"])

# %%
max_alpha = 1
min_alpha = 0.3

map = folium.Map(location=[(converter.lat_min + converter.lat_max)/2, (converter.lon_min + converter.lon_max)/2], zoom_start=11)
df_areas_with_error["polygone"] = df_areas_with_error.apply(lambda row : Polygon(converter.area2perim_lla((row.x, row.y))), axis=1)
df_areas_with_error["alpha"] = min_alpha + (max_alpha - min_alpha) * ((df_areas_with_error["n_error"] - df_areas_with_error["n_error"].min()) / (df_areas_with_error["n_error"].max() - df_areas_with_error["n_error"].min()))
colormap = cm.LinearColormap(colors=['green', 'yellow', 'red'], vmin=df_areas_with_error["rmse"].min()/1000, vmax=df_areas_with_error["rmse"].max()/1000, caption="Error (km)")
map = folium.Map(location=[(converter.lat_min + converter.lat_max)/2, (converter.lon_min + converter.lon_max)/2], zoom_start=11)
colormap.add_to(map)
for row in df_areas_with_error.itertuples():
    couleur = colormap(row.rmse / 1000)
    folium.GeoJson(
        row.polygone,
        style_function=lambda _, couleur=couleur, alpha=row.alpha: {
            'fillColor': couleur,
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7,#alpha,
        }
    ).add_to(map)

map.save("../results/tests2.html")

# %%
