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
from shapely.geometry import Polygon, mapping
import geopandas as gpd
import branca.colormap as cm
import folium

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
import torch
from torch.optim.lr_scheduler import StepLR

# %%
import regression
import importlib
importlib.reload(regression)

# %%
from scipy import optimize

# %%
import matplotlib.pyplot as plt

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
serie_in_map = df_trajs.apply(lambda row : calc_mu.in_map(row.x, row.y), axis=1)
for nct in df_trajs["NCT"].unique():
    l = df_trajs[df_trajs["NCT"] == nct].index
    for i, e in enumerate(l[:-1]):
        if (serie_in_map.loc[e] != serie_in_map.loc[e+1]):
            df_trajs.loc[l[i+1:], "NCT"] = max(df_trajs["NCT"][df_trajs["Année"] == df_trajs.loc[e, "Année"]]) + 1

# %%
df_trajs = df_trajs[serie_in_map].reset_index(drop=True)

# %%
df_trajs

# %%
df_trajs["cos_directionCourant"] = np.cos(df_trajs["directionCourant"]*np.pi/180)
df_trajs["sin_directionCourant"] = np.sin(df_trajs["directionCourant"]*np.pi/180)

df_trajs.loc[df_trajs["maree"] == "R", "angleMaree"] = 0
df_trajs.loc[df_trajs["maree"] == "H", "angleMaree"] = 90
df_trajs.loc[df_trajs["maree"] == "E", "angleMaree"] = 180
df_trajs.loc[df_trajs["maree"] == "L", "angleMaree"] = 270
df_trajs["cos_maree"] = np.cos(df_trajs["angleMaree"]*np.pi/180)
df_trajs["sin_maree"] = np.sin(df_trajs["angleMaree"]*np.pi/180)

# %%
step_reg = 10 # in min
step_err = 2 # * step_reg must be an int
df_reg_trajs = regression.create_df_newtrajs(step_reg, step_err, df_trajs, converter, other_columns=["pcG", "Taille", "Rayon", "cos_directionCourant", "sin_directionCourant", "cos_maree", "sin_maree", "vitesseCourant"])

# %%
df_reg_trajs[["areax", "areay"]] = pd.Series(converter.utm2area(df_reg_trajs["x"], df_reg_trajs["y"]))

# %%
max_alpha = 1
min_alpha = 0.3
n = converter.n_areas_x * converter.n_areas_y
x_values = np.arange(converter.n_areas_x)
y_values = np.arange(converter.n_areas_y)
xy_pairs = [(x, y) for x in x_values for y in y_values]

df_areas = pd.DataFrame(xy_pairs, columns=['x', 'y'])
df_areas["number"] = 0

for row in df_reg_trajs.itertuples():
    df_areas.loc[(df_areas["x"] == row.areax) & (df_areas["y"] == row.areay), "number"] += 1

df_areas["polygone"] = None
df = df_areas[df_areas["number"] != 0]

map = folium.Map(location=[(converter.lat_min + converter.lat_max)/2, (converter.lon_min + converter.lon_max)/2], zoom_start=11)
df["polygone"] = df.apply(lambda row : Polygon(converter.area2perim_lla((row.x, row.y))), axis=1)
colormap = cm.LinearColormap(colors=['red', 'yellow', 'green'], vmin=df["number"].min(), vmax=df["number"].max())
map = folium.Map(location=[(converter.lat_min + converter.lat_max)/2, (converter.lon_min + converter.lon_max)/2], zoom_start=11)
colormap.add_to(map)
for row in df.itertuples():
    couleur = colormap(row.number)
    folium.GeoJson(
        row.polygone,
        style_function=lambda _, couleur=couleur, alpha=0.7: {
            'fillColor': couleur,
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7,#alpha,
        }
    ).add_to(map)

map.save("../results/repartition.html")

# %%
df_reg_trajs["directionCourant"] = np.arctan2(df_reg_trajs["sin_directionCourant"], df_reg_trajs["cos_directionCourant"]) * 180/np.pi
df_reg_trajs["angleMaree"] = np.arctan2(df_reg_trajs["sin_maree"], df_reg_trajs["cos_maree"]) * 180/np.pi
df_reg_trajs["maree"] = ""
df_reg_trajs.loc[(-45 <= df_reg_trajs["angleMaree"]) & (df_reg_trajs["angleMaree"] < 45), "maree"] = "R"
df_reg_trajs.loc[(45 <= df_reg_trajs["angleMaree"]) & (df_reg_trajs["angleMaree"] < 135), "maree"] = "H"
df_reg_trajs.loc[(135 <= df_reg_trajs["angleMaree"]) | (df_reg_trajs["angleMaree"] < -135), "maree"] = "E"
df_reg_trajs.loc[(-135 <= df_reg_trajs["angleMaree"]) & (df_reg_trajs["angleMaree"] < -45), "maree"] = "L"
#df_reg_trajs.drop(columns = ["sin_courant", "cos_courant", "sin_maree", "cos_maree"], inplace =True)

# %%
df_reg_trajs[["v", "theta_v"]] = [0., 0.]

# %%
for nct in df_reg_trajs["NCT"].unique():
    l = df_reg_trajs[df_reg_trajs["NCT"] == nct].index
    for i in l[1:-1]:
        deltat = df_reg_trajs.loc[i, "Time"] - df_reg_trajs.loc[i-1, "Time"]
        vx = (df_reg_trajs.loc[i, "x"] - df_reg_trajs.loc[i-1, "x"]) / deltat
        vy = (df_reg_trajs.loc[i, "y"] - df_reg_trajs.loc[i-1, "y"]) / deltat
        df_reg_trajs.loc[i, "v"] = np.sqrt(vx**2 + vy**2)
        df_reg_trajs.loc[i, "theta_v"] = np.arctan2(vy, vx) * 180/np.pi
        deltatp1 = df_reg_trajs.loc[i+1, "Time"] - df_reg_trajs.loc[i, "Time"]
        df_reg_trajs.loc[i, "vxp1"] = (df_reg_trajs.loc[i+1, "x"] - df_reg_trajs.loc[i, "x"]) / deltatp1
        df_reg_trajs.loc[i, "vyp1"] = (df_reg_trajs.loc[i+1, "y"] - df_reg_trajs.loc[i, "y"]) / deltatp1
        
    df_reg_trajs.drop(l[0], inplace =True)
    if len(l) > 1:
        df_reg_trajs.drop(l[-1], inplace =True)

# %%
df_reg_trajs["cos_theta_v"] = np.cos(df_reg_trajs["theta_v"] * np.pi/180)
df_reg_trajs["sin_theta_v"] = np.sin(df_reg_trajs["theta_v"] * np.pi/180)

# %%
df_reg_trajs["vp1"] = np.sqrt(df_reg_trajs["vxp1"] **2 + df_reg_trajs["vyp1"] **2)
df_reg_trajs["theta_vp1"] = np.arctan2(df_reg_trajs["vyp1"], df_reg_trajs["vxp1"])
df_reg_trajs["cos_theta_vp1"] = np.cos(df_reg_trajs["theta_vp1"])
df_reg_trajs["sin_theta_vp1"] = np.sin(df_reg_trajs["theta_vp1"])

# %%
df_reg_trajs.reset_index(drop = True, inplace = True)

# %%
ind_train = random.sample(list(df_reg_trajs.index), 500)
serie_train = pd.Series(index=df_reg_trajs.index, dtype=bool)
for ind in serie_train.index:
    serie_train.loc[ind] = ind in ind_train
df_train = df_reg_trajs[serie_train]
df_test = df_reg_trajs[~serie_train]

# %%
from sklearn.ensemble import RandomForestRegressor

# %%
labels_x = ["x", "y", "pcG", "Taille", "Rayon", "cos_directionCourant", "sin_directionCourant", "cos_maree", "sin_maree", "vitesseCourant", "v", "cos_theta_v", "sin_theta_v"]
labels_y = ["vxp1", "vyp1"]

# %%
regr = RandomForestRegressor()
regr.fit(df_train[labels_x], df_train[labels_y])

# %%
importances = regr.feature_importances_
feature_names = labels_x  # ou ["feature1", "feature2", ...] si X est un array

# Affiche les résultats
plt.figure(figsize=(10, 6))
plt.title("Importance des variables (Random Forest)")
plt.bar(range(len(labels_x)), importances, align="center")
plt.xticks(range(len(labels_x)), labels_x, rotation=90)
plt.tight_layout()
plt.show()

# %%
predictions = regr.predict(df_test[labels_x])

# %%
((np.array(df_test[labels_y]) - predictions)**2).sum(axis = 1).mean()

# %%
labels_x = ["x", "y", "pcG", "Taille", "Rayon", "cos_directionCourant", "sin_directionCourant", "cos_maree", "sin_maree", "vitesseCourant", "v", "cos_theta_v", "sin_theta_v"]
labels_y = ["vp1", "theta_vp1"]

# %%
regr = RandomForestRegressor()
regr.fit(df_train[labels_x], df_train[labels_y])

# %%
importances = regr.feature_importances_
feature_names = labels_x  # ou ["feature1", "feature2", ...] si X est un array

# Affiche les résultats
plt.figure(figsize=(10, 6))
plt.title("Importance des variables (Random Forest)")
plt.bar(range(len(labels_x)), importances, align="center")
plt.xticks(range(len(labels_x)), labels_x, rotation=90)
plt.tight_layout()
plt.show()

# %%
predictions = regr.predict(df_test[labels_x])

# %%
cos_tehta_pred = np.cos(predictions[:, 1])
sin_tehta_pred = np.sin(predictions[:, 1])
cos_tehta_pred, sin_tehta_pred

# %%
trues = np.array(df_test[labels_y])

# %%
vxp1 = predictions[:,0] * cos_tehta_pred
vyp1 = predictions[:,0] * sin_tehta_pred

# %%
((trues[:, 0] - vxp1) **2 + (trues[:, 1] - vyp1)**2).mean()

# %%
labels_x = ["x", "y", "pcG", "Taille", "Rayon", "cos_directionCourant", "sin_directionCourant", "cos_maree", "sin_maree", "vitesseCourant", "v", "cos_theta_v", "sin_theta_v"]
labels_y = ["vp1", "cos_theta_vp1", "sin_theta_vp1"]

# %%
regr = RandomForestRegressor()
regr.fit(df_train[labels_x], df_train[labels_y])

# %%
importances = regr.feature_importances_
feature_names = labels_x  # ou ["feature1", "feature2", ...] si X est un array

# Affiche les résultats
plt.figure(figsize=(10, 6))
plt.title("Importance des variables (Random Forest)")
plt.bar(range(len(labels_x)), importances, align="center")
plt.xticks(range(len(labels_x)), labels_x, rotation=90)
plt.tight_layout()
plt.show()

# %%
predictions = regr.predict(df_test[labels_x])

# %%
vxp1 = predictions[:,0] * predictions[:, 1]
vyp1 = predictions[:,0] * predictions[:, 2]

# %%
((trues[:, 0] - vxp1) **2 + (trues[:, 1] - vyp1)**2).mean()

# %%
