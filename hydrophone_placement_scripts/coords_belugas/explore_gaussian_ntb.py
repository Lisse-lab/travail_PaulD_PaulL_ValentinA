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
df_trajs["cos_courant"] = np.cos(df_trajs["directionCourant"]*np.pi/180)
df_trajs["sin_courant"] = np.sin(df_trajs["directionCourant"]*np.pi/180)

df_trajs.loc[df_trajs["maree"] == "R", "angleMaree"] = 0
df_trajs.loc[df_trajs["maree"] == "H", "angleMaree"] = 90
df_trajs.loc[df_trajs["maree"] == "E", "angleMaree"] = 180
df_trajs.loc[df_trajs["maree"] == "L", "angleMaree"] = 270
df_trajs["cos_maree"] = np.cos(df_trajs["angleMaree"]*np.pi/180)
df_trajs["sin_maree"] = np.sin(df_trajs["angleMaree"]*np.pi/180)

# %% [markdown]
# ### voir fin !!!

# %%
step_reg = 15 # in min
step_err = 1 # * step_reg must be an int
df_reg_trajs = regression.create_df_newtrajs(step_reg, step_err, df_trajs, converter, other_columns=["pcG", "Taille", "Rayon", "cos_courant", "sin_courant", "cos_maree", "sin_maree", "vitesseCourant"])

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
df_reg_trajs["directionCourant"] = np.arctan2(df_reg_trajs["sin_courant"], df_reg_trajs["cos_courant"]) * 180/np.pi
df_reg_trajs["angleMaree"] = np.arctan2(df_reg_trajs["sin_maree"], df_reg_trajs["cos_maree"]) * 180/np.pi
df_reg_trajs["maree"] = ""
df_reg_trajs.loc[(-45 <= df_reg_trajs["angleMaree"]) & (df_reg_trajs["angleMaree"] < 45), "maree"] = "R"
df_reg_trajs.loc[(45 <= df_reg_trajs["angleMaree"]) & (df_reg_trajs["angleMaree"] < 135), "maree"] = "H"
df_reg_trajs.loc[(135 <= df_reg_trajs["angleMaree"]) | (df_reg_trajs["angleMaree"] < -135), "maree"] = "E"
df_reg_trajs.loc[(-135 <= df_reg_trajs["angleMaree"]) & (df_reg_trajs["angleMaree"] < -45), "maree"] = "L"
df_reg_trajs.drop(columns = ["sin_courant", "cos_courant", "sin_maree", "cos_maree"], inplace =True)

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
df_reg_trajs.reset_index(drop = True, inplace = True)


# %%
def abs_diff_angle(diff):
    return abs(np.arctan2(np.sin(diff), np.cos(diff))) * 180 / np.pi


# %%
def create_values_norm(areax = None, areay = None, show = True, stds = None):
    if (not areax is None) & (not areay is None):
        df = df_reg_trajs[(df_reg_trajs["areax"] == areax) & (df_reg_trajs["areay"] == areay)]
    else :
        df = df_reg_trajs
    if len(df) >= 2:
        x = df["x"].values
        y = df["y"].values
        dist_pos = (x[:, np.newaxis] - x) ** 2 + (y[:, np.newaxis] - y) ** 2
        l_dist_pos = dist_pos[np.triu_indices(len(dist_pos), k=1)]
        std_pos = np.std(np.sqrt(l_dist_pos)) if stds is None else stds[0]
        l_dist_pos_norm = l_dist_pos/ std_pos**2

        v = df["v"].values
        dist_v = (v[:, np.newaxis] - v) ** 2
        l_dist_v = dist_v[np.triu_indices(len(dist_v), k=1)]
        std_v = np.std(np.sqrt(l_dist_v)) if stds is None else stds[1]
        l_dist_v_norm = l_dist_v / std_v**2

        theta_v = df["theta_v"].values
        dist_theta_v = abs_diff_angle(theta_v[:, np.newaxis] - theta_v) ** 2
        l_dist_theta_v = dist_theta_v[np.triu_indices(len(dist_theta_v), k=1)]
        std_theta_v = np.std(np.sqrt(l_dist_theta_v)) if stds is None else stds[2]
        l_dist_theta_v_norm = l_dist_theta_v / std_theta_v**2

        pcG = df["pcG"].values
        dist_pcG = (pcG[:, np.newaxis] - pcG) ** 2
        l_dist_pcG = dist_pcG[np.triu_indices(len(dist_pcG), k=1)]
        std_pcG = np.std(np.sqrt(l_dist_pcG)) if stds is None else stds[3]
        l_dist_pcG_norm = l_dist_pcG / std_pcG**2

        Taille = df["Taille"].values
        dist_Taille = (Taille[:, np.newaxis] - Taille) ** 2
        l_dist_Taille = dist_Taille[np.triu_indices(len(dist_Taille), k=1)]
        std_Taille = np.std(np.sqrt(l_dist_Taille)) if stds is None else stds[4]
        l_dist_Taille_norm = l_dist_Taille / std_Taille**2 

        Rayon = df["Rayon"].values
        dist_Rayon = (Rayon[:, np.newaxis] - Rayon) ** 2
        l_dist_Rayon = dist_Rayon[np.triu_indices(len(dist_Rayon), k=1)]
        std_Rayon = np.std(np.sqrt(l_dist_Rayon)) if stds is None else stds[5]
        l_dist_Rayon_norm = l_dist_Rayon / std_Rayon**2

        vitesseCourant = df["vitesseCourant"].values
        dist_vitesseCourant = (vitesseCourant[:, np.newaxis] - vitesseCourant) ** 2
        l_dist_vitesseCourant = dist_vitesseCourant[np.triu_indices(len(dist_vitesseCourant), k=1)]
        std_vitesseCourant = np.std(np.sqrt(l_dist_vitesseCourant)) if stds is None else stds[6]
        l_dist_vitesseCourant_norm = l_dist_vitesseCourant / std_vitesseCourant**2
        
        directionCourant = df["directionCourant"].values
        dist_directionCourant = abs_diff_angle(directionCourant[:, np.newaxis] - directionCourant) ** 2
        l_dist_directionCourant = dist_directionCourant[np.triu_indices(len(dist_directionCourant), k=1)]
        std_directionCourant = np.std(np.sqrt(l_dist_directionCourant)) if stds is None else stds[7]
        l_dist_directionCourant_norm = l_dist_directionCourant / std_directionCourant**2

        angleMaree = df["angleMaree"].values
        dist_angleMaree = abs_diff_angle(angleMaree[:, np.newaxis] - angleMaree) ** 2
        l_dist_angleMaree = dist_angleMaree[np.triu_indices(len(dist_angleMaree), k=1)]
        std_angleMaree = np.std(np.sqrt(l_dist_angleMaree)) if stds is None else stds[8]
        l_dist_angleMaree_norm = l_dist_angleMaree / std_angleMaree**2

        vxp1 = df["vxp1"].values
        vyp1 = df["vyp1"].values
        dist_vp1 = (vxp1[:, np.newaxis] - vxp1) ** 2 + (vyp1[:, np.newaxis] - vyp1) ** 2
        l_dist_vp1 = dist_vp1[np.triu_indices(len(dist_vp1), k=1)]

        values_norm = np.array([l_dist_pos_norm, l_dist_v_norm, l_dist_theta_v_norm, l_dist_pcG_norm, l_dist_Taille_norm, l_dist_Rayon_norm, l_dist_vitesseCourant_norm, l_dist_directionCourant_norm, l_dist_angleMaree_norm])

        if show :
            print("std_pos :", std_pos)
            print("std_v :", std_v)
            print("std_theta_v :", std_theta_v)
            print("std_pcG :", std_pcG)
            print("std_Taille :", std_Taille)
            print("std_Rayon :", std_Rayon)
            print("std_vitesseCourant :", std_vitesseCourant)    
            print("std_directionCourant :", std_directionCourant)
            print("std_angleMaree :", std_angleMaree)

        return values_norm, l_dist_vp1, [std_pos, std_v, std_theta_v, std_pcG, std_Taille, std_Rayon, std_vitesseCourant, std_directionCourant, std_angleMaree]
    else :
        return None, None, stds


# %%
def gauss_func(params, values_norm):
    return np.exp(-1/2 * params.T @ values_norm)


# %%
values_norm, l_dist_vp1, stds = create_values_norm()

# %%
meanings = ["Position", "Vitesse", "Angle vitesse", "%G", "Taille", "Rayon", "vitesse Courant", "directionCourant", "angle maree"]


# %%
def loss(params):
    g = gauss_func(params, values_norm)
    return g.T @ l_dist_vp1 / g.sum()


# %%
bounds = [(0, None) for _ in range(len(values_norm))]
result = optimize.minimize(loss, [1.0] * len(values_norm), bounds=bounds, method='L-BFGS-B')
print(result)

# %%
# results = []
# bounds = [(0, None) for _ in range(len(values_norm))]

# result = optimize.minimize(loss, [1.0] * len(values_norm), bounds=bounds, method='L-BFGS-B')
# results.append(result.x)
# print(result)

# result = optimize.minimize(loss, [1.0] * len(values_norm), bounds=bounds, method='Nelder-Mead')
# results.append(result.x)
# print(result)

# result = optimize.minimize(loss, [1.0] * len(values_norm), bounds=bounds, method='Powell')
# results.append(result.x)
# print(result)

# %%
plt.figure(figsize=(8, 5))
plt.bar(meanings, result.x)

# Ajout des titres et labels
plt.title("Valeurs des poids optimisés")
plt.ylabel("Valeur du poids")
plt.xlabel("Type de poids")
plt.xticks(rotation=45, ha='right')
# Affichage du graphique
plt.show()

# %%
values_norm.mean(axis = 1)

# %%
n = converter.n_areas_x * converter.n_areas_y
x_values = np.arange(converter.n_areas_x)
y_values = np.arange(converter.n_areas_y)
xy_pairs = [(x, y) for x in x_values for y in y_values]

df_areas = pd.DataFrame(xy_pairs, columns=['x', 'y'])
df_areas["error"] = None
df_areas["n_error"] = None


# %%
def create_values_norm(areax = None, areay = None, show = True, stds = None):
    if (not areax is None) & (not areay is None):
        df = df_reg_trajs[(df_reg_trajs["areax"] == areax) & (df_reg_trajs["areay"] == areay)]
    else :
        df= df_reg_trajs
    if len(df) >= 2:
        x = df["x"].values
        y = df["y"].values
        dist_pos = (x[:, np.newaxis] - x) ** 2 + (y[:, np.newaxis] - y) ** 2
        l_dist_pos = dist_pos[np.triu_indices(len(dist_pos), k=1)]
        std_pos = np.std(np.sqrt(l_dist_pos)) if stds is None else stds[0]
        l_dist_pos_norm = l_dist_pos/ std_pos**2

        v = df["v"].values
        dist_v = (v[:, np.newaxis] - v) ** 2
        l_dist_v = dist_v[np.triu_indices(len(dist_v), k=1)]
        std_v = np.std(np.sqrt(l_dist_v)) if stds is None else stds[1]
        l_dist_v_norm = l_dist_v / std_v**2

        theta_v = df["theta_v"].values
        dist_theta_v = abs_diff_angle(theta_v[:, np.newaxis] - theta_v) ** 2
        l_dist_theta_v = dist_theta_v[np.triu_indices(len(dist_theta_v), k=1)]
        std_theta_v = np.std(np.sqrt(l_dist_theta_v)) if stds is None else stds[2]
        l_dist_theta_v_norm = l_dist_theta_v / std_theta_v**2

        pcG = df["pcG"].values
        dist_pcG = (pcG[:, np.newaxis] - pcG) ** 2
        l_dist_pcG = dist_pcG[np.triu_indices(len(dist_pcG), k=1)]
        std_pcG = np.std(np.sqrt(l_dist_pcG)) if stds is None else stds[3]
        l_dist_pcG_norm = l_dist_pcG / std_pcG**2

        Taille = df["Taille"].values
        dist_Taille = (Taille[:, np.newaxis] - Taille) ** 2
        l_dist_Taille = dist_Taille[np.triu_indices(len(dist_Taille), k=1)]
        std_Taille = np.std(np.sqrt(l_dist_Taille)) if stds is None else stds[4]
        l_dist_Taille_norm = l_dist_Taille / std_Taille**2 

        Rayon = df["Rayon"].values
        dist_Rayon = (Rayon[:, np.newaxis] - Rayon) ** 2
        l_dist_Rayon = dist_Rayon[np.triu_indices(len(dist_Rayon), k=1)]
        std_Rayon = np.std(np.sqrt(l_dist_Rayon)) if stds is None else stds[5]
        l_dist_Rayon_norm = l_dist_Rayon / std_Rayon**2

        vitesseCourant = df["vitesseCourant"].values
        dist_vitesseCourant = (vitesseCourant[:, np.newaxis] - vitesseCourant) ** 2
        l_dist_vitesseCourant = dist_vitesseCourant[np.triu_indices(len(dist_vitesseCourant), k=1)]
        std_vitesseCourant = np.std(np.sqrt(l_dist_vitesseCourant)) if stds is None else stds[6]
        l_dist_vitesseCourant_norm = l_dist_vitesseCourant / std_vitesseCourant**2
        
        directionCourant = df["directionCourant"].values
        dist_directionCourant = abs_diff_angle(directionCourant[:, np.newaxis] - directionCourant) ** 2
        l_dist_directionCourant = dist_directionCourant[np.triu_indices(len(dist_directionCourant), k=1)]
        std_directionCourant = np.std(np.sqrt(l_dist_directionCourant)) if stds is None else stds[7]
        l_dist_directionCourant_norm = l_dist_directionCourant / std_directionCourant**2

        angleMaree = df["angleMaree"].values
        dist_angleMaree = abs_diff_angle(angleMaree[:, np.newaxis] - angleMaree) ** 2
        l_dist_angleMaree = dist_angleMaree[np.triu_indices(len(dist_angleMaree), k=1)]
        std_angleMaree = np.std(np.sqrt(l_dist_angleMaree)) if stds is None else stds[8]
        l_dist_angleMaree_norm = l_dist_angleMaree / std_angleMaree**2

        vxp1 = df["vxp1"].values
        vyp1 = df["vyp1"].values
        dist_vp1 = (vxp1[:, np.newaxis] - vxp1) ** 2 + (vyp1[:, np.newaxis] - vyp1) ** 2
        l_dist_vp1 = dist_vp1[np.triu_indices(len(dist_vp1), k=1)]

        values_norm = np.array([l_dist_pos_norm, l_dist_v_norm, l_dist_theta_v_norm, l_dist_pcG_norm, l_dist_Taille_norm, l_dist_Rayon_norm, l_dist_vitesseCourant_norm, l_dist_directionCourant_norm, l_dist_angleMaree_norm])

        if show :
            print("std_pos :", std_pos)
            print("std_v :", std_v)
            print("std_theta_v :", std_theta_v)
            print("std_pcG :", std_pcG)
            print("std_Taille :", std_Taille)
            print("std_Rayon :", std_Rayon)
            print("std_vitesseCourant :", std_vitesseCourant)    
            print("std_directionCourant :", std_directionCourant)
            print("std_angleMaree :", std_angleMaree)

        return values_norm, l_dist_vp1, [std_pos, std_v, std_theta_v, std_pcG, std_Taille, std_Rayon, std_vitesseCourant, std_directionCourant, std_angleMaree]
    else :
        return None, None, stds


# %%
#for i, xy in enumerate(list(set(zip(df_reg_trajs["areax"], df_reg_trajs["areay"])))):
for i in df_areas.index:
    areax, areay = df_areas.loc[i, ["x","y"]]
    values_norm, l_dist_vp1, _ = create_values_norm(areax, areay, show = False, stds=stds)
    if (not values_norm is None) & (not l_dist_vp1 is None):
        g = gauss_func(result.x, values_norm)
        if g.sum() > 0:
            df_areas.loc[i, "error"] = g.T @ l_dist_vp1 / g.sum()
            df_areas.loc[i, "n_error"] = len(values_norm[0])

# %%
df_areas

# %%
max_alpha = 1
min_alpha = 0.3

map = folium.Map(location=[(converter.lat_min + converter.lat_max)/2, (converter.lon_min + converter.lon_max)/2], zoom_start=11)
df_areas["polygone"] = df_areas.apply(lambda row : Polygon(converter.area2perim_lla((row.x, row.y))), axis=1)
df_areas["alpha"] = min_alpha + (max_alpha - min_alpha) * ((df_areas["n_error"] - df_areas["n_error"].min()) / (df_areas["n_error"].max() - df_areas["n_error"].min()))
df = df_areas[~df_areas["error"].isna()]
colormap = cm.LinearColormap(colors=['green', 'yellow', 'red'], vmin=df["error"].min(), vmax=df["error"].max(), caption="Error (km)")
map = folium.Map(location=[(converter.lat_min + converter.lat_max)/2, (converter.lon_min + converter.lon_max)/2], zoom_start=11)
colormap.add_to(map)
for row in df.itertuples():
    couleur = colormap(row.error)
    folium.GeoJson(
        row.polygone,
        style_function=lambda _, couleur=couleur, alpha=row.alpha: {
            'fillColor': couleur,
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7,#alpha,
        }
    ).add_to(map)

map.save("../results/tests3.html")


# %%

# %%

# %%
def create_values_norm2(areax = None, areay = None, show = True, stds = None):
    if (not areax is None) & (not areay is None):
        df = df_reg_trajs[(df_reg_trajs["areax"] == areax) & (df_reg_trajs["areay"] == areay)]
    else :
        df = df_reg_trajs
    if len(df) >= 2:
        x = df["x"].values
        y = df["y"].values
        dist_pos = (x[:, np.newaxis] - x) ** 2 + (y[:, np.newaxis] - y) ** 2
        l_dist_pos = dist_pos[np.triu_indices(len(dist_pos), k=1)]
        std_pos = np.std(np.sqrt(l_dist_pos)) if stds is None else stds[0]
        dist_pos_norm = dist_pos/ std_pos**2

        v = df["v"].values
        dist_v = (v[:, np.newaxis] - v) ** 2
        l_dist_v = dist_v[np.triu_indices(len(dist_v), k=1)]
        std_v = np.std(np.sqrt(l_dist_v)) if stds is None else stds[1]
        dist_v_norm = dist_v / std_v**2

        theta_v = df["theta_v"].values
        dist_theta_v = abs_diff_angle(theta_v[:, np.newaxis] - theta_v) ** 2
        l_dist_theta_v = dist_theta_v[np.triu_indices(len(dist_theta_v), k=1)]
        std_theta_v = np.std(np.sqrt(l_dist_theta_v)) if stds is None else stds[2]
        dist_theta_v_norm = dist_theta_v / std_theta_v**2

        pcG = df["pcG"].values
        dist_pcG = (pcG[:, np.newaxis] - pcG) ** 2
        l_dist_pcG = dist_pcG[np.triu_indices(len(dist_pcG), k=1)]
        std_pcG = np.std(np.sqrt(l_dist_pcG)) if stds is None else stds[3]
        dist_pcG_norm = dist_pcG / std_pcG**2

        Taille = df["Taille"].values
        dist_Taille = (Taille[:, np.newaxis] - Taille) ** 2
        l_dist_Taille = dist_Taille[np.triu_indices(len(dist_Taille), k=1)]
        std_Taille = np.std(np.sqrt(l_dist_Taille)) if stds is None else stds[4]
        dist_Taille_norm = dist_Taille / std_Taille**2 

        Rayon = df["Rayon"].values
        dist_Rayon = (Rayon[:, np.newaxis] - Rayon) ** 2
        l_dist_Rayon = dist_Rayon[np.triu_indices(len(dist_Rayon), k=1)]
        std_Rayon = np.std(np.sqrt(l_dist_Rayon)) if stds is None else stds[5]
        dist_Rayon_norm = dist_Rayon / std_Rayon**2

        vitesseCourant = df["vitesseCourant"].values
        dist_vitesseCourant = (vitesseCourant[:, np.newaxis] - vitesseCourant) ** 2
        l_dist_vitesseCourant = dist_vitesseCourant[np.triu_indices(len(dist_vitesseCourant), k=1)]
        std_vitesseCourant = np.std(np.sqrt(l_dist_vitesseCourant)) if stds is None else stds[6]
        dist_vitesseCourant_norm = dist_vitesseCourant / std_vitesseCourant**2
        
        directionCourant = df["directionCourant"].values
        dist_directionCourant = abs_diff_angle(directionCourant[:, np.newaxis] - directionCourant) ** 2
        l_dist_directionCourant = dist_directionCourant[np.triu_indices(len(dist_directionCourant), k=1)]
        std_directionCourant = np.std(np.sqrt(l_dist_directionCourant)) if stds is None else stds[7]
        dist_directionCourant_norm = dist_directionCourant / std_directionCourant**2

        angleMaree = df["angleMaree"].values
        dist_angleMaree = abs_diff_angle(angleMaree[:, np.newaxis] - angleMaree) ** 2
        l_dist_angleMaree = dist_angleMaree[np.triu_indices(len(dist_angleMaree), k=1)]
        std_angleMaree = np.std(np.sqrt(l_dist_angleMaree)) if stds is None else stds[8]
        dist_angleMaree_norm = dist_angleMaree / std_angleMaree**2

        vxp1 = df["vxp1"].values
        vyp1 = df["vyp1"].values
        diff_vp1x = vxp1[:, np.newaxis] - vxp1
        diff_vp1y = vyp1[:, np.newaxis] - vyp1 

        values_norm = np.array([dist_pos_norm, dist_v_norm, dist_theta_v_norm, dist_pcG_norm, dist_Taille_norm, dist_Rayon_norm, dist_vitesseCourant_norm, dist_directionCourant_norm, dist_angleMaree_norm])

        if show :
            print("std_pos :", std_pos)
            print("std_v :", std_v)
            print("std_theta_v :", std_theta_v)
            print("std_pcG :", std_pcG)
            print("std_Taille :", std_Taille)
            print("std_Rayon :", std_Rayon)
            print("std_vitesseCourant :", std_vitesseCourant)    
            print("std_directionCourant :", std_directionCourant)
            print("std_angleMaree :", std_angleMaree)

        return values_norm, diff_vp1x, diff_vp1y, [std_pos, std_v, std_theta_v, std_pcG, std_Taille, std_Rayon, std_vitesseCourant, std_directionCourant, std_angleMaree]
    else :
        return None, None, None, stds


# %%
values_norm, diff_vp1x, diff_vp1y, stds = create_values_norm2()

# %%
values_norm.shape


# %%
def gauss_func(params, values_norm):
    return np.exp(-1/2 * np.sum(params[:, np.newaxis, np.newaxis] * values_norm, axis=0))


# %%
g = gauss_func(result.x, values_norm)

# %%
g.shape

# %%
diff_vp1x.shape

# %%
c = ((g * diff_vp1x) **2 + (g * diff_vp1y) **2)

# %%
(((g * diff_vp1x) **2 + (g * diff_vp1y) **2).sum(axis = 0) / g.sum(axis = 1)).sum()


# %%
def loss2(params):
    g = gauss_func(params, values_norm)
    return (((g * diff_vp1x) **2 + (g * diff_vp1y) **2).sum(axis = 0) / (g.sum(axis = 0) -1)).sum()


# %%
g = gauss_func(np.array([1] * values_norm.shape[0]), values_norm)
g

# %%
g = gauss_func(result.x, values_norm)
g

# %%
g.sum(axis = 0) - 1

# %%
bounds = [(0, None) for _ in range(len(values_norm))]
result = optimize.minimize(loss2, [1.0] * len(values_norm), bounds=bounds, method='L-BFGS-B')
print(result)

# %%
plt.figure(figsize=(8, 5))
plt.bar(meanings, result.x)

# Ajout des titres et labels
plt.title("Valeurs des poids optimisés")
plt.ylabel("Valeur du poids")
plt.xlabel("Type de poids")
plt.xticks(rotation=45, ha='right')
# Affichage du graphique
plt.show()

# %%
n = converter.n_areas_x * converter.n_areas_y
x_values = np.arange(converter.n_areas_x)
y_values = np.arange(converter.n_areas_y)
xy_pairs = [(x, y) for x in x_values for y in y_values]

df_areas = pd.DataFrame(xy_pairs, columns=['x', 'y'])
df_areas["error"] = 0
df_areas["n_error"] = 0

# %%
for i in df_areas.index:
    areax, areay = df_areas.loc[i, ["x","y"]]
    values_norm, diff_vp1x, diff_vp1y, _ = create_values_norm2(areax, areay, show = False, stds=stds)
    if (not values_norm is None): # & (not l_dist_vp1 is None):
        g = gauss_func(result.x, values_norm)
        if g.sum() > 0:
            print()
            df_areas.loc[i, "error"] = (((g * diff_vp1x) **2 + (g * diff_vp1y) **2).sum(axis = 0) / g.sum(axis = 1)).sum()
            df_areas.loc[i, "n_error"] = len(values_norm[0])

# %%
df_areas["mean_error"] = df_areas["error"] / df_areas["n_error"]

# %%
max_alpha = 1
min_alpha = 0.3

map = folium.Map(location=[(converter.lat_min + converter.lat_max)/2, (converter.lon_min + converter.lon_max)/2], zoom_start=11)
df_areas["polygone"] = df_areas.apply(lambda row : Polygon(converter.area2perim_lla((row.x, row.y))), axis=1)
df_areas["alpha"] = min_alpha + (max_alpha - min_alpha) * ((df_areas["n_error"] - df_areas["n_error"].min()) / (df_areas["n_error"].max() - df_areas["n_error"].min()))
df = df_areas[df_areas["n_error"] > 0]
colormap = cm.LinearColormap(colors=['green', 'yellow', 'red'], vmin=df["mean_error"].min(), vmax=df["mean_error"].max(), caption="Error (km)")
map = folium.Map(location=[(converter.lat_min + converter.lat_max)/2, (converter.lon_min + converter.lon_max)/2], zoom_start=11)
colormap.add_to(map)
for row in df.itertuples():
    couleur = colormap(row.mean_error)
    folium.GeoJson(
        row.polygone,
        style_function=lambda _, couleur=couleur, alpha=row.alpha: {
            'fillColor': couleur,
            'color': 'black',
            'weight': 1,
            'fillOpacity': alpha,
        }
    ).add_to(map)

map.save("../results/tests4.html")

# %%
df_areas[df_areas["n_error"] > 0]

# %%

# %%

# %%

# %%
step_reg = 10 # in min
step_err = 2 # * step_reg must be an int
df_reg_trajs = regression.create_df_newtrajs(step_reg, step_err, df_trajs, converter, other_columns=["pcG", "Taille", "Rayon", "cos_courant", "sin_courant", "cos_maree", "sin_maree", "vitesseCourant"])

# %%
df_reg_trajs[["areax", "areay"]] = pd.Series(converter.utm2area(df_reg_trajs["x"], df_reg_trajs["y"]))

# %%
df_reg_trajs["directionCourant"] = np.arctan2(df_reg_trajs["sin_courant"], df_reg_trajs["cos_courant"]) * 180/np.pi
df_reg_trajs["angleMaree"] = np.arctan2(df_reg_trajs["sin_maree"], df_reg_trajs["cos_maree"]) * 180/np.pi
df_reg_trajs["maree"] = ""
df_reg_trajs.loc[(-45 <= df_reg_trajs["angleMaree"]) & (df_reg_trajs["angleMaree"] < 45), "maree"] = "R"
df_reg_trajs.loc[(45 <= df_reg_trajs["angleMaree"]) & (df_reg_trajs["angleMaree"] < 135), "maree"] = "H"
df_reg_trajs.loc[(135 <= df_reg_trajs["angleMaree"]) | (df_reg_trajs["angleMaree"] < -135), "maree"] = "E"
df_reg_trajs.loc[(-135 <= df_reg_trajs["angleMaree"]) & (df_reg_trajs["angleMaree"] < -45), "maree"] = "L"
df_reg_trajs.drop(columns = ["sin_courant", "cos_courant", "sin_maree", "cos_maree"], inplace =True)

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
df_reg_trajs.reset_index(drop = True, inplace = True)


# %%
def abs_diff_angle(diff):
    return abs(np.arctan2(np.sin(diff), np.cos(diff))) * 180 / np.pi


# %%
df_train = df_reg_trajs[df_reg_trajs["NCT"] % 2 == 0].reset_index(drop = True)
df_test = df_reg_trajs[df_reg_trajs["NCT"] % 2 == 1].reset_index(drop = True)


# %%
def create_values_norm(df, areax = None, areay = None, show = True, stds = None):
    if (not areax is None) & (not areay is None):
        df = df[(df["areax"] == areax) & (df["areay"] == areay)]
    if len(df) >= 2:
        x = df["x"].values
        y = df["y"].values
        dist_pos = (x[:, np.newaxis] - x) ** 2 + (y[:, np.newaxis] - y) ** 2
        l_dist_pos = dist_pos[np.triu_indices(len(dist_pos), k=1)]
        std_pos = np.std(np.sqrt(l_dist_pos)) if stds is None else stds[0]
        dist_pos_norm = dist_pos/ std_pos**2

        v = df["v"].values
        dist_v = (v[:, np.newaxis] - v) ** 2
        l_dist_v = dist_v[np.triu_indices(len(dist_v), k=1)]
        std_v = np.std(np.sqrt(l_dist_v)) if stds is None else stds[1]
        dist_v_norm = dist_v / std_v**2

        theta_v = df["theta_v"].values
        dist_theta_v = abs_diff_angle(theta_v[:, np.newaxis] - theta_v) ** 2
        l_dist_theta_v = dist_theta_v[np.triu_indices(len(dist_theta_v), k=1)]
        std_theta_v = np.std(np.sqrt(l_dist_theta_v)) if stds is None else stds[2]
        dist_theta_v_norm = dist_theta_v / std_theta_v**2

        pcG = df["pcG"].values
        dist_pcG = (pcG[:, np.newaxis] - pcG) ** 2
        l_dist_pcG = dist_pcG[np.triu_indices(len(dist_pcG), k=1)]
        std_pcG = np.std(np.sqrt(l_dist_pcG)) if stds is None else stds[3]
        dist_pcG_norm = dist_pcG / std_pcG**2

        Taille = df["Taille"].values
        dist_Taille = (Taille[:, np.newaxis] - Taille) ** 2
        l_dist_Taille = dist_Taille[np.triu_indices(len(dist_Taille), k=1)]
        std_Taille = np.std(np.sqrt(l_dist_Taille)) if stds is None else stds[4]
        dist_Taille_norm = dist_Taille / std_Taille**2 

        Rayon = df["Rayon"].values
        dist_Rayon = (Rayon[:, np.newaxis] - Rayon) ** 2
        l_dist_Rayon = dist_Rayon[np.triu_indices(len(dist_Rayon), k=1)]
        std_Rayon = np.std(np.sqrt(l_dist_Rayon)) if stds is None else stds[5]
        dist_Rayon_norm = dist_Rayon / std_Rayon**2

        vitesseCourant = df["vitesseCourant"].values
        dist_vitesseCourant = (vitesseCourant[:, np.newaxis] - vitesseCourant) ** 2
        l_dist_vitesseCourant = dist_vitesseCourant[np.triu_indices(len(dist_vitesseCourant), k=1)]
        std_vitesseCourant = np.std(np.sqrt(l_dist_vitesseCourant)) if stds is None else stds[6]
        dist_vitesseCourant_norm = dist_vitesseCourant / std_vitesseCourant**2
        
        directionCourant = df["directionCourant"].values
        dist_directionCourant = abs_diff_angle(directionCourant[:, np.newaxis] - directionCourant) ** 2
        l_dist_directionCourant = dist_directionCourant[np.triu_indices(len(dist_directionCourant), k=1)]
        std_directionCourant = np.std(np.sqrt(l_dist_directionCourant)) if stds is None else stds[7]
        dist_directionCourant_norm = dist_directionCourant / std_directionCourant**2

        angleMaree = df["angleMaree"].values
        dist_angleMaree = abs_diff_angle(angleMaree[:, np.newaxis] - angleMaree) ** 2
        l_dist_angleMaree = dist_angleMaree[np.triu_indices(len(dist_angleMaree), k=1)]
        std_angleMaree = np.std(np.sqrt(l_dist_angleMaree)) if stds is None else stds[8]
        dist_angleMaree_norm = dist_angleMaree / std_angleMaree**2

        vxp1 = df["vxp1"].values
        vyp1 = df["vyp1"].values
        diff_vp1x = vxp1[:, np.newaxis] - vxp1
        diff_vp1y = vyp1[:, np.newaxis] - vyp1 

        values_norm = [dist_pos_norm, dist_v_norm, dist_theta_v_norm, dist_pcG_norm, dist_Taille_norm, dist_Rayon_norm, dist_vitesseCourant_norm, dist_directionCourant_norm, dist_angleMaree_norm]

        if show :
            print("std_pos :", std_pos)
            print("std_v :", std_v)
            print("std_theta_v :", std_theta_v)
            print("std_pcG :", std_pcG)
            print("std_Taille :", std_Taille)
            print("std_Rayon :", std_Rayon)
            print("std_vitesseCourant :", std_vitesseCourant)    
            print("std_directionCourant :", std_directionCourant)
            print("std_angleMaree :", std_angleMaree)

        return values_norm, diff_vp1x, diff_vp1y, [std_pos, std_v, std_theta_v, std_pcG, std_Taille, std_Rayon, std_vitesseCourant, std_directionCourant, std_angleMaree]
    else :
        return None, None, None, stds


# %%
values_norm, diff_vp1x, diff_vp1y, stds = create_values_norm(df_train)


# %%
def gauss_func(params, values_norm):
    return np.exp(-1/2 * np.sum(params[:, np.newaxis, np.newaxis] * values_norm, axis=0))


# %%
def loss(params):
    g = gauss_func(params, values_norm)
    return (((g * diff_vp1x) **2 + (g * diff_vp1y) **2).sum(axis = 1)).sum()


# %%
bounds = [(0, None) for _ in range(len(values_norm))]
result = optimize.minimize(loss, [1.0] * len(values_norm), bounds=bounds, method='L-BFGS-B')
print(result)

# %%
meanings = ["Position", "Vitesse", "Angle vitesse", "%G", "Taille", "Rayon", "vitesse Courant", "directionCourant", "angle maree"]

# %%
plt.figure(figsize=(8, 5))
plt.bar(meanings, result.x)

# Ajout des titres et labels
plt.title("Valeurs des poids optimisés")
plt.ylabel("Valeur du poids")
plt.xlabel("Type de poids")
plt.xticks(rotation=45, ha='right')
# Affichage du graphique
plt.show()

# %%
n = converter.n_areas_x * converter.n_areas_y
x_values = np.arange(converter.n_areas_x)
y_values = np.arange(converter.n_areas_y)
xy_pairs = [(x, y) for x in x_values for y in y_values]

df_areas = pd.DataFrame(xy_pairs, columns=['x', 'y'])
df_areas["error"] = 0
df_areas["n_error"] = 0


# %%
def create_values_norm_area(df_train, df_test, areax, areay, stds):
    if (not areax is None) & (not areay is None):
        df = df_test[(df_test["areax"] == areax) & (df_test["areay"] == areay)]
    if len(df) >= 1:
        x_train = df_train["x"].values
        y_train = df_train["y"].values
        x = df["x"].values
        y = df["y"].values
        dist_pos = (x[:, np.newaxis] - x_train) ** 2 + (y[:, np.newaxis] - y_train) ** 2
        dist_pos_norm = dist_pos/ stds[0]**2

        v_train = df_train["v"].values
        v = df["v"].values
        dist_v = (v[:, np.newaxis] - v_train) ** 2
        dist_v_norm = dist_v / stds[1]**2

        theta_v_train = df_train["theta_v"].values
        theta_v = df["theta_v"].values
        dist_theta_v = abs_diff_angle(theta_v[:, np.newaxis] - theta_v_train) ** 2
        dist_theta_v_norm = dist_theta_v / stds[2]**2

        pcG_train = df_train["pcG"].values
        pcG = df["pcG"].values
        dist_pcG = (pcG[:, np.newaxis] - pcG_train) ** 2
        dist_pcG_norm = dist_pcG / stds[3]**2

        Taille_train = df_train["Taille"].values
        Taille = df["Taille"].values
        dist_Taille = (Taille[:, np.newaxis] - Taille_train) ** 2
        dist_Taille_norm = dist_Taille / stds[4]**2 

        Rayon_train = df_train["Rayon"].values
        Rayon = df["Rayon"].values
        dist_Rayon = (Rayon[:, np.newaxis] - Rayon_train) ** 2
        dist_Rayon_norm = dist_Rayon / stds[5]**2

        vitesseCourant_train = df_train["vitesseCourant"].values
        vitesseCourant = df["vitesseCourant"].values
        dist_vitesseCourant = (vitesseCourant[:, np.newaxis] - vitesseCourant_train) ** 2
        dist_vitesseCourant_norm = dist_vitesseCourant / stds[6]**2
        
        directionCourant_train = df_train["directionCourant"].values
        directionCourant = df["directionCourant"].values
        dist_directionCourant = abs_diff_angle(directionCourant[:, np.newaxis] - directionCourant_train) ** 2
        dist_directionCourant_norm = dist_directionCourant / stds[7]**2

        angleMaree_train = df_train["angleMaree"].values
        angleMaree = df["angleMaree"].values
        dist_angleMaree = abs_diff_angle(angleMaree[:, np.newaxis] - angleMaree_train) ** 2
        dist_angleMaree_norm = dist_angleMaree / stds[8]**2

        vxp1_train = df_train["vxp1"].values
        vyp1_train = df_train["vyp1"].values
        vxp1 = df["vxp1"].values
        vyp1 = df["vyp1"].values
        diff_vp1x = vxp1[:, np.newaxis] - vxp1_train
        diff_vp1y = vyp1[:, np.newaxis] - vyp1_train

        values_norm = [dist_pos_norm, dist_v_norm, dist_theta_v_norm, dist_pcG_norm, dist_Taille_norm, dist_Rayon_norm, dist_vitesseCourant_norm, dist_directionCourant_norm, dist_angleMaree_norm]

        return values_norm, diff_vp1x, diff_vp1y
    else :
        return None, None, None


# %%
for row in df_areas.itertuples():
    values_norm, diff_vp1x, diff_vp1y = create_values_norm_area(df_train, df_test, row.x, row.y, stds=stds)
    if not values_norm is None:
        g = gauss_func(result.x, values_norm)
        df_areas.loc[row.Index, "error"] = (((g * diff_vp1x) **2 + (g * diff_vp1y) **2).sum(axis = 1)).mean()

# %%
max_alpha = 1
min_alpha = 0.7

map = folium.Map(location=[(converter.lat_min + converter.lat_max)/2, (converter.lon_min + converter.lon_max)/2], zoom_start=11)
df_areas["polygone"] = df_areas.apply(lambda row : Polygon(converter.area2perim_lla((row.x, row.y))), axis=1)
df_areas["alpha"] = min_alpha + (max_alpha - min_alpha) * ((df_areas["n_error"] - df_areas["n_error"].min()) / (df_areas["n_error"].max() - df_areas["n_error"].min()))
df = df_areas[df_areas["error"] > 0]
colormap = cm.LinearColormap(colors=['green', 'yellow', 'red'], vmin=df["error"].min(), vmax=df["error"].max(), caption="Error (km)")
map = folium.Map(location=[(converter.lat_min + converter.lat_max)/2, (converter.lon_min + converter.lon_max)/2], zoom_start=11)
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

map.save("../results/tests4.html")

# %%

# %%
df_train
df_val
df_test

# %%
