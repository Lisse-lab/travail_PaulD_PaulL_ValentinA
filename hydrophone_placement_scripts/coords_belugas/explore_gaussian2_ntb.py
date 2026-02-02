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
import class_forecast

# %%
import importlib
importlib.reload(conv)
importlib.reload(topo)
importlib.reload(clc_mu)
importlib.reload(class_forecast)

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
df_reg_trajs.reset_index(drop = True, inplace = True)


# %%
def abs_diff_angle(diff):
    return abs(np.arctan2(np.sin(diff), np.cos(diff))) * 180 / np.pi


# %%
# ncts_train = random.sample(list(df_reg_trajs["NCT"].unique()), int(len(df_reg_trajs["NCT"].unique()) / 4))
# serie_train = df_reg_trajs.apply(lambda row : row.NCT in ncts_train, axis=1)
ind_train = random.sample(list(df_reg_trajs.index), 5)#int(len(df_reg_trajs) / 200))
serie_train = pd.Series(index=df_reg_trajs.index, dtype=bool)
for ind in serie_train.index:
    serie_train.loc[ind] = ind in ind_train
df_train = df_reg_trajs[serie_train]
# ncts_val = random.sample(list(df_reg_trajs[~serie_train]["NCT"].unique()), int(len(df_reg_trajs[~serie_train]["NCT"].unique()) / 3))
# serie_val = df_reg_trajs.apply(lambda row : row.NCT in ncts_val, axis=1)
# # df_val = df_reg_trajs[serie_val]
# df = df_reg_trajs[~serie_val]
df = df_reg_trajs[~serie_train]


# %%
def create_dists_torch(df, show = False):
    x = torch.tensor(df["x"].values)
    std_x = torch.std(x, dim=0)
    x /= std_x
    y = torch.tensor(df["y"].values)
    std_y = torch.std(y, dim=0)
    y /= std_y
    dist_pos = (x.unsqueeze(0) - x.unsqueeze(1)) ** 2 + (y.unsqueeze(0) - y.unsqueeze(1)) ** 2

    v = torch.tensor(df["v"].values)
    std_v = torch.std(v, dim=0)
    v /= std_v
    dist_v = (v.unsqueeze(0) - v.unsqueeze(1)) ** 2

    cos_theta_v = torch.tensor(df["cos_theta_v"].values)
    std_cos_theta_v = torch.std(cos_theta_v, dim=0)
    cos_theta_v /= np.sqrt(2)*std_cos_theta_v
    sin_theta_v = torch.tensor(df["sin_theta_v"].values)
    std_sin_theta_v = torch.std(sin_theta_v, dim=0)
    sin_theta_v /= np.sqrt(2)*std_sin_theta_v
    dist_theta_v = (cos_theta_v.unsqueeze(0) - cos_theta_v.unsqueeze(1)) ** 2 + (sin_theta_v.unsqueeze(0) - sin_theta_v.unsqueeze(1)) ** 2

    pcG = torch.tensor(df["pcG"].values)
    std_pcG = torch.std(pcG, dim=0)
    pcG /= std_pcG
    dist_pcG = (pcG.unsqueeze(0) - pcG.unsqueeze(1)) ** 2

    Taille = torch.tensor(df["Taille"].values)
    std_Taille = torch.std(Taille, dim=0)
    Taille /= std_Taille
    dist_Taille = (Taille.unsqueeze(0) - Taille.unsqueeze(1)) ** 2
    
    Rayon = torch.tensor(df["Rayon"].values)
    std_Rayon = torch.std(Rayon, dim=0)
    Rayon /= std_Rayon
    dist_Rayon = (Rayon.unsqueeze(0) - Rayon.unsqueeze(1)) ** 2
    
    vitesseCourant = torch.tensor(df["vitesseCourant"].values)
    std_vitesseCourant = torch.std(vitesseCourant, dim=0)
    vitesseCourant /= std_vitesseCourant
    dist_vitesseCourant = (vitesseCourant.unsqueeze(0) - vitesseCourant.unsqueeze(1)) ** 2
    
    cos_directionCourant = torch.tensor(df["cos_directionCourant"].values)
    std_cos_directionCourant = torch.std(cos_directionCourant, dim=0)
    cos_directionCourant /= np.sqrt(2)*std_cos_directionCourant
    sin_directionCourant = torch.tensor(df["sin_directionCourant"].values)
    std_sin_directionCourant = torch.std(sin_directionCourant, dim=0)
    sin_directionCourant /= np.sqrt(2)*std_sin_directionCourant
    dist_directionCourant = (cos_directionCourant.unsqueeze(0) - cos_directionCourant.unsqueeze(1)) ** 2 + (sin_directionCourant.unsqueeze(0) - sin_directionCourant.unsqueeze(1)) ** 2

    cos_maree = torch.tensor(df["cos_maree"].values)
    std_cos_maree = torch.std(cos_maree, dim=0)
    cos_maree /= np.sqrt(2)*std_cos_maree
    sin_maree = torch.tensor(df["sin_maree"].values)
    std_sin_maree = torch.std(sin_maree, dim=0)
    sin_maree /= np.sqrt(2)*std_sin_maree
    dist_maree = (cos_maree.unsqueeze(0) - cos_maree.unsqueeze(1)) ** 2 + (sin_maree.unsqueeze(0) - sin_maree.unsqueeze(1)) ** 2

    vxp1 = torch.tensor(df["vxp1"].values)
    vyp1 = torch.tensor(df["vyp1"].values)
    diff_vp1 = torch.stack((vxp1 - vxp1.mean(), vyp1 - vyp1.mean()), dim=0)

    dists = torch.stack((dist_pos, dist_v, dist_theta_v, dist_pcG, dist_Taille, dist_Rayon, dist_vitesseCourant, dist_directionCourant, dist_maree), dim=0)

    if show :
        print("std_x :", std_x)
        print("std_y :", std_y)
        print("std_v :", std_v)
        print("std_cos_theta_v :", std_cos_theta_v)
        print("std_sin_theta_v :", std_sin_theta_v)
        print("std_pcG :", std_pcG)
        print("std_Taille :", std_Taille)
        print("std_Rayon :", std_Rayon)
        print("std_vitesseCourant :", std_vitesseCourant)    
        print("std_cos_directionCourant :", std_cos_directionCourant)
        print("std_sin_directionCourant :", std_sin_directionCourant)
        print("std_cos_angleMaree :", std_cos_maree)
        print("std_sin_angleMaree :", std_sin_maree)

    return dists, diff_vp1


# %%
dists, diff_vp1 = create_dists_torch(df_train, show=True)

# %%
params = torch.tensor([1, 1e-2] + [1. for i in range(len(dists))], dtype=torch.double)


# %%
def get_Sigma_torch(params, dists):
    tens_params = torch.tensor(params, dtype=torch.double)
    K = params[0] * torch.exp(-1/2 * torch.einsum('i,ijk->jk', tens_params[2:], dists))
    return K + params[1] * torch.eye(K.shape[0], dtype=torch.double)


# %%
def neg_log_likelihood_torch(params):
    Sigma = get_Sigma_torch(params, dists)
    return (diff_vp1 @torch.linalg.inv(Sigma)@diff_vp1.T).trace() + np.log(torch.linalg.det(Sigma))


# %%
ind_train = random.sample(list(df_reg_trajs.index), 500)
serie_train = pd.Series(index=df_reg_trajs.index, dtype=bool)
for ind in serie_train.index:
    serie_train.loc[ind] = ind in ind_train
df_train = df_reg_trajs[serie_train]
df_test = df_reg_trajs[~serie_train]

# %%
dists, diff_vp1 = create_dists_torch(df_train)

# %%
params = np.array([1, 1e-2] + [1. for _ in range(len(dists))])
bounds = [(0, None) for _ in range(2+len(dists))]
result = optimize.minimize(neg_log_likelihood_torch, params, bounds=bounds, method='L-BFGS-B')
print(result)

# %%
meanings = ["Sigma", "Sigmaf", "Position", "Vitesse", "Angle vitesse", "%G", "Taille", "Rayon", "vitesse Courant", "directionCourant", "angle maree"]
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
def create_dists_test(df_train, df, areax, areay, show = False):
    df = df[(df["areax"] == areax) & (df["areay"] == areay)]

    x_train = torch.tensor(df_train["x"].values)
    std_x_train = torch.std(x_train, dim=0)
    x_train /= std_x_train
    y_train = torch.tensor(df_train["y"].values)
    std_y_train = torch.std(y_train, dim=0)
    y_train /= std_y_train
    x_test = torch.tensor(df["x"].values)
    x_test /= std_x_train
    y_test = torch.tensor(df["y"].values)
    y_test /= std_y_train
    dist_pos = (x_test.unsqueeze(0) - x_train.unsqueeze(1)) ** 2 + (y_test.unsqueeze(0) - y_train.unsqueeze(1)) ** 2

    v_train = torch.tensor(df_train["v"].values)
    std_v_train = torch.std(v_train, dim=0)
    v_train /= std_v_train
    v_test = torch.tensor(df["v"].values)
    v_test /= std_v_train
    dist_v = (v_test.unsqueeze(0) - v_train.unsqueeze(1)) ** 2

    cos_theta_v_train = torch.tensor(df_train["cos_theta_v"].values)
    std_cos_theta_v_train = torch.std(cos_theta_v_train, dim=0)
    cos_theta_v_train /= np.sqrt(2)*std_cos_theta_v_train
    sin_theta_v_train = torch.tensor(df_train["sin_theta_v"].values)
    std_sin_theta_v_train = torch.std(sin_theta_v_train, dim=0)
    sin_theta_v_train /= np.sqrt(2)*std_sin_theta_v_train
    cos_theta_v_test = torch.tensor(df["cos_theta_v"].values)
    cos_theta_v_test /= np.sqrt(2)*std_cos_theta_v_train
    sin_theta_v_test = torch.tensor(df["sin_theta_v"].values)
    sin_theta_v_test /= np.sqrt(2)*std_sin_theta_v_train
    dist_theta_v = (cos_theta_v_test.unsqueeze(0) - cos_theta_v_train.unsqueeze(1)) ** 2 + (sin_theta_v_test.unsqueeze(0) - sin_theta_v_train.unsqueeze(1)) ** 2

    pcG_train = torch.tensor(df_train["pcG"].values)
    std_pcG_train = torch.std(pcG_train, dim=0)
    pcG_train /= std_pcG_train
    pcG_test = torch.tensor(df["pcG"].values)
    pcG_test /= std_pcG_train
    dist_pcG = (pcG_test.unsqueeze(0) - pcG_train.unsqueeze(1)) ** 2
    
    Taille_train = torch.tensor(df_train["Taille"].values)
    std_Taille_train = torch.std(Taille_train, dim=0)
    Taille_train /= std_Taille_train
    Taille_test = torch.tensor(df["Taille"].values)
    Taille_test /= std_Taille_train
    dist_Taille = (Taille_test.unsqueeze(0) - Taille_train.unsqueeze(1)) ** 2
    
    Rayon_train = torch.tensor(df_train["Rayon"].values)
    std_Rayon_train = torch.std(Rayon_train, dim=0)
    Rayon_train /= std_Rayon_train
    Rayon_test = torch.tensor(df["Rayon"].values)
    Rayon_test /= std_Rayon_train
    dist_Rayon = (Rayon_test.unsqueeze(0) - Rayon_train.unsqueeze(1)) ** 2
    
    vitesseCourant_train = torch.tensor(df_train["vitesseCourant"].values)
    std_vitesseCourant_train = torch.std(vitesseCourant_train, dim=0)
    vitesseCourant_train /= std_vitesseCourant_train
    vitesseCourant_test = torch.tensor(df["vitesseCourant"].values)
    vitesseCourant_test /= std_vitesseCourant_train
    dist_vitesseCourant = (vitesseCourant_test.unsqueeze(0) - vitesseCourant_train.unsqueeze(1)) ** 2
    
    cos_directionCourant_train = torch.tensor(df_train["cos_directionCourant"].values)
    std_cos_directionCourant_train = torch.std(cos_directionCourant_train, dim=0)
    cos_directionCourant_train /= np.sqrt(2)*std_cos_directionCourant_train
    sin_directionCourant_train = torch.tensor(df_train["sin_directionCourant"].values)
    std_sin_directionCourant_train = torch.std(sin_directionCourant_train, dim=0)
    sin_directionCourant_train /= np.sqrt(2)*std_sin_directionCourant_train
    cos_directionCourant_test = torch.tensor(df["cos_directionCourant"].values)
    cos_directionCourant_test /= np.sqrt(2)*std_cos_directionCourant_train
    sin_directionCourant_test = torch.tensor(df["sin_directionCourant"].values)
    sin_directionCourant_test /= np.sqrt(2)*std_sin_directionCourant_train
    dist_directionCourant = (cos_directionCourant_test.unsqueeze(0) - cos_directionCourant_train.unsqueeze(1)) ** 2 + (sin_directionCourant_test.unsqueeze(0) - sin_directionCourant_train.unsqueeze(1)) ** 2

    cos_maree_train = torch.tensor(df_train["cos_maree"].values)
    std_cos_maree_train = torch.std(cos_maree_train, dim=0)
    cos_maree_train /= np.sqrt(2)*std_cos_maree_train
    sin_maree_train = torch.tensor(df_train["sin_maree"].values)
    std_sin_maree_train = torch.std(sin_maree_train, dim=0)
    sin_maree_train /= np.sqrt(2)*std_sin_maree_train
    cos_maree_test = torch.tensor(df["cos_maree"].values)
    cos_maree_test /= np.sqrt(2)*std_cos_maree_train
    sin_maree_test = torch.tensor(df["sin_maree"].values)
    sin_maree_test /= np.sqrt(2)*std_sin_maree_train
    dist_maree = (cos_maree_test.unsqueeze(0) - cos_maree_train.unsqueeze(1)) ** 2 + (sin_maree_test.unsqueeze(0) - sin_maree_train.unsqueeze(1)) ** 2

    vxp1_train = torch.tensor(df_train["vxp1"].values)
    vyp1_train = torch.tensor(df_train["vyp1"].values)
    vxp1_test = torch.tensor(df["vxp1"].values)
    vyp1_test = torch.tensor(df["vyp1"].values)
    diff_vp1 = torch.stack((vxp1_train - vxp1_train.mean(), vyp1_train - vyp1_train.mean()), dim=0)

    dists = torch.stack((dist_pos, dist_v, dist_theta_v, dist_pcG, dist_Taille, dist_Rayon, dist_vitesseCourant, dist_directionCourant, dist_maree), dim=0)

    return dists, diff_vp1


# %%
dists, diff_vp1 = create_dists_test(df_train, df_test, 32,47)

# %%
dists.shape
