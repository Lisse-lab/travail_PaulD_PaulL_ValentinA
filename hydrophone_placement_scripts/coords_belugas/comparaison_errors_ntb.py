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
sys.path.append("../..")

# %%
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

# %%
import hydrophone_placement_scripts.utils_scripts.conversions_coordinates as conv
import hydrophone_placement_scripts.to_optimise.topo as tp
import hydrophone_placement_scripts.coords_belugas.calc_mu as clc_mu
import hydrophone_placement_scripts.coords_belugas.regression as regr
import hydrophone_placement_scripts.coords_belugas.forecasting_errors as fe

# %%
import importlib
importlib.reload(conv)
importlib.reload(clc_mu)
importlib.reload(regr)
importlib.reload(fe)

# %%
args = {
    "lat_min" : 47.65,
    "lat_max" : 48.07,
    "lon_min" : -70.04,
    "lon_max" : -69.28,
    "width_area" : 1000,
    "depth_area" : 2
}
geotiff_path = "../datas/BelugaRelativeDens/BelugaRelativeDens.tif"
step = 100

converter = conv.Conv(**args)
calc_mu = clc_mu.Calc_mu(geotiff_path, step)
topo = tp.Topo(converter, new_dic_depths=True, substrat=False, save=False)

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
df_trajs = df_trajs[serie_in_map & serie_in_area].reset_index(drop=True)

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
step_err = 2 # step_reg / step_err must be an int
df_reg_trajs = regr.create_df_newtrajs(step_reg, step_err, df_trajs, converter, other_columns=["pcG", "Taille", "Rayon", "cos_directionCourant", "sin_directionCourant", "cos_maree", "sin_maree", "vitesseCourant"])

# %%
df_reg_trajs[["areax", "areay"]] = pd.Series(converter.utm2area(df_reg_trajs["x"], df_reg_trajs["y"]))

# %%
df_reg_trajs["directionCourant"] = np.arctan2(df_reg_trajs["sin_directionCourant"], df_reg_trajs["cos_directionCourant"]) * 180/np.pi
df_reg_trajs["angleMaree"] = np.arctan2(df_reg_trajs["sin_maree"], df_reg_trajs["cos_maree"]) * 180/np.pi
df_reg_trajs["maree"] = ""
df_reg_trajs.loc[(-45 <= df_reg_trajs["angleMaree"]) & (df_reg_trajs["angleMaree"] < 45), "maree"] = "R"
df_reg_trajs.loc[(45 <= df_reg_trajs["angleMaree"]) & (df_reg_trajs["angleMaree"] < 135), "maree"] = "H"
df_reg_trajs.loc[(135 <= df_reg_trajs["angleMaree"]) | (df_reg_trajs["angleMaree"] < -135), "maree"] = "E"
df_reg_trajs.loc[(-135 <= df_reg_trajs["angleMaree"]) & (df_reg_trajs["angleMaree"] < -45), "maree"] = "L"

# %%
#df_reg_trajs[["v", "theta_v"]] = [0., 0.]

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

# %%
df_reg_trajs["cos_theta_v"] = np.cos(df_reg_trajs["theta_v"] * np.pi/180)
df_reg_trajs["sin_theta_v"] = np.sin(df_reg_trajs["theta_v"] * np.pi/180)

# %%
df_reg_trajs["depth"] = df_reg_trajs.apply(lambda row : topo.dic_depths[(row.areax, row.areay)], axis=1)

# %%
df_reg_trajs.reset_index(drop = True, inplace = True)

# %%
l_ncts = list(df_reg_trajs["NCT"].unique())
len(l_ncts)

# %%
part_test = 1/2

# %%
ncts = random.sample(l_ncts, int(len(df_reg_trajs["NCT"].unique()) * part_test))
df_train = df_reg_trajs[df_reg_trajs["NCT"].apply(lambda nct : nct in ncts)]
df_test =  df_reg_trajs[df_reg_trajs["NCT"].apply(lambda nct : not nct in ncts)]

# %%
mod_test = fe.forecasting_model(df_train, converter, True)
mod_test.set_df_test(df_test)
mod_test.create_df_areas()
df_areas = mod_test.df_areas

# %%
mod_test.create_repartition_map()

# %%
l_rmse = []

# %%
for b_mean in [True, False]:
    for b_cat in [True, False]:
        gm = fe.gaussian_model(df_train, converter, mean = b_mean, beluga_features = True, categories = b_cat)
        gm.train(100, True)
        gm.create_batches()
        l_rmse.append(gm.test(df_test))
        df_areas[gm.title] = gm.get_rmse(df_areas[["x", "y", "number_train", "number_test", "polygone"]])
        print("done")

# %%
gm = fe.gaussian_model(df_train, converter, beluga_features = False)
gm.train(100, True)
gm.create_batches()
l_rmse.append(gm.test(df_test))
df_areas[gm.title] = gm.get_rmse(df_areas[["x", "y", "number_train", "number_test", "polygone"]])
print("done")

# %%
for b_cat in [True, False]:
    rf = fe.random_forest(df_train, converter, beluga_features = True, categories = b_cat)
    rf.train()
    rf.show_importance()
    l_rmse.append(rf.test(df_test))
    df_areas[rf.title] = rf.get_rmse(df_areas[["x", "y", "number_train", "number_test", "polygone"]])
    print("done")

# %%
rf = fe.random_forest(df_train, converter, beluga_features = False)
rf.train()
rf.show_importance()
l_rmse.append(rf.test(df_test))
df_areas[rf.title] = rf.get_rmse(df_areas[["x", "y", "number_train", "number_test", "polygone"]])
print("done")

# %%
mod_ou = fe.ou_model(df_train, converter, calc_mu, False)
mod_ou.train()
l_rmse.append(mod_ou.test(df_test))
df_areas[mod_ou.title] = mod_ou.get_rmse(df_areas[["x", "y", "number_train", "number_test", "polygone"]])

# %%
mod_pers = fe.pers_model(df_train, converter)
l_rmse.append(mod_pers.test(df_test))
df_areas[mod_pers.title] = mod_pers.get_rmse(df_areas[["x", "y", "number_train", "number_test", "polygone"]])

# %%
corr = df_areas[["number_train", "number_test"] + list(df_areas.columns[5:])].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Corrélation entre les erreurs des modèles")
plt.show()

# %%
sns.pairplot(df_areas[["number_train", "number_test"] + list(df_areas.columns[5:])])
plt.suptitle("Matrice de scatterplots des erreurs par modèle", y=1.02)
plt.show()

# %%
plt.figure(figsize=(8, 5))
plt.bar(list(df_areas.columns[5:]), l_rmse)
plt.title("RMSEs")
plt.ylabel("RMSE (m/s)")
plt.xlabel("Model")
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
l_rmse

# %%
mod_best = df_areas.columns[5:][np.argmin(l_rmse)]
mod_best

# %%
df_areas.rename(columns = {mod_best : "forecasting_error"}, inplace = True)

# %%
known_points = df_areas.dropna(subset=["forecasting_error"])[["x", "y"]].values
known_values = df_areas.dropna(subset=["forecasting_error"])["forecasting_error"].values

missing_points = df_areas[df_areas["forecasting_error"].isna()][["x", "y"]].values

# %%
from scipy.interpolate import griddata

interpolated_values = griddata(known_points, known_values, missing_points, method="linear")

df_areas.loc[df_areas["forecasting_error"].isna(), "forecasting_error"] = interpolated_values

# %%
df_areas[df_areas["forecasting_error"].isna(), "forecasting_error"] = df_areas["forecasting_error"].max()

# %%
df_areas[["polygone", "forecasting_error"]].to_csv("../datas/for_model/df_forecasting_errors.csv", sep=";")
