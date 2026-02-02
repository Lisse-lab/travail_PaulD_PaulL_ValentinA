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
import pandas as pd
import numpy as np
import random
import class_forecast
import importlib
import calc_mu
from scipy import optimize
import optuna
import folium
import regression
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# %%
importlib.reload(class_forecast)
importlib.reload(calc_mu)

# %%
geotiff_path = 'BelugaRelativeDens\BelugaRelativeDens.tif'
step = 100
#geotiff_path = "MedianKernelSum_allver_visual_st01st02_1990-2022_JulySept_sansZero_normalized_20250514.tiff"
#step = 1000

# %%
lat_min = 47.65
lat_max = 48.07
lon_min = -70.04
lon_max = -69.28
utm_zone = 19
from pyproj import Transformer
transformer_utm = Transformer.from_crs("EPSG:4326", f"+proj=utm +zone={utm_zone} +ellps=WGS84", always_xy=True)
x_min, y_min = transformer_utm.transform(lon_min, lat_min)
x_max, y_max = transformer_utm.transform(lon_max, lat_max)

# %%
calc = calc_mu.Calc_mu(geotiff_path, step)


# %%
def sep_if_not_in_area(df, nct):
    year = df["Année"][df["NCT"] == nct].mean()
    bool_new_nct = False
    for i in df["Année"][df["NCT"] == nct].index:
        if bool_new_nct:
            df.loc[i, "NCT"] = new_nct
        if (not ((0 <= df.loc[i, "x"] <= x_max-x_min) & (0 <= df.loc[i, "y"] <= y_max - y_min))) | ((df.loc[i, "long"]<-69.68227817588027) & (df.loc[i, "lat"]>48.11295396835736)):
            new_nct = df["NCT"][df["Année"] == year].max() + 1
            df.drop(i, inplace = True)
            bool_new_nct = True


# %%
def create_dfs():
    df = pd.read_csv("clean_coords", sep = ";").drop("Unnamed: 0", axis = 1)
    for nct in df["NCT"].unique():
        sep_if_not_in_area(df, nct)
    for nct in df["NCT"].unique():
        l = df[df["NCT"] == nct].index
        if len(l) == 1:
            df.drop(l[0], inplace = True)
    l_ncts = list(df["NCT"].unique())
    ncts = random.sample(l_ncts, int(len(df["NCT"].unique()) / 10))
    df_train1 = df[df["NCT"].apply(lambda nct : nct in ncts)]
    df2 =  df[df["NCT"].apply(lambda nct : not nct in ncts)]

    l_ncts2 = list(df2["NCT"].unique())
    ncts2 = random.sample(l_ncts2, int(len(df2["NCT"].unique()) / 2))
    df_train2 = df2[df2["NCT"].apply(lambda nct : nct in ncts2)]
    df_test =  df2[df2["NCT"].apply(lambda nct : not nct in ncts2)]
    return df_train1, df_train2, df_test


# %%
#df_train1, df_train2, df_test = create_dfs()
#df_train1.to_csv("df_train1.csv", sep = ";")
#df_train2.to_csv("df_train2.csv", sep = ";")
#df_test.to_csv("df_test.csv", sep = ";")

# %%
df_train1, df_train2, df_test = create_dfs()

# %%
len(df_train1), len(df_train2), len(df_test)

# %%
df_train1 = pd.read_csv("df_train1.csv", sep = ";")
df_train2 = pd.read_csv("df_train2.csv", sep = ";")
df_test = pd.read_csv("df_test.csv", sep = ";")

# %%
len(df_train1), len(df_train2), len(df_test)


# %%
def create_dfs2():
    df = pd.read_csv("clean_coords", sep = ";").drop("Unnamed: 0", axis = 1)
    for nct in df["NCT"].unique():
        sep_if_not_in_area(df, nct)
    for nct in df["NCT"].unique():
        l = df[df["NCT"] == nct].index
        if len(l) == 1:
            df.drop(l[0], inplace = True)
    l_ncts = list(df["NCT"].unique())
    ncts = random.sample(l_ncts, int(len(df["NCT"].unique()) / 4))
    df_train1 = df[df["NCT"].apply(lambda nct : nct in ncts)]
    df2 =  df[df["NCT"].apply(lambda nct : not nct in ncts)]

    l_ncts2 = list(df2["NCT"].unique())
    ncts2 = random.sample(l_ncts2, int(len(df2["NCT"].unique()) / 3))
    df_train2 = df2[df2["NCT"].apply(lambda nct : nct in ncts2)]
    df_test =  df2[df2["NCT"].apply(lambda nct : not nct in ncts2)]
    return df_train1, df_train2, df_test


# %%
df_train1, df_train2, df_test = create_dfs2()

# %%
len(df_train1), len(df_train2), len(df_test)

# %%
delta2 = 200 #(d/np.sqrt(2))^2

# %%
sigma2 = class_forecast.optimise_sigma2(df_train1, delta2)
sigma2

# %%
calc.set_sigma2(sigma2)

# %%
inv_tau0 = class_forecast.get_inv_tau0(df_train1, calc.mu)
inv_tau0

# %%
deltav2 = class_forecast.get_deltav2(df_train1)
deltav2

# %%
sigmav2 = class_forecast.get_sigmav2(df_train2, inv_tau0, calc)
sigmav2

# %%
importlib.reload(class_forecast)
importlib.reload(regression)
importlib.reload(calc_mu)
calc = calc_mu.calc_mu(geotiff_path, x_min , y_min, step, sigma2 = sigma2)

# %%
transformer_wgs = Transformer.from_crs(f"+proj=utm +zone={utm_zone} +ellps=WGS84", "EPSG:4326", always_xy=True)
def to_wgs(x, y):
    return transformer_wgs.transform(x + x_min, y + y_min)


# %%
def test(nct, df, bool_map, timestep = np.inf, name = "ma_carte", plot_color="red"):
    assert timestep > 0, "the timestep must be positive" 
    l = df[df["NCT"] == nct].index
    traj = class_forecast.forecast_trajectory(df["x"].loc[l[0]], df["y"].loc[l[0]], delta2, deltav2, inv_tau0, sigmav2, calc, to_wgs, bool_map, name, plot_color)
    l_se = []
    l_ae = []
    for i in range (l[0], l[-1]):
        deltat = df["Time"].loc[i+1] - df["Time"].loc[i]
        while deltat > timestep:
            _ = traj.predict(timestep)
            deltat -= timestep
        x = traj.predict(deltat)
        x_o = (df["x"].loc[i+1], df["y"].loc[i+1])
        l_se.append((x[0] - x_o[0]) ** 2 + (x[1] - x_o[1]) ** 2)
        l_ae.append(np.sqrt((x[0] - x_o[0]) ** 2 + (x[1] - x_o[1]) ** 2))
        _ = traj.set_to_obs(x_o[0], x_o[1])

    return (l_se, l_ae)



# %%
def create_l_mse(df, bool_timestep, timestep_min = np.inf):
    l_ncts = df["NCT"].unique()
    l_mse = []
    for nct in l_ncts:
        if bool_timestep:
            df_traj2 = regression.create_newtraj(timestep_min, df, nct, to_wgs)
            mse, _ = test(nct, df_traj2, False)
        else:
            mse, _ = test(nct, df, False)
        
        l_mse += mse
    return l_mse


# %%
def test_v(nct, df, bool_map, timestep = np.inf, name = "ma_carte", plot_color="red"):
    assert timestep > 0, "the timestep must be positive" 
    l = df[df["NCT"] == nct].index
    n = len(l) - 1
    traj = class_forecast.forecast_trajectory(df["x"].loc[l[0]], df["y"].loc[l[0]], delta2, deltav2, inv_tau0, sigmav2, calc, to_wgs, bool_map, name, plot_color)
    l_se = []
    l_ae = []
    for i in range (l[0], l[-1]):
        deltat = df["Time"].loc[i+1] - df["Time"].loc[i]
        while deltat > timestep:
            _ = traj.predict(timestep)
            deltat -= timestep
        x = traj.predict(deltat)
        v_o = ((df["x"].loc[i+1]-df["x"].loc[i])/deltat, (df["y"].loc[i+1]-df["y"].loc[i])/deltat)
        l_se.append((x[2] - v_o[0]) ** 2 + (x[3] - v_o[1]) ** 2)
        l_ae.append(np.sqrt((x[2] - v_o[0]) ** 2 + (x[3] - v_o[1]) ** 2))
        _ = traj.set_to_obs(df["x"].loc[i+1], df["y"].loc[i+1])

    return (l_se, l_ae)



# %%
def create_l_mse_v(df, bool_timestep, timestep_min = np.inf):
    l_ncts = df["NCT"].unique()
    l_rse = []
    for nct in l_ncts:
        if bool_timestep:
            df_traj2 = regression.create_newtraj(timestep_min, df, nct, to_wgs)    
            rse, _ = test_v(nct, df_traj2, False)
        else:
            rse, _ = test_v(nct, df, False)
        
        l_rse += rse
    return l_rse


# %%
l_rse_old = create_l_rse(df_test, False)
l_rse_resampled = create_l_rse(df_test, True, 15)

indices = list(range(len(l_rse_old)))

plt.scatter(indices, l_rse_old, color='red', label='Old RMSE of NCT', s=5)
plt.scatter(indices, l_rse_resampled, color='blue', label='Resampled RMSE of NCT', s= 5)

plt.axhline(y=np.mean(l_rse_old), color='red', linewidth=2, label=f'Old RMSE: {np.mean(l_rse_old):.3f}')
plt.axhline(y=np.mean([e for e in l_rse_resampled if e is not None]), color='blue', linewidth=2, label=f'Resampled RMSE: {np.mean([e for e in l_rse_resampled if e is not None]):.3f}')

plt.ylabel('RMSE')
plt.title('RMSE for each NCT')
plt.legend()
plt.yscale('log')
plt.show()

# %%
l = np.linspace(5, 30, 26)
dic_mse = {}
for i in l:
    dic_mse[i] = create_l_mse(df_test, True, i)

# %%
l_reshaped = 60 * (np.array(l)).reshape(-1, 1)
l_reg = np.linspace(min(l), max(l), 100).reshape(-1, 1)

l_mse = [np.mean(dic_mse[i]) for i in l]
l_med_se = [np.median(dic_mse[i]) for i in l]

model_2 = LinearRegression(fit_intercept=False)
model_2.fit(l_reshaped**2, l_mse)
errs_reg_2 = model_2.predict((60*l_reg)**2)
print("Error of quadratic regression :", np.mean(np.square(l_mse - model_2.predict(l_reshaped**2))))

model_3 = LinearRegression(fit_intercept=False)
model_3.fit(l_reshaped**3, l_mse)
errs_reg_3 = model_3.predict((60*l_reg)**3)
print("Error of cubic regression :", np.mean(np.square(l_mse - model_3.predict(l_reshaped**3))))

model_med = LinearRegression(fit_intercept=False)
model_med.fit(l_reshaped**3, l_med_se)
errs_reg_med = model_med.predict((60*l_reg)**3)
print("Error of cubic regression :", np.mean(np.square(l_med_se - model_med.predict(l_reshaped**3))))

#model_sqrt = LinearRegression(fit_intercept=False)
#model_sqrt.fit(l_reshaped**0.5, l_rmse)
#errs_reg_sqrt = model_sqrt.predict((60*l_reg)**0.5)
#print("Error of sqrt regression :", np.mean(np.square(l_rmse - model_sqrt.predict(l_reshaped**0.5))))

# %%
fig, ax1 = plt.subplots(figsize=(10, 6))

boxplot = ax1.boxplot(dic_mse.values(), positions=dic_mse.keys(), showfliers=False, label = "Boxplot")
for box in boxplot['boxes']:
    box.set(color='red', linewidth=2) 
for median in boxplot['medians']:
    median.set(color='red', linewidth=2)
for whisker in boxplot['whiskers']:
    whisker.set(color='red', linewidth=2)
for cap in boxplot['caps']:
    cap.set(color='red', linewidth=2)

ax1.scatter(l, l_mse, label = "MSE", s= 50, zorder=3)
ax1.plot(l_reg, errs_reg_3, color="green", linewidth = 3, label=f'Cubic Regression of MSE (coef:{np.round(model_3.coef_[0], 3)}m²/s⁻³)')
#plt.plot(l_reg, errs_reg_med, color="orange", linewidth = 3, label=f'Cubic Regression of MSE')

ax1.set_xlabel('Timestep (min)')
ax1.set_ylabel('MSE (m²)')
ax1.set_xticks(l, labels=[str(int(e)) for e in l])
plt.title('Square error of position according timestep')
ax1.set_ylim(bottom = 0)
ax1.set_xlim(left=0)
ax1.grid(True, which='both', axis='y')

ax2 = ax1.twinx()
yticks_left = ax1.get_yticks()
ax2.set_yticks(yticks_left)
ax2.set_yticklabels([f"{np.sqrt(val):.2f}" for val in yticks_left])
ax2.set_ylabel('RMSE (m)')
ax2.set_ylim(ax1.get_ylim())

ax1.legend()
plt.show()


# %%

# %%
len(df_test["NCT"].unique())

# %%
l = np.linspace(5, 30, 26)
dic_mse_v = {}
for i in l:
    dic_mse_v[i] = create_l_mse_v(df_test, True, i)

# %%
l_reshaped = 60 * (np.array(l)).reshape(-1, 1)
l_reg = np.linspace(min(l), max(l), 100).reshape(-1, 1)

l_mse_v = [np.mean(dic_mse_v[i]) for i in l]

model_v = LinearRegression(fit_intercept=False)
model_v.fit(l_reshaped, l_mse_v)
errs_reg_v = model_v.predict(60 * l_reg)
np.mean(np.square(l_mse_v - model_v.predict(l_reshaped)))


# %%
def get_var(df):
    v = 0
    n = 0
    for nct in df["NCT"].unique():
        l = df[df["NCT"] == nct].index
        for i in range (l[0], l[-1]):
            deltat = df["Time"].loc[i+1] - df["Time"].loc[i]
            v_o = ((df["x"].loc[i+1] - df["x"].loc[i])/deltat, (df["y"].loc[i+1] - df["y"].loc[i])/deltat)
            v += np.sum(np.square(v_o))
            n += 1
    return v/n


# %%
variance = get_var(df_test)
variance

# %%
fig, ax1 = plt.subplots(figsize=(10, 6))

boxplot = ax1.boxplot(dic_mse_v.values(), positions=dic_mse_v.keys(), showfliers=False, label = "Boxplot")
for box in boxplot['boxes']:
    box.set(color='red', linewidth=2) 
for median in boxplot['medians']:
    median.set(color='red', linewidth=2)
for whisker in boxplot['whiskers']:
    whisker.set(color='red', linewidth=2)
for cap in boxplot['caps']:
    cap.set(color='red', linewidth=2)

ax1.scatter(l, l_mse_v, label = "MSE", s= 50, zorder=3)
ax1.plot(l_reg, errs_reg_v, color="green", linewidth = 3, label=f'Linear Regression of MSE (coef:{np.round(model_v.coef_[0], 4)}m²/s⁻³)')
#plt.plot(l_reg, errs_reg_med, color="orange", linewidth = 3, label=f'Cubic Regression of median of square error')
ax1.axhline(y=variance, color='orange', linewidth=2, label=f'Variance of velocity')

ax1.set_xlabel('Timestep (min)')
ax1.set_ylabel('MSE ((m/s)²)')
ax1.set_xticks(l, labels=[str(int(e)) for e in l])
plt.title('Square error of velocity according timestep')
ax1.set_ylim(bottom = 0)
ax1.set_xlim(left=0)

ax2 = ax1.twinx()
yticks_left = ax1.get_yticks()
ax2.set_yticks(yticks_left)
ax2.set_yticklabels([f"{np.sqrt(val):.2f}" for val in yticks_left])
ax2.set_ylabel('RMSE (m/s)')
ax2.set_ylim(ax1.get_ylim())

ax1.grid(True)
ax1.legend()
plt.show()

# %%
5*60*0.0012

# %%
len(df_test)

# %%
len(df_test["NCT"].unique())

# %%
plt.scatter(l, l_rmse_v, label='RMSE')
plt.ylabel('RMSE (m/s)')
#plt.plot(l_reg, errs_reg_1_v, color="red", label=f'Linear Regression')
#plt.plot(l_reg, errs_reg_15_v, color="green", label=f'1.5 Regression')
plt.plot(l_reg, errs_reg_sqrt_v, color="orange", label=f'Square Root Regression')
plt.xlabel('Timestep (min)')
plt.title('RMSE of velocity according timestep')
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.legend()
plt.show()

# %%
plt.scatter(l, 60*l*l_rmse_v, label='timestep * RMSE of velocity')
plt.scatter(l, l_rmse, label='RMSE of position')
plt.title('Comparison of the RMSEs between velocity and position')
plt.legend()
plt.xlabel('Timestep (min)')
plt.ylabel('RMSE (m)')
plt.ylim(bottom=0)
plt.xlim(left=0)


# %%

# %%

# %%

# %%
def func_to_optimise(timestep_any_type):
    if isinstance(timestep_any_type, np.ndarray):
        timestep = timestep_any_type[0]
    else:
        timestep = timestep_any_type
    
    l_ncts = df_train2["NCT"].unique()
    se_tot = 0

    for nct in l_ncts:
        se, _ = test(nct, df_train2, False, timestep)
        se_tot += se
    return se_tot/len(l_ncts) #np.sqrt(se_tot/(len(l_ncts)))


# %%
l_ncts = df_test["NCT"].unique()
se_tot = 0
ae_tot = 0

for nct in l_ncts:
    se, ae = test(nct, df_test, False, ideal_timestep)
    se_tot += se
    ae_tot += ae

print("MSE :", se_tot/len(l_ncts))
print("RMSE :", np.sqrt(se_tot/len(l_ncts)))
print("MAE :", ae_tot/len(l_ncts))

# %%
func_to_optimise(1711)

# %%
np.sqrt(6881342)


# %%
#bounds = [(60, 7200)]
#results = optimize.minimize(func_to_optimise, 1800, bounds=bounds, method='Nelder-Mead')
#print(results)
#ideal_timestep_simpl = results.x[0]

# %%
def objective(trial):
    timestep = trial.suggest_float("timestep", 60, 7200)
    return func_to_optimise(timestep)
study = optuna.create_study(direction='minimize')
study.optimize(objective)
ideal_timestep_baye = study.best_params

# %%
1406.6311536175692
