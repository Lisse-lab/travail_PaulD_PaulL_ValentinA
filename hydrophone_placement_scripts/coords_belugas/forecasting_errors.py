import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import folium
import os
from shapely.geometry import Polygon
import branca.colormap as cm
import random
from sklearn.ensemble import RandomForestRegressor
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import hydrophone_placement_scripts.coords_belugas.ornstein_uhlenbeck as ou

def supr_first_last(df):
    for nct in df["NCT"].unique():
        l = df[df["NCT"] == nct].index
        df.drop(l[0], inplace =True)
        if len(l) > 1:
            df.drop(l[-1], inplace =True)
    return df

class forecasting_model : 
    df_areas_created = False
    calc_mse_in_df_areas = False

    def __init__(self, df_train, converter, suppr):
        if suppr:
            self.df_train = supr_first_last(df_train.copy(deep = True))
        else:
            self.df_train = df_train.copy(deep = True)
        self.suppr = suppr
        self.converter = converter
        return None
    
    def set_df_test(self, df_test): #Only for an empty model
        if self.suppr:
            self.df_test = supr_first_last(df_test.copy(deep=True))
        else:
            self.df_test = df_test.copy(deep=True)
        
    def fill_mean_train(self):
        self.l_means = []
        for feature in self.l_beluga_features:
            mean = np.round(self.df_train.groupby("NCT")[feature].mean().mean())
            self.df_train.loc[self.df_train[feature].isna(), feature] = mean
            self.l_means.append(mean)
        return None
    
    def fill_categories_train(self):
        for feature in self.dic_categories.keys():
            l_values, l_labels = self.dic_categories[feature]
            for i in range (len(l_labels)):
                mask = (self.df_train[feature] >= l_values[i]) & (self.df_train[feature] < l_values[i+1])
                self.df_train.loc[mask, feature + "_cat"] = l_labels[i]
        return None
    
    def fill_mean_test(self):
        for feature, mean in zip(self.l_beluga_features, self.l_means):
            self.df_test.loc[self.df_test[feature].isna(), feature] = mean
        return None

    def fill_categories_test(self):
        for feature in self.dic_categories.keys():
            l_values, l_labels = self.dic_categories[feature]
            for i in range (len(l_labels)):
                mask= (self.df_test[feature] >= l_values[i]) & (self.df_test[feature] < l_values[i+1])
                self.df_test.loc[mask, feature + "_cat"] = l_labels[i]
        return None
        
    def create_df_areas(self, df_areas = None):
        if not self.df_areas_created:
            self.df_areas_created = True
            if df_areas is None:
                x_values = np.arange(self.converter.n_areas_x)
                y_values = np.arange(self.converter.n_areas_y)
                xy_pairs = [(x, y) for x in x_values for y in y_values]

                self.df_areas = pd.DataFrame(xy_pairs, columns=['x', 'y'])
                self.df_areas["number_train"] = 0
                for row in self.df_train.itertuples():
                    if ((self.df_areas["x"] == row.areax) & (self.df_areas["y"] == row.areay)).sum() > 0:
                        self.df_areas.loc[(self.df_areas["x"] == row.areax) & (self.df_areas["y"] == row.areay), "number_train"] += 1
                self.df_areas["number_test"] = 0
                for row in self.df_test.itertuples():
                    if ((self.df_areas["x"] == row.areax) & (self.df_areas["y"] == row.areay)).sum() > 0:
                        self.df_areas.loc[(self.df_areas["x"] == row.areax) & (self.df_areas["y"] == row.areay), "number_test"] += 1
                self.df_areas["polygone"] = None
                self.df_areas["polygone"] = self.df_areas.apply(lambda row : Polygon(self.converter.area2perim_lla((row.x, row.y))), axis=1)
            else:
                self.df_areas = df_areas.copy(deep=True)
        return None

    def create_repartition_map(self):
        self.create_df_areas()
        df = self.df_areas[self.df_areas["number_test"] > 0]
        map = folium.Map(location=[(self.converter.lat_min + self.converter.lat_max)/2, (self.converter.lon_min + self.converter.lon_max)/2], zoom_start=11)
        colormap = cm.LinearColormap(colors=['red', 'yellow', 'green'], vmin=df["number_test"].min(), vmax=df["number_test"].max())
        colormap.add_to(map)
        for row in df.itertuples():
            couleur = colormap(row.number_test)
            folium.GeoJson(
                row.polygone,
                style_function=lambda _, couleur=couleur, alpha=0.7: {
                    'fillColor': couleur,
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.7,
                }
            ).add_to(map)

        map.save(os.path.join(os.path.dirname(__file__), "../results/repartition.html"))
        return None
        
    def get_rmse(self, df_areas = None):
        if not self.calc_mse_in_df_areas:
            self.calc_mse_in_df_areas = True
            self.create_df_areas(df_areas)
            self.df_areas["MSE"] = 0.
            for row in self.df_test.itertuples():
                if ((self.df_areas["x"] == row.areax) & (self.df_areas["y"] == row.areay)).sum() > 0:
                    self.df_areas.loc[(self.df_areas["x"] == row.areax) & (self.df_areas["y"] == row.areay), "MSE"] += row.SE
            mask = self.df_areas["number_test"] > 0
            self.df_areas.loc[mask, "MSE"] = self.df_areas.loc[mask, "MSE"] / self.df_areas.loc[mask, "number_test"]
            self.df_areas.loc[~mask, "MSE"] = float("nan")
            self.df_areas["RMSE"] = np.sqrt(self.df_areas["MSE"])
        return self.df_areas["RMSE"]

    def create_errors_map(self, with_alpha = False, min_alpha = 0.5):
        self.get_rmse()
        df = self.df_areas[self.df_areas["number_test"] > 0]
        map = folium.Map(location=[(self.converter.lat_min + self.converter.lat_max)/2, (self.converter.lon_min + self.converter.lon_max)/2], zoom_start=11)
        if with_alpha:
            df["alpha"] = min_alpha + (1 - min_alpha) * ((df["number_train"] - df["number_train"].min()) / (df["number_train"].max() - df["number_train"].min()))
        else:
            df["alpha"] = 1
        colormap = cm.LinearColormap(colors=['red', 'yellow', 'green'], vmin=df["RMSE"].min(), vmax=df["RMSE"].max())
        colormap.add_to(map)
        for row in df.itertuples():
            couleur = colormap(row.RMSE)
            folium.GeoJson(
                row.polygone,
                style_function=lambda _, couleur=couleur, alpha=row.alpha: {
                    'fillColor': couleur,
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': row.alpha,
                }
            ).add_to(map)

        map.save(os.path.join(os.path.dirname(__file__), f"../results/{self.title}_errors_map.html"))
        return None


class ou_model (forecasting_model):
    delta2 = 200

    def __init__(self, df_train, converter, calc_mu, with_first = False, **kwargs):
        for attr, value in kwargs.items():
            if not hasattr(self, attr):
                raise AttributeError(f"OU_model has no attribute '{attr}'.")
            else:
                setattr(self, attr, value)
        if with_first:
            self.title = "ou_with_first"
        else:
            self.title = "ou"
        self.with_first = with_first
        self.calc_mu = calc_mu
        super().__init__(df_train, converter, False)
        return None

    def train(self):
        sigma2 = ou.optimise_sigma2(self.df_train, self.delta2)
        self.calc_mu.set_sigma2(sigma2)
        self.inv_tau0 = ou.get_inv_tau0(self.df_train, self.calc_mu.mu, with_first=self.with_first)
        self.deltav2 = ou.get_deltav2(self.df_train,  with_first=self.with_first)
        print(f"Sigma2 : {sigma2}, Inv_tau0 : {self.inv_tau0}, Deltav2 : {self.deltav2}")
        return None
    
    def test(self, df_test):
        self.set_df_test(df_test)
        self.df_test["SE"] = np.nan
        for nct in self.df_test["NCT"].unique():
            l = self.df_test[self.df_test["NCT"] == nct].index
            if self.with_first:
                offset = 0
                traj = ou.ou_forecast_trajectory(self.df_test["x"].loc[l[0]], self.df_test["y"].loc[l[0]], 0, 0, self.inv_tau0, 0, self.calc_mu, lambda x : None, False)
            else:
                if len(l) > 1:
                    offset = 1
                    deltat = self.df_test["Time"].loc[l[1]] - self.df_test["Time"].loc[l[0]]
                    traj = ou.ou_forecast_trajectory(self.df_test["x"].loc[l[1]], self.df_test["y"].loc[l[1]], (self.df_test["x"].loc[l[1]] - self.df_test["x"].loc[l[0]])/deltat, (self.df_test["y"].loc[l[1]]-self.df_test["y"].loc[l[0]])/deltat, self.inv_tau0, 0, self.calc_mu, lambda x : None, False)
                    for i in range (l[0] + offset, l[-1]):
                        deltat = self.df_test["Time"].loc[i+1] - self.df_test["Time"].loc[i]
                        _, err_v = traj.err_set(deltat, np.array([self.df_test["x"].loc[i+1], self.df_test["y"].loc[i+1]]))          
                        self.df_test.loc[i, "SE"] = err_v
        return self.df_test["SE"].mean()

    def create_df_areas(self, df_areas=None):
        if not self.df_areas_created:
            self.df_areas_created = True
            if df_areas is None:
                x_values = np.arange(self.converter.n_areas_x)
                y_values = np.arange(self.converter.n_areas_y)
                xy_pairs = [(x, y) for x in x_values for y in y_values]

                self.df_areas = pd.DataFrame(xy_pairs, columns=['x', 'y'])
                self.df_areas["number_train"] = 0
                self.df_areas["number_test"] = 0
                if self.with_first:
                    offset = 0
                else:
                    offset = 1
                for nct in self.df_train["NCT"].unique():
                    l = self.df_train[self.df_train["NCT"] == nct].index
                    for i in range (l[0] + offset, l[-1]):
                        areax = self.df_train.loc[i, "areax"]
                        areay = self.df_train.loc[i, "areay"]
                        if ((self.df_areas["x"] == areax) & (self.df_areas["y"] == areay)).sum() > 0:
                            self.df_areas.loc[(self.df_areas["x"] == areax) & (self.df_areas["y"] == areay), "number_train"] += 1
                for nct in self.df_test["NCT"].unique():
                    l = self.df_test[self.df_test["NCT"] == nct].index
                    for i in range (l[0] + offset, l[-1]):
                        areax = self.df_test.loc[i, "areax"]
                        areay = self.df_test.loc[i, "areay"]
                        if ((self.df_areas["x"] == areax) & (self.df_areas["y"] == areay)).sum() > 0:
                            self.df_areas.loc[(self.df_areas["x"] == areax) & (self.df_areas["y"] == areay), "number_test"] += 1
                self.df_areas = self.df_areas[self.df_areas["number_test"] != 0]
                self.df_areas["polygone"] = None
                self.df_areas["polygone"] = self.df_areas.apply(lambda row : Polygon(self.converter.area2perim_lla((row.x, row.y))), axis=1)
            else:
                self.df_areas = df_areas.copy(deep=True)
        return None    

    def get_rmse(self, df_areas=None):
        if not self.calc_mse_in_df_areas:
            self.calc_mse_in_df_areas = True
            self.create_df_areas(df_areas)
            self.df_areas["MSE"] = 0
            if self.with_first:
                offset = 0
            else:
                offset = 1
            for nct in self.df_test["NCT"].unique():
                l = self.df_test[self.df_test["NCT"] == nct].index
                for i in range (l[0] + offset, l[-1]):
                    areax = self.df_test.loc[i, "areax"]
                    areay = self.df_test.loc[i, "areay"]
                    if ((self.df_areas["x"] == areax) & (self.df_areas["y"] == areay)).sum() > 0:
                        self.df_areas.loc[(self.df_areas["x"] == areax) & (self.df_areas["y"] == areay), "MSE"] += self.df_test.loc[i, "SE"]

            self.df_areas["MSE"] = self.df_areas["MSE"] / self.df_areas["number_test"]
            self.df_areas["RMSE"] = np.sqrt(self.df_areas["MSE"])
        return self.df_areas["RMSE"]

class gaussian_model (forecasting_model):
    l_beluga_features = ["pcG", "Taille", "Rayon"]
    dic_categories = {
        "pcG" : ([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]),
        "Taille" : ([0, 10, 20, 30, 50, 75, 100, np.inf], [5, 15, 25, 40, 62, 87, 105]),
        "Rayon" : ([0, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, np.inf], [5, 17, 37, 62, 87, 125, 175, 250, 400, 625, 875, 1250, 1600])
    }
    learning_rate_initial = 0.01
    step_learning_rate = 50
    mult_learning_rate = 0.8
    n_steps = 2000
    log_params_initial_sigmas = np.log([1, 0.1])
    log_params_initial_others = np.log(1)
    n_try = 5
    seed = 73

    def __init__(self, df_train, converter, beluga_features, mean=False, categories=False, position=True, **kwargs):
        self.title = "gaussian"
        for attr, value in kwargs.items():
            if not hasattr(self, attr):
                raise AttributeError(f"Gaussian model has no attribute '{attr}'.")
            else:
                setattr(self, attr, value)
        super().__init__(df_train, converter, True)
        if position:
            self.title += "_position"
            self.features = ["x", "y", "depth"]
            self.meanings = ["pos", "depth"]
        else:
            self.features = []
            self.meanings = []
        self.features += ["v", "cos_theta_v", "sin_theta_v", "vitesseCourant", "cos_directionCourant", "sin_directionCourant", "cos_maree", "sin_maree"]
        self.meanings += ["v", "theta_v", "vitesseCourant", "directionCourant", "maree"]
        self.beluga_features = beluga_features
        if beluga_features:
            self.title += "_beluga_features"
            if mean :
                self.title += "_mean"
                self.fill_mean_train()
            if categories:
                self.title += "_cat"
                self.fill_categories_train()
                self.features += ["pcG_cat", "Taille_cat", "Rayon_cat"]
                self.meanings += ["pcG_cat", "Taille_cat", "Rayon_cat"]
            else:
                self.features += ["pcG", "Taille", "Rayon"]
                self.meanings += ["pcG", "Taille", "Rayon"]
        self.mean = mean
        self.position = position
        self.categories = categories
        return None
    
    def neg_2log_likelihood_step(self, params):
        ind =  np.random.choice(len(self.df_train), size=self.batch_size, replace=False)
        dists = torch.stack([torch.tensor(dist[np.ix_(ind, ind)]) for dist in self.dists])
        vp1 = torch.stack((torch.tensor(self.vp1[0][ind]), torch.tensor(self.vp1[1][ind])), dim = 0)
        K = params[0] * torch.exp( -1/2 * torch.einsum('i,ijk->jk', params[2:], dists))
        Sigma = K + params[1] * torch.eye(K.shape[0], dtype=torch.double)
        return (vp1 @torch.linalg.inv(Sigma)@vp1.T).trace() + torch.log(torch.linalg.det(Sigma))

    def plot_results(self, params):
        meanings = ["Sigma", "Sigmaf"] + self.meanings
        plt.figure(figsize=(8, 5))
        plt.bar(meanings, [e.item() for e in params])
        plt.title("Poids optimisés pour " + self.title)
        plt.ylabel("Valeur du poid")
        plt.xlabel("Poids")
        plt.xticks(rotation=45, ha='right')
        plt.show()
        return None

    def create_stds(self):
        self.stds = {}
        for feat in self.features:
            self.stds[feat] = self.df_train[feat].std()
        return self.stds

    def create_diff_for_train(self, feature):
        return self.df_train[feature].values[:, np.newaxis] - self.df_train[feature].values

    def create_dists(self, diffs):
        if self.position:
            dist_pos = diffs["x"] ** 2 / self.stds["x"]**2 + diffs["y"] ** 2 / self.stds["y"]**2
            dist_depth = diffs["depth"] ** 2 /self.stds["depth"]
        dist_v = diffs["v"] ** 2 / self.stds["v"]**2
        dist_theta_v = diffs["cos_theta_v"] ** 2 / self.stds["cos_theta_v"]**2 / 2 + diffs["sin_theta_v"] ** 2 / self.stds["sin_theta_v"]**2 / 2
        dist_vitesseCourant = diffs["vitesseCourant"] ** 2 / self.stds["vitesseCourant"]**2
        dist_directionCourant = diffs["cos_directionCourant"] ** 2 / self.stds["cos_directionCourant"]**2 / 2 + diffs["sin_directionCourant"] ** 2 / self.stds["sin_directionCourant"]**2 / 2
        dist_maree = diffs["cos_maree"] ** 2 / self.stds["cos_maree"]**2 / 2 + diffs["sin_maree"] ** 2 / self.stds["sin_maree"]**2 / 2
        if self.beluga_features:
            if self.categories:
                dist_pcG = diffs["pcG_cat"] ** 2 / self.stds["pcG_cat"]**2
                dist_Taille = diffs["Taille_cat"] ** 2 / self.stds["Taille_cat"]**2
                dist_Rayon = diffs["Rayon_cat"] ** 2 / self.stds["Rayon_cat"]**2
            else:
                dist_pcG = diffs["pcG"] ** 2 / self.stds["pcG"]**2
                dist_Taille = diffs["Taille"] ** 2 / self.stds["Taille"]**2
                dist_Rayon = diffs["Rayon"] ** 2 / self.stds["Rayon"]**2
        if self.position:
            dists = [dist_pos, dist_depth, dist_v, dist_theta_v, dist_vitesseCourant, dist_directionCourant, dist_maree]
        else:
            dists = [dist_v, dist_theta_v, dist_vitesseCourant, dist_directionCourant, dist_maree]
        if self.beluga_features :
            dists += [dist_pcG, dist_Taille, dist_Rayon]
        dists = np.array(dists)
        dists[np.isnan(dists)] = 0
        return dists


    def train(self, batch_size, show_optim = False, **kwargs):
        for attr, value in kwargs.items():
            if not hasattr(self, attr):
                raise AttributeError(f"OU_model has no attribute '{attr}'.")
            else:
                setattr(self, attr, value)
        
        self.batch_size = batch_size
        self.create_stds()
        self.dists = self.create_dists({feat : self.create_diff_for_train(feat) for feat in self.features})

        self.vp1 = (self.df_train["vxp1"].values, self.df_train["vyp1"].values)

        l_params = []
        self.l_losses = []
        for t in range(self.n_try):
            print("try : ", t + 1)
            log_params_initial = np.concatenate((self.log_params_initial_sigmas, np.array([self.log_params_initial_others] * len(self.dists))))
            log_params = torch.tensor(log_params_initial, requires_grad=True, dtype=torch.double)
            optimizer = torch.optim.SGD([log_params], lr=self.learning_rate_initial)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_learning_rate, gamma=self.mult_learning_rate)
            losses = []

            for step in range(self.n_steps):
                neg_2log_likelihood = self.neg_2log_likelihood_step(torch.exp(log_params))
                if (step % 1000 == 0) & show_optim:
                    print(f"{step} / {self.n_steps}")
                    print("loss : ", neg_2log_likelihood.item())
                optimizer.zero_grad()
                neg_2log_likelihood.backward()
                optimizer.step()
                scheduler.step()
                losses.append(neg_2log_likelihood.item())
            
            l_params.append([e.item() for e in torch.exp(log_params)])
            if show_optim:
                self.plot_results(torch.exp(log_params))
            self.l_losses.append(losses)
        self.np_params = np.array(l_params)
        self.np_params = self.np_params[~np.any(np.isnan(self.np_params), axis = 1)]
        ind = []
        for i, params in enumerate(self.np_params):
            params_median = np.median(self.np_params, axis = 0)
            if (np.abs(params - params_median) < 0.5 * params_median).all():
                ind.append(i)
        self.select_params(ind)
        return self.params

    def select_params(self, ind):
        self.selected_ind = ind
        self.params = self.np_params[ind].mean(axis = 0)
        self.plot_results(self.params)

    def create_values_for_batch(self, feature, indices):
        return self.df_train.loc[indices, feature].values

    def create_batches(self):
        indices = list(range(0, len(self.df_train)))
        random.seed(self.seed)
        random.shuffle(indices)
        self.l_batches = []
        for i in range(np.floor(len(self.df_train) / self.batch_size).astype(int)):
            ind_batches = indices[i * self.batch_size: (i+1) * self.batch_size]
            dic = {feat : self.create_values_for_batch(feat, self.df_train.index[ind_batches]) for feat in self.features}
            dists = np.stack([np.array(dist[np.ix_(ind_batches, ind_batches)]) for dist in self.dists])
            K = self.params[0] * np.exp( -1/2 * np.einsum('i,ijk->jk', self.params[2:], dists))
            sigma = K + self.params[1] * np.eye(K.shape[0])
            dic["inv_sigma_vxp1"] = np.linalg.solve(sigma, self.vp1[0][ind_batches])
            dic["inv_sigma_vyp1"] = np.linalg.solve(sigma, self.vp1[1][ind_batches])
            self.l_batches.append(dic)
        return None
    
    def predict_single_row(self, row):
        l_vx = []
        l_vy = []
        l_w = []
        for dic_batch in self.l_batches:
            diff = {feat : dic_batch[feat] - row[feat] for feat in self.features}
            dists = self.create_dists(diff)
            k = np.exp(-1/2 * self.params[2:]@dists)
            l_vx.append(k@dic_batch["inv_sigma_vxp1"])
            l_vy.append(k@dic_batch["inv_sigma_vyp1"])
            l_w.append(k.sum())
        vxs = np.array(l_vx)
        vys = np.array(l_vy)
        ws = np.array(l_w)
        vxp1_pred = (ws * vxs).sum() / ws.sum()
        vyp1_pred = (ws * vys).sum() / ws.sum()
        return (row.vxp1 - vxp1_pred) **2 + (row.vyp1 - vyp1_pred) **2
               
    def test(self, df_test):
        self.set_df_test(df_test)
        if self.mean:
            self.fill_mean_test()
        if self.categories:
            self.fill_categories_test()
        self.df_test["SE"] = self.df_test.apply(lambda row : self.predict_single_row(row), axis =1)
        return self.df_test["SE"].mean()

    def show_stds(self):
        for feat in self.features:
            print("std" + feat + " : ", self.stds[feat])
        return None

    def show_std_params(self):
        meanings = ["Sigma", "Sigmaf"] + self.meanings
        plt.figure(figsize=(8, 5))
        plt.bar(meanings, self.np_params[self.selected_ind].std(axis = 0))
        plt.title("Std des valeurs des poids pour "  + self.title)
        plt.ylabel("Stds")
        plt.xlabel("Poids")
        plt.xticks(rotation=45, ha='right')
        plt.show()
        return None

    def show_learning_losses(self, mean):
        for losses in self.l_losses:
            losses_bis = []
            for i in range (np.ceil(len(losses) / mean).astype(int)):
                losses_bis.append(np.mean(losses[mean * i: mean * (i+1)]))
            plt.figure(figsize=(10, 6))  # Taille de la figure
            plt.plot(range(1 * mean, (len(losses_bis) + 1) * mean, mean), losses_bis, label='Loss', color='blue', marker='o', linestyle='-')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('Evolution of losses for '  + self.title)
            plt.legend()
            plt.grid(True)
            plt.show()
        return None

class random_forest (forecasting_model):
    labels_y = ["vxp1", "vyp1"]
    l_beluga_features = ["pcG", "Taille", "Rayon"]
    dic_categories = {
        "pcG" : ([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]),
        "Taille" : ([0, 10, 20, 30, 50, 75, 100, np.inf], [5, 15, 25, 40, 62, 87, 105]),
        "Rayon" : ([0, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, np.inf], [5, 17, 37, 62, 87, 125, 175, 250, 400, 625, 875, 1250, 1600])
    }

    def __init__(self, df_train, converter, beluga_features, categories=None, position=True, **kwargs):
        self.title = "randomForest"
        for attr, value in kwargs.items():
            if not hasattr(self, attr):
                raise AttributeError(f"RF_model has no attribute '{attr}'.")
            else:
                setattr(self, attr, value)
        super().__init__(df_train, converter, True)
        self.fill_mean_train()
        if position:
            self.title += "_position"
            self.features = ["x", "y", "depth"]
        else:
            self.features = []
        self.beluga_features = beluga_features
        self.features += ["v", "cos_theta_v", "sin_theta_v", "vitesseCourant", "cos_directionCourant", "sin_directionCourant", "cos_maree", "sin_maree"]
        if beluga_features:
            self.title += "_beluga_features"
            if categories:
                self.title += "_cat"
                self.fill_categories_train()
                self.features += ["pcG_cat", "Taille_cat", "Rayon_cat"]
            else:
                self.features += ["pcG", "Taille", "Rayon"]
        self.position = position
        self.categories = categories
        return None
    
    def train(self):
        self.regr = RandomForestRegressor()
        self.regr.fit(self.df_train[self.features], self.df_train[self.labels_y])

    def test(self, df_test):
        self.set_df_test(df_test)
        self.fill_mean_test()
        if self.categories:
            self.fill_categories_test()
        predictions =  self.regr.predict(self.df_test[self.features])
        self.df_test["SE"] = (self.df_test["vxp1"] - predictions[:,0]) ** 2 + (self.df_test["vyp1"] - predictions[:,1]) ** 2
        return self.df_test["SE"].mean()

    def show_importance(self):
        importances = self.regr.feature_importances_
        feature_names = self.features
        plt.figure(figsize=(10, 6))
        plt.title("Importance des variables for " + self.title)
        plt.bar(range(len(feature_names)), importances, align="center")
        plt.xticks(range(len(feature_names)), feature_names, rotation=90)
        plt.tight_layout()
        plt.show()

class pers_model (forecasting_model):

    def __init__(self, df_train, converter):
        self.title = "persistence"
        super().__init__(df_train, converter, True)
    
    def test(self, df_test):
        self.set_df_test(df_test)
        self.df_test["SE"] = (self.df_test["vxp1"] - self.df_test["v"] * self.df_test["cos_theta_v"]) ** 2 + (self.df_test["vyp1"] - self.df_test["v"] * self.df_test["sin_theta_v"]) ** 2
        return self.df_test["SE"].mean()
    

if __name__ == "__main__":
    import sys
    sys.path.append("../..")
    from scipy.interpolate import griddata
    import hydrophone_placement_scripts.utils_scripts.conversions_coordinates as conv
    import hydrophone_placement_scripts.coords_belugas.calc_mu as clc_mu
    import hydrophone_placement_scripts.coords_belugas.regression as regr
    args = {
        "lat_min" : 47.65,
        "lat_max" : 48.07,
        "lon_min" : -70.04,
        "lon_max" : -69.28,
        "width_area" : 1000,
        "depth_area" : 2,
    }
    geotiff_path = os.path.join(os.path.dirname(__file__), "../datas/BelugaRelativeDens/BelugaRelativeDens.tif")
    step = 100

    converter = conv.Conv(**args)
    calc_mu = clc_mu.Calc_mu(geotiff_path, step)
    df_trajs = pd.read_csv(os.path.join(os.path.dirname(__file__), "../datas/coords_belugas/cleaned_coords.csv"), sep=";").drop(columns=["Unnamed: 0"])
    serie_in_area = converter.in_area(df_trajs["x"], df_trajs["y"])
    serie_in_map = df_trajs.apply(lambda row : calc_mu.in_map(row.x, row.y), axis=1)
    for nct in df_trajs["NCT"].unique():
        l = df_trajs[df_trajs["NCT"] == nct].index
        for i, e in enumerate(l[:-1]):
            if (serie_in_area.loc[e] != serie_in_area.loc[e+1]) | (serie_in_map.loc[e] != serie_in_map.loc[e+1]):
                df_trajs.loc[l[i+1:], "NCT"] = max(df_trajs["NCT"][df_trajs["Année"] == df_trajs.loc[e, "Année"]]) + 1
    df_trajs = df_trajs[serie_in_map & serie_in_area].reset_index(drop=True)
    step_reg = 10 # in min
    step_err = 2 # step_reg / step_err must be an int
    df_reg_trajs = regr.create_df_newtrajs(step_reg, step_err, df_trajs, converter)
    df_reg_trajs[["areax", "areay"]] = pd.Series(converter.utm2area(df_reg_trajs["x"], df_reg_trajs["y"]))
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
    df_reg_trajs["cos_theta_v"] = np.cos(df_reg_trajs["theta_v"] * np.pi/180)
    df_reg_trajs["sin_theta_v"] = np.sin(df_reg_trajs["theta_v"] * np.pi/180)
    mod_pers = pers_model(pd.DataFrame({col:[] for col in df_reg_trajs.columns}), converter)
    mod_pers.test(df_reg_trajs)
    df_reg_trajs[mod_pers.title] = mod_pers.get_rmse()
    mod_pers.create_errors_map(True)
    mod_pers.df_areas.rename(columns = {"RMSE" : "forecasting_error"}, inplace = True)
    known_points = mod_pers.df_areas.dropna(subset=["forecasting_error"])[["x", "y"]].values
    known_values = mod_pers.df_areas.dropna(subset=["forecasting_error"])["forecasting_error"].values

    missing_points = mod_pers.df_areas[mod_pers.df_areas["forecasting_error"].isna()][["x", "y"]].values

    interpolated_values = griddata(known_points, known_values, missing_points, method="linear")
    mod_pers.df_areas.loc[mod_pers.df_areas["forecasting_error"].isna(), "forecasting_error"] = interpolated_values
    mod_pers.df_areas.loc[mod_pers.df_areas["forecasting_error"].isna(), "forecasting_error"] = mod_pers.df_areas["forecasting_error"].max()
    mod_pers.df_areas[["polygone", "forecasting_error"]].to_csv(os.path.join(os.path.dirname(__file__), "../datas/for_model/df_forecasting_error.csv"), sep=";")