"""
Le fichier le plus important qui comprend les fonctions qui permettent de créer le df_areas, trouver la distance maximale à laquelle on ne peut plus entendre de bélugas, puis de faire les différents calculs permettant de trouver la valeur coût de notre système d'optimisation
"""

import pandas as pd
import numpy as np
import os
import arlpy.uwapm as pm
import geopandas as gpd
from shapely.geometry import Polygon, Point, Linestring
from shapely import wkt
from multiprocessing import Pool, cpu_count
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../beluga_watch"))
import hydrophone_placement_scripts.utils_scripts.utils as ut
import hydrophone_placement_scripts.utils_scripts.class_points as cls_points
from . import topo
from . import sound_velocity

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from hydrophone_placement_scripts.utils_scripts.conversions_coordinates import Conv # Only for typing

from hydrophone_placement_scripts.coords_belugas.calc_mu import Calc_mu

from beluga_watch.src.tests.ziv_zakai import load_noises, spectral_power
from beluga_watch.src.utils.sub_classes import AudioMetadata, Tetrahedra, Hydrophone, HydrophonePair, Environment
from beluga_watch.src.location_bricks.tdoa_brick import crb_from_pair
from beluga_watch.src.location_bricks.low_level_fusion import pos_crb_matrix, weight_matrices

class Calculator:
    
    l_fcs = [4000]
    l_freqs_range=[(3700, 4300)]
    boat_freqs = [500]
    l_boat_freqs_range = [(490, 510)]
    source_level_beluga = 143.8 #dB
    boat_noise = 160
    noise_threshold = 50
    gain = 5
    n_parts = 64
    weights = {"density" : True, "navigation" : True, "forecasting error" : False}
    err_max = 5000

    snr_min_dB = 0
    noises = None
    noises_for_nav = None
    mult_range = 1.25
    n_calc_ranges = 30
    range_ultra_max = 50000
    n_processes = cpu_count() - 1 if cpu_count()>1 else 1
    #Path of files
    traversier_path = os.path.join(os.path.dirname(__file__), "../datas/route_traversier/Scénarios_routes.shp")
    noises_folder= os.path.join(os.path.dirname(__file__), "../../beluga_watch/noises")
    geotiff_path = os.path.join(os.path.dirname(__file__), "../datas/BelugaRelativeDens/BelugaRelativeDens.tif")
    path = os.path.join(os.path.dirname(__file__), "../datas/for_model")
    sound_velocity_path = os.path.join(os.path.dirname(__file__), "../datas/sound_velocity_new.csv")
    resol_sound_speed=2
    #For alpha, transmission loss
    ph = 8
    temperature = 3
    salinity = 30
    #Hydrophones
    system_sensitivity = -165.2
    audio_system_gain = 32768.0 
    sample_rate = 384000
    estimated_tdoa_std = (1. / sample_rate) / np.sqrt(12.)
    one_meter = False #configuration of tetrahedras (one_meter or 30cm)
    height_sensor = 0.85

    #---Initialisation---
    def __init__(self, converter : "Conv", new_dics:bool=False, **kwargs):
        for attr, value in kwargs.items():
            setattr(self, attr, value)
        self.args_alpha = self.get_params_alpha() #for the transmission loss in the water due to the chimical components
        self.resol = self.resol_sound_speed
        self.sound_speeds = np.array(sound_velocity.get_sound_velocities(self.sound_velocity_path, depth_max=200, resol = self.resol))
        self.converter = converter
        
        self.calc_mu = Calc_mu(self.geotiff_path)

        self.sample_rate = self.sample_rate
        if self.noises is None:
            self.noises = self.get_noises(load_noises(self.noises_folder, self.sample_rate), self.l_fcs, self.l_freqs_range) #we only use one sec because there are not a lot of differences with more time
        if self.noises_for_nav is None:
            self.noises_for_nav = self.get_noises(load_noises(self.noises_folder, self.sample_rate), self.boat_freqs, self.l_boat_freqs_range)
        self.topo = topo.Topo(self.converter, new_dic_depths=new_dics, new_dic_substrats=new_dics)
        self.create_df_areas(new = new_dics)

        if self.one_meter:
            self.env = Environment(os.path.join(os.path.dirname(__file__), "../environments/basic_env_one_meter.json"), use_h4=True)
        else:
            self.env = Environment(os.path.join(os.path.dirname(__file__), "../environments/basic_env.json"), use_h4=True)
        cls_points.Point.set_topo(self.topo, self.height_sensor)
        self.max_range, mean_max_range = self.get_max_range()
        cls_points.NPoint.set_range(mean_max_range)
        self.max_range *= self.mult_range
        self.n_areas = 1 + np.ceil(self.max_range / self.converter.width_area).astype(int)
        self.create_l_to_calculate()
        self.create_l_to_avr()

    #---Ranges---
    def get_max_range(self, new : bool = False):
        if (not new) & ("df_ranges.csv" in os.listdir(self.path)):
            df_ranges = pd.read_csv(os.path.join(self.path, "df_ranges.csv"), sep=";").drop("Unnamed: 0", axis = 1)
        else:
            df_ranges = pd.DataFrame(columns=["source_level_beluga", "freq", "freq_range_min", "freq_range_max", "noise", "range_mean", "range_var", "range_max"])
        l_to_calculate = []
        for freq, freq_range in zip(self.l_fcs, self.l_freqs_range):
            mask = (abs(df_ranges["source_level_beluga"] - self.source_level_beluga) < 0.5) & (df_ranges["freq"] == freq) & (df_ranges["freq_range_min"] == freq_range[0]) & (df_ranges["freq_range_max"] == freq_range[1]) & (abs(df_ranges["noise"] - self.noises[freq])<0.5)
            if not len(df_ranges[mask]) > 0:
                l_to_calculate.append((freq, freq_range))

        if len(l_to_calculate) > 0:
            n_tetrahedras = cls_points.NPoint.n_tetrahedras
            cls_points.NPoint.set_n_tetrahedras(self.n_calc_ranges)
            npoint = cls_points.NPointBayesian()
            self.modify_environnement(npoint)
            cls_points.NPoint.set_n_tetrahedras(n_tetrahedras)
            self.n_areas = 1 + np.ceil(self.range_ultra_max / self.converter.width_area).astype(int)
            self.create_l_to_calculate()
            self.create_l_to_avr()
            args = [(req_freq_range[0], req_freq_range[1], k, npoint.points[k]) for req_freq_range in l_to_calculate for k in range(self.n_calc_ranges)]
            if min(self.n_processes, len(args)) > 1:
                with Pool(processes=min(self.n_processes, len(args)),  initializer=self.init_worker) as pool:
                    results = list(pool.imap_unordered(self.get_range, args))
            else:
                self.init_worker()
                results = [self.get_range(arg) for arg in args]
            dic_ranges = {freq_freq_range[0] : [] for freq_freq_range in l_to_calculate}
            for freq, r in results:
                dic_ranges[freq].append(r)   
            for freq_freq_range in l_to_calculate:
                df_ranges.loc[len(df_ranges)] = [self.source_level_beluga, freq_freq_range[0], freq_freq_range[1][0], freq_freq_range[1][1], self.noises[freq], np.mean(dic_ranges[freq_freq_range[0]]), np.var(dic_ranges[freq_freq_range[0]], ddof=1), np.max(dic_ranges[freq_freq_range[0]])]
            df_ranges.to_csv(os.path.join(self.path, "df_ranges.csv"), sep=";")
            print("df_ranges created/modified")
        for freq, freq_range in zip(self.l_fcs, self.l_freqs_range):
            mask = (abs(df_ranges["source_level_beluga"] - self.source_level_beluga) < 0.5) & (df_ranges["freq"] == freq) & (df_ranges["freq_range_min"] == freq_range[0]) & (df_ranges["freq_range_max"] == freq_range[1]) & (abs(df_ranges["noise"] - self.noises[freq])<0.5)
        return df_ranges.loc[mask, "range_max"].max(), df_ranges.loc[mask, "range_mean"].max()
    
    def get_range(self, arg : tuple[float, tuple[float, float], int, "Point"]):
        freq, freq_range, k, point = arg
        l_ranges = []
        areat = self.converter.utm2area(point.coords[0], point.coords[1])
        
        l_theta_q = [(0, 0), (np.pi/4, 1), (np.pi/2, 0), (np.pi/4, 1), (0, 2), (np.pi/4, 2), (np.pi/2, 2), (np.pi/4, 3)]
        for theta, q in l_theta_q:
            l_dists, rx_ranges, areas_visited = self.create_l_dists(theta)
            l_depths = self.create_l_depths(l_dists, areat, q)
            depths, substrat = self.topo.depths_and_substrat(l_depths)
            depthsb = self.converter.create_depthsb(depths[:,1].max())
            col_max = 0
            if len (depthsb) > 0:
                tlosses = self.tloss(0.5, depthsb, rx_ranges, freq, depths, substrat)
                for col, area in zip(tlosses.columns, areas_visited):
                    area2 = ut.modif_area_q(area, areat, q)
                    mask = ((self.df_areas["x"] == area2[0]) & (self.df_areas["y"] == area2[1]))
                    if mask.sum() >0:
                        s_snr_dB = tlosses[col].iloc[:mask.sum()].apply(lambda val : self.calc_snr_dB(val, freq))
                        results = s_snr_dB.apply(lambda row : self.calc_tdoa_errors_hydro(row, k, freq, freq_range))
                        if not np.array([r[1] for r in results]).any():
                            break
                        else:
                            col_max = col
                    else :
                        break
            l_ranges.append(col_max)
        return freq, np.max(l_ranges)

    #---Creation od df_areas---
    def create_df_areas(self, new : bool = False):
        if (not new) & ("df_areas.csv" in os.listdir(self.path)):
            self.df_areas = pd.read_csv(os.path.join(self.path, "df_areas.csv"), sep = ";")
            self.df_areas.drop(columns="Unnamed: 0", inplace = True)
        else: 
            n = 0
            for i in range (self.converter.n_areas_x):
                for j in range(self.converter.n_areas_y):
                    d = self.topo.dic_depths[(i,j)]
                    if d>0:
                        n += self.converter.n_depth_area(d)
            self.df_areas = pd.DataFrame({"x": np.empty(n, dtype=int), "y": np.empty(n, dtype=int), "d": np.empty(n, dtype=int), "w": np.empty(n, dtype=float)})   #x, y, depth, weight
            n = 0
            for i in range (self.converter.n_areas_x):
                for j in range(self.converter.n_areas_y):
                    d = self.topo.dic_depths[(i,j)]
                    if d>0:
                        m = self.converter.n_depth_area(d)
                        for k in range (m):
                            self.df_areas.at[n, "x"] = i
                            self.df_areas.at[n, "y"] = j
                            self.df_areas.at[n, "d"] = k
                            n += 1
        
        self.df_areas["w"] = 1.
        if self.weights["density"]:
            if "weight_density" not in self.df_areas.columns:
                self.df_areas["weight_density"] = self.df_areas.apply(lambda row : self.weight_density(row.x, row.y, row.d), axis = 1) #les points sont en coordoonées d'area et non en coordonnées angulaire
            self.df_areas["w"] *= self.df_areas["weight_density"]
        if self.weights["navigation"]:
            if "nav_weight" not in self.df_areas.columns:
                self.calc_nav_weight()
            self.df_areas["w"] *= self.df_areas["nav_weight"]
        if self.weights["forecasting error"]:
            if "forecasting_error" not in self.df_areas.columns:
                self.fill_forecasting_errors()
            self.df_areas["w"] *= self.df_areas["forecasting_error"]
        print("df_areas created")
        self.df_areas.to_csv(os.path.join(self.path, "df_areas.csv"), sep = ";")
        return None

    #---Weights Density---  
    def weight_density(self, x : int, y : int, d : int):
        #les points sont en coordoonées d'area et non en coordonnées angulaire
        #à vérifier si quand on prend toute la zone on a des valeurs cohérentes notamment si on a des poids nuls ou quasi (epsilon)
        n_area = self.converter.n_depth_area(self.topo.dic_depths[(x,y)])
        xmin, xmax, ymin, ymax = self.converter.min_max_area((x,y))
        return self.calc_mu.h_area(xmin, xmax, ymin, ymax) / n_area

    #---Forecsting errors---
    def fill_forecasting_errors(self):
        assert ("df_forecasting_errors.csv" in os.listdir(os.path.dirname(__file__), "../datas/for_model")), "you need to make the forecasting errors dataframe by executing the scripts coords_belugas/forecasting_errors.py or coords_belugas/comparaison_errors.ipynb" 
        df_forecasting_error = pd.read_csv("../datas/for_model/df_forecasting_error.csv", sep=";").drop("Unnamed: 0", axis = 1)
        df_forecasting_error["polygone"] = df_forecasting_error["polygone"].apply(wkt.loads)
        df_forecasting_error["geometry"] = df_forecasting_error["polygone"].apply(lambda poly : Polygon([(lat, lon) for lon, lat in poly.exterior.coords]))
        gdf_errors = gpd.GeoDataFrame(df_forecasting_error, crs="EPSG:4326")
        self.df_areas["geometry"] = self.df_areas.apply(lambda row : Point(self.converter.area2lla((row.x, row.y))), axis=1)
        gdf_areas = gpd.GeoDataFrame(self.df_areas, crs="EPSG:4326")
        result = gpd.sjoin(gdf_areas, gdf_errors, how="left", predicate="within")
        self.df_areas.drop(columns="geometry", inplace=True)
        self.df_areas["forecasting_error"] = result["forecasting_error"]
        return None

    #---Navigation Weights---
    def zone_intersects_line(self, row : Any, linestring : Linestring, l_nav : list[tuple[int,int]]):
        if row.d == 0:
            polygon = Polygon([
                self.converter.area2utm((row.x - 0.5, row.y - 0.5)),
                self.converter.area2utm((row.x + 0.5, row.y - 0.5)),
                self.converter.area2utm((row.x + 0.5, row.y + 0.5)),
                self.converter.area2utm((row.x - 0.5, row.y + 0.5))
            ])
            if polygon.intersects(linestring):
                l_nav.append((int(row.x), int(row.y)))
        return None

    def calc_nav_weight_multi_pros(self, args : tuple[tuple[int,int], float]):
        """
        Find the areas where the boat is audible when its in areat
        """
        areat, freq = args
        noise_threshold = max(self.noise_threshold, self.noises_for_nav[freq])
        excess_noise = self.boat_noise - noise_threshold
        s_tloss_dB = pd.Series(index=self.df_areas.index, dtype=float)
        results = pd.Series(0, index=self.df_areas.index, dtype=float)
        s_count_areas = pd.Series(0, index=self.df_areas.index, dtype=int)

        mask = ((self.df_areas["x"] == areat[0]) & (self.df_areas["y"] == areat[1]))
        s_tloss_dB[mask] = 0
        s_count_areas[mask] += 1
    
        for theta in self.l_to_calculate:
            l_dists, rx_ranges, areas_visited = self.create_l_dists(theta)
            for q in [0,1,2,3]:
                l_depths = self.create_l_depths(l_dists, areat, q)
                depths, substrat = self.topo.depths_and_substrat(l_depths)
                depthsb = self.converter.create_depthsb(depths[:,1].max())
                if len (depthsb) > 0:
                    tlosses = self.tloss(0.5, depthsb, rx_ranges, freq, depths, substrat)
                    for col, area in zip(tlosses.columns, areas_visited):
                        area2 = ut.modif_area_q(area, areat, q)
                        mask = ((self.df_areas["x"] == area2[0]) & (self.df_areas["y"] == area2[1]))
                        if mask.sum() >0:
                            s_tloss_dB[mask] = tlosses[col].iloc[:mask.sum()].values
                            s_count_areas[mask] += 1
        mask = s_count_areas > 0
        s_tloss_dB[mask] /= s_count_areas[mask]
        results[mask] = excess_noise - s_tloss_dB[mask]
        results[results < 0] = 0     

        #Function to fill one empty area by interpolation
        def f(areax, areay, prevx, prevy, nextx, nexty, q):
            u0, v0 = ut.modif_area_q((areax, areay), areat, q)
            u1, v1 = ut.modif_area_q((prevx, prevy), areat, q)
            u2, v2 = ut.modif_area_q((nextx, nexty), areat, q)

            mask0 = ((self.df_areas["x"] == u0) & (self.df_areas["y"] == v0))
            mask1 = ((self.df_areas["x"] == u1) & (self.df_areas["y"] == v1))
            mask2 = ((self.df_areas["x"] == u2) & (self.df_areas["y"] == v2))
            indexes = results[mask0].index
            n0, n1, n2 = mask0.sum(), mask1.sum(), mask2.sum()
            values1 = results[mask1].values
            values2 = results[mask2].values

            if n0 == 0:
                return None

            min_n = min(n0, n1, n2)
            results.loc[indexes[:min_n]] = (values1[:min_n] + values2[:min_n]) / 2

            if n0 > min_n:
                if n1 <= n2:
                    min_n_2 = min(n0, n2)
                    results.loc[indexes[min_n:min_n_2]] = values2[min_n:min_n_2]
                    if (n0 > min_n_2) & (min_n_2 > 0):
                        results.loc[indexes[min_n_2:n0]] = results[mask0].iloc[min_n_2-1]
                else:
                    min_n_2 = min(n0, n1)
                    results.loc[indexes[min_n:min_n_2]] = values1[min_n:min_n_2]
                    if (n0 > min_n_2) & (min_n_2 > 0):
                        results.loc[indexes[min_n_2:n0]] = results[mask0].iloc[min_n_2-1]
        
            return None
        
        vectorized_f = np.vectorize(f)
        
        for q in [0,1,2,3]:
            l_thetas = []
            #To know at which range boats are audible in each direction
            for theta in self.l_to_calculate:
                _, _, areas_visited = self.create_l_dists(theta)
                n = -1
                boolean = True
                while (n < len(areas_visited) -1) & (boolean):
                    n += 1
                    area = areas_visited[n]
                    area2 = ut.modif_area_q(area, areat, q)
                    boolean = (results[((self.df_areas["x"] == area2[0]) & (self.df_areas["y"] == area2[1]))] > 0).any()
                l_thetas.append(ut.norme(areas_visited[n][0], areas_visited[n][1]))
            #To fill the empty areas
            for i, l in enumerate(self.l_to_avr):
                range_max = max(l_thetas[i], l_thetas[i+1])
                to_avr, preds, nexts = l
                if len(to_avr) > 0:
                    ind = np.where(ut.norme(to_avr[:, 0], to_avr[:, 1]) < range_max)[0]
                    if len(ind) > 0:
                        vectorized_f(to_avr[ind][:, 0], to_avr[ind][:, 1], preds[ind][:, 0], preds[ind][:, 1], nexts[ind][:, 0], nexts[ind][:, 1], np.repeat(q, len(to_avr[ind])))

        print("area done")
        return results

    def calc_nav_weight(self):
        gdf = gpd.read_file(self.traversier_path)
        l_nav = []
        _ = self.df_areas.apply(lambda row: self.zone_intersects_line(row, gdf.loc[0, "geometry"], l_nav), axis=1)
        _ = self.df_areas.apply(lambda row: self.zone_intersects_line(row, gdf.loc[1, "geometry"], l_nav), axis=1)
        print(f"There are {len(l_nav)} areas visited by a boat.")
        self.n_areas = 1 + np.ceil(self.range_ultra_max / self.converter.width_area).astype(int)
        self.create_l_to_calculate()
        self.create_l_to_avr()
        args = [(areat, freq) for areat in l_nav for freq in self.boat_freqs]
        if min(self.n_processes, len(args)) > 1:
            with Pool(processes=min(self.n_processes, len(args))) as pool:
                results = list(pool.imap_unordered(self.calc_nav_weight_multi_pros, args))
        else:
            results = [self.calc_nav_weight_multi_pros(arg) for arg in args]
        nav_exp = pd.Series(0, index=self.df_areas.index, dtype=float)
        for exp in results:
            nav_exp += exp
        self.df_areas["nav_weight"] = 1 + (self.gain - 1) * nav_exp / len(self.boat_freqs) / len(l_nav)
        return None

    #---Main Function    
    def main_func(self, npoint : "cls_points.NPointBayesian"):
        self.modify_environnement(npoint)
        args = [(npoint.points[k], freq, freq_range, k) for k in range (cls_points.NPoint.n_tetrahedras) for freq, freq_range in zip(self.l_fcs, self.l_freqs_range)]

        if min(self.n_processes, len(args)) > 1:
            with Pool(processes=min(self.n_processes, len(args)),  initializer=self.init_worker) as pool:
                results = list(pool.imap_unordered(self.calc_tdoa_errors, args))
            print("All processes done")
        else:
            self.init_worker()
            results = [self.calc_tdoa_errors(arg) for arg in args]
        for colonne1, serie1, colonne2, serie2 in results:
            self.df_areas[colonne1] = serie1
            self.df_areas[colonne2] = serie2
        self.df_areas["error"] = 0.
        self.df_areas["count"] = 0
        for freq in self.l_fcs:
            cols = [str(k) + "_use_crb" + str(freq) for k in range (cls_points.NPoint.n_tetrahedras)]
            mask = self.df_areas[cols].any(axis=1)
            results = self.df_areas[mask].apply(lambda row: self.calc_meter_errors(row, freq), axis=1)
            self.df_areas.loc[mask, "error"] += [r[0] for r in results]
            self.df_areas.loc[mask, "count"] += [r[1] for r in results]
        mask = self.df_areas["count"] > 0
        self.df_areas.loc[mask, "error"] /= self.df_areas.loc[mask, "count"]
        mask = mask & (self.df_areas["error"] <= self.err_max)
        print("Value : ", (self.df_areas.loc[mask, "w"] / (1+self.df_areas.loc[mask, "error"])).sum())
        return 1e-6 + (self.df_areas.loc[mask, "w"] / (1+self.df_areas.loc[mask, "error"])).sum()

    def modify_environnement(self, npoint : "cls_points.NPointBayesian"):
        coordinates_hydrophones = self.env.tetrahedras['0'].relative_hydro_coords
        rotation_matrix = self.env.tetrahedras['0'].rotation_matrix
        use_h4 = self.env.tetrahedras['0'].use_h4
        self.env.tetrahedras = {}

        coords = np.array([(point.coords[0], point.coords[1], point.depth()) for point in npoint.points])
        x_mean, y_mean, depth_mean = coords.mean(axis=0)
        lat_mean, lon_mean = self.converter.utm2lla(x_mean, y_mean)
        self.env.enu_ref = np.array([lat_mean, lon_mean, -depth_mean])
        self.converter.update_enu_ref(lat_mean, lon_mean, -depth_mean)

        for k in range (cls_points.NPoint.n_tetrahedras):
            coords = self.converter.utm2lla(npoint.points[k].coords[0], npoint.points[k].coords[1])
            origin_enu = [coords[0], coords[1], - npoint.points[k].depth()]
            tetra_data = {
                "coordinates_lla":origin_enu,
                "rotation_matrix": rotation_matrix,
                "coordinates_hydrophones": coordinates_hydrophones,
            }
            self.env.tetrahedras[str(k)] = Tetrahedra(str(k), tetra_data, use_h4, self.env.sound_speed, self.env.enu_ref)
        return None

    #---Position and tdoas errors---
    def calc_meter_errors(self, row : Any, freq : float):
        variances = []
        truth_mask = []
        for k in range (cls_points.NPoint.n_tetrahedras):
            if row[str(k) + "_tdoa_error" + str(freq)] is None:
                variances.append(np.array(6*[self.estimated_tdoa_std**2]))
                truth_mask.append(np.array(6*[False]))
            else:
                variances.append(np.array(6*[row[str(k) + "_tdoa_error" + str(freq)]]))
                truth_mask.append(np.array(6*[row[str(k) + "_use_crb" + str(freq)]]))
        w = weight_matrices(variances, truth_mask)
        pos = self.converter.area2enu((row.x, row.y), row.d)
        crb_matrix = pos_crb_matrix(pos, w, truth_mask, self.env)
        err =  np.sqrt(np.sum(np.diag(crb_matrix)))
        if np.isnan(err):
            return (0, 0)
        else:
            return (err, 1)
    
    def calc_tdoa_errors(self, args : tuple["cls_points.Point", float, tuple[float,float], int]):
        point, freq, freq_range, k = args
        s_snr_dB = pd.Series(self.snr_min_dB - 1, index=self.df_areas.index, dtype=float)
        s_count_areas = pd.Series(0, index=self.df_areas.index, dtype=int)
        areat = self.converter.utm2area(point.coords[0], point.coords[1])
        s_tdoa_errors = pd.Series(index=self.df_areas.index, dtype=float)
        s_use_crb = pd.Series(False, index=self.df_areas.index)

        mask = ((self.df_areas["x"] == areat[0]) & (self.df_areas["y"] == areat[1]))
        s_snr_dB[mask] = self.calc_snr_dB(0, freq)
        s_count_areas[mask] += 1
    
        for theta in self.l_to_calculate:
            l_dists, rx_ranges, areas_visited = self.create_l_dists(theta)
            for q in [0,1,2,3]:
                l_depths = self.create_l_depths(l_dists, areat, q)
                depths, substrat = self.topo.depths_and_substrat(l_depths)
                depthsb = self.converter.create_depthsb(depths[:,1].max())
                if len (depthsb) > 0:
                    tlosses = self.tloss(0.5, depthsb, rx_ranges, freq, depths, substrat)
                    for col, area in zip(tlosses.columns, areas_visited):
                        area2 = ut.modif_area_q(area, areat, q)
                        mask = ((self.df_areas["x"] == area2[0]) & (self.df_areas["y"] == area2[1]))
                        if mask.sum() >0:
                            s_snr_dB[mask] = self.calc_snr_dB(tlosses[col].iloc[:mask.sum()].values, freq)
                            s_count_areas[mask] += 1
        
        mask = s_count_areas > 0
        s_snr_dB[mask] /= s_count_areas[mask]
        results = s_snr_dB[mask].apply(lambda row : self.calc_tdoa_errors_hydro(row, k, freq, freq_range))
        s_tdoa_errors[mask] = [r[0] for r in results]
        s_use_crb[mask] = [r[1] for r in results]

        #Function to fill one empty area by interpolation               
        def f(areax, areay, prevx, prevy, nextx, nexty, q):
            u0, v0 = ut.modif_area_q((areax, areay), areat, q)
            u1, v1 = ut.modif_area_q((prevx, prevy), areat, q)
            u2, v2 = ut.modif_area_q((nextx, nexty), areat, q)

            mask0 = ((self.df_areas["x"] == u0) & (self.df_areas["y"] == v0))
            mask1 = ((self.df_areas["x"] == u1) & (self.df_areas["y"] == v1))
            mask2 = ((self.df_areas["x"] == u2) & (self.df_areas["y"] == v2))
            indexes = s_snr_dB[mask0].index
            n0, n1, n2 = mask0.sum(), mask1.sum(), mask2.sum()
            values1 = s_snr_dB[mask1].values
            values2 = s_snr_dB[mask2].values

            if n0 == 0:
                return None

            min_n = min(n0, n1, n2)
            s_snr_dB.loc[indexes[:min_n]] = (values1[:min_n] + values2[:min_n]) / 2

            if n0 > min_n:
                if n1 <= n2:
                    min_n_2 = min(n0, n2)
                    s_snr_dB.loc[indexes[min_n:min_n_2]] = values2[min_n:min_n_2]
                    if (n0 > min_n_2) & (min_n_2 > 0):
                        s_snr_dB.loc[indexes[min_n_2:n0]] = s_snr_dB[mask0].iloc[min_n_2-1]
                else:
                    min_n_2 = min(n0, n1)
                    s_snr_dB.loc[indexes[min_n:min_n_2]] = values1[min_n:min_n_2]
                    if (n0 > min_n_2) & (min_n_2 > 0):
                        s_snr_dB.loc[indexes[min_n_2:n0]] = s_snr_dB[mask0].iloc[min_n_2-1]
            
            results = s_snr_dB[mask0].apply(lambda row : self.calc_tdoa_errors_hydro(row, k, freq, freq_range))
            s_tdoa_errors[mask0] = [r[0] for r in results]
            s_use_crb[mask0] = [r[1] for r in results]
            
            return None
        
        vectorized_f = np.vectorize(f)
        
        for q in [0,1,2,3]:
            l_thetas = []
            #To know at which range belugas are audible in each direction
            for theta in self.l_to_calculate:
                _, _, areas_visited = self.create_l_dists(theta)
                n = -1
                boolean = True
                while (n < len(areas_visited) -1) & (boolean):
                    n += 1
                    area = areas_visited[n]
                    area2 = ut.modif_area_q(area, areat, q)
                    boolean = s_use_crb[((self.df_areas["x"] == area2[0]) & (self.df_areas["y"] == area2[1]))].any()
                l_thetas.append(ut.norme(areas_visited[n][0], areas_visited[n][1]))
            #To fill the empty areas
            for i, l in enumerate(self.l_to_avr):
                to_avr, preds, nexts = l
                range_max = max(l_thetas[i], l_thetas[i+1])
                if len(to_avr) > 0:
                    ind = np.where(ut.norme(to_avr[:, 0], to_avr[:, 1]) < range_max)[0]
                    if len(ind) > 0:
                        vectorized_f(to_avr[ind][:, 0], to_avr[ind][:, 1], preds[ind][:, 0], preds[ind][:, 1], nexts[ind][:, 0], nexts[ind][:, 1], np.repeat(q, len(to_avr[ind])))

        return str(k) + "_tdoa_error" + str(freq), s_tdoa_errors, str(k) + "_use_crb" + str(freq), s_use_crb  

    def calc_tdoa_errors_hydro(self, snr_dB : float, k : int, freq : float, freq_range : tuple[float,float]):
        if snr_dB >= self.snr_min_dB:
            snr_power = 10**(snr_dB/10)
            #We can make the calcul on the 6 pairs of hydrophones but because it has the same SNR the tdoa error will be the same
            hydrophoneref = Hydrophone(0, None, (self.env.tetrahedras[str(k)].relative_hydro_coords + self.env.tetrahedras[str(k)].origin_enu)[0], snr_power)
            hydrophonedelta = Hydrophone(1, None, (self.env.tetrahedras[str(k)].relative_hydro_coords + self.env.tetrahedras[str(k)].origin_enu)[1], snr_power)
            hydrophonepair = HydrophonePair(hydrophoneref, hydrophonedelta, int(next(iter(self.env.tetrahedras.values())).max_delay_seconds*384000))
            metadata = AudioMetadata('', "Whistle", 1, 0, None, self.sample_rate, freq_range, freq)
            error, use_crb = crb_from_pair(metadata, 1, hydrophonepair, freq_range[1]-freq_range[0])
            return error, use_crb
        else:
            return np.nan, False

    #---Acoustic functions---
    def calc_snr_dB(self, tl:float, freq:float):
        return self.source_level_beluga - tl - self.noises[freq]

    def tloss(self, depthh : float, depthsb : np.ndarray, rx_ranges : np.ndarray, freq : float, depths : np.ndarray, substrat : np.ndarray):
        args={
            "frequency" : freq,
            "depth": depths,
            "soundspeed": self.sound_speeds,
            "bottom_soundspeed" : substrat[0],
            "bottom_density" : substrat[1],
            "bottom_absorption" : substrat[2],
            "tx_depth" : depthh,
            "rx_range" : rx_ranges,
            "rx_depth" : depthsb,
            "min_angle" : -89,
            "max_angle" : 89,
            }
       
        env_acc = pm.create_env2d(**args)
        if env_acc['soundspeed'].shape[0] <= 3:
            env_acc['soundspeed'] = env_acc['soundspeed'][:,1].mean() #if there is still some issues with the shape of sound speed, a unique value of sound speed is given
        tlosses = pm.compute_transmission_loss(env_acc, mode='semicoherent')
        if tlosses is None:
            env_acc["soundspeed"] = self.sound_speeds[:, 1].mean()
            tlosses = pm.compute_transmission_loss(env_acc, mode='semicoherent')
        if tlosses is None:
            return pd.DataFrame({})
        sound_speed = env_acc['soundspeed'][:,1].mean()
        tlosses_dB = -20 * ut.log10(np.abs(tlosses))
        for i, alpha_loss in enumerate(self.alpha(freq, depthsb.max(), sound_speed) * rx_ranges[:-1]):
            tlosses_dB[tlosses_dB.columns[i]] += alpha_loss
        return  tlosses_dB

    #---Others---
    def init_worker(self):
        """When Multiprocessing is used, at the creation of a worker, the variables in the other files are "reseted" that's why you have to initialise again"""
        cls_points.Point.set_topo(self.topo, self.height_sensor)
        cls_points.Point.update_point(self.converter.n_areas_x, self.converter.n_areas_y, self.converter.width_area)

    def get_params_alpha(self):
        """To make the calculus of alpha easier"""
        a1 = 8.86 * (10 **(0.78*self.ph - 5)) / 1000 #A1*cw/1000
        p1 = 1
        f1 = 2.8 * np.sqrt(self.salinity/35) * 10 ** (4-1245/(self.temperature+273))
        f12 = f1 ** 2
        a2 = 21.44 * self.salinity * (1 + 0.025*self.temperature) / 1000
        p2 = np.array([1, -1.37*1e-4, 6.2*1e-9])
        f2 = 8.17 * (10 ** (8-1990/(self.temperature+273))) / (1+0.0018*(self.salinity-35))
        f22 = f2 ** 2
        if self.temperature <= 20:
            a3 = 4.937*(1e-4)-2.59*(1e-5)*self.temperature + 9.11*(1e-7)*(self.temperature**2) - 1.5*(1e-8)*(self.temperature**3)
        else:
            a3 = 3.964*(1e-4) - 1.146*(1e-5)*self.temperature + 1.45*(1e-7)*(self.temperature**2) - 6.5*(1e-10)*(self.temperature**3)
        a3 = a3/1000
        p3 = np.array([1, -3.83*1e-5, 4.9*1e-10])
        t1 = a1*p1*f1
        t2 = a2*p2*f2
        t3 = a3*p3
        return t1, t2, t3, f12, f22

    def alpha(self, freq : float, depth : float, soundspeed : float):
        t1, t2, t3, f12, f22 = self.args_alpha
        f2 = (freq/1000)**2 #need to have it in kHz
        z = np.array([1, depth, depth**2])
        return (t1/(f2+f12)/soundspeed + (t2/(f2+f22)/soundspeed + t3)@z) * f2

    def get_noises(self, list_noises : list[str], l_fcs : list[float], l_freqs_range : list[tuple[float, float]]):
        noises = {}
        for f, freq_range in zip(l_fcs, l_freqs_range):
                noises[f] =np.mean([10 * np.log10(spectral_power(noise/self.audio_system_gain, self.sample_rate, freq_range) / (freq_range[1] - freq_range[0]) / noise.shape[1]*self.sample_rate) - self.system_sensitivity for noise in list_noises])  #/ noise.shape[1]*self.sample_rate is for normalisation
        return noises

    def create_l_to_calculate(self):
        """Creates a dictionary where each key is a distance and each value is the list of areas at that distance (sorted by increasing angle), along with the list of angles to traverse"""
        self.dic_dists = {d : [] for d in range (self.n_areas)}
        for i in range(self.n_areas):
            for j in range (self.n_areas):
                if ut.norme(i, j) <= self.n_areas-1:
                    self.dic_dists[np.ceil(ut.norme(i, j))].append((i, j))
        for _, l_k in self.dic_dists.items():
            l_k.sort(key=lambda p: np.arctan2(p[1], p[0]))
        self.l_to_calculate = np.linspace(0, np.pi/2, np.ceil(self.n_parts / 4).astype(int) +1)
        return None
    
    def create_l_dists(self, theta : float):
        """"
        To calculate the transmission loss, you need the topology as well as the locations at which the loss is evaluated (rx_ranges).
        For the topology, a list called l_dists is created; each of its elements is a triplet: the first item is the area crossed, the second is the distance from (0,0), and the last indicates whether the depth should be taken on the crossed area or averaged between the crossed area and the next one along x and/or y.
        This function also returns all the areas that are crossed
        """
        diff = 1/self.converter.width_area
        l_dists = []
        if theta != 0:
            a = 1/np.tan(theta) if theta != np.pi/2 else 0
            x = 0.5
            while ut.norme(int(x-0.5), int(a*x -0.5)) <= self.n_areas - 1:
                if (a*x % 0.5 == 0) & (a*x % 1 != 0):
                    ut.sort_insert_p_dist([(x, a*x), ut.norme(x, a*x), (2, 2)], l_dists, 1/self.converter.width_area)    
                else:
                    ut.sort_insert_p_dist([(x, a*x), ut.norme(x, a*x), (2, 1)], l_dists, 1/self.converter.width_area)
                x += 1
        if theta != np.pi /2:
            b = np.tan(theta)
            y = 0.5
            while ut.norme(int(b*y-0.5), int(y-0.5)) <= self.n_areas - 1:
                ut.sort_insert_p_dist([(b*y, y), ut.norme(b*y, y), (1,2)], l_dists, 1/self.converter.width_area)
                y += 1
        l_dists_temp = []
        for e1, e2 in zip(l_dists, l_dists[1:]):
            x = (e1[0][0] + e2[0][0]) / 2
            y = (e1[0][1] + e2[0][1]) / 2
            if ut.norme(e2[0][0], e2[0][1]) - ut.norme(e1[0][0], e1[0][1]) > 2 * diff:
                l_dists_temp.append((x,y))
            else :
                e1[2] = (2,2)
        rx_ranges = []
        areas_visited = []
        for e in l_dists_temp:
            ut.sort_insert_p_dist([e, ut.norme(e[0],e[1]), (1,1)], l_dists, 1/self.converter.width_area)
            rx_ranges.append(np.round(self.converter.width_area * ut.norme(e[0], e[1])))
            areas_visited.append((int(e[0] + 0.5), int(e[1] + 0.5)))
        ut.sort_insert_p_dist([(0,0), 0, (1,1)], l_dists, 1/self.converter.width_area)
        return l_dists, np.array(rx_ranges), areas_visited

    def create_l_depths(self, l_dists : list, area : tuple[int,int], q : int):
        """Transforms the list l_dists into a list of pairs: the distance to the area, and the areas over which the depth must be averaged"""
        l_depths = []
        for e in l_dists:
            xs, ys = [], []
            if e[2][0] == 1:
                xs.append(int(e[0][0] + 0.5))
            else:
                xs.append(np.round(e[0][0] - 0.5).astype(int))
                xs.append(np.round(e[0][0] + 0.5).astype(int))
            if e[2][1] == 1:
                ys.append(int(e[0][1] + 0.5))
            else:
                ys.append(np.round(e[0][1] - 0.5).astype(int))
                ys.append(np.round(e[0][1] + 0.5).astype(int))
            l = []
            for x in xs:
                for y in ys:
                    l.append(ut.modif_area_q((x, y), area, q))
            l_depths.append((self.converter.width_area * e[1], l))
        return l_depths

    def create_l_to_avr(self):
        """Create three lists (combined in one) of the not visited areas and the two closest one to average on their value"""
        mat = np.zeros((self.n_areas, self.n_areas))
        for theta in self.l_to_calculate:
            _, _, areas_visited = self.create_l_dists(theta)
            for area_v in areas_visited:
                mat[area_v[0], area_v[1]] = 1
        l_areas, l_prevs, l_nexts = [], [], []
        for k in range(1, self.n_areas):
            l_k = self.dic_dists[k]
            def rec_f (i, previous):
                if mat[l_k[i]] == 1:
                    if i + 1 < len(l_k):
                        rec_f(i+1, l_k[i])
                    return l_k[i]
                else:
                    next = rec_f(i+1, previous)
                    l_areas.append(l_k[i])
                    l_prevs.append(previous)
                    l_nexts.append(next)
                    return next
            rec_f(0, None)
        n_parts_q = len(self.l_to_calculate) - 1
        l_to_avr = [[[[], []] for _ in range (3)] for _ in range(n_parts_q)]
        for area, prev, next in zip(l_areas, l_prevs, l_nexts):
            theta = np.arctan2(area[1], area[0])
            for i in [0, 1]:
                l_to_avr[int(theta *2 / np.pi * n_parts_q)][0][i].append(area[i])
                l_to_avr[int(theta *2 / np.pi * n_parts_q)][1][i].append(prev[i])
                l_to_avr[int(theta *2 / np.pi * n_parts_q)][2][i].append(next[i])
        for i in range (len(l_to_avr)):
            for j in range (len(l_to_avr[i])):
                l_to_avr[i][j] = np.array(l_to_avr[i][j])
        self.l_to_avr = l_to_avr    
        return None  
    
    def save_error(self, path : str):
        mask = (self.df_areas["count"] == 0) | (self.df_areas["error"] > self.err_max)
        self.df_areas.loc[mask, "error"] = np.nan
        self.df_areas["error"].to_csv(os.path.join(path, "error.csv"), sep=";")
        return None

    def load_error(self, path : str):
        error = pd.read_csv(os.path.join(path, "error.csv"), sep=";")
        error.drop(columns="Unnamed: 0", inplace = True)
        self.df_areas["error"] = error
        return None

"""
    def update_ranges(self, results):
        dic_ranges = {freq : [] for freq in self.l_fcs}    
        for _, _, _, _, freq, range in results:
            dic_ranges[freq].append(range)
            
        df_results = pd.DataFrame()
        for freq, freq_range in zip(self.l_fcs, self.l_freqs_range):
            mask = (self.df_ranges["source_level_beluga"] == self.source_level_beluga) & (self.df_ranges["freq"] == freq) & (self.df_ranges["freq_range_min"] == freq_range[0]) & (self.df_ranges["freq_range_max"] == freq_range[1]) & (self.df_ranges["noise"] == self.noises[freq])
            m_new, var_new, n_new = np.mean(dic_ranges[freq]), np.var(dic_ranges[freq], ddof=1), len(dic_ranges[freq])
            m_old, var_old, n_old = self.df_areas["range_mean", "range_var", "n_range"][mask].iloc[0]
            n = n_new + n_old
            m = (n_new * m_new + n_old * m_old) / n
            var = ((n_new - 1) * var_new + (n_old - 1) * var_old) / (n - 1) + n_new * n_old / (n*(n-1)) * (m_new - m_old)**2
            self.df_ranges["range_mean", "range_var", "n_range"][mask] = m, var, n
            df_results = pd.concat([df_results, self.df_ranges[mask]])
        self.df_ranges.to_csv(os.path.join(self.path, "df_ranges.csv"), sep=";")
        cls_points.NPoint.set_range(df_results["range_mean"].max())
        return None
    
    def tloss_old(self, deptht, depthr, freq, depths, substrat):
        args={
            "frequency" : freq,
            "depth": depths,
            "soundspeed": self.sound_speeds,
            "bottom_soundspeed" : substrat[0],
            "bottom_density" : substrat[1],
            "bottom_absorption" : substrat[2],
            "tx_depth" : deptht,
            "rx_range" : depths[-2][0],
            "rx_depth" : depthr,
            "min_angle" : -89,
            "max_angle" : 89,
            }
        env_acc = pm.create_env2d(**args)
        if env_acc['soundspeed'].shape[0] <= 3:
            env_acc['soundspeed'] = env_acc['soundspeed'][:,1].mean() #if there is still some issues with the shape of sound speed, a unique value of sound speed is given
        i_soundspeed = np.floor(depthr/self.resol).astype(int)
        try : 
            tloss = pm.compute_transmission_loss(env_acc, mode='semicoherent').values[0,0]
        except:
            env_acc["soundspeed"] = self.sound_speeds[:i_soundspeed, 1].mean()
            tloss = pm.compute_transmission_loss(env_acc, mode='semicoherent').values[0,0]
        sound_speed = (self.sound_speeds[i_soundspeed+1][1] -  self.sound_speeds[i_soundspeed+1][1]) / self.resol * (depthr - self.resol*i_soundspeed) + self.sound_speeds[i_soundspeed+1][1]
        return -20 * ut.log10(np.abs(tloss)) + self.alpha(freq, depthr, sound_speed) * depths[-2][0]

    
    
    def calc_tdoa_errors_old(self, args):
        point, freq, freq_range, k = args
        s_tdoa_errors = pd.Series(index=self.df_areas.index, dtype=float)
        s_use_crb = pd.Series(False, index=self.df_areas.index, dtype=bool)
        area = self.converter.utm2area(point.coords[0], point.coords[1])
        i = 0
        boolean = True
        while boolean:
            j = 0
            while (j <= i) & boolean:
                boolean = False
                l = ut.comb(i,j, area)
                for u,v in l:
                    ds = self.df_areas["d"][(self.df_areas["x"] == u) & (self.df_areas["y"] == v)]
                    if len(ds) > 0:
                        depths, substrat = self.topo.depths_and_substrat((u,v), self.converter.utm2units_area(point.coords[0], point.coords[1]))
                        for d in ds:
                            snr_power = self.calc_snr_power(self.area_depth(u,v,d), point.depth(), freq, depths, substrat) if i != 0 else self.source_level_beluga - self.noises[freq]
                            tdoa_errors, use_crb = self.calc_tdoa_errors_hydro(snr_power, k, freq, freq_range)
                            ind = self.df_areas[((self.df_areas["x"] == u) & (self.df_areas["y"] == v) & (self.df_areas["d"] == d))].index
                            s_tdoa_errors.loc[ind] = tdoa_errors
                            s_use_crb.loc[ind] = use_crb
                            if use_crb:
                                boolean = True
                j+=1
            i+=1
        if self.bool_update_ranges:
                    if i==j:
                        i-=1
                        j-=1
                    else:
                        j-=1
                    l = ut.comb(i,j,area)
                    return str(k) + "_tdoa_error" + str(freq), s_tdoa_errors, str(k) + "_use_crb" + str(freq), s_use_crb, freq, np.max([self.converter.dist_areas((u,v), area) for u,v in l])
        else:
            return str(k) + "_tdoa_error" + str(freq), s_tdoa_errors, str(k) + "_use_crb" + str(freq), s_use_crb, None, None

            
    
    def calc_snr_power(self, deptht, depthr, freq, depths, substrat):
        tl = self.tloss(deptht, depthr, freq, depths, substrat)
        return 10 ** ((self.source_level_beluga - tl - self.noises[freq]) /10)

        """


"""
    def calc_nav_weight_multi_pros(self, args):
        areat, freq, dic_to_calc, mat_which_value = args
        s_tloss = pd.Series(0, index=self.df_areas.index, dtype=float)
        l_angles = [-np.round(np.pi/2, 8), 0, np.round(np.pi/2, 8), np.round(np.pi,8)]
        dic_angles_range = {angle : self.range_ultra_max for angle in l_angles}
        
        def calc_nav_weight_single_area(u, v, n2):
            depths, substrat = self.topo.depths_and_substrat(areat, (areat[0] + u*n2, areat[1] + v*n2))
            depthsb = self.converter.create_depthsb(depths[:,1].max())
            if len(depthsb) > 0:
                tlosses = self.tloss(0, depthsb, n2, freq, depths, substrat)
                for m, col in enumerate(tlosses.columns):
                    x, y = areat[0] + u*(m+1), areat[1] + v*(m+1)
                    mask = ((self.df_areas["x"] == x) & (self.df_areas["y"] == y))
                    if mask.sum() > 0:
                        s_tloss[mask] = tlosses[col].iloc[:mask.sum()].apply(lambda val : max(val, 0))
                        if (s_tloss[mask] == 0).all():
                            break
                    else:
                        break
                return tlosses.columns[m], (m == len(tlosses.columns) - 1)
            else :
                return 0, False
            
        for ij, n in dic_to_calc.items():
            i, j = ij[0], ij[1]
            if i == 0:
                s_tloss[((self.df_areas["x"] == areat[0]) & (self.df_areas["y"] == areat[1]))] = self.source_level_beluga - self.noises_for_nav[freq]
            else:
                l = ut.comb(i, j, (0, 0))
                norme = np.ceil(ut.norme(i, j)).astype(int)
                for u, v in l:        
                    theta = np.round(np.arctan2(v, u), 8)
                    ind = ut.find_indice_angles(theta, l_angles)
                    max_range = max(dic_angles_range[l_angles[ind]], dic_angles_range[l_angles[ind-1]])
                    n2 = min(n, int(max_range/norme/self.converter.width_area))
                    if n2 > 0:
                        range_max_reached = True
                        while range_max_reached:
                            range_max, range_max_reached = calc_nav_weight_single_area(u, v, n2)
                            n2 *= 2
                        dic_angles_range[theta] = range_max * self.mult_range
                        l_angles.insert(ind, theta)

        for i in range(mat_which_value.shape[0]):
            for j in range(mat_which_value.shape[1]):
                if mat_which_value[i,j] is not None and len(mat_which_value[i,j]) > 2:
                    for uv0, uv1, uv2 in zip(ut.comb(i,j,areat), ut.comb(mat_which_value[i,j][0][0], mat_which_value[i,j][0][1],areat), ut.comb(mat_which_value[i,j][1][0],mat_which_value[i,j][1][1],areat)):
                        mask0 = ((self.df_areas["x"] == uv0[0]) & (self.df_areas["y"] == uv0[1]))
                        mask1 = ((self.df_areas["x"] == uv1[0]) & (self.df_areas["y"] == uv1[1]))
                        mask2 = ((self.df_areas["x"] == uv2[0]) & (self.df_areas["y"] == uv2[1]))
                        if mask0.sum() > 0:
                                indexes = s_tloss[mask0].index
                                if (mask0.sum() <= mask1.sum()) & (mask0.sum() <= mask2.sum()):
                                    s_tloss.loc[indexes] = (s_tloss[mask1].iloc[:mask0.sum()].values + s_tloss[mask2].iloc[:mask0.sum()].values)/2
                                elif (mask0.sum() > mask1.sum()) & (mask0.sum() <= mask2.sum()):
                                    s_tloss.loc[indexes[:mask1.sum()]] = (s_tloss[mask1].values + s_tloss[mask2].iloc[:mask1.sum()].values) /2
                                    s_tloss.loc[indexes[mask1.sum():]] = s_tloss[mask2].iloc[mask1.sum():mask0.sum()].values
                                elif (mask0.sum() <= mask1.sum()) & (mask0.sum() > mask2.sum()):
                                    s_tloss.loc[indexes[:mask2.sum()]] = (s_tloss[mask1].iloc[:mask2.sum()].values + s_tloss[mask2].values) /2
                                    s_tloss.loc[indexes[mask2.sum():]] = s_tloss[mask1].iloc[mask2.sum():mask0.sum()].values
                                else :
                                    if mask1.sum()<= mask2.sum():
                                        s_tloss.loc[indexes[:mask1.sum()]] = (s_tloss[mask1].values + s_tloss[mask2].iloc[:mask1.sum()].values) /2
                                        s_tloss.loc[indexes[mask1.sum():mask2.sum()]] = s_tloss[mask2].iloc[mask1.sum():].values
                                        s_tloss.loc[indexes[mask2.sum():]] = s_tloss[mask0].iloc[mask2.sum()]
                                    else:
                                        s_tloss.loc[indexes[:mask2.sum()]] = (s_tloss[mask1].iloc[:mask2.sum()].values + s_tloss[mask2].values) /2
                                        s_tloss.loc[indexes[mask2.sum():mask1.sum()]] = s_tloss[mask1].iloc[mask2.sum():].values
                                        s_tloss.loc[indexes[mask1.sum():]] = s_tloss[mask0].iloc[mask1.sum()]
        return s_tloss
    """



"""def calc_tdoa_errors(self, args):
        point, freq, freq_range, k = args
        l_angles = [-np.round(np.pi/2, 8), 0, np.round(np.pi/2, 8), np.round(np.pi,8)]
        dic_angles_range = {angle : self.range_ultra_max for angle in l_angles}
        s_snr_dB = pd.Series(index=self.df_areas.index, dtype=float)
        s_tdoa_errors = pd.Series(index=self.df_areas.index, dtype=float)
        s_use_crb = pd.Series(index=self.df_areas.index, dtype=bool)
        area = self.converter.utm2area(point.coords[0], point.coords[1])

        def calc_tdoa_error_single_area(u, v, n2):
            depths, substrat = self.topo.depths_and_substrat(area, (area[0] + u*n2, area[1] + v*n2))
            depthsb = self.converter.create_depthsb(depths[:,1].max())
            if len(depthsb) > 0:
                snr_dB = self.calc_snr_dB(point.depth(), depthsb, n2, freq, depths, substrat)
                for m, col in enumerate(snr_dB.columns):
                    x, y = area[0] + u*(m+1), area[1] + v*(m+1)
                    mask = ((self.df_areas["x"] == x) & (self.df_areas["y"] == y))
                    if mask.sum() >0:
                        s_snr_dB[mask] = snr_dB[col].iloc[:mask.sum()].values
                        results = s_snr_dB[mask].apply(lambda row : self.calc_tdoa_errors_hydro(row, k, freq, freq_range))
                        s_tdoa_errors[mask], s_use_crb[mask] = [r[0] for r in results], [r[1] for r in results]
                        if not (s_use_crb[mask]).any():
                            break
                    else:
                        break
                return snr_dB.columns[m], (m == len(snr_dB.columns) - 1)
            else :
                return 0, False
            
        for ij, n in self.dic_to_calc.items():
            i, j = ij[0], ij[1]
            if i == 0:
                mask = ((self.df_areas["x"] == area[0]) & (self.df_areas["y"] == area[1]))
                s_snr_dB[mask] = self.source_level_beluga - self.noises[freq]
                results = s_snr_dB[mask].apply(lambda row : self.calc_tdoa_errors_hydro(row, k, freq, freq_range))
                s_tdoa_errors[mask], s_use_crb[mask] = [r[0] for r in results], [r[1] for r in results]
            else:
                l = ut.comb(i, j, (0, 0))
                norme = np.ceil(ut.norme(i, j)).astype(int)
                for u, v in l:        
                    theta = np.round(np.arctan2(v, u), 8)
                    ind = ut.find_indice_angles(theta, l_angles)
                    max_range = max(dic_angles_range[l_angles[ind]], dic_angles_range[l_angles[ind-1]])
                    n2 = min(n, int(max_range/norme/self.converter.width_area))
                    if n2 > 0:
                        range_max_reached = True
                        while range_max_reached:
                            range_max, range_max_reached = calc_tdoa_error_single_area(u, v, n2)
                            n2 *= 2
                        dic_angles_range[theta] = range_max * self.mult_range
                        l_angles.insert(ind, theta)


        for i in range(self.mat_which_value.shape[0]):
            for j in range(self.mat_which_value.shape[1]):
                if self.mat_which_value[i,j] is not None and len(self.mat_which_value[i,j]) >= 2:
                    for uv0, uv1, uv2 in zip(ut.comb(i,j,area), ut.comb(self.mat_which_value[i,j][0][0],self.mat_which_value[i,j][0][1],area), ut.comb(self.mat_which_value[i,j][1][0],self.mat_which_value[i,j][1][1],area)):
                        mask0 = ((self.df_areas["x"] == uv0[0]) & (self.df_areas["y"] == uv0[1]))
                        mask1 = ((self.df_areas["x"] == uv1[0]) & (self.df_areas["y"] == uv1[1]))
                        mask2 = ((self.df_areas["x"] == uv2[0]) & (self.df_areas["y"] == uv2[1]))
                        indexes = s_snr_dB[mask0].index
                        if mask0.sum() > 0:
                            if (mask0.sum() <= mask1.sum()) & (mask0.sum() <= mask2.sum()):
                                s_snr_dB.loc[indexes] = (s_snr_dB[mask1].iloc[:mask0.sum()].values + s_snr_dB[mask2].iloc[:mask0.sum()].values)/2
                            elif (mask0.sum() > mask1.sum()) & (mask0.sum() <= mask2.sum()):
                                s_snr_dB.loc[indexes[:mask1.sum()]] = (s_snr_dB[mask1].values + s_snr_dB[mask2].iloc[:mask1.sum()].values) /2
                                s_snr_dB.loc[indexes[mask1.sum():]] = s_snr_dB[mask2].iloc[mask1.sum():mask0.sum()].values
                            elif (mask0.sum() <= mask1.sum()) & (mask0.sum() > mask2.sum()):
                                s_snr_dB.loc[indexes[:mask2.sum()]] = (s_snr_dB[mask1].iloc[:mask2.sum()].values + s_snr_dB[mask2].values) /2
                                s_snr_dB.loc[indexes[mask2.sum():]] = s_snr_dB[mask1].iloc[mask2.sum():mask0.sum()].values
                            else :
                                if mask1.sum()<= mask2.sum():
                                    s_snr_dB.loc[indexes[:mask1.sum()]] = (s_snr_dB[mask1].values + s_snr_dB[mask2].iloc[:mask1.sum()].values) /2
                                    s_snr_dB.loc[indexes[mask1.sum():mask2.sum()]] = s_snr_dB[mask2].iloc[mask1.sum():].values
                                    s_snr_dB.loc[indexes[mask2.sum():]] = s_snr_dB[mask0].iloc[mask2.sum()]
                                else:
                                    s_snr_dB.loc[indexes[:mask2.sum()]] = (s_snr_dB[mask1].iloc[:mask2.sum()].values + s_snr_dB[mask2].values) /2
                                    s_snr_dB.loc[indexes[mask2.sum():mask1.sum()]] = s_snr_dB[mask1].iloc[mask2.sum():].values
                                    s_snr_dB.loc[indexes[mask1.sum():]] = s_snr_dB[mask0].iloc[mask1.sum()]
                            results = s_snr_dB[mask0].apply(lambda row : self.calc_tdoa_errors_hydro(row, k, freq, freq_range))
                            s_tdoa_errors[mask0], s_use_crb[mask0] = [r[0] for r in results], [r[1] for r in results]
        return str(k) + "_tdoa_error" + str(freq), s_tdoa_errors, str(k) + "_use_crb" + str(freq), s_use_crb"""
