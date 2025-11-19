import pandas as pd
import numpy as np
import os
import arlpy.uwapm as pm
import geopandas as gpd
from shapely.geometry import Polygon
from multiprocessing import Pool, cpu_count
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../beluga_watch"))
import hydrophone_placement_scripts.utils_scripts.utils as ut
import hydrophone_placement_scripts.utils_scripts.class_points as cls_points
from . import topo
from . import sound_velocity

from hydrophone_placement_scripts.coords_belugas.calc_mu import Calc_mu

from beluga_watch.src.tests.ziv_zakai import load_noises, spectral_power
from beluga_watch.src.utils.sub_classes import AudioMetadata, Tetrahedra, Hydrophone, HydrophonePair, Environment
from beluga_watch.src.location_bricks.tdoa_brick import crb_from_pair
from beluga_watch.src.location_bricks.low_level_fusion import pos_crb_matrix, weight_matrices

class Calculator:
    system_sensitivity = -165.2
    audio_system_gain = 32768.0 
    sample_rate = 384000
    estimated_tdoa_std = (1. / sample_rate) / np.sqrt(12.)
    source_level_beluga = 143.8 #dB
    ph = 8
    temperature = 3
    salinity = 30
    traversier_path = os.path.join(os.path.dirname(__file__), "../datas/route_traversier/Scénarios_routes.shp")
    boat_noise = 160
    noise_threshold = 50
    gain = 10
    boat_freqs = [500]
    l_boat_freqs_range = [(490, 510)]
    l_fcs = [4000]
    l_freqs_range=[(3700, 4300)]
    noises_folder= os.path.join(os.path.dirname(__file__), "../../beluga_watch/noises")
    sample_rate = 384000
    resol_sound_speed=2
    geotiff_path = os.path.join(os.path.dirname(__file__), "../datas/BelugaRelativeDens/BelugaRelativeDens.tif")
    one_meter = False
    n_processes = cpu_count() - 1 if cpu_count()>1 else 1
    height_sensor = 0.85
    n_calc_ranges = 30
    angle_calc = 2*np.pi/64
    mult_range = 1.25
    range_ultra_max = 50000
    snr_min_dB = 0
    sound_velocity_path = os.path.join(os.path.dirname(__file__), "../datas/sound_velocity_new.csv")
    path = os.path.join(os.path.dirname(__file__), "../datas/for_model")
    noises = None
    noises_for_nav = None

    def __init__(self, converter, new_dics=False, **kwargs):
        for attr, value in kwargs.items():
            setattr(self, attr, value)
        self.args_alpha = self.get_params_alpha()
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
        self.dic_to_calc, self.mat_which_value = ut.get_to_calculate(self.angle_calc, 1 + np.ceil(self.max_range / self.converter.width_area).astype(int))

    def get_params_alpha(self):
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

    def alpha(self, freq, depth, soundspeed):
        t1, t2, t3, f12, f22 = self.args_alpha
        f2 = (freq/1000)**2 #need to have it in kHz
        z = np.array([1, depth, depth**2])
        return (t1/(f2+f12)/soundspeed + (t2/(f2+f22)/soundspeed + t3)@z) * f2

    def init_worker(self):
        cls_points.Point.set_topo(self.topo, self.height_sensor)
        cls_points.Point.update_point(self.converter.xmin, self.converter.ymin, self.converter.xmax, self.converter.ymax, self.converter.accuracy)
        
    def main_func(self, npoint):
        self.modify_environnement(npoint)
        self.df_areas["error"] = 0
        args = [(npoint.points[k], freq, freq_range, k) for k in range (cls_points.NPoint.n_tetrahedras) for freq, freq_range in zip(self.l_fcs, self.l_freqs_range)]

        if min(self.n_processes, len(args)) > 1:
            with Pool(processes=min(self.n_processes, len(args)),  initializer=self.init_worker) as pool:
                results = list(pool.imap_unordered(self.calc_tdoa_errors, args))
            print("All processes done")
        else:
            self.init_worker()
            results = [self.calc_tdoa_errors(arg) for arg in args]
        # if self.bool_update_ranges:
        #     self.update_ranges(results)
        for colonne1, serie1, colonne2, serie2 in results:
            self.df_areas[colonne1] = serie1
            self.df_areas[colonne2] = serie2
        for freq in self.l_fcs:
            self.df_areas["error"] += self.df_areas.apply(lambda row: self.calc_meter_errors(row, freq), axis=1)
        self.df_areas["error"] /= len(self.l_fcs )#mean over the frequencies but maybe we can ponderate it
        print((self.df_areas["w"] / (1+self.df_areas["error"])).sum())
        return (self.df_areas["w"] / (1+self.df_areas["error"])).sum()

    def save_error(self, path):
        self.df_areas["error"].to_csv(os.path.join(path, "error.csv"), sep=";")
        return None

    def load_error(self, path):
        error = pd.read_csv(os.path.join(path, "error.csv"), sep=";")
        error.drop(columns="Unnamed: 0", inplace = True)
        self.df_areas["error"] = error
        return None

    def modify_environnement(self, npoint):
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
    
    # Area has no a depth of d
    # def area_depth(self, x, y, d):
    #     if (d + 1) * self.converter.depth_area > self.topo.dic_depths[(x,y)]:
    #         depth = 1/2 * (d*self.converter.depth_area + self.topo.dic_depths[(x,y)])
    #     else:
    #         depth = (d + 0.5) * self.converter.depth_area
    #     return depth
    
###Construction of df_areas
    def weight_density(self, x, y, d):
            #les points sont en coordoonées d'area et non en coordonnées angulaire
            #à vérifier si quand on prend toute la zone on a des valeurs cohérentes notamment si on a des poids nuls ou quasi (epsilon)
            #n_area = np.ceil(self.topo.dic_depths[(x,y)] / self.converter.depth_area)
            n_area = self.converter.n_depth_area(self.topo.dic_depths[(x,y)])
            xmin, xmax, ymin, ymax = self.converter.min_max_area((x,y))
            return self.calc_mu.h_area(xmin, xmax, ymin, ymax) / n_area

    def zone_intersects_line(self, row, linestring, l_nav):
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

    def calc_nav_weight(self):
        excess_noise = self.boat_noise - self.noise_threshold
        gdf = gpd.read_file(self.traversier_path)
        l_nav = []
        _ = self.df_areas.apply(lambda row: self.zone_intersects_line(row, gdf.loc[0, "geometry"], l_nav), axis=1)
        _ = self.df_areas.apply(lambda row: self.zone_intersects_line(row, gdf.loc[1, "geometry"], l_nav), axis=1)
        print(f"There are {len(l_nav)} areas visited by a boat.")
        n_areas = int(self.range_ultra_max / self.converter.width_area)
        dic_to_calc, mat_which_value = ut.get_to_calculate(self.angle_calc, 1 + n_areas)
        args = [(areat, freq, dic_to_calc, mat_which_value) for areat in l_nav for freq in self.boat_freqs]
        if min(self.n_processes, len(args)) > 1:
            with Pool(processes=min(self.n_processes, len(args))) as pool:
                results = list(pool.imap_unordered(self.calc_nav_weight_multi_pros, args))
        else:
            results = [self.calc_nav_weight_multi_pros(arg) for arg in args]
        nav_exp = pd.Series(0, index=self.df_areas.index, dtype=float)
        for s_tloss in results:
            nav_exp += s_tloss
        self.df_areas["nav_weight"] = 1 + (self.gain - 1) / excess_noise * nav_exp / len(self.boat_freqs) / len(l_nav)
        return None

    def create_df_areas(self, new = False):
        if (not new) & ("df_areas.csv" in os.listdir(self.path)):
            self.df_areas = pd.read_csv(os.path.join(self.path, "df_areas.csv"), sep = ";")
            self.df_areas.drop(columns="Unnamed: 0", inplace = True)
        else: 
            n = 0
            for i in range (self.converter.n_areas_x):
                for j in range(self.converter.n_areas_y):
                    d = self.topo.dic_depths[(i,j)]
                    if d>0:
                        #n += np.ceil(d/self.converter.depth_area).astype(int)
                        n += self.converter.n_depth_area(d)
            self.df_areas = pd.DataFrame({"x": np.empty(n, dtype=int), "y": np.empty(n, dtype=int), "d": np.empty(n, dtype=int), "w": np.empty(n, dtype=float)})   #x, y, depth, weight
            n = 0
            for i in range (self.converter.n_areas_x):
                for j in range(self.converter.n_areas_y):
                    d = self.topo.dic_depths[(i,j)]
                    if d>0:
                        #d = np.floor(d/self.converter.depth_area).astype(int)
                        m = self.converter.n_depth_area(d)
                        for k in range (m):
                            self.df_areas.at[n, "x"] = i
                            self.df_areas.at[n, "y"] = j
                            self.df_areas.at[n, "d"] = k
                            self.df_areas.at[n, "w"] = 1
                            self.df_areas.at[n, "weight_density"] = self.weight_density(i, j, k) #les points sont en coordoonées d'area et non en coordonnées angulaire
                            n += 1
            self.calc_nav_weight()
            self.df_areas["w"] = self.df_areas["weight_density"] * self.df_areas["nav_weight"]
            print("df_areas created")
            self.df_areas.to_csv(os.path.join(self.path, "df_areas.csv"), sep = ";")
        return None

### Acoustic losses
    def get_noises(self, list_noises, l_fcs, l_freqs_range):
        noises = {}
        for f, freq_range in zip(l_fcs, l_freqs_range):
                noises[f] =np.mean([10 * np.log10(spectral_power(noise/self.audio_system_gain, self.sample_rate, freq_range) / (freq_range[1] - freq_range[0]) / noise.shape[1]*self.sample_rate) - self.system_sensitivity for noise in list_noises])  #/ noise.shape[1]*self.sample_rate is for normalisation
        return noises
    
    def calc_snr_dB(self, depthh, depthsb, n_areas, freq, depths, substrat):
        tl = self.tloss(depthh, depthsb, n_areas, freq, depths, substrat)
        return self.source_level_beluga - tl -self.noises[freq]

    def tloss(self, depthh, depthsb, n_areas, freq, depths, substrat):
        args={
            "frequency" : freq,
            "depth": depths,
            "soundspeed": self.sound_speeds,
            "bottom_soundspeed" : substrat[0],
            "bottom_density" : substrat[1],
            "bottom_absorption" : substrat[2],
            "tx_depth" : depthh,
            "rx_range" : np.linspace(0 , int(depths[-2][0]), n_areas + 1)[1:],
            "rx_depth" : depthsb,
            "min_angle" : -89,
            "max_angle" : 89,
            }
        env_acc = pm.create_env2d(**args)
        if env_acc['soundspeed'].shape[0] <= 3:
            env_acc['soundspeed'] = env_acc['soundspeed'][:,1].mean() #if there is still some issues with the shape of sound speed, a unique value of sound speed is given
        i_soundspeed = np.floor(depthsb.max()/self.resol).astype(int)
        try : 
            tloss = pm.compute_transmission_loss(env_acc, mode='semicoherent')
        except:
            env_acc["soundspeed"] = self.sound_speeds[:i_soundspeed, 1].mean()
            tloss = pm.compute_transmission_loss(env_acc, mode='semicoherent')
        sound_speed = (self.sound_speeds[i_soundspeed+1][1] -  self.sound_speeds[i_soundspeed+1][1]) / self.resol * (depthsb.max() - self.resol*i_soundspeed) + self.sound_speeds[i_soundspeed+1][1]
        return -20 * ut.log10(np.abs(tloss)) + self.alpha(freq, depthsb.max(), sound_speed) * np.linspace(0 , int(depths[-2][0]), n_areas + 1)[1:]
    
###Error :
    def calc_meter_errors(self, row, freq):
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
        return np.sqrt(np.sum(np.diag(crb_matrix)))
    
    
    def calc_tdoa_errors(self, args):
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
        return str(k) + "_tdoa_error" + str(freq), s_tdoa_errors, str(k) + "_use_crb" + str(freq), s_use_crb

    def calc_tdoa_errors_hydro(self, snr_dB, k, freq, freq_range):
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
            return float("nan"), False
    
    def get_max_range(self, new = False):
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
    
    def get_range(self, arg):
        freq, freq_range, k, point = arg
        l_ranges = []
        area = self.converter.utm2area(point.coords[0], point.coords[1])
        n = np.round(self.range_ultra_max / self.converter.width_area).astype(int)
        l = ut.comb(n,0,area) + ut.comb(n,n, area)
        for u, v in l:
            depths, substrat = self.topo.depths_and_substrat(area, (u,v))
            depthsb = self.converter.create_depthsb(depths[:,1].max())
            if len(depthsb) > 0:
                snr_dB = self.calc_snr_dB(point.depth(), depthsb, n, freq, depths, substrat)
                col_max = snr_dB.columns[0]
                for col in snr_dB.columns:
#                    if snr_dB.apply(lambda row : self.calc_tdoa_errors_hydro(row[col], k, freq, freq_range)[1], axis=1).any():
                    if snr_dB.apply(lambda row : row[col] >= self.snr_min_dB, axis=1).any():
                        col_max = col
                l_ranges.append(col_max)
        return freq, np.max(l_ranges)

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