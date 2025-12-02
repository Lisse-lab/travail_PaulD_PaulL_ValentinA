"""
Ce fichier contient les différentes fonctions pour la topologie : il y a une fonction qui créer un dictionnaire, où à chaque zone on associe la profondeur, un autre où on associe le substrat et le dernier qui permet de créer la liste exploitable pour bellhop
This file contains the various functions related to topology: one function creates a dictionary mapping each zone to its depth, another maps each zone to its substrate type, and the last one generates the list that can be used by Bellhop 
"""


import pickle
import os
import pandas as pd
import numpy as np

if __name__ == "__main__":
    import folium
    from shapely.geometry import Polygon
    import branca.colormap as cm
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
    import hydrophone_placement_scripts.utils_scripts.conversions_coordinates as conv

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from  hydrophone_placement_scripts.utils_scripts.conversions_coordinates import Conv # Only for typing

import hydrophone_placement_scripts.utils_scripts.utils as ut

class Topo:
    path = os.path.join(os.path.dirname(__file__), "../datas/")

    def __init__(self, converter : "Conv", new_dic_depths: bool = False, new_dic_substrats : bool = False, substrat : bool = True, save : bool = True):
        self.converter = converter
        self.save = save
        self.create_dic_depths(new_dic_depths)
        if substrat:
            self.create_dic_substrats(new_dic_substrats)

    def create_dic_depths(self, new : bool = False):
        if (not new) & ("dic_depths.pkl" in os.listdir(os.path.join(self.path, "for_model"))):
            with open(os.path.join(self.path, "for_model", "dic_depths.pkl"), "rb") as f:
                dic_depths = pickle.load(f)
        else:
            list_df = []
            for file in os.listdir(os.path.join(self.path, "CSV_Bathymetry")):
                df = pd.read_csv(os.path.join(self.path, "CSV_Bathymetry", file), sep = ";")
                list_df.append(df)
            df = pd.concat(list_df)
            df["Lat"] = df["Lat (DMS)"].apply(ut.dms_to_dd)
            df["Lon"] = df["Long (DMS)"].apply(ut.dms_to_dd)
            df = df[(self.converter.lat_min-0.1<df["Lat"]) & (df["Lat"]<self.converter.lat_max+0.1) & (self.converter.lon_min-0.1<df["Lon"]) & (df["Lon"]<self.converter.lon_max+0.1)] #because of the projection it is not enough to compare with lat_min and lat_max, because some points whcich respect the constraint on latitude and longitude may not respect the constraints on x and y, and vice versa 
            df.rename(columns={"Depth (m)" : "Depth"}, inplace = True)

            dic_depths = {(i,j) : np.array([0.,0]) for i in range (self.converter.n_areas_x) for j in range (self.converter.n_areas_y)}
            for row in df.itertuples():
                if self.converter.lla2area(row.Lat, row.Lon) in dic_depths.keys():
                    dic_depths[self.converter.lla2area(row.Lat, row.Lon)] += [row.Depth, 1]
            for key in dic_depths.keys():
                if dic_depths[key][1]!=0:
                    dic_depths[key][0] = dic_depths[key][0]/dic_depths[key][1]
            for key in dic_depths.keys():
                dic_depths[key] = dic_depths[key][0]
            if self.save:
                with open(os.path.join(self.path, "for_model", "dic_depths.pkl"), "wb") as f:
                    pickle.dump(dic_depths, f)

            print("dic_depths created")
        self.dic_depths = dic_depths    
        return None
    
    def create_dic_substrats(self, new : bool = False):
        if (not new) & ("dic_substrats.pkl" in os.listdir(os.path.join(self.path, "for_model"))):
            with open(os.path.join(self.path, "for_model", "dic_substrats.pkl"), "rb") as f:
                dic_substrats = pickle.load(f)
            self.dic_substrats = dic_substrats
        else:
            df = pd.read_csv(os.path.join(self.path, "substrat.csv")).drop(columns = "Unnamed: 0")
            df = df[(self.converter.xmin<=df["UTMx"])&(df["UTMx"]<=self.converter.xmax)&(self.converter.ymin<=df["UTMy"])&(df["UTMy"]<=self.converter.ymax)]
            dic_substrats = {(i,j) : np.array([0.0,0.0,0.0,0]) for i in range (self.converter.n_areas_x) for j in range (self.converter.n_areas_y)}
            for row in df.itertuples():
                if row.sound_speed != 0:
                    dic_substrats[self.converter.utm2area(row.UTMx, row.UTMy)] += [row.sound_speed, row.density, row.absorption, 1]
            for key in dic_substrats.keys():
                if dic_substrats[key][3]!=0:
                    dic_substrats[key] = dic_substrats[key]/dic_substrats[key][3]
            for key in dic_substrats.keys():
                dic_substrats[key] = dic_substrats[key][:-1]
            self.dic_substrats = dic_substrats
            if self.save:
                with open(os.path.join(self.path, "for_model", "dic_substrats.pkl"), "wb") as f:
                    pickle.dump(dic_substrats, f)
            print("dic_substrats created")
        return None

    def depths_and_substrat(self, l_depths : list[float, list[tuple[int, int]]]):
        depths = []
        substrat = np.array([0.,0.,0.])
        compt = 0
        for dist, areas in l_depths:
            depth = 0
            for area in areas:
                depth += max(self.dic_depths.get(area, 0), 0)
                if (area in self.dic_substrats.keys()) and (self.dic_substrats[area][0] != 0):
                    substrat += self.dic_substrats[area]
                    compt += 1
            depth /= len(areas)
            depths.append([np.round(dist), depth])
        depths.append([np.round(dist)+1, depth])
        if compt != 0:
            return (np.array(depths), substrat/compt)
        else :
            return (np.array(depths), substrat)

"""
    def depths_and_substrat(self, areat, arear):
        x1, y1 = areat
        x2, y2 = arear
        if x1 <= x2:
            xmin, xmax, ymin, ymax = x1, x2, y1, y2 
        else:
            xmin, xmax, ymin, ymax = x2, x1, y2, y1

        a = xmax - xmin
        b = ymax - ymin
        depths = []
        substrat = np.array([0.,0.,0.])
        compt = 0
        
        pgcd = np.gcd(x2-x1, y2-y1)
        vx, vy = (x2-x1)/pgcd, (y2-y1)/pgcd
        for k in range (pgcd+1):
            area = (areat[0] + k*vx, areat[1] + k*vy)
            depth = max(self.dic_depths.get(area, 0), 0)
            ut.sort_insert_depth([np.round(self.converter.dist_areas(areat, area)).astype(int), depth], depths)
            if area in self.dic_substrats.keys():
                substrat += self.dic_substrats[area]
                compt += 1
        ut.sort_insert_depth([np.round(self.converter.dist_areas(areat, arear)+1).astype(int), depth], depths) #only to be sure that the last point is after the reception point


        for x in np.arange((0.5 + np.floor(xmin+0.5).astype(int)), - 0.5 + np.ceil(xmax+0.5).astype(int)):
            t = (x - xmin) / a
            y = b * t + ymin
            dist = np.round(self.converter.dist_areas(areat, (x,y))).astype(int)
            if y % 1 == 0.5:
                depth = max(1/4 * (self.dic_depths.get((int(x),int(y)), 0) + self.dic_depths.get((int(x+1),int(y)), 0) + self.dic_depths.get((int(x),int(y+1)),0) + self.dic_depths.get((int(x+1),int(y+1)), 0)), 0)
                if ut.sort_insert_depth([dist, depth], depths):
                    bool, sub = self.additional_substrat(int(x), int(y), "xy")
                    if bool:
                        substrat += sub
                        compt += 1
            else: 
                depth = max(1/2 * (self.dic_depths.get((int(x),int(y)), 0) + self.dic_depths.get((int(x+1),int(y)), 0)), 0)
                if ut.sort_insert_depth([dist, depth], depths):
                    bool, sub = self.additional_substrat(int(x), int(y), "x")
                    if bool:
                        substrat += sub
                        compt += 1
        
        if y1 <= y2:
            xmin, xmax, ymin, ymax = x1, x2, y1, y2
        else:
            xmin, xmax, ymin, ymax = x2, x1, y2, y1
        
        for y in np.arange(0.5 + np.floor(ymin+0.5).astype(int), - 0.5 + np.ceil(ymax+0.5).astype(int)):
            t = (y - ymin) / b
            x = a * t + xmin
            dist = np.round(self.converter.dist_areas(areat, (x,y))).astype(int)
            if x % 1 != 0.5:
                depth = max(1/2 * (self.dic_depths.get((int(x),int(y)), 0) + self.dic_depths.get((int(x),int(y+1)), 0)), 0)
                if ut.sort_insert_depth([dist, depth], depths):
                    bool, sub = self.additional_substrat(int(x), int(y), "y")
                    if bool:
                        substrat += sub
                        compt += 1

        #print(depths)
        return (np.array(depths), substrat/compt)         
    
    def additional_substrat(self, x, y, str):
        substrat = np.array([0.,0.,0.])
        compt = 0
        if (x, y) in self.dic_substrats.keys():
            if str == "x":
                if self.dic_substrats[(x,y)][0] != 0:
                    substrat += self.dic_substrats[(x,y)]
                    compt += 1
                if (x+1, y) in self.dic_substrats.keys():
                    if self.dic_substrats[(x+1,y)][0] != 0:
                        substrat += self.dic_substrats[(x+1,y)]
                        compt += 1
            elif str == "y":
                if self.dic_substrats[(x,y)][0] != 0:
                    substrat += self.dic_substrats[(x,y)]
                    compt += 1
                if (x, y+1) in self.dic_substrats.keys():
                    if self.dic_substrats[(x,y+1)][0] != 0:
                        substrat += self.dic_substrats[(x,y+1)]
                        compt += 1
            else:
                if self.dic_substrats[(x,y)][0] != 0:
                    substrat += self.dic_substrats[(x,y)]
                    compt += 1
                if (x+1, y) in self.dic_substrats.keys():
                    if self.dic_substrats[(x+1,y)][0] != 0:
                        substrat += self.dic_substrats[(x+1,y)]
                        compt += 1
                if (x, y+1) in self.dic_substrats.keys():
                    if self.dic_substrats[(x,y+1)][0] != 0:
                        substrat += self.dic_substrats[(x,y+1)]
                        compt += 1
                if (x+1, y+1) in self.dic_substrats.keys():
                    if self.dic_substrats[(x+1,y+1)][0] != 0:
                        substrat += self.dic_substrats[(x+1,y+1)]
                        compt += 1
        if compt >0:
            return (True, substrat / compt)
        else:
            return (False, substrat)
"""       
if __name__ == "__main__":
    lat_min = 47.65
    lat_max = 48.07
    lon_min = -70.04
    lon_max = -69.28
    width_area = 500
    depth_area = 2

    converter = conv.Conv(lat_min, lat_max, lon_min, lon_max, width_area, depth_area)
    topo = Topo(converter, save = False)
    df = pd.DataFrame([(x, y, d) for (x, y), d in topo.dic_depths.items()], columns=['x', 'y', 'depth'])
    df = df[df.apply(lambda row : topo.converter.area_in_area((row.x, row.y)), axis = 1)]
    df["polygone"] = df.apply(lambda row : Polygon(topo.converter.area2perim_lla((row.x, row.y))), axis=1)
    map = folium.Map(location=[(topo.converter.lat_min + topo.converter.lat_max)/2, (topo.converter.lon_min + topo.converter.lon_max)/2], zoom_start=11)
    colormap = cm.LinearColormap(colors=['blue', 'red'], vmin=df["depth"].min(), vmax=df["depth"].max(), caption="Depth")
    colormap.add_to(map)
    for row in df.itertuples():
        couleur = colormap(row.depth)
        folium.GeoJson(
            row.polygone,
            style_function=lambda _, couleur=couleur, alpha=0.7: {
                'fillColor': couleur,
                'color': 'black',
                'weight': 1,
                'fillOpacity': alpha,
            }
        ).add_to(map)
    map.save(os.path.join(os.path.dirname(__file__), "../results/map_depths.html"))

    #A map would be created in results with the different depths, if there is an issue it might be because a dic_depths has been saved with other specifications (lat_min, lat_max, ...)