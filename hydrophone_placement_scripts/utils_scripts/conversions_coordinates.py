"""
Ce module fait toutes les conversions entre les différents systèmes de coordonnées
This model deals with all the conversions between the different coordinates systems
"""

from pyproj import Transformer
import numpy as np


#It is not include in the class of points because area is a concept that is used outside of points
#And it seems more logical (for me) to put conversions for areas and points together

#By convention depth is positive downward and altitude is positive upward

class Conv :

    def __init__(self, lat_min = None, lat_max = None, lon_min = None, lon_max = None, width_area = None, depth_area = None):
        self.transformer_lla2utm = Transformer.from_crs("EPSG:4326", "EPSG:26919")
        self.transformer_utm2lla = Transformer.from_crs("EPSG:26919", "EPSG:4326")
        self.transformer_lla2ecef = Transformer.from_crs('epsg:4326', 'epsg:4978')
        self.transformer_ecef2lla = Transformer.from_crs("epsg:4978", "epsg:4326")
        if not ((lat_min is None) | (lat_max is None) | (lon_min is None) | (lon_max is None)):
            self.lat_min = lat_min
            self.lat_max = lat_max
            self.lon_min = lon_min
            self.lon_max = lon_max
            self.xmin, self.ymin = self.transformer_lla2utm.transform(lat_min, lon_min)
            self.xmax, self.ymax = self.transformer_lla2utm.transform(lat_max, lon_max)
            if not ((width_area is None) | (depth_area is None)):
                self.width_area = width_area
                self.depth_area = depth_area
                self.n_areas_x, self.n_areas_y = np.ceil((self.xmax - self.xmin)/self.width_area).astype(int), np.ceil((self.ymax-self.ymin)/self.width_area).astype(int)
        return None

    def in_area(self, x, y):
        return (x >= self.xmin) & (x <= self.xmax) & (y >= self.ymin) & (y <= self.ymax)

    def lla2utm (self, lat, lon):
        x, y = self.transformer_lla2utm.transform(lat, lon)
        return (x, y)

    def utm2lla(self, x, y):
        return self.transformer_utm2lla.transform(x, y)

    def utm2area(self, x, y):
        return (np.floor((x - self.xmin)/self.width_area).astype(int), np.floor((y - self.ymin)/self.width_area).astype(int))
    
    def utm2units_area(self, x, y):
        return ((x - self.xmin)/self.width_area - 0.5, (y - self.ymin)/self.width_area - 0.5) #The first area is 0 but has a coordinate of 0.5*self.width_area

    def round_area(self, area):
        return (int(area[0]+0.5), int(area[1]+0.5))

    def area2utm(self, area):
        return ((area[0]+0.5) * self.width_area + self.xmin, (area[1]+0.5) * self.width_area + self.ymin)
    
    def lla2area(self, lat, lon):
        x, y = self.lla2utm(lat, lon)
        return self.utm2area(x, y)
    
    def area2lla(self, area):
        x, y = self.area2utm(area)
        return self.utm2lla(x, y)
    
    def area2perim_lla(self, area):
        "GeoJson takes longitude then latitude and not the opposite"
        lat1, lon1 = self.area2lla((area[0]-0.5, area[1]-0.5))
        lat2, lon2 = self.area2lla((area[0]+0.5, area[1]-0.5))
        lat3, lon3 = self.area2lla((area[0]+0.5, area[1]+0.5))
        lat4, lon4 = self.area2lla((area[0]-0.5, area[1]+0.5))
        return [(lon1, lat1), (lon2, lat2), (lon3, lat3), (lon4, lat4), (lon1, lat1)]
    
    def min_max_area(self, area):
        xmin, ymin = self.area2utm((area[0]-0.5, area[1]-0.5))
        xmax, ymax= self.area2utm((area[0]+0.5, area[1]+0.5))
        return xmin, xmax, ymin, ymax
    
    def area_in_area(self, area):
        x, y = self.area2utm(area)
        return self.in_area(x, y)

    def n_depth_area(self, depth):
        return 1 + np.floor(depth/self.depth_area - 0.5).astype(int)

    def depth_m2area(self, depth):
        return np.floor(depth/self.depth_area).astype(int)
    
    def depth_area2m(self, d):
        return (d+0.5)*self.depth_area

    def create_depthsb(self, dmax):
        return np.arange(0.5, 1 + np.floor(dmax/self.depth_area - 0.5).astype(int), 1) * self.depth_area

    #Even though it's already implemented in other parts of the code, I put it here for clarity
    def lla2enu(self, lat, lon, alt=0):
        x, y, z = self.transformer_lla2ecef.transform(lat, lon, alt)
        ecef_diff = np.array([x - self.x_ref, y - self.y_ref, z - self.z_ref])
        enu_coords = self.R @ ecef_diff
        return enu_coords

    def enu2lla(self, x, y, z=0):
        d = self.R.T @ np.array([x, y, z])
        x = d[0] + self.x_ref
        y = d[1] + self.y_ref
        z = d[2] + self.z_ref
        lon, lat, alt = self.transformer_ecef2lla.transform(x, y, z)
        return np.array([lat, lon, alt])
    
    def update_enu_ref(self, lat, lon, alt=0):
        self.x_ref, self.y_ref, self.z_ref = self.transformer_lla2ecef.transform(lat, lon, alt)
        self.lat_ref_rad = np.radians(lat)
        self.lon_ref_rad = np.radians(lon)
        self.R = np.array([
            [-np.sin(self.lon_ref_rad),  np.cos(self.lon_ref_rad), 0],
            [-np.sin(self.lat_ref_rad) * np.cos(self.lon_ref_rad), -np.sin(self.lat_ref_rad) * np.sin(self.lon_ref_rad), np.cos(self.lat_ref_rad)],
            [ np.cos(self.lat_ref_rad) * np.cos(self.lon_ref_rad),  np.cos(self.lat_ref_rad) * np.sin(self.lon_ref_rad), np.sin(self.lat_ref_rad)]
        ])

    def utm2enu(self, x, y, d=0): #d is depth so it is positive downward and and we need an altitude
        lat, lon = self.utm2lla(x, y)
        return self.lla2enu(lat, lon, -d)
    
    def enu2utm(self, x, y, z=0):
        lat, lon, a = self.enu2lla(x, y, z)
        x, y = self.lla2utm(lat, lon)
        return (x, y, -a)

    def area2enu(self, area, d=0):
        x, y = self.area2utm(area)
        return self.utm2enu(x, y, d)
    
    def dist_areas(self, area1, area2):
        return self.width_area * np.sqrt((area1[0] - area2[0])**2 + (area1[1] - area2[1])**2)