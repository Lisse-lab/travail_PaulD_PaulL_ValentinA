import rasterio
import numpy as np
import itertools

class Calc_mu:
    def __init__(self, geotiff_path, sigma2 = None):
        dataset = rasterio.open(geotiff_path)
        self.data = dataset.read(1)
        self.height = dataset.height
        self.width = dataset.width
        self.transform = dataset.transform
        self.setp_x = np.abs(self.transform[0]) # Note: The real-world and dataset frames are assumed to be aligned.
        self.setp_y = np.abs(self.transform[4])
        if not sigma2 is None:
            self.sigma2 = sigma2
    
    def set_sigma2(self, sigma2):
        self.sigma2 = sigma2

    def create_xs(self, x):
        l = []
        if 0 <= x <self.width:
            l.append(x)
        if 0 <= x+1 < self.width:
            l.append(x+1)
        return l
    
    def create_ys(self, y):
        l = []
        if 0 <= y <self.height:
            l.append(y)
        if 0 <= y+1 < self.height:
            l.append(y+1)
        return l

    def find_square(self, x, y):
        pixel_coords = ~self.transform * (x, y)
        if ((pixel_coords[0]-0.5) % 1 == 0) & ((pixel_coords[1]-0.5) % 1 == 0):
            return [(pixel_coords[0]-0.5, pixel_coords[1] - 0.5, 1)]
        elif (pixel_coords[0]-0.5) % 1 == 0:
            x = pixel_coords[0]-0.5
            ys = self.create_ys(np.floor(pixel_coords[1]-0.5))
            ws = [1/(np.abs(pixel_coords[1] - y - 0.5)) for y in ys]
            return [(x, y, w/np.sum(ws)) for y, w in zip(ys, ws)]
        elif (pixel_coords[1]-0.5) % 1 == 0:
            xs = self.create_xs(np.floor(pixel_coords[0]-0.5))
            y = pixel_coords[1]-0.5
            ws = [1/(np.abs(pixel_coords[0] - x - 0.5)) for x in xs]
            return [(x, y, w/np.sum(ws)) for x, w in zip(xs, ws)]
        else :
            xs = self.create_xs(np.floor(pixel_coords[0]-0.5))
            ys = self.create_ys(np.floor(pixel_coords[1]-0.5))
            ws = [1/(np.sqrt((pixel_coords[0] - x - 0.5)**2 + (pixel_coords[1] - y - 0.5)**2)) for (x, y) in itertools.product(xs, ys)]
            return [(x, y, w/np.sum(ws)) for (x, y), w in zip(itertools.product(xs, ys), ws)]

    def h_pix(self, x, y):
        assert (x % 1 == 0) & (y % 1 == 0), "x and y must be integers (float but representing an integer)"
        if self.data[int(y),int(x)] == 0:
            return 2 * np.finfo(float).eps
        elif self.data[int(y),int(x)] < 0:
            return np.finfo(float).eps
        else:
            return self.data[int(y), int(x)]

    def hx_pix(self, x, y):
        if x-1 < 0:
            return (self.h_pix(x+1, y) - self.h_pix(x, y))/self.step_x
        elif x+1 >= self.width:
            return (self.h_pix(x, y) - self.h_pix(x-1, y))/self.step_x
        else:
            return (self.h_pix(x+1, y) - self.h_pix(x-1, y))/(2*self.step_x)

    def hy_pix(self, x, y):
        if y-1 < 0:
            return (self.h_pix(x, y+1) - self.h_pix(x, y))/self.step_y
        elif y+1 >= self.height:
            return (self.h_pix(x, y) - self.h_pix(x, y-1))/self.step_y
        else:
            return (self.h_pix(x, y+1) - self.h_pix(x, y-1))/(2*self.step_y)

    def hx2_pix(self, x, y):
        if x-1 < 0:
            return (self.hx_pix(x+1, y) - self.hx_pix(x,y))/self.step_x
        elif x+1 >= self.width:
            return (self.hx_pix(x, y) - self.hx_pix(x-1, y))/self.step_x
        else:
            return (self.hx_pix(x+1, y) - self.hx_pix(x-1, y))/(2*self.step_x)

    def hy2_pix(self, x, y):
        if y-1 < 0:
            return (self.hy_pix(x, y+1) - self.hy_pix(x, y))/self.step_y
        elif y+1 >= self.height:
            return (self.hy_pix(x, y) - self.hy_pix(x, y-1))/self.step_y
        else:
            return (self.hy_pix(x, y+1) - self.hy_pix(x, y-1))/(2*self.step_y)

    def hxy_pix(self, x, y):
        if x-1 < 0:
            return (self.hx_pix(x, y+1) - self.hx_pix(x, y))/self.step_y
        elif x+1 >= self.width:
            return (self.hx_pix(x, y) - self.hx_pix(x, y-1))/self.step_y
        else:
            return (self.hx_pix(x, y+1) - self.hx_pix(x, y-1))/(2*self.step_y)

    def mu_pix(self, x, y):
        return self.sigma2 / 2 / self.h_pix(x, y) * np.array([self.hx_pix(x,y), self.hy_pix(x,y)])
    
    def dmu_pix(self, x, y):
        m = np.zeros([2,2])
        m[0,0] = self.hx2_pix(x,y) - self.hx_pix(x,y) / self.h_pix(x,y)
        m[0,1] = self.hxy_pix(x,y) - self.hx_pix(x,y) * self.hy_pix(x,y) / self.h_pix(x,y)
        m[1,0] = m[0,1]
        m[1,1] = self.hy2_pix(x,y) - self.hy_pix(x,y) / self.h_pix(x,y)
        return self.sigma2 / 2 / self.h_pix(x,y) * m

    def h(self, x, y):
        l = self.find_square(x,y)
        s = 0
        for e in l:
            s += self.h_pix(e[0], e[1]) * e[2]
        return s

    def mu(self, x, y):
        l = self.find_square(x,y)
        s = np.array([0., 0.])
        for e in l:
            s += self.mu_pix(e[0], e[1]) * e[2]
        return s
    
    def dmu(self, x, y):
        l = self.find_square(x,y)
        s = np.zeros([2,2])
        for e in l:
            s += self.dmu_pix(e[0], e[1]) * e[2]
        return s
    
    def in_map_pix(self, x, y):
        assert (x % 1 == 0) & (y % 1 == 0), "x and y must be integers (float but representing an integer)"
        return self.data[int(y), int(x)] > 0

    def in_map(self, x, y):
        l = self.find_square(x, y)
        if l == []:
            return False
        bool = True
        for e in l:
            bool = bool & self.in_map_pix(e[0], e[1])
        return bool
    
    def new_point(self, x, y, fvx, fvy, deltat):
        x_pix, y_pix =  ~self.transform * (x, y)
        fvx_pix = fvx / self.transform.a
        fvy_pix = fvy / self.transform.e
        return self.rec_new_point(x_pix, y_pix, fvx_pix, fvy_pix, deltat, 0, 0, [])


    def rec_new_point(self, x_pix, y_pix, fvx_pix, fvy_pix, deltat, old_deltat, old_theta, l):
            assert deltat >= 0, "deltat must be positive"
            if (int(x_pix - 0.5) == int(x_pix + deltat*fvx_pix - 0.5)) & (int(y_pix - 0.5) == int(y_pix + deltat*fvy_pix - 0.5)):
                x, y = self.transform * (x_pix + deltat*fvx_pix, y_pix + deltat*fvy_pix)
                fvx = fvx_pix * self.transform.a
                fvy = fvy_pix * self.transform.e
                l.append((old_theta, old_deltat))
                return (np.array([x, y, fvx, fvy]), l)
            else:
                deltat_x = get_deltat(x_pix, fvx_pix)
                deltat_y = get_deltat(y_pix, fvy_pix)
                assert (deltat_x != np.inf) | (deltat_y != np.inf), "velocity is null"
                if deltat_x < deltat_y:
                    deltat_min = min(deltat_x, deltat)
                    x_pix += deltat_min*fvx_pix
                    y_pix += deltat_min*fvy_pix
                    bool, theta = self.new_point_x(x_pix, y_pix, np.arctan2(fvy_pix, fvx_pix))
                else:
                    deltat_min = min(deltat_y, deltat)
                    x_pix += deltat_min*fvx_pix
                    y_pix += deltat_min*fvy_pix
                    bool, theta = self.new_point_y(x_pix, y_pix, np.arctan2(fvy_pix, fvx_pix))
                old_deltat += deltat
                if bool :
                    l.append((old_theta, old_deltat))
                    old_deltat = 0
                    old_theta = theta
                    fvx_pix, fvy_pix = np.array([[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ np.array([fvx_pix, fvy_pix])
                return self.rec_new_point(x_pix, y_pix, fvx_pix, fvy_pix, deltat - deltat_min, old_deltat, old_theta, l)

    def new_point_x(self, x, y, theta):
        if int(y - 0.5) == y - 0.5:
            if self.data[np.floor(y).astype(int), np.floor(x).astype(int)] <= 0:
                #print("bong")
                return (True, np.pi/2 - 2 * theta)
            else:
                return (False, 0)
        else:
            if (self.data[np.floor(y-0.5).astype(int), np.floor(x).astype(int)] <= 0) & (self.data[(np.ceil(y-0.5).astype(int), np.floor(x).astype(int))] <= 0):
                #print("bong")
                return (True, np.pi - 2 * theta)
            else:
                return (False, 0)
    
    def new_point_y(self, x, y, theta):
        if int(x - 0.5) == x - 0.5:
            if self.data[np.floor(y).astype(int), np.floor(x).astype(int)] <= 0:
                #print("bong")
                return (True, np.pi/2 - 2 * theta)
            else:
                return (False, 0)
        else:
            if (self.data[np.floor(y).astype(int), np.floor(x-0.5).astype(int)] <= 0) & (self.data[(np.floor(y).astype(int), np.ceil(x-0.5).astype(int))] <= 0):
                #print("bong")
                return (True, - 2 * theta)
            else:
                return (False, 0)
    

def get_deltat(z, v):
    if np.sign(v) == 1:
        return (np.floor(z-0.5) + 1.5 - z)/v
    elif np.sign(v) == -1:
        return (np.ceil(z-0.5) - 0.5 -z)/v
    else:
        return np.inf        
