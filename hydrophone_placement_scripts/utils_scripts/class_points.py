#implémenter update npoint

import numpy as np
import random
import hydrophone_placement_scripts.utils_scripts.utils as ut

class Point:
    xmin = 0
    ymin = 0
    xmax = 1
    ymax = 1
    accuracy = 1 # units : area
    lx = np.ceil(np.log2(1 + (xmax-xmin)/accuracy)).astype(int)
    ly = np.ceil(np.log2(1 + (ymax-ymin)/accuracy)).astype(int)
    dx = (xmax-xmin)/(2**lx-1)
    dy = (ymax-ymin)/(2**ly-1)
    log_params_cor = np.log([2, 2, 1/2/(40**2), 2, 2, 2]) #utils.esp_diff(xmax-xmin), 40 is an average 
    topo = None
    height_sensor = None
    
    @classmethod
    def update_point(cls, xmin, ymin, xmax, ymax, accuracy = 1):
        cls.accuracy = accuracy
        cls.xmin = xmin
        cls.ymin = ymin
        cls.xmax = xmax
        cls.ymax = ymax
        cls.lx = np.ceil(np.log2(1 + (xmax-xmin)/cls.accuracy)).astype(int)
        cls.ly = np.ceil(np.log2(1 + (ymax-ymin)/cls.accuracy)).astype(int)
        cls.dx = (xmax - xmin)/(2**cls.lx-1)
        cls.dy = (ymax - ymin)/(2**cls.ly-1)
        cls.log_params_cor = np.log([2/((xmax - xmin)**2), 2/((ymax - ymin)**2), 1/2/(40**2)])
        return None

    @classmethod
    def set_topo(cls, topo, height_sensor):
        cls.topo = topo
        cls.height_sensor = height_sensor
        return None
    
    def __init__(self, bin_coords=None):
        if bin_coords is None:
            bin_coords = [random.randint(0, 1) for _ in range(Point.lx + Point.ly)]
        else:
            assert len(bin_coords) == Point.lx + Point.ly, "bin_coords does not have the right size"
        self.bin_coords = bin_coords
        self.coords = (self.xmin + ut.bin2int(self.bin_coords[0: Point.lx]) * self.dx, self.ymin + ut.bin2int(self.bin_coords[Point.lx: ])*self.dy)
    
    def __repr__(self):
        return self.topo.converter.utm2lla(self.coords[0], self.coords[1]).__repr__()

    def in_water(self):
        return (self.depth()>0)
    
    def depth(self):
        x, y = self.coords
        return self.topo.dic_depths[self.topo.converter.utm2area(x,y)] - self.height_sensor

    def dist_xy(self, x, y):
        return np.sqrt((self.coords[0] - x)**2 + (self.coords[1] - y)**2)

    def dist_point_xy(self, other):
        c = other.coords
        return self.dist_xy(c[0], c[1])
    
    def dist_withdepth(self, x, y, d):
        return np.sqrt((self.coords[0] - x)**2 + (self.coords[1] - y)**2 + (self.depth() - d)**2)

    def dist_point_withdepth(self, other):
        c = other.coords
        d = other.depth()
        return self.dist_withdepth(c[0], c[1], d)
        
    def norme(self, other):
        return (np.exp(self.log_params_cor[0]) * ((abs(self.coords[0] - other.coords[0])) ** 2)
                + np.exp(self.log_params_cor[1]) * ((abs(self.coords[1] - other.coords[1])) ** 2)
                + np.exp(self.log_params_cor[2]) * ((abs(self.depth() - other.depth())) ** 2))

    def inf(self, other):
        return ut.bin_inf(self.bin_coords, other.bin_coords)
            
    def __eq__(self, other):
        return self.bin_coords == other.bin_coords

class PointBayesian (Point):
    max_compt = 100

    def __init__(self, bin_coords = None):
        if bin_coords is None:
            compt = 0
            super().__init__()
            while not self.in_water():
                assert compt < PointBayesian.max_compt, "too many attempts to create a point in water"
                compt +=1
                super().__init__()
        else:
            super().__init__(bin_coords)
            assert self.in_water(), "the PointBayesian is not bayesian"
        
    def create_random_point(self, range):
        compt = 0
        d = np.sqrt(random.uniform(0, (2*range)**2))
        theta = random.uniform(0, 2*np.pi)
        x = np.round((d * np.cos(theta) + self.coords[0] - self.xmin)/self.dx)
        y = np.round((d * np.sin(theta) + self.coords[1] - self.ymin)/self.dy)
        bin_coords = ut.float2bin(x, self.lx) + ut.float2bin(y, self.ly)
        while len(bin_coords) != self.lx + self.ly:
            compt +=1
            assert compt < PointBayesian.max_compt, "too many attempts to create a new point in the area"
            d = np.sqrt(random.uniform(0, (2*range)**2))
            theta = random.uniform(0, 2*np.pi)
            x = np.round((d * np.cos(theta) + self.coords[0] - self.xmin)/self.dx)
            y = np.round((d * np.sin(theta) + self.coords[1] - self.ymin)/self.dy)
            bin_coords = ut.float2bin(x, self.lx) + ut.float2bin(y, self.ly)
        return bin_coords
    
    def create_new_pointbayesian(self, range):
        bin_coords = self.create_random_point(range)
        compt = 0
        while not Point(bin_coords=bin_coords).in_water():
            assert compt < PointBayesian.max_compt, "too many attempts to create a new point in water"
            compt +=1
            bin_coords = self.create_random_point(range)
        return PointBayesian(bin_coords=bin_coords)


class PointGenetic (Point):

    def __init__(self, bin_coords = None):
        super().__init__(bin_coords)
        
class NPoint :
    n_tetrahedras = 2
    range = np.inf

    def __init__(self, points=None, nbin_coords=None): #nbin_coords is a list and not a matrix !!! 
        if points is None:
            self.points = []
            if nbin_coords is None:
                for i in range (NPoint.n_tetrahedras):
                    self.points.append(Point())
            else:
                assert len(nbin_coords) == NPoint.n_tetrahedras * (Point.lx + Point.ly), "nbin_coords is not the right size"
                for i in range (NPoint.n_tetrahedras):
                    self.points.append(Point(nbin_coords[i*(Point.lx+Point.ly) : (i+1)*(Point.lx+Point.ly)]))
        else:
            assert len(points) == NPoint.n_tetrahedras, "There is not the good amount of points"
            self.points = points

    @classmethod
    def set_n_tetrahedras(cls, n_tetrahedras):
        cls.n_tetrahedras = n_tetrahedras

    @classmethod
    def set_range(cls, range):
        cls.range = range

    def __repr__(self):
        return self.points.__repr__()

    def coords(self):
        coords = []
        for i in range (NPoint.n_tetrahedras):
            coords.append(self.points[i].coords)
        return coords
    
    def nbin_coords(self):
        nbin_coords = []
        for i in range (NPoint.n_tetrahedras):
            nbin_coords += self.points[i].bin_coords
        return nbin_coords

    def corr(self, other):
        e1 = self.points
        e2 = other.points
        assert len(e1) == len(e2), "e1 and e2 do not have the same size"
        s = 0
        for i in range (NPoint.n_tetrahedras):
            s += e1[i].norme(e2[i])
        return np.exp(-1/2 * s)

    def in_water(self):
        for point in self.points:
            if not point.in_water():
                return False
        return True
    
    def verify_range(self):
        t = np.array([False] * self.n_tetrahedras)
        for i in range(len(self.points)):
            for j in range(i):
                if self.points[i].dist_point_xy(self.points[j]) <= 2*self.range:
                    t[i] = True
                    t[j] = True
        return t.all()

    @classmethod
    def set_value(cls, value):
        cls.value = lambda self : value(self)
        return None
        
    def __eq__(self, other):
        return self.points == other.points

    def is_in(self, set_of_npoints):
        for npoint in set_of_npoints:
            if self == npoint:
                return True
        return False

class NPointBayesian (NPoint):

    def __init__(self, points = None, nbin_coords=None):
        assert self.n_tetrahedras > 1, "the number of sensors must be at least 2"
        if points is None:
            points = []
            if nbin_coords is None:
                t = np.array([False] * NPoint.n_tetrahedras)
                points = []
                for _ in range (NPoint.n_tetrahedras):
                    points.append(PointBayesian())
                for i in range(NPoint.n_tetrahedras):
                    for j in range(i):
                        if points[i].dist_point_xy(points[j]) <= 2*self.range:
                            t[i] = True
                            t[j] = True
                tfalse = np.where(~t)[0]
                while len(tfalse) != 0:
                    i = tfalse[0]
                    j = random.choice([x for x in range(NPoint.n_tetrahedras) if x != i])
                    points[i] = points[j].create_new_pointbayesian(self.range)
                    for j in range(NPoint.n_tetrahedras):
                        if points[i].dist_point_xy(points[j]) <= 2*self.range: 
                            t[j] = True
                    tfalse = np.where(~t)[0]
                points2 = []
                for point in points:
                    ut.sort_insert_point(point, points2)
                super().__init__(points = points)
            else:
                assert len(nbin_coords) == NPoint.n_tetrahedras * (Point.lx + Point.ly), "nbin_coords is not the right size"
                for i in range (NPoint.n_tetrahedras):
                    ut.sort_insert_point(PointBayesian(nbin_coords[i*(Point.lx+Point.ly) : (i+1)*(Point.lx+Point.ly)]), points)
                super().__init__(points = points)
        else:
            super().__init__(points = points)
            assert self.in_water() & self.verify_range(), "the NPointBayesian is not Bayesian"

    def to_genetic(self):
        return NPointGenetic(nbin_coords=self.nbin_coords())

class NPointGenetic (NPoint):

    def __init__(self, points = None, nbin_coords=None):
        if points is None:
            points = []
            if nbin_coords is None:
                for _ in range (NPoint.n_tetrahedras):
                    points.append(PointGenetic())
                super().__init__(points = points)
            else:
                assert len(nbin_coords) == NPoint.n_tetrahedras * (Point.lx + Point.ly), "nbin_coords is not the right size"
                for i in range (NPoint.n_tetrahedras):
                    points.append(PointGenetic(nbin_coords[i*(Point.lx+Point.ly) : (i+1)*(Point.lx+Point.ly)]))
                super().__init__(points = points)
        else:
            for point in points:
                assert isinstance(point, PointGenetic), "one point is not Genetic"
            super()._init(points = points)

    @classmethod
    def set_p_mut(cls, p_mut):
        cls.p_mut = p_mut
        return None

    def esp_improv(self):
        return 0
        
    def breed(self, other, ks):
        b1 = self.nbin_coords()
        b2 = other.nbin_coords()
        p_mut = self.p_mut
        assert len(b1) == len(b2), "the two nbin_coords of the parents do not have the same size"
        new_nbin_coords = [0] * len(b1)
        bool_par = True
        for i in range (len(b1)):
            if i in ks:
                bool_par = not bool_par
            if bool_par:
                new_nbin_coords[i] = b1[i] if random.random()>p_mut else 1-b1[i]
            else:
                new_nbin_coords[i] = b2[i] if random.random()>p_mut else 1-b2[i]
        return NPointGenetic(nbin_coords = new_nbin_coords)

    def corr(self, other):
        e1 = self.points
        e2 = other.points
        assert len(e1) == len(e2), "e1 and e2 do not have the same size"
        l=[]
        for i in range (NPoint.n_tetrahedras):
            ut.sort_insert_point(e1[i], l)
        s = 0
        for i in range (NPoint.n_tetrahedras):
            s += l[i].norme(e2[i])
        return np.exp(-1/2 * s)

    def to_bayesian(self):
        return NPointBayesian(nbin_coords=self.nbin_coords())

class Set_Of_NPoints:

    def __init__(self, set_of_npoints = None, values = None, size = 0, l_nbin_coords = None):
        if set_of_npoints is None:
            set_of_npoints = []
            if l_nbin_coords is None:
                for _ in range (size):
                    set_of_npoints.append(NPoint())
            else:
                for nbin_coords in l_nbin_coords:
                    set_of_npoints.append(NPoint(nbin_coords = nbin_coords))
        self.set_of_npoints = set_of_npoints
        self.size = len(set_of_npoints)
        if values is None:
            self.values = [npoint.value() for npoint in self.set_of_npoints]      
        else:
            assert len(set_of_npoints) == len(values), "set_of_npoints and values don't have the same length"
            self.values = values

    def __repr__(self):
        return self.set_of_npoints.__repr__()

    def add_npoint(self, npoint, value = None):
        if npoint is None:
            npoint = NPoint()
        self.set_of_npoints.append(npoint)
        if value is None:
            self.values.append(npoint.value())
        else:
            self.values.append(value)
        self.size += 1        

    def argmax(self):
        arg = ut.argmax(self.values)
        return arg

    def argmin(self):
        arg = ut.argmin(self.values)
        return arg

    def k_best(self, k):
        l = ut.k_best(self.values, k)
        return self.__class__(set_of_npoints = [self.set_of_npoints[i] for i in l])

    def sort(self):
        l = ut.k_best(self.values, self.size)
        set_of_npoints = [self.set_of_npoints[i] for i in l]
        values = [self.values[i] for i in l]
        self.set_of_npoints = set_of_npoints
        self.values = values

    def l_nbin_coords_values(self):
        l_nbin_coords = []
        for npoint in self.set_of_npoints:
            l_nbin_coords.append(npoint.nbin_coords())
        return (l_nbin_coords, self.values)

class Set_of_NPointsBayesian (Set_Of_NPoints):
    max_compt = 5

    def __init__(self, set_of_npoints = None, values = None, size = 0, l_nbin_coords = None):
        super().__init__()
        if set_of_npoints is None:
            if l_nbin_coords is None:
                for _ in range(size):
                    self.add_npoint()
            else:
                if values is None:
                    for nbin_coords in l_nbin_coords:
                        self.add_npoint(NPointBayesian(nbin_coords=nbin_coords))
                else:
                    assert len(l_nbin_coords) == len(values), "l_nbin_coords and values don't have the same length" 
                    for nbin_coords, value in zip(l_nbin_coords, values):
                        self.add_npoint(NPointBayesian(nbin_coords=nbin_coords), value = value)
        else:
            if values is None:
                for npoint in set_of_npoints:
                    assert isinstance(npoint, NPointBayesian),  "one npoint is not a bayesian one"
                    self.add_npoint(npoint)
            else:
                assert len(set_of_npoints) == len(values), "set_of_npoints and values don't have the same length" 
                for npoint, v in zip(set_of_npoints, values):
                    assert isinstance(npoint, NPointBayesian),  "one npoint is not a bayesian one"
                    self.add_npoint(npoint, value = v)


    def add_npoint(self, npoint = None, value =  None):
        if npoint is None:
            npoint = NPointBayesian()
            compt = 0
            while npoint.is_in(self.set_of_npoints):
                assert compt < Set_of_NPointsBayesian.max_compt, "too many tries to get a npoint bayesian not in the set of points"
                compt+=1
                npoint = NPointBayesian()
        else:
            assert isinstance(npoint, NPointBayesian), "the new npoint is not bayesian"
            assert not npoint.is_in(self.set_of_npoints), "the npoint is already in the set of points, R is now singular"
        super().add_npoint(npoint, value)

class Set_of_NPointsGenetic (Set_Of_NPoints):
    max_compt_parents = 100
    max_compt_ks = 10

    def __init__(self, set_of_npoints = None, values = None, size = 0, l_nbin_coords = None):
        if set_of_npoints is None:
            set_of_npoints = []
            if l_nbin_coords is None:
                for _ in range (size):
                    set_of_npoints.append(NPointBayesian().to_genetic())
            else:
                for nbin_coords in l_nbin_coords:
                    set_of_npoints.append(NPointGenetic(nbin_coords=nbin_coords))
        else :
            for npoint in set_of_npoints:
                assert isinstance(npoint, NPointGenetic), "one npoint is not a genetic one"
        super().__init__(set_of_npoints = set_of_npoints, values = values)

    def add_npoint(self, npoint, value = None):   
        if npoint is None:
            npoint = NPointGenetic()         
        assert isinstance(npoint, NPointGenetic), "the new npoint is not Genetic"
        return super().add_npoint(npoint, value)

    def k_best(self, k):
        return super().k_best(k)
    
    @classmethod
    def set_probs_n_sep(cls, probs, n_sep):
        cls.probs = probs
        cls.n_sep = n_sep

    def breed(self, set_of_parents):
        i_par1 = random.choices(range(set_of_parents.size), self.probs)[0]
        i_par2 = random.choices(range(set_of_parents.size), self.probs)[0]
        compt = 0
        while i_par2 == i_par1:
            assert compt < Set_of_NPointsGenetic.max_compt_parents, "too many attempts to get a second parent"
            compt += 1
            i_par2 = random.choices(range(set_of_parents.size), self.probs)[0]

        ks = []
        for _ in range (self.n_sep):
            compt = 0
            k = random.randint(0, NPoint.n_tetrahedras * (Point.lx + Point.ly) - 1)
            while k in ks:
                assert compt < Set_of_NPointsGenetic.max_compt_ks, "too many attemps to get the ks"
                compt +=1
                k = random.randint(0, NPoint.n_tetrahedras * (Point.lx + Point.ly) - 1)      
                ks.append(k)
        ks.sort()
        self.add_npoint(set_of_parents.set_of_npoints[i_par1].breed(set_of_parents.set_of_npoints[i_par2], ks))

    def nb_bayesian(self):
        return len([x for x in self.values if x>=0])