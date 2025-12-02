"""
Ce module est celui qui contient les éléments de bases au niveau de la logique pour les différents algorithmes.
Les Points représentent les tétrahèdres, les NPoints les ensembles de n tétrahèdres qui forment une configuration de réseau et enfin les Set of NPoints qui sont un ensemble des réseauw qui ont été testés soit pour l'algorithme Bayesien soit pour l'algorithme génétique
Il y a, à chaque fois une classe normale, une classe Bayesienne et une Génétique car les deux dernières ont des spécificités différentes :
- Un Point Bayesien est dans le fleuve, il ne peut pas être sur Terre
- Dans un NPoint Bayesien, chaque tétrahèdre est à une distance inférieure à 2 range_max (range_max correspond à la distance maximale à laquelle un béluga ne peut être détecté), cela permet d'être sur que chaque tétrahèdre est utile (un tétrahèdre isolé étant inutile)
- Dans un NPoint Bayesien les tétrahdères sont classés par ordre croissant (en fonction de leurs coordonnées), cela sert à évitér de calculer plusieurs fois la même chose : chaque tétrahèdre est inter-changeable
- Dans un Set of NPoints Bayesien il ne faut pas qu'il y ait deux fois le même point sinon la matrice de covariance est non inversible

This module contains the basic logical components used by the different algorithms.
Points represent the individual tetrahedra, NPoints represent sets of n tetrahedra forming a network configuration, and Sets of NPoints are collections of networks that have been tested either by the Bayesian algorithm or by the genetic algorithm
For each of these, there are three versions: a standard class, a Bayesian class, and a Genetic class, because the last two have specific constraints:
- A Bayesian Point must lie in the river; it cannot be located on land
- In a Bayesian NPoint, each tetrahedron must be within a distance smaller than 2 range_max (where range_max is the maximum distance at which a beluga can be detected). This ensures that every tetrahedron is useful — an isolated one would be useless
- In a Bayesian NPoint, the tetrahedra are sorted in ascending order (based on their coordinates). This avoids recomputing equivalent configurations, since tetrahedra are interchangeable.
- In a Bayesian Set of NPoints, no configuration may appear twice; otherwise, the covariance matrix becomes singular
"""

import numpy as np
import random
import hydrophone_placement_scripts.utils_scripts.utils as ut


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from  hydrophone_placement_scripts.to_optimise.topo import Topo# Only for typing
from typing import Callable

class Point:
    # xmin = 0
    # ymin = 0
    # xmax = 1
    # ymax = 1
    # lx = np.ceil(np.log2(1 + (xmax-xmin))).astype(int)
    # ly = np.ceil(np.log2(1 + (ymax-ymin))).astype(int)
    # dx = (xmax-xmin)/(2**lx-1)
    # dy = (ymax-ymin)/(2**ly-1)
    # params_cor = np.array([2, 2, 1/2/(40**2)]) #utils.esp_diff(xmax-xmin), 40 is an average 
    topo = None
    height_sensor = None
    
    @classmethod
    def update_point(cls, n_areas_x : int, n_areas_y : int, width_area : float):
        cls.n_areas_x = n_areas_x
        cls.n_areas_y = n_areas_y
        cls.lx = np.ceil(np.log2(n_areas_x)).astype(int)
        cls.ly = np.ceil(np.log2(n_areas_y)).astype(int)
        cls.params_cor = np.array([2/((n_areas_x * width_area)**2), 2/((n_areas_y * width_area)**2), 1/2/(40**2)])
        return None

    @classmethod
    def set_topo(cls, topo : "Topo", height_sensor : float):
        cls.topo = topo
        cls.height_sensor = height_sensor
        return None
    
    def __init__(self, area : tuple[int, int] | None = None):
        if area is None:
            self.area = (random.randint(0, self.n_areas_x), random.randint(0, self.n_areas_x))
        else:
            self.area = area
        self.coords = self.topo.converter.area2utm(self.area)
    
    def modif(self, type):
        if type == 0:
            self.area[0] += 1
        elif type == 1:
            self.area[0] -= 1
        elif type == 2:
            self.area[1] += 1
        else :
            self.area[1] -= 1
        self.coords = self.topo.converter.area2utm(self.area)

    def __repr__(self):
        return self.topo.converter.utm2lla(self.coords[0], self.coords[1]).__repr__()

    def in_water(self):
        return (self.depth()>0)
    
    def depth(self):
        return self.topo.dic_depths.get(self.area, 0) - self.height_sensor

    def dist_xy(self, x : float, y : float):
        return np.sqrt((self.coords[0] - x)**2 + (self.coords[1] - y)**2)

    def dist_point_xy(self, other : "Point"):
        c = other.coords
        return self.dist_xy(c[0], c[1])
    
    def dist_withdepth(self, x : float, y : float, d : float):
        return np.sqrt((self.coords[0] - x)**2 + (self.coords[1] - y)**2 + (self.depth() - d)**2)

    def dist_point_withdepth(self, other : "Point"):
        c = other.coords
        d = other.depth()
        return self.dist_withdepth(c[0], c[1], d)
        
    def near(self, other :  "Point", range : float):
        return self.dist_point_xy(other) <= 2*range

    def norme(self, other : "Point"):
        """
        Calcule la norme que l'on utilisera pour la corrélation dans l'algorithme bayesien
        Compute the norm that will be used for correlation in the Bayesian algorithm
        """
        return (self.params_cor[0] * ((abs(self.coords[0] - other.coords[0])) ** 2)
                + self.params_cor[1] * ((abs(self.coords[1] - other.coords[1])) ** 2)
                + self.params_cor[2] * ((abs(self.depth() - other.depth())) ** 2))

    # def inf(self, other : "Point"):
    #     return ut.bin_inf(self.bin_coords, other.bin_coords)
            
    def __eq__(self, other : "Point"):
        return self.area == other.area

class PointBayesian (Point):
    max_compt = 10000

    def __init__(self, area : tuple[int, int] | None = None):
        if area is None:
            compt = 0
            super().__init__()
            while not self.in_water():
                assert compt < PointBayesian.max_compt, "too many attempts to create a point in water"
                compt +=1
                super().__init__()
        else:
            super().__init__(area)
            assert self.in_water(), "the PointBayesian is not bayesian"
        
    def create_random_point(self, range : float):
        """
        Utile pour create_new_pointbayesian pour créer des cooordonnées qui sont dans la zone à une distance inférieure de range au Point
        Useful for create_new_pointbayesian to generate coordinates that lie within the area at a distance smaller than range from the Point
        """
        d = np.sqrt(random.uniform(0, (2*range)**2))
        theta = random.uniform(0, 2*np.pi)
        area = self.topo.converter.utm2area(d * np.cos(theta) + self.coords[0], d * np.sin(theta) + self.coords[1])
        return area
    
    def create_new_pointbayesian(self, range : float):
        """
        Crée un Point Bayesien a proximité, pour s'assurer que les points ne soient pas isolés
        Create a nearby Bayesian Point to ensure that the points are not isolated
        """
        area = self.create_random_point(range)
        compt = 0
        while not Point(area=area).in_water():
            assert compt < PointBayesian.max_compt, "too many attempts to create a new point in water"
            compt +=1
            area = self.create_random_point(range)
        return PointBayesian(area=area)

    def to_genetic(self):
        return PointGenetic(area = self.area)

class PointGenetic (Point):

    def __init__(self, area : tuple[int, int] | None = None):
        super().__init__(area)
        
class NPoint :
    n_tetrahedras = 2
    range = np.inf

    def __init__(self, points : list[Point] | None = None, nareas : list[tuple[int,int]] | None = None):
        if points is None:
            self.points = []
            if nareas is None:
                for i in range (NPoint.n_tetrahedras):
                    self.points.append(Point())
            else:
                assert len(nareas) == NPoint.n_tetrahedras, "nareas is not the right size"
                for i in range (NPoint.n_tetrahedras):
                    self.points.append(Point(nareas[i]))
        else:
            assert len(points) == NPoint.n_tetrahedras, "There is not the good amount of points"
            self.points = points

    @classmethod
    def set_n_tetrahedras(cls, n_tetrahedras : int):
        cls.n_tetrahedras = n_tetrahedras

    @classmethod
    def set_range(cls, range : float):
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

    def get_nareas(self):
        nareas = []
        for i in range (NPoint.n_tetrahedras):
            nareas.append(self.points[i].area)
            return nareas

    def corr(self, other : "NPoint"):
        """
        Calcule la corrélation utilisée dans l'algorithme Bayesien entre le NPoint et un autre NPoint
        Computes the correlation used in the Bayesian algorithm between the NPoint and another NPoint
        """
        e1 = self.points
        e2 = other.points
        assert len(e1) == len(e2), "e1 and e2 do not have the same size"
        mat = np.zeros((len(e1), len(e2)))
        for i in range (len(e1)):
            for j in range (len(2)):
                mat[i,j] = e1[i].norme(e2[j])
        return np.exp(-1/2 * mat.sum(axis = 0).min())

    def in_water(self):
        for point in self.points:
            if not point.in_water():
                return False
        return True
    
    def verify_range(self):
        """
        Pour vérifier si aucun point n'est isolé
        To check wether one Point is isolated
        """
        t = np.array([False] * self.n_tetrahedras)
        for i in range(len(self.points)):
            for j in range(i):
                if self.points[i].near(self.points[j]):
                    t[i] = True
                    t[j] = True
        return t.all()

    @classmethod
    def set_value(cls, value : Callable[["NPoint"], float]):
        cls.value = lambda self : value(self)
        return None
        
    def __eq__(self, other : "NPoint"):
        return self.points == other.points

    def is_in(self, set_of_npoints : "Set_Of_NPoints"):
        """
        Pour savoir si un NPoint est dans un set_of_npoints
        To know if a NPoint is in a set_of_npoints or not"""
        for npoint in set_of_npoints.set_of_npoints:
            if self == npoint:
                return True
        return False

class NPointBayesian (NPoint):

    def __init__(self, points : list[Point] | None = None, nareas : list[tuple[int,int]] | None = None):
        assert self.n_tetrahedras > 1, "the number of sensors must be at least 2"
        if points is None:
            points = []
            if nareas is None:
                t = np.array([False] * NPoint.n_tetrahedras) # To check wether the Points are not isolated
                points = []
                for _ in range (NPoint.n_tetrahedras):
                    points.append(PointBayesian())
                for i in range(NPoint.n_tetrahedras):
                    for j in range(i):
                        if points[i].near(points[j]): # To check wether the Points are not isolated
                            t[i] = True
                            t[j] = True
                tfalse = np.where(~t)[0] #While some points are isolated, we replace the isolated points by a non isolated one
                while len(tfalse) != 0:
                    i = tfalse[0]
                    j = random.choice([x for x in range(NPoint.n_tetrahedras) if x != i])
                    points[i] = points[j].create_new_pointbayesian(self.range)
                    for j in range(NPoint.n_tetrahedras):
                        if points[i].near(points[j]): 
                            t[j] = True
                    tfalse = np.where(~t)[0]
                super().__init__(points = points)
            else:
                assert len(nareas) == NPoint.n_tetrahedras, "nareas is not the right size"
                for i in range (NPoint.n_tetrahedras):
                    ut.sort_insert_point(PointBayesian(nareas[i]), points)
                super().__init__(points = points)
        else:
            super().__init__(points = points)
            assert self.in_water() & self.verify_range(), "the NPointBayesian is not Bayesian"

    # def to_genetic(self):
    #     return NPointGenetic(nareas=self.nareas())

class NPointGenetic (NPoint):

    def __init__(self, points : list[Point] | None = None, nareas : list[tuple[int,int]] | None = None):
        if points is None:
            points = []
            if nareas is None:
                t = np.array([False] * NPoint.n_tetrahedras) # To check wether the Points are not isolated
                points = []
                for _ in range (NPoint.n_tetrahedras):
                    points.append(PointBayesian().to_genetic())
                for i in range(NPoint.n_tetrahedras):
                    for j in range(i):
                        if points[i].near(points[j]): # To check wether the Points are not isolated
                            t[i] = True
                            t[j] = True
                tfalse = np.where(~t)[0] #While some points are isolated, we replace the isolated points by a non isolated one
                while len(tfalse) != 0:
                    i = tfalse[0]
                    j = random.choice([x for x in range(NPoint.n_tetrahedras) if x != i])
                    points[i] = points[j].create_new_pointbayesian(self.range)
                    for j in range(NPoint.n_tetrahedras):
                        if points[i].dist_point_xy(points[j]) <= 2*self.range: 
                            t[j] = True
                    tfalse = np.where(~t)[0]
                super().__init__(points = points)
            else:
                assert len(nareas) == NPoint.n_tetrahedras, "nareas is not the right size"
                for i in range (NPoint.n_tetrahedras):
                    ut.sort_insert_point(PointBayesian(nareas[i]), points)
                super().__init__(points = points)
        else:
            super().__init__(points = points)
            assert self.in_water() & self.verify_range(), "the NPointBayesian is not Bayesian"

    def __init__(self, points : list[Point] | None = None, nareas : list[tuple[int,int]] | None = None):
        if points is None:
            points = []
            if nareas is None:
                for _ in range (NPoint.n_tetrahedras):
                    points.append(PointGenetic())
                super().__init__(points = points)
            else:
                assert len(nareas) == NPoint.n_tetrahedras, "nareas is not the right size"
                for i in range (NPoint.n_tetrahedras):
                    points.append(PointGenetic(nareas[i]))
                super().__init__(points = points)
        else:
            for point in points:
                assert isinstance(point, PointGenetic), "one point is not Genetic"
            super()._init(points = points)

    @classmethod
    def set_p_mut(cls, p_mut : float):
        cls.p_mut = p_mut
        return None
        
    def breed(self, other : "NPointGenetic", ks : list[int]):
        """
        Pour créer un nouveau NPoint à partir de deux parents
        To create a new NPoint thanks to two parents
        """
        
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

    # def corr(self, other : "NPoint"):
    #     """
    #     Calcule la corrélation entre deux NPoints pour après la mettre dans la matrice de covariance
    #     Computes the correlation between two NPoints to then place it in the covariance matrix
    #     """
    #     e1 = self.points
    #     e2 = other.points
    #     assert len(e1) == len(e2), "e1 and e2 do not have the same size"
    #     l=[]
    #     for i in range (NPoint.n_tetrahedras): #I am not sure why I wrote that because the points should already be sorted
    #         ut.sort_insert_point(e1[i], l)
    #     s = 0
    #     for i in range (NPoint.n_tetrahedras):
    #         s += l[i].norme(e2[i])
    #     return np.exp(-1/2 * s)

    def to_bayesian(self):
        return NPointBayesian(nareas=self.nareas())

class Set_Of_NPoints:

    def __init__(self, set_of_npoints : list[NPoint] | None = None , values : list[float] | None = None, size : int = 0, l_nareas : list[list[tuple[int,int]]] | None = None):
        if set_of_npoints is None:
            set_of_npoints = []
            if l_nareas is None:
                for _ in range (size):
                    set_of_npoints.append(NPoint())
            else:
                for nareas in l_nareas:
                    set_of_npoints.append(NPoint(nareas = nareas))
        self.set_of_npoints = set_of_npoints
        self.size = len(set_of_npoints)
        if values is None:
            self.values = [npoint.value() for npoint in self.set_of_npoints]      
        else:
            assert len(set_of_npoints) == len(values), "set_of_npoints and values don't have the same length"
            self.values = values

    def __repr__(self):
        return self.set_of_npoints.__repr__()

    def add_npoint(self, npoint : NPoint, value: float | None = None):
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

    def k_best(self, k : int):
        l = ut.k_best(self.values, k)
        return self.__class__(set_of_npoints = [self.set_of_npoints[i] for i in l])

    def sort(self):
        l = ut.k_best(self.values, self.size)
        set_of_npoints = [self.set_of_npoints[i] for i in l]
        values = [self.values[i] for i in l]
        self.set_of_npoints = set_of_npoints
        self.values = values

    def l_nareas_values(self):
        l_nareas = []
        for npoint in self.set_of_npoints:
            l_nareas.append(npoint.nareas())
        return (l_nareas, self.values)

class Set_of_NPointsBayesian (Set_Of_NPoints):
    max_compt = 5

    def __init__(self, set_of_npoints : list[NPoint] | None = None , values : list[float] | None = None, size : int = 0, l_nareas : list[list[tuple[int,int]]] | None = None):
        super().__init__()
        if set_of_npoints is None:
            if l_nareas is None:
                for _ in range(size):
                    self.add_npoint()
            else:
                if values is None:
                    for nareas in l_nareas:
                        self.add_npoint(NPointBayesian(nareas=nareas))
                else:
                    assert len(l_nareas) == len(values), "l_nareas and values don't have the same length" 
                    for nareas, value in zip(l_nareas, values):
                        self.add_npoint(NPointBayesian(nareas=nareas), value = value)
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


    def add_npoint(self, npoint : NPoint | None = None, value : float | None =  None):
        if npoint is None:
            npoint = NPointBayesian()
            compt = 0
            while npoint.is_in(self):
                assert compt < Set_of_NPointsBayesian.max_compt, "too many tries to get a npoint bayesian not in the set of points"
                compt+=1
                npoint = NPointBayesian()
        else:
            assert isinstance(npoint, NPointBayesian), "the new npoint is not bayesian"
            assert not npoint.is_in(self), "the npoint is already in the set of points, Sigma is now singular"
        super().add_npoint(npoint, value)

class Set_of_NPointsGenetic (Set_Of_NPoints):
    max_compt_parents = 100
    max_compt_ks = 10

    def __init__(self, set_of_npoints : list[NPoint] | None = None , values : list[float] | None = None, size : int = 0, l_nareas : list[list[tuple[int,int]]] | None = None):
        if set_of_npoints is None:
            set_of_npoints = []
            if l_nareas is None:
                for _ in range (size):
                    set_of_npoints.append(NPointBayesian().to_genetic())
            else:
                for nareas in l_nareas:
                    set_of_npoints.append(NPointGenetic(nareas=nareas))
        else :
            for npoint in set_of_npoints:
                assert isinstance(npoint, NPointGenetic), "one npoint is not a genetic one"
        super().__init__(set_of_npoints = set_of_npoints, values = values)

    def add_npoint(self, npoint : NPoint | None = None, value : float | None = None):   
        if npoint is None:
            npoint = NPointGenetic()         
        assert isinstance(npoint, NPointGenetic), "the new npoint is not Genetic"
        return super().add_npoint(npoint, value)

    def k_best(self, k : int):
        return super().k_best(k)
    
    @classmethod
    def set_probs_n_sep(cls, probs : list[float], n_sep : int):
        cls.probs = probs
        cls.n_sep = n_sep

    def breed(self, set_of_parents : "Set_of_NPointsGenetic"):
        """
        Pour créer un nouvel set_of_individuals à partir d'un set_of_parents
        To create a new set_of_individuals thanks to a set_of_parents
        """
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
            k = random.randint(0, NPoint.n_tetrahedras)
            while k in ks:
                assert compt < Set_of_NPointsGenetic.max_compt_ks, "too many attemps to get the ks"
                compt +=1
                k = random.randint(0, NPoint.n_tetrahedras)      
                ks.append(k)
        ks.sort()
        self.add_npoint(set_of_parents.set_of_npoints[i_par1].breed(set_of_parents.set_of_npoints[i_par2], ks))

    def nb_bayesian(self):
        """
        C'est un indicateur pour connaitre le nombre de NPoints dont tous les points sont dans l'eau et non isolés
        It is an indicator used to determine the number of NPoints for which all points lie in the water and none are isolated
        """
        return len([x for x in self.values if x>=0])