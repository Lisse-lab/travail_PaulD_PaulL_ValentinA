import numpy as np
import math
from scipy import optimize
import pickle
import os

import hydrophone_placement_scripts.utils_scripts.utils as ut
import hydrophone_placement_scripts.utils_scripts.class_points as cls_points
import hydrophone_placement_scripts.optimisation.genetic_algo as ga

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from hydrophone_placement_scripts.utils_scripts.conversions_coordinates import Conv # Only for typing
    from hydrophone_placement_scripts.to_optimise.func_to_optimise import Calculator
    from hydrophone_placement_scripts.utils_scripts.class_points import NPoint

class Bayesian_Process:
    n_first_points = 50
    min_expected_improvement = 0.001
    max_iter = 250
    ksi = 0.01
    sigmaf = 1
    sigman = 0.1

    def __init__(self, path : str = None, l_nareas : list[list[tuple[int,int]]] | None = None, values : list[float] | None = None, expected_improvements : list[float] | None = None):
        self.path = path
        if l_nareas is None:
            self.iter = 0
            self.expected_improvements = []
            self.set_of_npointsbayesian = cls_points.Set_of_NPointsBayesian()
        else :
            assert not values is None, "values is None while l_nareas is not"
            assert not expected_improvements is None, "expected_improvements is None while l_nareas is not"
            self.set_of_npointsbayesian = cls_points.Set_of_NPointsBayesian(l_nareas=l_nareas, values=values)
            self.expected_improvements = expected_improvements
            self.iter = len(values)

    def modify(self, **kwargs):
        kwargs_b = {k[2:]: v for k, v in kwargs.items() if k.startswith("b_")}
        for attr, value in kwargs_b.items():
            if not hasattr(self, attr):
                raise AttributeError(f"Bayesian_Process has no attribute '{attr}'.")
            setattr(self, attr, value)
        self.kwargs_ga = {k[3:]: v for k, v in kwargs.items() if k.startswith("ga_")}

    def update_Sigma_invSigma(self):
        self.Sigma = self.sigmaf ** 2 * np.matrix([[npoint1.corr(npoint2) for npoint2 in self.set_of_npointsbayesian.set_of_npoints] for npoint1 in self.set_of_npointsbayesian.set_of_npoints]) + self.sigman ** 2 * np.eye(self.set_of_npointsbayesian.size)
        self.invSigma = np.linalg.inv(self.Sigma)

    def update_mu(self):
        self.mu = (np.ones(self.set_of_npointsbayesian.size)@self.invSigma@self.set_of_npointsbayesian.values)[0,0] / (np.ones(self.set_of_npointsbayesian.size)@self.invSigma@np.ones(self.set_of_npointsbayesian.size))[0,0]
        
    def neg_log_likelihood(self, params : list[float]):
        cls_points.Point.params_cor = params[2:]
        self.sigmaf = params[0]
        self.sigman = params[1]
        self.update_Sigma_invSigma()
        self.update_mu()
        return 1/2* ut.log(np.linalg.det(self.Sigma)) + 1/2*([v - self.mu for v in self.set_of_npointsbayesian.values]@self.invSigma@[v - self.mu for v in self.set_of_npointsbayesian.values])[0,0]

    def max_likelihood(self):
        print("Maximisation of likelihood")
        bounds = [(1e-6, None) for _ in range(5)]
        params = np.concatenate([np.array([self.sigmaf, self.sigman]), cls_points.Point.params_cor])
        result = optimize.minimize(self.neg_log_likelihood, params, method="Nelder-Mead", bounds = bounds, options={"disp" : True, "maxiter" : 1200})
        self.neg_log_likelihood(result.x)        
    
    def find_max(self, converter : "Conv", calculator : "Calculator"):
        while self.iter < self.n_first_points:
            self.iter += 1
            self.set_of_npointsbayesian.add_npoint()
            with open(os.path.join(self.path, "model.pkl"), "wb") as f:
                pickle.dump((self.set_of_npointsbayesian.l_nareas_values(), self.expected_improvements), f)
        self.val_max = max(self.set_of_npointsbayesian.values)
        expected_improvement = abs(self.val_max) if self.expected_improvements == [] else self.expected_improvements[-1]
        while (self.iter<self.max_iter + self.n_first_points) & (expected_improvement/abs(self.val_max)>self.min_expected_improvement):
            self.iter += 1
            print("Itération : " + str(self.iter))
            self.max_likelihood()
            self.genetic_algo = ga.Genetic_Algo(str(self.iter), converter, calculator, cls_points.NPoint.n_tetrahedras, cls_points.NPoint.range, self.esp_improv, self.path, **self.kwargs_ga)
            npoint, expected_improvement = self.genetic_algo.find_max()
            print("expected improvement", expected_improvement)
            npoint_gauss = npoint.to_bayesian()
            self.set_of_npointsbayesian.add_npoint(npoint_gauss)
            self.expected_improvements.append(expected_improvement)
            with open(os.path.join(self.path, "model.pkl"), "wb") as f:
                pickle.dump((self.set_of_npointsbayesian.l_nareas_values(), self.expected_improvements), f)
        print(expected_improvement, self.val_max)
        return self.best_npoint()

    def best_npoint(self):
        return self.set_of_npointsbayesian.set_of_npoints[self.set_of_npointsbayesian.argmax()]

    def best_value(self):
        return self.set_of_npointsbayesian.values[self.set_of_npointsbayesian.argmax()]

    def best_iteration(self):
        return 1 + self.set_of_npointsbayesian.argmax()

    def esp_improv(self, npoint : "NPoint"):
        if npoint.is_in(self.set_of_npointsbayesian):
            return 0
        elif npoint.in_water() & npoint.verify_range():
            k = self.sigmaf ** 2 * np.matrix([npoint.corr(e) for e in self.set_of_npointsbayesian.set_of_npoints])
            fexpec = self.mu + (k @ self.invSigma @ [v - self.mu for v in self.set_of_npointsbayesian.values])[0,0]
            s = (self.sigmaf ** 2 - k @ self.invSigma @ k.T)[0,0]
            a = (fexpec - self.val_max - self.ksi)
            return max(a / 2 * (1 + math.erf(a/(2*s))) + s * 1/math.sqrt(2*math.pi) * math.exp(-1/2 * (a/s)**2), 0) #the values should be strictly positive
        else:
            return -1

    def max_values(self):
        l = []
        m = self.set_of_npointsbayesian.values[0]
        for i in range(len(self.set_of_npointsbayesian.values)):
            if m < self.set_of_npointsbayesian.values[i]:
                m = self.set_of_npointsbayesian.values[i]
            l.append(m)
        return l

def load(path : str, **kwargs):
    with open(os.path.join(path, "model.pkl"), "rb") as f:
        l_nareas_values, expected_improvements = pickle.load(f)
        l_nareas, values = l_nareas_values
    return Bayesian_Process(path, l_nareas, values, expected_improvements, **kwargs)