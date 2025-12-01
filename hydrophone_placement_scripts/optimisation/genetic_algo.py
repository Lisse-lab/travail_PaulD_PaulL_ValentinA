from multiprocessing import Pool, cpu_count
import pickle
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import hydrophone_placement_scripts.utils_scripts.class_points as cls_points

class Genetic_Algo:
    n_individuals = 20
    n_parents = 10
    C_prob = 1.9
    n_sep = 4
    p_mut = 0.01
    n_run = 20
    max_iter = 200
    n_improvements = 20
    n_processes = cpu_count() - 1 if cpu_count()>1 else 1
    save_run = True

    probs = None
    
    def __init__(self, id, converter, calculator, n_tetrahedras, range, esp_improv, path, **kwargs_ga):
        for attr, value in kwargs_ga.items():
            if not hasattr(self, attr):
                raise AttributeError(f"Genetic_Algo has no attribute '{attr}'.")
            setattr(self, attr, value)
        self.converter = converter
        self.calculator = calculator
        self.n_tetrahedras = n_tetrahedras
        self.range = range
        self.esp_improv = esp_improv
        self.calculate_probs()
        self.set_of_best = cls_points.Set_of_NPointsGenetic()
        self.path = os.path.join(path, f"Runs/Iteration_{id}")
        os.makedirs(self.path)

        

    def calculate_probs (self):
        self.probs = [self.C_prob/self.n_parents + ((2-2*self.C_prob) / (self.n_parents * (self.n_parents - 1))) * i for i in range (self.n_parents)]
        
    def set_probs(self):
        cls_points.NPointGenetic.set_p_mut(self.p_mut)
        cls_points.Set_of_NPointsGenetic.set_probs_n_sep(self.probs, self.n_sep)

    def find_max(self):
        args = [(str(k+1), self.path) for k in range(self.n_run)]
        if min(self.n_processes, len(args)) > 1:
            with Pool(processes=min(self.n_processes, len(args)), initializer=self.init_worker) as pool:
                results = list(pool.imap_unordered(self.find_max_run, args))
        else:
            self.init_worker()
            results = [self.find_max_run(arg) for arg in args]
        for max, val_max in results:
            self.set_of_best.add_npoint(max, value=val_max)
        arg = self.set_of_best.argmax()
        return (self.set_of_best.set_of_npoints[arg], self.set_of_best.values[arg])

    def find_max_run(self, args):
        run = Run(args)
        return run.find_max()
    
    def init_worker(self):
        cls_points.NPoint.set_n_tetrahedras(self.n_tetrahedras)
        self.set_probs()
        cls_points.Point.update_point(self.converter.xmin, self.converter.ymin, self.converter.xmax, self.converter.ymax)
        cls_points.NPoint.set_range(self.range)
        cls_points.NPointGenetic.set_value(self.esp_improv)
        cls_points.Point.set_topo(self.calculator.topo, self.calculator.height_sensor)
        Run.max_iter = self.max_iter
        Run.n_improvements = self.n_improvements
        Run.n_individuals = self.n_individuals
        Run.n_parents = self.n_parents

class Run:

    def __init__(self, args):
        self.max = None
        self.val_max = None
        self.individuals = cls_points.Set_of_NPointsGenetic(size = self.n_individuals)
        self.path = os.path.join(args[1], f"Run_{args[0]}.pkl")
        self.id = args[0]

    def find_max(self):
        self.bests = cls_points.Set_of_NPointsGenetic()
        self.l_nb_bayesian = [self.individuals.nb_bayesian()]
        iter = 0
        improv = 0
        while (iter < self.max_iter) & (improv < self.n_improvements):
            iter += 1
            self.parents = self.individuals.k_best(self.n_parents)
            self.parents.sort()
            self.max = self.parents.set_of_npoints[0]
            self.val_max = self.parents.values[0]
            self.bests.add_npoint(self.max, self.val_max)
            new_set_of_individuals = cls_points.Set_of_NPointsGenetic()
            for _ in range (self.n_individuals):
                new_set_of_individuals.breed(self.parents)
            new_max = max(new_set_of_individuals.values)
            if self.val_max > new_max:
                improv += 1
                i = new_set_of_individuals.argmin()
                new_set_of_individuals.set_of_npoints[i] = self.max
                new_set_of_individuals.values[i] = self.val_max
            elif self.val_max == new_max:
                improv += 1
                i = new_set_of_individuals.argmax()
                self.max = new_set_of_individuals.set_of_npoints[i]
            else :
                improv = 0
                i = new_set_of_individuals.argmax()
                self.max = new_set_of_individuals.set_of_npoints[i]
                self.val_max = new_set_of_individuals.values[i]
            self.individuals = new_set_of_individuals
            self.l_nb_bayesian.append(self.individuals.nb_bayesian())
        self.bests.add_npoint(self.max, self.val_max)
        with open(self.path, "wb") as f:
            pickle.dump(self, f)
        return (self.max, self.val_max)
    
    def display(self):
        plt.figure(figsize=(8, 5)) 
        plt.plot(self.bests.values, marker='o', linestyle='-', color='blue', label='Expected Improvement')
        plt.plot(self.l_nb_bayesian, marker='o', linestyle='-', color='red', label=f'Number of Bayesian individuals (Min : {min(self.l_nb_bayesian)})')
        plt.xlabel('Iteration')
        plt.title(f'Informations about the run {self.id}')
        plt.legend()
        plt.ylim(bottom = 0)
        plt.show()
        return None

    def display_plotly(self, max_lim = None):
        fig = go.Figure()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Ajout des courbes
        fig.add_trace(go.Scatter(
            x=list(range(len(self.bests.values))),
            y=self.bests.values,
            mode='lines+markers',
            marker=dict(color='blue'),
            line=dict(color='blue'),
            name='Expected Improvement'
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=list(range(len(self.l_nb_bayesian))),
            y=self.l_nb_bayesian,
            mode='lines+markers',
            marker=dict(color='red'),
            line=dict(color='red'),
            name=f'Number of Bayesian individuals (Min: {min(self.l_nb_bayesian)})'
        ), secondary_y=True)

        # Mise en forme
        fig.update_layout(
            title=f'Informations about the run {self.id}',
            xaxis_title='Iteration',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(title_text="Number of Bayesian individuals", range=[0, None], secondary_y=True)
        fig.update_yaxes(title_text="Expected Improvement", range=[0, max_lim], secondary_y=False)
        
        return fig

def load_run(file):
    with open(file, "rb") as f:
        run = pickle.load(f)
    return run

def load_runs(path, iteration):
    l_runs = []
    for file in os.listdir(os.path.join(path, f"Runs/Iteration_{iteration}")):
        l_runs.append(load_run(os.path.join(path, f"Runs/Iteration_{iteration}", file)))
    return l_runs