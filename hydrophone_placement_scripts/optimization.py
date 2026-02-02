import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../beluga_watch"))

import hydrophone_placement_scripts.optimisation.class_mod as cls_mod
import hydrophone_placement_scripts.optimisation.genetic_algo as ga

args = {
    "n_tetrahedras" : 10,
    "lat_min" : 47.65,
    "lat_max" : 48.07,
    "lon_min" : -70.04,
    "lon_max" : -69.28,
    "width_area" : 500,
    "depth_area" : 2,
    "n_processes" : 15,
}

args_find_max = {
    "b_n_first_points" : 30,
    "b_max_iter" : 100,
    "ga_max_iter" : 500,
    "ga_n_run" : 60,
    "ga_n_improvements" : 10,
    "ga_n_individuals" : 100,
    "ga_n_parents" : 30,
}

if __name__ == '__main__':
    mod = cls_mod.Model(**args)
    print("model created")
    mod.find_max(**args_find_max)
    # mod.calculator.df_areas.to_csv("df_areas_after_optim.csv")
    mod.display()
    mod.create_optimisation_history(True)
    mod.create_maps()