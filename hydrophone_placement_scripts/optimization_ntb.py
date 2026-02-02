# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import sys
sys.path.append("../")

# %%
import hydrophone_placement_scripts.optimisation.class_mod as cls_mod
import hydrophone_placement_scripts.to_optimise.func_to_optimise as f
import hydrophone_placement_scripts.utils_scripts.class_points as cls_points
import hydrophone_placement_scripts.utils_scripts.conversions_coordinates as conv

# %%
import importlib
importlib.reload(cls_mod)
importlib.reload(f)

# %%
import hydrophone_placement_scripts.to_optimise.topo
importlib.reload(hydrophone_placement_scripts.to_optimise.topo)

# %%
args = {
    "n_tetrahedras" : 5,
    "lat_min" : 47.65,
    "lat_max" : 48.07,
    "lon_min" : -70.04,
    "lon_max" : -69.28,
    "width_area" : 500,
    "depth_area" : 2,
    "height_sensor" : 0.85,
    "n_processes" : 2,
}

# %%
mod = cls_mod.Model(**args)
