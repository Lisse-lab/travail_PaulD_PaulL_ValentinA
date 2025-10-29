from main_module import positions_from_audio
from src.utils.dashboard import set_dashboard
from src.utils.sub_classes import Environment, Parameters
import numpy as np
from pyproj import Transformer
from math import radians, sin, cos, sqrt, atan2


audio_path =  [r"C:\Users\pauld\OneDrive\Bureau\Césure\Stage UQO\beluga-watch\test_data\full audios\8296.240729065600.wav", r"C:\Users\pauld\OneDrive\Bureau\Césure\Stage UQO\beluga-watch\test_data\full audios\8295.240729065600.wav"]
model_path = 'jsons/models/mobile_net_overlaps.pt'
param_path = 'jsons/parameters/default_parameters.json'
env_path = 'jsons/environments/env_cacouna.json'

parameters = Parameters(param_path)
environment = Environment(env_path, parameters.location_parameters.use_h4)

positions, errors, timestamps, durations, call_types = positions_from_audio(model_path, env_path, param_path, audio_path)

if __name__ == "__main__":
    app = set_dashboard(audio_path, positions, errors, timestamps, durations, call_types, [(47.940056, -69.530229), (47.939948, -69.530897)], environment)
    app.run(debug=False)