import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import librosa
import threading
import time
import folium
import os
from datetime import timedelta, datetime
import numpy as np
from pyproj import Transformer
from math import radians, sin, cos, sqrt, atan2

def enu_to_lla(e, n, u, lat_ref, lon_ref, alt_ref):
    """Fonction pour convertir ENU centre sur un point de reference a LLA

    Args:
        e (float): coordonnee est
        n (float): coordonnee nord
        u (float): coordonnee up
        lat_ref (float): latitude (degres decimaux)
        lon_ref (float): longitude (degres decimaux)
        alt_ref (float, optional): altitude. Defaults to 0.

    Returns:
        float: coorodnnees en lat, long, alt
    """
    transformer = Transformer.from_crs('epsg:4326', 'epsg:4978')
    X_ref, Y_ref, Z_ref  = transformer.transform(lat_ref, lon_ref, alt_ref)

    lat_ref = np.radians(lat_ref)
    lon_ref = np.radians(lon_ref)

    R = np.array([[-np.sin(lon_ref), np.cos(lon_ref), 0],
                  [-np.sin(lat_ref)*np.cos(lon_ref), -np.sin(lat_ref)*np.sin(lon_ref), np.cos(lat_ref)],
                  [np.cos(lat_ref)*np.cos(lon_ref), np.cos(lat_ref)*np.sin(lon_ref), np.sin(lat_ref)]])

    enu = np.array([e, n, u])
    d = R.T @ enu

    X = d[0] + X_ref
    Y = d[1] + Y_ref
    Z = d[2] + Z_ref

    transformer = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)
    lon, lat, alt = transformer.transform(X, Y, Z)

    return lat, lon, alt

def haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    R = 6371.0

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

def set_dashboard(audio_files, positions, errors, timestamps, durations, call_types, groundtruths, environment):
    """
    Create a dashboard from the data of the main pipeline. So far, it can only be used with 2 tetrahedrons.

    Parameters:
    - audio_files: List of audio_files.
    - positions, errors, timestamps, durations, call_types: Outputs of the positions_from_audio function.
    - groundtruths: List of groundtruths with format (latitude, longitude) for each groundtruth.
    - environment: Environment containing tetrahedra information.

    Returns:
    - app: Dashboard to launch.
    """

    # Load audio files
    audio_path1, audio_path2 = audio_files
    year, month, day, hour, minute, second = 2000+int(audio_path1[-16:-14]), int(audio_path1[-14:-12]), int(audio_path1[-12:-10]), int(audio_path1[-10:-8]), int(audio_path1[-8:-6]), int(audio_path1[-6:-4])
    audio_start_time = f"{audio_path1[-10:-8]}:{audio_path1[-8:-6]}:{audio_path1[-6:-4]}" 

    y, sr1 = librosa.load(audio_path1, sr=None, mono=False)
    y1 = y[1] if y.ndim > 1 and y.shape[0] > 1 else y
    y, sr2 = librosa.load(audio_path2, sr=None, mono=False)
    y2 = y[1] if y.ndim > 1 and y.shape[0] > 1 else y

    min_len = min(len(y1), len(y2))
    y1 = y1[:min_len]
    y2 = y2[:min_len]

    # Real-time streaming of audios
    class AudioStreamer:
        def __init__(self, data1, data2, sample_rate, chunk_duration=1.0, window_duration=3.0):
            self.data1 = data1
            self.data2 = data2
            self.sample_rate = sample_rate
            self.chunk_samples = int(chunk_duration * sample_rate)
            self.window_samples = int(window_duration * sample_rate)
            self.total_samples = len(data1)
            self.index = 0
            self.lock = threading.Lock()
            self.running = True
            self.thread = threading.Thread(target=self.update_index)
            self.thread.start()
            self.rms_times = []
            self.rms1_values = []
            self.rms2_values = []

        def update_index(self):
            while self.running and self.index + self.chunk_samples < self.total_samples:
                time.sleep(1.0)
                with self.lock:
                    self.index += self.chunk_samples

        def get_current_chunk(self):
            with self.lock:
                start = max(0, self.index - self.window_samples)
                end = self.index
                return self.data1[start:end], self.data2[start:end], start / self.sample_rate

        def stop(self):
            self.running = False
            self.thread.join()

    streamer = AudioStreamer(y1, y2, sr1)

    # Input preparation
    tetrahedrons_coords = [v.origin_lla[:2] for v in environment.tetrahedras.values()]
    T1_lat, T1_lon = tetrahedrons_coords[0]
    #positions = [positions[i] for i in range(len(positions)) if call_types[i]=='Whistle']
    #errors = [errors[i] for i in range(len(errors)) if call_types[i]=='Whistle']
    #timestamps = [timestamps[i] for i in range(len(timestamps)) if call_types[i]=='Whistle']
    #highlight_durations = [durations[i] for i in range(len(durations)) if call_types[i]=='Whistle']
    lla_positions = [enu_to_lla(*position, T1_lat, T1_lon, 0) for position in positions]

    # Inputs
    highlight_coords = [(position[0][0], position[1][0]) for position in lla_positions]
    errors = [(error[0], error[1]) for error in errors]
    highlight_timestamps = [(timestamp - datetime(year, month, day, hour, minute, second)).total_seconds() for timestamp in timestamps]

    intervals_nodes = [int(x) for x in highlight_timestamps]
    highlight_intervals = [(intervals_nodes[i], intervals_nodes[i+1]) for i in range(len(intervals_nodes)-1)]
    distances_to_groundtruths = [min(haversine(position, ref) * 1000 for ref in groundtruths) for position in highlight_coords]


    # Structure of the app
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.Div([
            dcc.Graph(id='spectrogram1', style={'width': '48%', 'height': '300px', 'display': 'inline-block'}),
            dcc.Graph(id='spectrogram2', style={'width': '48%', 'height': '300px', 'display': 'inline-block'})
        ], style={'display': 'flex', 'justifyContent': 'space-between'}),

        html.Div([
            html.Div([
                html.Iframe(id='map-frame', width='100%', height='100%', style={'border': 'none'})
            ], style={'width': '30%', 'marginLeft': '5%'}),

            html.Div([
                dash_table.DataTable(
                    id='highlight-table',
                    columns=[
                        {"name": "Temps", "id": "start"},
                        {"name": "Latitude", "id": "lat"},
                        {"name": "Longitude", "id": "lon"},
                        {"name": "Distance (m)", "id": "dis"},
                        {"name": "Erreur (m)", "id": "err"},
                        {"name": "Cri", "id": "call"}
                    ],
                    data=[],
                    style_table={'height': '100%', 'overflowY': 'auto'},
                    style_cell={'textAlign': 'center'},
                    style_header={'fontWeight': 'bold'}
                )
            ], style={'width': '60%', 'marginRight': '5%', 'overflowY': 'auto'})
        ], style={'display': 'flex', 'height': '250px', 'marginTop': '20px', 'justifyContent': 'space-between'}),

        dcc.Interval(id='interval', interval=1000, n_intervals=0)
    ])


    # Uodate at each second
    @app.callback(
        [Output('spectrogram1', 'figure'),
        Output('spectrogram2', 'figure'),
        Output('map-frame', 'srcDoc'),
        Output('highlight-table', 'data')],
        [Input('interval', 'n_intervals')]
    )

    def update_visuals(n):
        data1, data2, start_time = streamer.get_current_chunk()

        # Paramètres du spectrogramme
        n_fft = 4096
        hop_length = 1024

        # STFT et conversion en dB
        S1 = np.abs(librosa.stft(data1, n_fft=n_fft, hop_length=hop_length))
        S2 = np.abs(librosa.stft(data2, n_fft=n_fft, hop_length=hop_length))

        D1 = librosa.amplitude_to_db(S1, ref=np.max)
        D2 = librosa.amplitude_to_db(S2, ref=np.max)

        # Axes temps et fréquence
        times = librosa.frames_to_time(np.arange(D1.shape[1]), sr=sr1, hop_length=hop_length)
        times += start_time
        freqs = librosa.fft_frequencies(sr=sr1, n_fft=n_fft)

        # Filtrage des fréquences
        freq_mask = (freqs >= 125) & (freqs <= 60000)
        D1_filtered = D1[freq_mask, :]
        D2_filtered = D2[freq_mask, :]
        freqs_filtered = freqs[freq_mask]

        # Création des figures
        fig1 = go.Figure(data=go.Heatmap(
            z=D1_filtered, x=times, y=freqs_filtered,
            colorscale='magma', zmin=np.min(D1), zmax=np.max(D1)
        ))

        fig2 = go.Figure(data=go.Heatmap(
            z=D2_filtered, x=times, y=freqs_filtered,
            colorscale='magma', zmin=np.min(D2), zmax=np.max(D2)
        ))

        fig1.update_layout(title=f'T1 - {audio_path1[-21:-17]}', xaxis_title='Temps [s]', yaxis_title='Fréquence [Hz]')
        fig2.update_layout(title=f'T2 - {audio_path2[-21:-17]}', xaxis_title='Temps [s]', yaxis_title='Fréquence [Hz]')

        fig1.update_yaxes(type="log", range=[np.log10(125), np.log10(60000)])
        fig2.update_yaxes(type="log", range=[np.log10(125), np.log10(60000)])

        # Surlignages temporels
        current_start = times[0]
        current_end = times[-1]

        for i, start_timestamp in enumerate(highlight_timestamps):
            if start_timestamp + durations[i] >= current_start and start_timestamp <= current_end:
                x0 = max(start_timestamp, current_start)
                x1 = min(start_timestamp + durations[i], current_end)
                if call_types[i] == 'Whistle':
                    shape = dict(
                    type="rect", xref="x", yref="y",
                    x0=x0, x1=x1, y0=520, y1=9900,
                    line=dict(color="red", width=1),
                    fillcolor="rgba(255, 0, 0, 0.05)", layer="above"
                    )
                else:
                    shape = dict(
                    type="rect", xref="x", yref="y",
                    x0=x0, x1=x1, y0=10000, y1=60000,
                    line=dict(color="red", width=1),
                    fillcolor="rgba(255, 0, 0, 0.05)", layer="above"
                    )
                fig1.add_shape(shape)
                fig2.add_shape(shape)

        # Détection de l'intervalle actif
        active_index = None
        for i, (start_sec, end_sec) in enumerate(highlight_intervals):
            adjusted_start = start_sec + 1
            adjusted_end = end_sec + 1
            if current_start <= adjusted_start and current_end >= adjusted_end:
                active_index = i
            elif current_end < adjusted_start:
                active_index = i - 1 if i > 0 else None
                break
        else:
            if active_index is None and len(highlight_intervals) > 0:
                active_index = len(highlight_intervals) - 1

        # Carte Folium
        m = folium.Map(location=groundtruths[0], zoom_start=14)

        for lat, lon in groundtruths:
            folium.Marker(location=[lat, lon],
                        icon=folium.DivIcon(html='<div style="font-size: 20pt; color: orange;">+</div>')).add_to(m)

        for lat, lon in tetrahedrons_coords:
            folium.Marker(location=[lat, lon],
                        icon=folium.DivIcon(html='<div style="font-size: 20pt; color: indigo;">▲</div>')).add_to(m)

        if active_index is not None and active_index < len(highlight_coords):
            lat, lon = highlight_coords[active_index]
            folium.Marker(location=[lat, lon],
                        icon=folium.DivIcon(html='<div style="font-size: 20pt; color: red;">+</div>')).add_to(m)

        map_html = m.get_root().render()

        # Tableau de données
        if active_index is not None and active_index >= 0:
            table_data = [
                {
                    "start": (datetime.strptime(audio_start_time, "%H:%M:%S") + timedelta(seconds=highlight_intervals[i][0])).strftime("%H:%M:%S"),
                    "lat": highlight_coords[i][0],
                    "lon": highlight_coords[i][1],
                    "dis": distances_to_groundtruths[i],
                    "err": round(np.sqrt(errors[i][0]**2 + errors[i][1]**2), 3),
                    "call": call_types[i]
                }
                for i in range(active_index + 1)
            ]
        else:
            table_data = []

        return fig1, fig2, map_html, table_data

    return app