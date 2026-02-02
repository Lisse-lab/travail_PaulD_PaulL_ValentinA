import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline, interp1d
import folium

def ind(l, e):
    i = 0
    bool = True
    while (i < len(l)) & bool:
        if l[i] >= e:
            bool = False
        else:
            i+=1
    return i

def create_newtraj(timestep_min, df, nct, converter, with_old_points = False, display = False, name = "ma_carte", other_columns = []):
    df_traj = df[df["NCT"] == nct]
    if len(df_traj) <= 3:
        k = len(df_traj) - 1
    else:
        k=3
    spline_x = make_interp_spline(df_traj['Time'], df_traj['x'], k=k)
    spline_y = make_interp_spline(df_traj['Time'], df_traj['y'], k=k)
    ts = list(np.arange(df_traj["Time"].min(), df_traj["Time"].max(), timestep_min*60))
    xs = spline_x(ts)
    ys = spline_y(ts)
    lons, lats = converter.utm2lla(xs, ys)
    l_other_cols = []
    for col in other_columns:
        spline_col = interp1d(df_traj['Time'], df_traj[col], kind='linear')
        l_other_cols.append(spline_col(ts))
    l_new_traj = [True] * len(ts)
    l_ind = []
    for i in df_traj.index:
        j = ind(ts, df_traj.loc[i, "Time"])
        if (j >= len(ts)) or (ts[j] != df_traj.loc[i, "Time"]):
            ts = np.insert(ts, j, df_traj.loc[i, "Time"])
            lons = np.insert(lons, j, df_traj.loc[i, "Longitude"])
            lats = np.insert(lats, j, df_traj.loc[i, "Latitude"])
            xs = np.insert(xs, j, df_traj.loc[i, "x"])
            ys = np.insert(ys, j, df_traj.loc[i, "y"])
            for c, col in enumerate(other_columns):
                l_other_cols[c] = np.insert(l_other_cols[c], j, df_traj.loc[i, col])
            l_new_traj.insert(j, False)
        l_ind.append(j)

    l_old = [False] * len(ts)
    for i in range (len(ts)):
        if i in l_ind:
            l_old[i] = True
    df_traj2 = pd.DataFrame({"NCT":df_traj["NCT"].iloc[0], "Longitude" : lons, "Latitude" : lats, "Time": ts, "x":xs, "y":ys, "in_new_traj":l_new_traj, "old":l_old})
    for c, col in enumerate(other_columns):
        df_traj2[col] = l_other_cols[c]


    if display:
        show(df_traj2, name)
    
    if with_old_points:
        return df_traj2
    else:
        return df_traj2[df_traj2["in_new_traj"]].reset_index()

def divide(step_err, df):
    df["group"] = df.index % step_err
    df_sorted = df.sort_values(by=["group", "Time"])
    df_sorted["NCT"] = step_err * df_sorted["NCT"] + df_sorted["group"]
    df_sorted.drop(columns = ["group"], inplace = True)
    return df_sorted

def create_df_newtrajs(timestep_min, step_err, df, converter, other_columns = []):
    l_dfs = []
    for nct in df["NCT"].unique():
        l_dfs.append(divide(step_err, create_newtraj(timestep_min, df, nct, converter, other_columns=other_columns)))
    df_trajs = pd.concat(l_dfs, ignore_index=True)
    df_trajs.drop(columns=["in_new_traj", "old", "index"], inplace=True)
    return df_trajs

def show(df, name):
    map = folium.Map(location=[47.86, -69.66], zoom_start=12)

    for row in df.itertuples():
        if row.old:
            color = "red"
        else:
            color = "blue"
        folium.CircleMarker(
                    location=[row.Latitude, row.Longitude],
                    radius=2,
                    color=color,
                    fill=True,
                    fill_color=color,
                    tooltip=folium.Tooltip(str(row.Index), permanent=False)
                    ).add_to(map)
        
    map.save(name + ".html")