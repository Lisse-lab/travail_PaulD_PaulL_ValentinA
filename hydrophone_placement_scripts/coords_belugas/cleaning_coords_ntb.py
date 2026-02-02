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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../..")
import hydrophone_placement_scripts.utils_scripts.conversions_coordinates as conv

# %%
df_ct = pd.read_excel("../datas/coords_belugas/CtDL_DBase8923_VarOceano.xlsx")
df_or = pd.read_excel("../datas/coords_belugas/bdGremm_nePasDistribuer.xlsx")
df_ct["ref.line.2017"] = np.round(df_ct["ref.line.2017"], 4)
df = pd.merge(df_ct, df_or, left_on='ref.line.2017', right_on='refline2017', how='left', suffixes = ("_ct", "_or"))

# %%
df = df[df["Annee"] >= 1999]

# %%
(abs(df["Heure_ct"] - df["Heure_or"].apply(lambda row :(row.hour + row.minute / 60 + row.second / 3600 )/24)) > 1e-5).sum()

# %%
(df["NCT_or"] != df["NCT_ct"]).sum()

# %%
df = df[pd.to_numeric(df["Longitude"], errors='coerce').notna()]
df = df[pd.to_numeric(df["Latitude"], errors='coerce').notna()]

# %%
(abs(df["long"] - df["Longitude"].apply(lambda row : float(row))) > 1e-6).sum()

# %%
(abs(df["lat"] - df["Latitude"].apply(lambda row : float(row))) > 1e-6).sum()

# %%
df.columns

# %%
columns = ["Année", "Date_ct", "Heure_ct", "Date.heure.stamp", "NCT_ct", "vitesseCourant", "directionCourant", "maree", "Longitude", "Latitude", "%G", "%G min", "%G max", "G", "Taille", "Taille Min.", "TailleMax", "Rayon (m)", "Dist. CC Troupeau (m)", "Bearing CC troupeau"]

# %%
df = df[columns]

# %%
df.rename(columns={"NCT_ct" : "NCT", "Date_ct":"Date", "Heure_ct":"Heure"}, inplace = True)

# %%
df["Bearing CC troupeau"].unique()

# %%
df.loc[df["Bearing CC troupeau"] == "nd", "Bearing CC troupeau"] = float("nan")
df.loc[df["Bearing CC troupeau"] == "na", "Bearing CC troupeau"] = float("nan")
df.loc[df["Bearing CC troupeau"] == "O", "Bearing CC troupeau"] = 270
df.loc[df["Bearing CC troupeau"] == "N", "Bearing CC troupeau"] = 0
df.loc[df["Bearing CC troupeau"] == "SE", "Bearing CC troupeau"] = 135
df.loc[df["Bearing CC troupeau"] == "S", "Bearing CC troupeau"] = 180
df.loc[df["Bearing CC troupeau"] == "N-NO", "Bearing CC troupeau"] = 337.5
df.loc[df["Bearing CC troupeau"] == "NO", "Bearing CC troupeau"] = 315
df.loc[df["Bearing CC troupeau"] == "NE", "Bearing CC troupeau"] = 45
df.loc[df["Bearing CC troupeau"] == "N-NE", "Bearing CC troupeau"] = 22.5
df.loc[df["Bearing CC troupeau"] == "O-SO", "Bearing CC troupeau"] = 247.5
df.loc[df["Bearing CC troupeau"] == "E", "Bearing CC troupeau"] = 90
df.loc[df["Bearing CC troupeau"] == "E-NE", "Bearing CC troupeau"] = 22.5
df.loc[df["Bearing CC troupeau"] == "SO", "Bearing CC troupeau"] = 225
df.loc[df["Bearing CC troupeau"] == "E-SE", "Bearing CC troupeau"] = 112.5
df.loc[df["Bearing CC troupeau"] == "au S", "Bearing CC troupeau"] = 180
df.loc[df["Bearing CC troupeau"] == "O; 300", "Bearing CC troupeau"] = 300
df.loc[df["Bearing CC troupeau"] == "N; 0", "Bearing CC troupeau"] = 0
df.loc[df["Bearing CC troupeau"] == "O; 270", "Bearing CC troupeau"] = 270
df.loc[df["Bearing CC troupeau"] == "N, 0", "Bearing CC troupeau"] = 0
df.loc[df["Bearing CC troupeau"] == "S/SO", "Bearing CC troupeau"] = 202.5
df.loc[df["Bearing CC troupeau"] == "360, N", "Bearing CC troupeau"] = 0
df.loc[df["Bearing CC troupeau"] == "E,90", "Bearing CC troupeau"] = 90
df.loc[df["Bearing CC troupeau"] == "280 (O/NO)", "Bearing CC troupeau"] = 280
df.loc[df["Bearing CC troupeau"] == "O,270", "Bearing CC troupeau"] = 270
df.loc[df["Bearing CC troupeau"] == "E, 90", "Bearing CC troupeau"] = 90
df.loc[df["Bearing CC troupeau"] == "S?", "Bearing CC troupeau"] = 180
df.loc[df["Bearing CC troupeau"] == "SSE", "Bearing CC troupeau"] = 157.5
df.loc[df["Bearing CC troupeau"] == "NNO", "Bearing CC troupeau"] = 90
df.loc[df["Bearing CC troupeau"] == "NNE", "Bearing CC troupeau"] = 22.5
df.loc[df["Bearing CC troupeau"] == "NE, E, S, NO", "Bearing CC troupeau"] = float("nan")
df.loc[df["Bearing CC troupeau"] == "OSO", "Bearing CC troupeau"] = 247.5
df.loc[df["Bearing CC troupeau"] == "ENE", "Bearing CC troupeau"] = 67.5
df.loc[df["Bearing CC troupeau"] == "SSO", "Bearing CC troupeau"] = 202.5
df.loc[df["Bearing CC troupeau"] == "ONO", "Bearing CC troupeau"] = 292.5
df.loc[df["Bearing CC troupeau"] == "na ", "Bearing CC troupeau"] = float("nan")
df.loc[df["Bearing CC troupeau"] == "NO ", "Bearing CC troupeau"] = 315
df.loc[df["Bearing CC troupeau"] == "N,NO", "Bearing CC troupeau"] = 337.5
df.loc[df["Bearing CC troupeau"] == "N,NE", "Bearing CC troupeau"] = 22.5
df.loc[df["Bearing CC troupeau"] == "S, SE", "Bearing CC troupeau"] = 157.5
df.loc[df["Bearing CC troupeau"] == "N, NE", "Bearing CC troupeau"] = 22.5
df.loc[df["Bearing CC troupeau"] == "nd ", "Bearing CC troupeau"] = float("nan")
df.loc[df["Bearing CC troupeau"] == "S-SE", "Bearing CC troupeau"] = 157.5
df.loc[df["Bearing CC troupeau"] == "S-SO", "Bearing CC troupeau"] = 202.5
df.loc[df["Bearing CC troupeau"] == "ESE", "Bearing CC troupeau"] = 112.5
df.loc[df["Bearing CC troupeau"] == "N ", "Bearing CC troupeau"] = 0
df.loc[df["Bearing CC troupeau"] == "S ", "Bearing CC troupeau"] = 180
df.loc[df["Bearing CC troupeau"] == "W", "Bearing CC troupeau"] = 270

# %%
df["Bearing CC troupeau"].unique()

# %%
df.loc[~df["Bearing CC troupeau"].isna()]["Dist. CC Troupeau (m)"].unique()

# %%
df.loc[df["Dist. CC Troupeau (m)"] == "300 à 500", "Dist. CC Troupeau (m)"] = 400
df.loc[df["Dist. CC Troupeau (m)"] == "100aine", "Dist. CC Troupeau (m)"] = 100
df.loc[df["Dist. CC Troupeau (m)"] == "500 à 600", "Dist. CC Troupeau (m)"] = 550
df.loc[df["Dist. CC Troupeau (m)"] == "60aine", "Dist. CC Troupeau (m)"] = 60
df.loc[df["Dist. CC Troupeau (m)"] == "500-600", "Dist. CC Troupeau (m)"] = 550
df.loc[df["Dist. CC Troupeau (m)"] == "nd", "Dist. CC Troupeau (m)"] = float("nan")
df.loc[df["Dist. CC Troupeau (m)"] == "300-400", "Dist. CC Troupeau (m)"] = 350
df.loc[df["Dist. CC Troupeau (m)"] == "1000 à 1500", "Dist. CC Troupeau (m)"] = 1250
df.loc[df["Dist. CC Troupeau (m)"] == "700 à 800", "Dist. CC Troupeau (m)"] = 750
df.loc[df["Dist. CC Troupeau (m)"] == "na", "Dist. CC Troupeau (m)"] = float("nan")
df.loc[df["Dist. CC Troupeau (m)"] == "150 à 200", "Dist. CC Troupeau (m)"] = 175
df.loc[df["Dist. CC Troupeau (m)"] == "800 à 900", "Dist. CC Troupeau (m)"] = 850
df.loc[df["Dist. CC Troupeau (m)"] == "200 à 300", "Dist. CC Troupeau (m)"] = 250
df.loc[df["Dist. CC Troupeau (m)"] == "1000 à 2000", "Dist. CC Troupeau (m)"] = 1500
df.loc[df["Dist. CC Troupeau (m)"] == "100 à 200", "Dist. CC Troupeau (m)"] = 150
df.loc[df["Dist. CC Troupeau (m)"] == "400 à 500", "Dist. CC Troupeau (m)"] = 450
df.loc[df["Dist. CC Troupeau (m)"] == "300 à 400", "Dist. CC Troupeau (m)"] = 350
df.loc[df["Dist. CC Troupeau (m)"] == "1500 à 2000", "Dist. CC Troupeau (m)"] = 1750
df.loc[df["Dist. CC Troupeau (m)"] == "800 à 1000", "Dist. CC Troupeau (m)"] = 900
df.loc[df["Dist. CC Troupeau (m)"] == "600 à 700", "Dist. CC Troupeau (m)"] = 750
df.loc[df["Dist. CC Troupeau (m)"] == "1700 à 1800", "Dist. CC Troupeau (m)"] = 1750

# %%
df.loc[~df["Bearing CC troupeau"].isna()]["Dist. CC Troupeau (m)"].unique()

# %%
df.loc[df["Dist. CC Troupeau (m)"] == "S", ["Dist. CC Troupeau (m)", "Bearing CC troupeau"]]

# %%
df.loc[11648, ["Dist. CC Troupeau (m)", "Bearing CC troupeau"]] = [200, 180]
df.loc[11649, ["Dist. CC Troupeau (m)", "Bearing CC troupeau"]] = [50, 180]

# %%
df.loc[df["Dist. CC Troupeau (m)"] == "d", ["Dist. CC Troupeau (m)", "Bearing CC troupeau"]]

# %%
df.loc[df["Dist. CC Troupeau (m)"] == "d", ["Dist. CC Troupeau (m)", "Bearing CC troupeau"]] = [float("nan"), float("nan")]

# %%
df.loc[df["Bearing CC troupeau"].isna()]["Dist. CC Troupeau (m)"].unique()

# %%
df[df.apply(lambda row : (not isinstance(row["Bearing CC troupeau"], (float, int))) | (not isinstance(row["Dist. CC Troupeau (m)"], (float, int))), axis = 1)]

# %%
df.loc[df["Dist. CC Troupeau (m)"].isna(), ["Dist. CC Troupeau (m)", "Bearing CC troupeau"]] = [0,0]
df.loc[df["Bearing CC troupeau"].isna(), ["Dist. CC Troupeau (m)", "Bearing CC troupeau"]] = [0,0]

# %% [markdown]
# ### Part two

# %%
df["Time"] = df["Heure"] * 60 * 60 * 24


# %%
def create_date_heure_stamp(row):
    year, month, day = str(row.Date.year), str(row.Date.month), str(row.Date.day)
    if len(month) == 1:
        month = "0" + month
    if len(day) == 1:
        day = "0" + day
    y = year[2:]
    h = int((row.Time / 60) // 60)
    m = np.round((row.Time / 60) % 60).astype(int)
    if m == 60:
        m = 0
        h += 1
    m = str(m)
    h = str(h)
    if len(h) == 1:
        h = "0" + h
    if len(m) == 1:
        m = "0" + m
    return  int(y + month + day + h + m)


# %%
df["date_heure_stamp"] = df.apply(lambda row : create_date_heure_stamp(row), axis = 1)
df["is_equal"] = df["Date.heure.stamp"] == df["date_heure_stamp"]
df[~df["is_equal"]]

# %%
df.drop(columns=["Date.heure.stamp", "is_equal"], inplace = True)
df

# %%
converter = conv.Conv()
df[["x", "y"]] = df.apply(lambda row : pd.Series(converter.lla2utm(row.Latitude, row.Longitude)), axis=1)

# %%
df["x"] += df.apply(lambda row : row["Dist. CC Troupeau (m)"] * np.cos(row["Bearing CC troupeau"]), axis = 1)
df["y"] += df.apply(lambda row : row["Dist. CC Troupeau (m)"] * np.sin(row["Bearing CC troupeau"]), axis = 1)

# %%
df.drop(columns=["Dist. CC Troupeau (m)", "Bearing CC troupeau"], inplace=True)

# %%
df[["Longitude", "Latitude"]] = df.apply(lambda row : pd.Series(converter.utm2lla(row.x, row.y)), axis=1)

# %%
df

# %%
l_nct_date = []
for nct in df["NCT"].unique():
    m = df["date_heure_stamp"][df["NCT"] == nct].min()
    M = df["date_heure_stamp"][df["NCT"] == nct].max()
    ma = df["Date"][df["NCT"] == nct].min()
    Ma = df["Date"][df["NCT"] == nct].max()
    if (M - m > 10000) | (Ma != ma): #same day on Date.heure.stamp, same day on Date
        print("problem of date : ", nct)
        l_nct_date.append(nct)

# %%
df[df["NCT"] == l_nct_date[0]]

# %%
m = max(df["NCT"][df["Année"]==2006])
df.loc[9858, "NCT"] = m + 1
df.loc[9866, "NCT"] = m + 1
df.loc[9867, "NCT"] = m + 1

# %%
df[df["NCT"] == l_nct_date[1]]

# %%
df.drop(14004, inplace = True)
df.drop(14005, inplace = True)

# %%
df[df["NCT"] == l_nct_date[2]]

# %%
df.drop(15923, inplace = True)
df.drop(15956, inplace = True)

# %%
df[df["NCT"] == l_nct_date[3]]

# %%
m = max(df["NCT"][df["Année"]==2020])
df.loc[17140, "NCT"] = m + 1
df.loc[17141, "NCT"] = m + 1
df.loc[17157, "NCT"] = m + 1
df.loc[17158, "NCT"] = m + 1
df.loc[17159, "NCT"] = m + 1

# %%
df[df["NCT"] == l_nct_date[4]]

# %%
df.drop(17345, inplace = True)
df.drop(17351, inplace = True)
df.drop(17352, inplace = True)

# %%
df = df.sort_values(by = ["NCT", "Année", "date_heure_stamp"]).reset_index(drop = True)

# %%
l_sup = []
for i in df.index[:-1]:
    if (df.loc[i, "Heure"] == df.loc[i+1, "Heure"]) & (df.loc[i, "NCT"] == df.loc[i+1, "NCT"]):
        if df.loc[i+1].isna().sum() < df.loc[i].isna().sum():
            l_sup.append(i)
        else:
            l_sup.append(i+1)

# %%
len(l_sup)

# %%
l_sup[0]

# %%
df.drop(l_sup, inplace = True)
df.reset_index(inplace=True, drop=True)
df

# %%
deltat_max = 2 * 60 * 60
for nct in df["NCT"].unique():
    l = df[df["NCT"] == nct].index
    for i, e in enumerate(l[:-1]):
        deltat = df.loc[e+1, "Time"] - df.loc[e, "Time"]
        if deltat > deltat_max:
            print(e)
            df.loc[l[i+1:], "NCT"] = max(df["NCT"][df["Année"] == df.loc[e, "Année"]]) + 1

# %%
l_vit = []
for nct in df["NCT"].unique():
    l = df[df["NCT"] == nct].index
    for i in l[:-1]:
        deltat = df.loc[i+1, "Time"] - df.loc[i, "Time"]
        vx = (df.loc[i+1, "x"] - df.loc[i, "x"]) / deltat
        vy = (df.loc[i+1, "y"] - df.loc[i, "y"]) / deltat
        l_vit.append(3.6 * np.sqrt(vx**2 + vy**2))

# %%
plt.scatter(np.arange(len(l_vit)), l_vit)
plt.ylabel("Speed (km/h)")

# %%
vit = 25 / 3.6
l_too_quick = []
for nct in df["NCT"].unique():
    l = df[df["NCT"] == nct].index
    for i in l[:-1]:
        deltat = df.loc[i+1, "Time"] - df.loc[i, "Time"]
        vx = (df.loc[i+1, "x"] - df.loc[i, "x"]) / deltat
        vy = (df.loc[i+1, "y"] - df.loc[i, "y"]) / deltat
        if vx**2 + vy**2 > vit ** 2:
            print("too quick : ", nct)
            l_too_quick.append(nct)
            print(i, i+1)
print(len(l_too_quick))

# %%
df.loc[df["NCT"] == l_too_quick[0]]

# %%
df.drop(2302, inplace = True)

# %%
df.loc[df["NCT"] == l_too_quick[1]]

# %%
df.drop(3126, inplace = True)

# %%
df.loc[df["NCT"] == l_too_quick[2]]

# %%
df.drop(3688, inplace = True)

# %%
df.loc[df["NCT"] == l_too_quick[3]]

# %%
df.drop(5859, inplace = True)
df.drop(5860, inplace = True)

# %%
df.loc[df["NCT"] == l_too_quick[4]]

# %%
df.drop(5861, inplace = True)
df.drop(5862, inplace = True)

# %%
df.loc[df["NCT"] == l_too_quick[5]]

# %%
df.drop(10319, inplace = True)

# %%
df.reset_index(drop=True, inplace=True)


# %% [markdown]
# ### About the Taille

# %%
def toint(row):
    try :
        return int(row.Taille)
    except:
        return row.Taille


# %%
df["Taille"] = df.apply(lambda row : toint(row), axis = 1)


# %%
def test(row):
    if isinstance(row.Taille, str):
        l = row.Taille.split("à")
        if len(l) == 2:
            try :
                return (int(l[0]) != int(row["Taille Min."])) | (int(l[1]) != int(row["TailleMax"]))
            except:
                return True
    return False


# %%
def test2(row):
    if isinstance(row.Taille, str):
        l = row.Taille.split("à")
        if len(l) == 2:
            try :
                if (int(l[0]) == int(row["Taille Min."])) & (int(l[1]) == int(row["TailleMax"])):
                    return int((int(l[0]) + int(l[1]))/2)
            except:
                return row.Taille
    return row.Taille


# %%
df["Taille"] = df.apply(lambda row : test2(row), axis = 1)


# %%
def modif2(row):
    if isinstance(row.Taille, str):
        if row.Taille[-1] == "+":
            try:
                return int(row.Taille[:-1])
            except:
                return row.Taille
        elif row.Taille.endswith("aine"):
            try:
                return int(row.Taille[:-4])
            except:
                return row.Taille
        elif row.Taille.endswith("aine "):
            try:
                return int(row.Taille[:-5])
            except:
                return row.Taille
        elif row.Taille.startswith("env."):
            try:
                return int(row.Taille[4:])
            except:
                return row.Taille
    return row.Taille


# %%
df["Taille"] = df.apply(lambda row : modif2(row), axis = 1)

# %%
l = df[~df["Taille"].apply(lambda val : isinstance(val, int))]["Taille"].unique()
l

# %%
i = -1

# %%
i += 1
print(i)
df.loc[df["Taille"] == l[i]]

# %%
df.loc[df["Taille"] == l[0], "Taille"] = float("nan")
df.loc[df["Taille"] == l[1], "Taille"] = 18
df.loc[df["Taille"] == l[2], "Taille"] = 1
df.loc[df["Taille"] == l[3], "Taille"] = 12
df.loc[df["Taille"] == l[4], "Taille"] = 10
df.loc[df["Taille"] == l[5], "Taille"] = 70
df.loc[df["Taille"] == l[6], "Taille"] = 7
df.loc[df["Taille"] == l[7], "Taille"] = 45
df.loc[df["Taille"] == l[8], "Taille"] = 7
df.loc[df["Taille"] == l[9], "Taille"] = 25
df.loc[df["Taille"] == l[10], "Taille"] = 3
df.loc[df["Taille"] == l[11], "Taille"] = float("nan")
df.loc[df["Taille"] == l[12], "Taille"] = 35
df.loc[df["Taille"] == l[13], "Taille"] = 17
df.loc[df["Taille"] == l[14], "Taille"] = 35
df.loc[df["Taille"] == l[15], "Taille"] = 6
df.loc[df["Taille"] == l[16], "Taille"] = 65
df.loc[df["Taille"] == l[17], "Taille"] = 4
df.loc[df["Taille"] == l[18], "Taille"] = 8
df.loc[df["Taille"] == l[19], "Taille"] = 2
df.loc[df["Taille"] == l[20], "Taille"] = float("nan")
df.loc[df["Taille"] == l[21], "Taille"] = 27
#df.loc[df["Taille"] == l[22], "Taille"] =    c'est nan
df.loc[df["Taille"] == l[23], "Taille"] = 30
df.loc[df["Taille"] == l[24], "Taille"] = 7
df.loc[df["Taille"] == l[25], "Taille"] = 3
df.loc[df["Taille"] == l[26], "Taille"] = 37
df.loc[df["Taille"] == l[27], "Taille"] = 14
df.loc[df["Taille"] == l[28], "Taille"] = 7
df.loc[df["Taille"] == l[29], "Taille"] = 10
df.loc[df["Taille"] == l[30], "Taille"] = 9
df.loc[df["Taille"] == l[31], "Taille"] = 20
df.loc[df["Taille"] == l[32], "Taille"] = 12
df.loc[df["Taille"] == l[33], "Taille"] = 17
df.loc[df["Taille"] == l[34], "Taille"] = float("nan")
df.loc[df["Taille"] == l[35], "Taille"] = 12
df.loc[df["Taille"] == l[36], "Taille"] = 9
df.loc[df["Taille"] == l[37], "Taille"] = 15
df.loc[df["Taille"] == l[38], "Taille"] = 22
df.loc[df["Taille"] == l[39], "Taille"] = 7
df.loc[df["Taille"] == l[40], "Taille"] = 27
df.loc[df["Taille"] == l[41], "Taille"] = 17
df.loc[df["Taille"] == l[42], "Taille"] = 1
df.loc[df["Taille"] == l[43], "Taille"] = 20
df.loc[df["Taille"] == l[44], "Taille"] = 2
df.loc[df["Taille"] == l[45], "Taille"] = 20
df.loc[df["Taille"] == l[46], "Taille"] = 3
df.loc[df["Taille"] == l[47], "Taille"] = 8
df.loc[df["Taille"] == l[48], "Taille"] = 8
df.loc[df["Taille"] == l[49], "Taille"] = 25
df.loc[df["Taille"] == l[50], "Taille"] = 19
df.loc[df["Taille"] == l[51], "Taille"] = 10
df.loc[df["Taille"] == l[52], "Taille"] = 25
df.loc[df["Taille"] == l[53], "Taille"] = 100
df.loc[df["Taille"] == l[54], "Taille"] = 7
df.loc[df["Taille"] == l[55], "Taille"] = 17
df.loc[df["Taille"] == l[56], "Taille"] = 7
df.loc[df["Taille"] == l[57], "Taille"] = 9
df.loc[df["Taille"] == l[58], "Taille"] = 11
df.loc[df["Taille"] == l[59], "Taille"] = 19
df.loc[df["Taille"] == l[60], "Taille"] = 18
df.loc[df["Taille"] == l[61], "Taille"] = 32
df.loc[df["Taille"] == l[62], "Taille"] = 7
df.loc[df["Taille"] == l[63], "Taille"] = 7
df.loc[df["Taille"] == l[64], "Taille"] = 12
df.loc[df["Taille"] == l[65], "Taille"] = 55
df.loc[df["Taille"] == l[66], "Taille"] = 11
df.loc[df["Taille"] == l[67], "Taille"] = 22
df.loc[df["Taille"] == l[68], "Taille"] = 25
df.loc[df["Taille"] == l[69], "Taille"] = float("nan")
df.loc[df["Taille"] == l[70], "Taille"] = 110
df.loc[df["Taille"] == l[71], "Taille"] = 35
df.loc[df["Taille"] == l[72], "Taille"] = 22
df.loc[df["Taille"] == l[73], "Taille"] = 190
df.loc[df["Taille"] == l[74], "Taille"] = 115
df.loc[df["Taille"] == l[75], "Taille"] = 28
df.loc[df["Taille"] == l[76], "Taille"] = float("nan")
df.loc[df["Taille"] == l[77], "Taille"] = 40
df.loc[df["Taille"] == l[78], "Taille"] = 135
df.loc[df["Taille"] == l[79], "Taille"] = 11

# %%
df[df["Taille"].apply(lambda val : (not isinstance(val, (float, int))))]

# %%
l_nct = df[df["Taille"] == 0]["NCT"].unique()
l_nct

# %%
df[df["NCT"] == l_nct[0]]

# %%
df.drop(5581, inplace = True)

# %%
df[df["NCT"] == l_nct[1]]

# %%
df.drop(11411, inplace = True)

# %%
df[df["NCT"] == l_nct[2]]

# %%
df.drop(11565, inplace = True)

# %%
df[df["NCT"] == l_nct[3]]

# %%
df.drop(11568, inplace = True)

# %%
df[df["NCT"] == l_nct[4]]

# %%
df.drop(11576, inplace = True)
df.drop(11577, inplace = True)

# %%
df[df["NCT"] == l_nct[5]]

# %%
df.drop(11631, inplace = True)

# %%
df.reset_index(drop=True, inplace=True)

# %%
df.drop(columns=["Taille Min.", "TailleMax"], inplace = True)


# %%
def rec_f_aux(i, imax, deltat_pred, taille_pred):
    if i == imax:
        return -1, 0
    elif np.isnan(df.loc[i, "Taille"]):
        taille_next, deltat_next = rec_f_aux(i+1, imax, deltat_pred + df.loc[i, "Time"], taille_pred)
        deltat_next += df.loc[i, "Time"]
        if (taille_pred == -1) & (taille_next != -1):
            df.loc[i, "Taille"] = taille_next
        elif (taille_pred != -1) & (taille_next == -1):
            df.loc[i, "Taille"] = taille_pred
        elif (taille_pred != -1) & (taille_next != -1):
            df.loc[i, "Taille"] = np.round((taille_pred * deltat_next + taille_next * deltat_pred) / (deltat_pred + deltat_next))
        return taille_next, deltat_next
    else :
        rec_f_aux(i+1, imax, df.loc[i, "Time"], df.loc[i, "Taille"])
        return df.loc[i, "Taille"], 0


# %%
for nct in df["NCT"].unique():
    if ((df["NCT"] == nct).sum() != df[df["NCT"] == nct]["Taille"].isna().sum()) & (df[df["NCT"] == nct]["Taille"].isna().sum() > 0):
        l = df[df["NCT"] == nct].index
        rec_f_aux(l[0], l[-1] + 1, 0, -1)


# %%
mean_Taille = np.round(df.groupby("NCT")["Taille"].mean().mean())
mean_Taille


# %%
#df.loc[df["Taille"].isna(), "Taille"] = mean_Taille

# %% [markdown]
# ### Concerning G

# %%
def tointG(row):
    try :
        return int(row["%G"])
    except:
        return row["%G"]


# %%
df["%G"] = df.apply(lambda row : tointG(row), axis = 1)


# %%
def test2G(row):
    if isinstance(row["%G"], str):
        l = row["%G"].split("à")
        if len(l) == 2:
            try :
                if (int(l[0]) == int(row["%G min"])) & (int(l[1]) == int(row["%G max"])):
                    return (int(l[0]) + int(l[1]))/2
            except:
                return row["%G"]
        l = row["%G"].split("et")
        if len(l) == 2:
            try :
                if (int(l[0]) == int(row["%G min"])) & (int(l[1]) == int(row["%G max"])):
                    return (int(l[0]) + int(l[1]))/2
            except:
                return row["%G"]
    return row["%G"]


# %%
df["%G"] = df.apply(lambda row : test2G(row), axis = 1)

# %%
l = df[~df["%G"].apply(lambda val : isinstance(val, (int, float)))]["%G"].unique()
l


# %%
def modif_G_nan(row):
    if row["%G"] in l:
      if isinstance(row["G"], int):
            if row["G"] != 0:
                return row["G"] / row["Taille"] * 100
    if isinstance(row["%G"], str):
        if row["%G"].endswith("aine"):
            return int(row["%G"][:-4])
    return row["%G"]


# %%
df["%G"] = df.apply(lambda row : modif_G_nan(row), axis = 1)

# %%
l2 = df[~df["%G"].apply(lambda val : isinstance(val, (int, float)))]["%G"].unique()
l2

# %%
i = -1

# %%
i += 1
print(i)
df.loc[df["%G"] == l2[i], ["%G", "%G min", "%G max", "G", "Taille"]]

# %%
df.loc[df["%G"] == l2[0], "%G"] = float("nan")
df.loc[df["%G"] == l2[1], "%G"] = 10
df.loc[df["%G"] == l2[2], "%G"] = 8
df.loc[df["%G"] == l2[3], "%G"] = 17
df.loc[df["%G"] == l2[4], "%G"] = 15
df.loc[df["%G"] == l2[5], "%G"] = float("nan")
df.loc[df["%G"] == l2[6], "%G"] = 35
df.loc[df["%G"] == l2[7], "%G"] = 75
df.loc[df["%G"] == l2[8], "%G"] = 40
df.loc[df["%G"] == l2[9], "%G"] = float("nan")
df.loc[df["%G"] == l2[10], "%G"] = 40
df.loc[df["%G"] == l2[11], "%G"] = 18
df.loc[df["%G"] == l2[12], "%G"] = 22
df.loc[df["%G"] == l2[13], "%G"] = 8
df.loc[df["%G"] == l2[14], "%G"] = 0
df.loc[df["%G"] == l2[15], "%G"] = float("nan")
df.loc[df["%G"] == l2[16], "%G"] = 12
df.loc[df["%G"] == l2[17], "%G"] = 15
df.loc[df["%G"] == l2[18], "%G"] = 12
df.loc[df["%G"] == l2[19], "%G"] = float("nan")
df.loc[df["%G"] == l2[20], "%G"] = 20
df.loc[df["%G"] == l2[21], "%G"] = float('nan')
df.loc[df["%G"] == l2[22], "%G"] = 23
df.loc[df["%G"] == l2[23], "%G"] = 0
df.loc[df["%G"] == l2[24], "%G"] = 12
df.loc[df["%G"] == l2[25], "%G"] = 50
df.loc[df["%G"] == l2[26], "%G"] = 8
df.loc[df["%G"] == l2[27], "%G"] = 15
df.loc[df["%G"] == l2[28], "%G"] = 40
df.loc[df["%G"] == l2[29], "%G"] = 50
df.loc[df["%G"] == l2[30], "%G"] = 45
df.loc[df["%G"] == l2[31], "%G"] = 15
df.loc[df["%G"] == l2[32], "%G"] = 12
df.loc[df["%G"] == l2[33], "%G"] = float("nan")
df.loc[df["%G"] == l2[34], "%G"] = 65
df.loc[df["%G"] == l2[35], "%G"] = float("nan")
df.loc[df["%G"] == l2[36], "%G"] = 78
df.loc[df["%G"] == l2[37], "%G"] = float("nan")

# %%
df[~df["%G"].apply(lambda val : isinstance(val, (int, float)))]

# %%
df.reset_index(drop=True, inplace=True)

# %%
df.drop(columns=["%G min", "%G max", "G"], inplace = True)


# %%
def rec_f_aux(i, imax, deltat_pred, taille_pred):
    if i == imax:
        return -1, 0
    elif np.isnan(df.loc[i, "Taille"]):
        taille_next, deltat_next = rec_f_aux(i+1, imax, deltat_pred + df.loc[i, "Time"], taille_pred)
        deltat_next += df.loc[i, "Time"]
        if (taille_pred == -1) & (taille_next != -1):
            df.loc[i, "Taille"] = taille_next
        elif (taille_pred != -1) & (taille_next == -1):
            df.loc[i, "Taille"] = taille_pred
        elif (taille_pred != -1) & (taille_next != -1):
            df.loc[i, "Taille"] = np.round((taille_pred * deltat_next + taille_next * deltat_pred) / (deltat_pred + deltat_next))
        return taille_next, deltat_next
    else :
        rec_f_aux(i+1, imax, df.loc[i, "Time"], df.loc[i, "Taille"])
        return df.loc[i, "Taille"], 0


# %%
def rec_f_aux_g(i, imax, deltat_pred, pcG_pred):
    if i == imax:
        return -1, 0
    elif np.isnan(df.loc[i, "%G"]):
        pcG_next, deltat_next = rec_f_aux_g(i+1, imax, deltat_pred + df.loc[i, "Time"], pcG_pred)
        deltat_next += df.loc[i, "Time"]
        if (pcG_pred == -1) & (pcG_next != -1):
            df.loc[i, "%G"] = pcG_next
        elif (pcG_pred != -1) & (pcG_next == -1):
            df.loc[i, "%G"] = pcG_pred
        elif (pcG_pred != -1) & (pcG_next != -1):
            df.loc[i, "%G"] = np.round((pcG_pred * deltat_next + pcG_next * deltat_pred) / (deltat_pred + deltat_next))
        return pcG_next, deltat_next
    else :
        rec_f_aux_g(i+1, imax, df.loc[i, "Time"], df.loc[i, "%G"])
        return df.loc[i, "%G"], 0


# %%
for nct in df["NCT"].unique():
    if ((df["NCT"] == nct).sum() != df[df["NCT"] == nct]["%G"].isna().sum()) & (df[df["NCT"] == nct]["%G"].isna().sum() > 0):
        l = df[df["NCT"] == nct].index
        rec_f_aux_g(l[0], l[-1] + 1, 0, -1)


# %%
mean_Gpc = df.groupby("NCT")["%G"].mean().mean()
mean_Gpc

# %%
#df.loc[df["%G"].isna(), "%G"] = mean_Gpc

# %%
df.rename(columns = {"%G" : "pcG"}, inplace = True)


# %% [markdown]
# ### Rayon

# %%
def tointR(row):
    try :
        return int(row["Rayon (m)"])
    except:
        return row["Rayon (m)"]


# %%
#df["Rayon (m)"] = df.apply(lambda row : tointR(row), axis = 1) I made an error and just under : all the case are made to match the different rayons man made

# %%
def test2Rayon(row):
    if isinstance(row["Rayon (m)"], str):
        l = row["Rayon (m)"].split("à")
        if len(l) == 2:
            try :
                return (int(l[0]) + int(l[1]))/2
            except :
                return row["Rayon (m)"]
    return row["Rayon (m)"]


# %%
df["Rayon (m)"] = df.apply(lambda row : test2Rayon(row), axis = 1)

# %%
l = df[~df["Rayon (m)"].apply(lambda val : isinstance(val, (int, float)))]["Rayon (m)"].unique()
l

# %%
i = -1

# %%
i += 1
print(i)
df.loc[df["Rayon (m)"] == l[i]]

# %%
df.loc[df["Rayon (m)"] == l[0], "Rayon (m)"] = float("nan")
df.loc[df["Rayon (m)"] == l[1], "Rayon (m)"] = 700
df.loc[df["Rayon (m)"] == l[2], "Rayon (m)"] = float("nan")
df.loc[df["Rayon (m)"] == l[3], "Rayon (m)"] = 20
df.loc[df["Rayon (m)"] == l[4], "Rayon (m)"] = 1100
df.loc[df["Rayon (m)"] == l[5], "Rayon (m)"] = 1100
df.loc[df["Rayon (m)"] == l[6], "Rayon (m)"] = float("nan")
df.loc[df["Rayon (m)"] == l[7], "Rayon (m)"] = 1500
df.loc[df["Rayon (m)"] == l[8], "Rayon (m)"] = 1500
df.loc[df["Rayon (m)"] == l[9], "Rayon (m)"] = 1600
df.loc[df["Rayon (m)"] == l[10], "Rayon (m)"] = 1500
df.loc[df["Rayon (m)"] == l[11], "Rayon (m)"] = 1100
df.loc[df["Rayon (m)"] == l[12], "Rayon (m)"] = 750
df.loc[df["Rayon (m)"] == l[13], "Rayon (m)"] = 1300
df.loc[df["Rayon (m)"] == l[14], "Rayon (m)"] = 20
df.loc[df["Rayon (m)"] == l[15], "Rayon (m)"] = 15
df.loc[df["Rayon (m)"] == l[16], "Rayon (m)"] = 80
df.loc[df["Rayon (m)"] == l[17], "Rayon (m)"] = 250
df.loc[df["Rayon (m)"] == l[18], "Rayon (m)"] = 20
df.loc[df["Rayon (m)"] == l[19], "Rayon (m)"] = 750
df.loc[df["Rayon (m)"] == l[20], "Rayon (m)"] = 1000
df.loc[df["Rayon (m)"] == l[21], "Rayon (m)"] = 1200
df.loc[df["Rayon (m)"] == l[22], "Rayon (m)"] = 2200
df.loc[df["Rayon (m)"] == l[23], "Rayon (m)"] = 180
df.loc[df["Rayon (m)"] == l[24], "Rayon (m)"] = 1200
df.loc[df["Rayon (m)"] == l[25], "Rayon (m)"] = 1100
df.loc[df["Rayon (m)"] == l[26], "Rayon (m)"] = float("nan")
df.loc[df["Rayon (m)"] == l[27], "Rayon (m)"] = float("nan")
df.loc[df["Rayon (m)"] == l[28], "Rayon (m)"] = 1000
df.loc[df["Rayon (m)"] == l[29], "Rayon (m)"] = float("nan")
df.loc[df["Rayon (m)"] == l[30], "Rayon (m)"] = float("nan")
df.loc[df["Rayon (m)"] == l[31], "Rayon (m)"] = 150
df.loc[df["Rayon (m)"] == l[32], "Rayon (m)"] = 100
df.loc[df["Rayon (m)"] == l[33], "Rayon (m)"] = 400
df.loc[df["Rayon (m)"] == l[34], "Rayon (m)"] = 40
df.loc[df["Rayon (m)"] == l[35], "Rayon (m)"] = float("nan")
df.loc[df["Rayon (m)"] == l[36], "Rayon (m)"] = 750
df.loc[df["Rayon (m)"] == l[37], "Rayon (m)"] = 20
df.loc[df["Rayon (m)"] == l[38], "Rayon (m)"] = 20
df.loc[df["Rayon (m)"] == l[39], "Rayon (m)"] = 450
df.loc[df["Rayon (m)"] == l[40], "Rayon (m)"] = 600
df.loc[df["Rayon (m)"] == l[41], "Rayon (m)"] = 1000
df.loc[df["Rayon (m)"] == l[42], "Rayon (m)"] = float("nan")
df.loc[df["Rayon (m)"] == l[43], "Rayon (m)"] = float("nan")
df.loc[df["Rayon (m)"] == l[44], "Rayon (m)"] = 600
df.loc[df["Rayon (m)"] == l[45], "Rayon (m)"] = float("nan")
df.loc[df["Rayon (m)"] == l[46], "Rayon (m)"] = 450
df.loc[df["Rayon (m)"] == l[47], "Rayon (m)"] = 175
df.loc[df["Rayon (m)"] == l[48], "Rayon (m)"] = float("nan")
df.loc[df["Rayon (m)"] == l[49], "Rayon (m)"] = float("nan")
df.loc[df["Rayon (m)"] == l[50], "Rayon (m)"] = 450
df.loc[df["Rayon (m)"] == l[51], "Rayon (m)"] = 175
df.loc[df["Rayon (m)"] == l[52], "Rayon (m)"] = float("nan")

# %%
df[~df["Rayon (m)"].apply(lambda val : isinstance(val, (int, float)))]["Rayon (m)"].unique()

# %%
# Je trouve ça bizarre d'extrapoler le rayon car on a aucune idée de si les animaux se sont ressérés ou non

# %%
mean_R = df.groupby("NCT")["Rayon (m)"].mean().mean()
mean_R

# %%
#df.loc[df["Rayon (m)"].isna(), "Rayon (m)"] = mean_R

# %%
df.rename(columns={"Rayon (m)" : "Rayon"}, inplace = True)

# %%
df

# %%
df.reset_index(inplace=True, drop = True)

# %%
df.to_csv("../datas/coords_belugas/cleaned_coords.csv" , sep = ";")

# %%
