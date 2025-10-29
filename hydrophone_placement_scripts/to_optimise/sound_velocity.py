import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def get_model(depths, sound_velocities, n):
    poly = PolynomialFeatures(degree=n)
    d_poly = poly.fit_transform(depths.reshape(-1,1))
    model = LinearRegression()
    model.fit(d_poly, sound_velocities)
    y_pred = model.predict(d_poly)
    return (r2_score(sound_velocities, y_pred), model)

def get_sound_velocities(file, depth_const = 17, depth_max=100, deg_max=20, display=False, resol = 1):
    df = pd.read_csv(file)
    df2 = df[df["Depth"] <=depth_const] #after this depth we will consider the sound velocity as constant
    depths = df2["Depth"].values.reshape(-1,1)
    sound_velocities = df2["Sound Velocity"]
    r2s = []
    for i in range(deg_max+1):
        r2, _ = get_model(depths, sound_velocities, i)
        r2s.append(r2)
    k = np.argmax(r2s)
    r2, model = get_model(depths, sound_velocities, k)
    poly = PolynomialFeatures(degree=k)
    d = np.arange(0, depth_const + 1)
    d_poly = poly.fit_transform(d.reshape(-1,1))
    sv = model.predict(d_poly)
    if depth_max > depth_const: #because we only have information down to 31m
        d2 = np.arange(d[-1] + 1, depth_max + 1)
        sv2 = np.full(d2.shape, df[df["Depth"] >=depth_const]["Sound Velocity"].mean())
        d = np.concatenate([d, d2])
        sv = np.concatenate([sv, sv2])
        depths = df["Depth"].values.reshape(-1,1)
        sound_velocities = df["Sound Velocity"]
    if display:
        disp(depths, sound_velocities, d, sv, k, r2) #r2 is only for the first part which is in the regression
    return [[d[i], sv[i]] for i in range(len(d))][::resol]

def disp(depths, sound_velocities, d, sv, k, r2):
    plt.scatter(sound_velocities, -depths, color='blue', label='Real Data')
    plt.plot(sv, -d, color='red', label='Polynomial regression')
    plt.xlabel('Sound velocity (m/s)')
    plt.ylabel('Depth (m)')
    plt.title(f'Poylinomial Regression (Degree : {k}, R2 : {np.round(r2,3)})')
    plt.legend()
    plt.grid(True)
    plt.show()
    return None

if __name__ == '__main__':
    import os
    file = os.path.join(os.path.dirname(__file__), "../datas/sound_velocity_new.csv")
    ssp = get_sound_velocities(file, display=True, resol=2)
    print(ssp)
