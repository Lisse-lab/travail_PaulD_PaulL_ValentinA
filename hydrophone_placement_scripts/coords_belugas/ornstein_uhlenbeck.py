import numpy as np
from scipy import optimize
import folium
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def optimise_sigma2(df, delta2):
    df = df.assign(a=np.nan, b=np.nan, p=np.nan)
    for nct in df["NCT"].unique():
        l = df[df["NCT"] == nct].index
        for i in range (l[0] + 1, l[-1], 2):
            alpha = (df.loc[i, "Time"] - df.loc[i-1, "Time"]) / (df.loc[i+1, "Time"] - df.loc[i-1, "Time"])
            x_pred = df.loc[i-1, "x"] * alpha + df.loc[i+1, "x"] * (1-alpha)
            y_pred = df.loc[i-1, "y"] * alpha + df.loc[i+1, "y"] * (1-alpha)
            df.loc[i, "p"] = ((df.loc[i,"x"] - x_pred)**2 + (df.loc[i,"y"] - y_pred)**2) / 2
            df.loc[i, "a"] = (df.loc[i+1, "Time"] - df.loc[i-1, "Time"]) * alpha * (1-alpha)
            df.loc[i, "b"] = ((1-alpha)**2 + alpha**2) * delta2
    df_sigma = df.dropna()

    def li_neg(sigma2, a, b, p):
        s2 = a * sigma2 + b
        return p / s2 + np.log(s2)

    def l_neg(sigma2):
        return df_sigma.apply(lambda row : li_neg(sigma2, row.a, row.b, row.p), axis = 1).sum()

    def gradi(sigma2, a, b, p):
        s2 = a * sigma2 + b
        return -a*p / (s2**2) + a / s2

    def grad(sigma2):
        return df_sigma.apply(lambda row : gradi(sigma2, row.a, row.b, row.p), axis = 1).sum()
    
    l_sigma2 = []
    bounds = [(0, None)]
    result = optimize.minimize(l_neg, 3000, jac = grad, bounds=bounds)
    l_sigma2.append(result.x)
    print(result)

    result = optimize.minimize(l_neg, 3000, bounds=bounds, method='Nelder-Mead')
    l_sigma2.append(result.x)
    print(result)

    result = optimize.minimize(l_neg, 3000, bounds=bounds, method='Powell')
    l_sigma2.append(result.x)
    print(result)

    sigma2 = np.array(l_sigma2).mean()
    
    df.drop(["a", "b", "p"], axis = 1, inplace = True)

    return sigma2

def constrained_phi(x):
    if x > 1:
        return 1
    elif x < 0:
        return 0
    else :
        return x

def get_inv_tau0(df, mu, with_first=False):
    if with_first:
        offset = 0
    else:
        offset = 1
    inv_tau = 0
    c = 0
    for nct in df["NCT"].unique():
        l = df[df["NCT"] == nct].index
        for i in range (l[0] + offset, l[-1]):
            if i == l[0]:
                v_ix, v_iy = 0, 0
            else:
                deltat_i = df.loc[i, "Time"] - df.loc[i-1, "Time"]
                v_ix = (df.loc[i, "x"] - df.loc[i-1, "x"]) / deltat_i
                v_iy = (df.loc[i, "y"] - df.loc[i-1, "y"]) / deltat_i
            deltat_ip1 = df.loc[i+1, "Time"] - df.loc[i, "Time"]
            v_ip1x = (df.loc[i+1, "x"] - df.loc[i, "x"]) / deltat_ip1
            v_ip1y = (df.loc[i+1, "y"] - df.loc[i, "y"]) / deltat_ip1
            m = mu(df.loc[i, "x"], df.loc[i, "y"])
            if (v_ix != m[0]) & (v_iy != m[1]):
                phi = ((v_ip1x - m[0])/(v_ix - m[0]) + (v_ip1y - m[1])/(v_iy - m[1])) / 2
                inv_tau += (1 - constrained_phi(phi)) / deltat_ip1
                c += 1
    return inv_tau/c

def get_deltav2(df, with_first = False):
    v = 0
    c = 0
    for nct in df["NCT"].unique(): 
        l = df[df["NCT"] == nct].index
        if with_first:
            c+=1 #to do like there was a velocity null at the begining
        for i in range (l[0], l[-1]):
            v += ((df.loc[i+1, "x"] - df.loc[i, "x"]) / (df.loc[i+1, "Time"] - df.loc[i, "Time"]))**2
            v += ((df.loc[i+1, "y"] - df.loc[i, "y"]) / (df.loc[i+1, "Time"] - df.loc[i, "Time"]))**2
            c += 1
    return v/c

def get_sigmav2(df, inv_tau0, calc_mu, display = True, with_first = False):#pour filtre de kalman ?
    if with_first:
        offset = 0
    else :
        offset = 1
    l_err_x = []
    l_err_v = []
    l_t = []
    for nct in df["NCT"].unique():
        l = df[df["NCT"] == nct].index
        if with_first:
            traj = ou_forecast_trajectory(df["x"].loc[l[0]], df["y"].loc[l[0]], 0, 0, inv_tau0, 0, calc_mu, lambda x : None, False)
        else:
            if len(l) > 1:
                deltat = df["Time"].loc[l[1]] - df["Time"].loc[l[0]]
                traj = ou_forecast_trajectory(df["x"].loc[l[1]], df["y"].loc[l[1]], (df["x"].loc[l[1]] - df["x"].loc[l[0]])/deltat, (df["y"].loc[l[1]]-df["y"].loc[l[0]])/deltat, inv_tau0, 0, calc_mu, lambda x : None, False)
        for i in range (l[0] + offset, l[-1]):
            deltat = df["Time"].loc[i+1] - df["Time"].loc[i]
            l_t.append(deltat)
            err_x, err_v = traj.err_set(deltat, np.array([df["x"].loc[i+1], df["y"].loc[i+1]]))          
            l_err_x.append(err_x)
            l_err_v.append(err_v)
            
    ts = (np.array(l_t)).reshape(-1, 1)
    ts_reg = np.linspace(min(ts), max(ts), 100).reshape(-1, 1)
    model_v = LinearRegression(fit_intercept=False)
    model_v.fit(ts, l_err_v)
    errs_reg = model_v.predict(ts_reg)
    
    if display:
        plt.scatter(l_t, l_err_v, color='red', s=5, label='Real Datas')
        plt.plot(ts_reg, errs_reg, color='blue', label=f'Linear Regression')
        plt.xlabel('Delta t')
        plt.ylabel('Error in v')
        plt.title('Linear Regression on deltat')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return model_v.coef_[0]

class ou_forecast_trajectory :
    
    def __init__(self, x, y, delta2, deltav2, inv_tau0, sigmav2, calc_mu, to_wgs, bool_map, name = "ma_carte", plot_color="blue"):
        self.X_lp = np.array([x, y, 0, 0]) #lp stands for last predicted, (np : new_predicted) if it was corrected it would be the last or the new corrected
        self.X_lc = np.array([x, y, 0, 0]) #lp stands for last corrected, (X_nc doesn't exist because X_np would just be modified)
        self.X_np = np.array([x, y, 0, 0])
        self.delta2 = delta2
        self.deltav2 = deltav2
        self.sigmav2 = sigmav2
        self.inv_tau = inv_tau0
        self.k = 0
        self.detltat_slp = 0 #slp stands for since last prediction
        self.deltat_slc = 0 #slc stands for since last correction
        self.create_P0()
        self.create_Q()
        self.create_R()
        self.calc_mu = calc_mu
        self.it_c = 0
        self.it_p = 0
        self.to_wgs = to_wgs
        self.name = name
        self.plot_color = plot_color
        self.bool_map = bool_map
        if self.bool_map:
            self.map = folium.Map(location=[47.86, -69.66], zoom_start=12)
        self.add_to_map(self.X_lp[0], self.X_lp[1], "obs")
        self.list_of_rot_and_deltat = []

    def add_to_map(self, x, y, type):
        if self.bool_map:
            lon, lat = self.to_wgs(x,y)
            if type == "pred":
                txt = "pred " + str(self.it_c) + "." + str(self.it_p)
                color = self.plot_color
            else :
                txt = type + " " + str(self.it_c)
                color = "green"
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color=color,
                fill=True,
                fill_color=color,
                tooltip=folium.Tooltip(txt, permanent=False)
                ).add_to(self.map)
            self.map.save(self.name + ".html")
        else:
            return None

    def create_P0(self):
        m = np.zeros([4,4])
        m[0,0] = self.delta2
        m[1,1] = self.delta2
        m[2,2] = self.deltav2
        m[3,3] = self.deltav2
        self.P = m
        
    def create_Q(self):
        self.Q = np.zeros([4,4])

    def create_R(self):
        m = np.zeros([4,4])
        m[0,0] = self.delta2
        m[1,1] = self.delta2
        self.R = m

    def phi(self):
        return constrained_phi(1 - self.deltat_slc * self.inv_tau)

    def update_inv_tau(self):
        m = self.calc_mu.mu(self.X_lc[0], self.X_lc[1])
        if (self.X_lc[2] != m[0]) & (self.X_lc[3] != m[1]): 
            phi = ((self.X_np[2] - m[0])/(self.X_lc[2] - m[0]) + (self.X_np[3] - m[1])/(self.X_lc[3] - m[1])) / 2
            self.inv_tau = (1 - constrained_phi(phi))
                            
    def f(self):
        m = self.calc_mu.mu(self.X_lp[0], self.X_lp[1])
        if m.T@m > 1e10:
            self.inv_tau = 0
        dm = self.calc_mu.dmu(self.X_lp[0], self.X_lp[1])
        if np.max(np.abs(dm))>1e6:
            self.inv_tau = 0
        fvx = (1-self.phi())*m[0] + self.phi() * self.X_lp[2]
        fvy = (1-self.phi())*m[1] + self.phi() * self.X_lp[3]
        self.X_np, self.list_of_rot_and_deltat = self.new_point(fvx, fvy)
        
    def df(self):
        dm = self.calc_mu.dmu(self.X_lp[0], self.X_lp[1])
        m = np.zeros([4,4])
        t = np.zeros([2,4])

        t[0,0] = (1 - self.phi()) * dm[0,0]
        t[0,1] = (1 - self.phi()) * dm[0,1]
        t[0,2] = self.phi()

        t[1,0] = (1 - self.phi()) * dm[1,0]
        t[1,1] = (1 - self.phi()) * dm[1,1]
        t[1,3] = self.phi()

        for theta, deltat in self.list_of_rot_and_deltat:
            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            m[0:2] += deltat * R@t
        m[2:4] = R@t
        return m
    
    def g(self):
        return self.X_np
    
    def dg(self):
        m = np.zeros([4,4])
        m[0,0] = 1
        m[1,1] = 1
        m[2,0] = 1 / self.deltat_slp
        m[3,1] = 1 / self.deltat_slp        
        return m
    
    def update_R(self):
        self.R[2,2] = self.delta2 / self.deltat_slc
        self.R[3,3] = self.delta2 / self.deltat_slc
    
    def update_Q(self):
        self.Q[0,0] = self.sigmav2 * self.deltat_slc**3
        self.Q[1,1] = self.sigmav2 * self.deltat_slc**3
        self.Q[2,2] = self.sigmav2 * self.deltat_slc
        self.Q[3,3] = self.sigmav2 * self.deltat_slc
    
    def predict(self, deltat):
        self.it_p += 1
        self.X_lp = self.X_np.copy()
        self.deltat_slp = deltat
        self.deltat_slc += deltat
        self.f()
        self.add_to_map(self.X_np[0], self.X_np[1], "pred")
        return self.X_np
        
    def correct(self, x, y):
        self.update_R()
        self.update_Q()
        X_o = np.array([x, y, (x - self.X_lc[0])/self.deltat_slc, (y - self.X_lc[1])/self.deltat_slc])#o stands for observed
        Y = X_o - self.g()
        self.add_to_map(X_o[0], X_o[1], "obs")
        F = self.df()
        G = self.dg()
        self.P = F@self.P@(F.T) + self.Q
        S = G@self.P@(G.T) + self.R
        K = self.P@(G.T) @ np.linalg.inv(S)
        if np.linalg.det(K) > 1:
            print("Hmmmmm, det K = ", np.linalg.det(K))
        if np.trace(K) > 4:
            print("Hmmmmm, trace K = ", np.trace(K))
        self.X_np += K@Y
        self.P = (np.eye(4) - K@G) @ self.P
        self.update_inv_tau()
        self.deltat_slc = 0
        self.X_lc = self.X_np.copy()
        self.it_c += 1
        self.it_p = 0
        self.add_to_map(self.X_np[0], self.X_np[1], "corr")
        return self.X_np
    
    def predict_correct(self, deltat, x, y):
        _ = self.predict(deltat)
        return self.correct(x, y)
    
    def err_set(self, deltat, x_o):
        x = self.predict(deltat)
        v_o = ((x_o[0] - self.X_lc[0])/deltat, (x_o[1] - self.X_lc[1])/deltat)
        err_x = (x[0] - x_o[0])**2 + (x[1] - x_o[1])**2
        err_v = (x[2] - v_o[0])**2 + (x[3] - v_o[1])**2
        self.X_np = np.array([x_o[0], x_o[1], v_o[0], v_o[1]])
        self.update_inv_tau()
        self.X_lc = self.X_np.copy()
        return (err_x, err_v)

    def new_point(self, fvx, fvy):
        return self.calc_mu.new_point(self.X_lp[0], self.X_lp[1], fvx, fvy, self.deltat_slp)
    
    def set_to_obs(self, x, y):
        X_o = np.array([x, y, (x - self.X_lc[0])/self.deltat_slc, (y - self.X_lc[1])/self.deltat_slc])#o stands for observed
        self.add_to_map(X_o[0], X_o[1], "obs")
        self.X_np = X_o
        self.update_inv_tau()
        self.deltat_slc = 0
        self.X_lc = self.X_np.copy()
        self.it_c += 1
        self.it_p = 0
        self.add_to_map(self.X_np[0], self.X_np[1], "corr")
        return self.X_np