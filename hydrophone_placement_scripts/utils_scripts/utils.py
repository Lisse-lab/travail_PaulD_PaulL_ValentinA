import numpy as np
import math
import matplotlib.pyplot as plt

def bin_inf(b1 , b2):
    assert len(b1) == len(b2), "two binaries do not have the same size"
    for i in range (len(b1)):
        if b1[i]<b2[i]:
            return True
        elif b1[i]>b2[i]:
            return False
    return True

def sort_insert_point (x, lx):
    i = 0
    bool = True
    m = len(lx)
    while (i<m) & bool:
        if x.inf(lx[i]):
            i+=1
        else:
            bool = False
    lx.insert(i, x)
    return None

def bin2int (l):
    puis = 1
    n = 0
    for i in range (len(l)):
        n += l[-i-1] * puis
        puis *= 2
    return n

def float2bin(x, n):
    puis = 2 ** (n-1)
    if (x // puis > 1) | (x < 0):
        return []
    l = [0] * n
    for i in range (n):
        l[i] = int(x // puis)
        x = x % puis
        puis /= 2
    return l

def argmax (l):
    assert len(l)>0, "List cannot be empty"  
    m = l[0]
    ind = 0
    for i in range (len(l)):
        if l[i] > m:
            m = l[i]
            ind = i
    return ind

def k_best(l, k):
    if k == 1:
        return [argmax(l)]
    else:
        i = argmax(l)
        l2 = k_best(l[:i] + l[i+1:], k-1)
        for j in range (len(l2)):
            if l2[j]>=i:
                l2[j] += 1
        return [i] + l2
    
def argmin (l):
    assert len(l)>0, "List cannot be empty"  
    m = l[0]
    ind = 0
    for i in range (len(l)):
        if l[i] < m:
            m = l[i]
            ind = i
    return ind 

def dms_to_dd (dms) :
    dir = dms[-1]
    d, m, s = dms[:-1].split("-")
    dd = float(d) + float(m)/60 + float(s)/3600
    if dir in ["W", "S"]:
        dd *= -1
    return dd

def sort_insert_depth(x, lx):
    i = 0
    bool = True
    m = len(lx)
    while (i<m) & bool:
        if x[0] > lx[i][0]:
            i+=1
        else:
            bool = False
    if i==m :
        lx.insert(i, x)
        return True
    elif x[0]!=lx[i][0]:
        lx.insert(i, x)
        return True
    return False

def log(x):
    return np.log(np.finfo(float).eps + x)

def log10(x):
    return np.log10(np.finfo(float).eps + x)
    
def fun_rep_norm(x, mean, stddev):
    return 1/2 * (1+math.erf((x-mean)/(stddev*np.sqrt(2))))

def comb(i, j, area):
    if (i == 0): #if i == 0 then j == 0 in our case
        return [area]
    elif (j == 0) | (i == j):
        return [(area[0] + i, area[1] + j), (area[0] - j, area[1] + i), (area[0] - i, area[1] - j), (area[0] + j, area[1] - i)]
    else:
        return [(area[0] + i, area[1] + j), (area[0] + j, area[1] + i), (area[0] - i, area[1] + j), (area[0] + j, area[1] - i), (area[0] + i, area[1] - j), (area[0] - j, area[1] + i), (area[0] - i,area[1] -j), (area[0] - j, area[1] - i)]

def norme(x, y):
    return np.sqrt(x**2+y**2)

def get_points_to_calculate(angle_max, points_tried, angles_diff, dic_to_calc, mat_which_value):
    def rec_f (ind, previous, angle):
        p = points_tried[ind]
        pgcd = np.gcd(p[0],p[1])
        if (p[0]//pgcd, p[1]//pgcd) in dic_to_calc.keys():
            dic_to_calc[(p[0]//pgcd, p[1]//pgcd)] += 1
            if ind + 1 < len(points_tried):
                rec_f(ind + 1, p, 0)
            mat_which_value[p] = [p] 
            return p
        else:
            if (previous is None) | (angle + angles_diff[ind] > angle_max):
                dic_to_calc[(p[0]//pgcd, p[1]//pgcd)] = 1
                if ind + 1 < len(points_tried):
                    rec_f(ind + 1, p, 0)
                mat_which_value[p] = [p] 
                return p
            else:
                if ind + 1 < len(points_tried):
                    next = rec_f(ind+1, previous, angle + angles_diff[ind])
                    mat_which_value[p] = [previous, next]
                    return next
                else:
                    dic_to_calc[(p[0]//pgcd, p[1]//pgcd)] = 1
                    if ind + 1 < len(points_tried):
                        rec_f(ind + 1, p, 0)
                    mat_which_value[p] = [p] 
                    return p
    rec_f(0, None, 0)
    return dic_to_calc, mat_which_value

def get_to_calculate(angle, n):
    mat_which_value = np.empty((n,n), dtype=object)
    dic_to_calc = {}
    dic_to_calc[0,0] = 1
    mat_which_value[0,0] = [(0,0)]
    for k in range (1,n):
        i = 0
        l = []
        while i <= k:
            j = 0
            while (j <= i):
                if k - 1< norme(i,j) <= k:
                    l.append((i,j))
                j+=1
            i+=1
        l_angles = [np.arctan(e[1]/e[0]) for e in l]
        ind_sorted = np.argsort(l_angles)
        points_tried = [l[i] for i in ind_sorted]
        angles_tried = [l_angles[i] for i in ind_sorted]
        angles_diff = [0] + [angles_tried[i+1] - angles_tried[i] for i in range (len(angles_tried) - 1)]
        get_points_to_calculate(angle, points_tried, angles_diff, dic_to_calc, mat_which_value)
    return dic_to_calc, mat_which_value

def visualize_mat_which_value(mat_which_value):
    n = mat_which_value.shape[0]
    mat_test = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if not mat_which_value[i, j] is None:
                mat_test[i,j] = len(mat_which_value[i,j])
    plt.imshow(mat_test)

def find_indice_angles(angle, l_angles):
    ind = 0
    boolean = True
    while (ind < len(l_angles)) & boolean:
        if l_angles[ind] >= angle:
            boolean = False
        else:
            ind += 1
    return ind