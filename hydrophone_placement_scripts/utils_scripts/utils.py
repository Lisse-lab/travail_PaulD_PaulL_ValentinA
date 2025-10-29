import numpy as np
import math

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
