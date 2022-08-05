#%%
#%%
# -*- coding: utf-8 -*-
import time
import math
#import logging soon!

import numpy as np
import numpy.linalg as la

from numba import njit
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

#import my_functions as fun (no use for this atm)

"""
#test values verified rahul on 2d-1d case (square lattice)
Nx, Ny = 500, 500
a, b = 1, 1 (b':1, θ=90)
-ac = -0.110836246
+ac = 0.377945157
"""

#variables defined here

#lattice vectors
v1 = [1, 0]
v2 = [0, 0]

#lattice resolution (how many atoms in x and y directions)
lat_size = [25, 31]


#functions defined here

# generates entire lattice and then slants it by v2
def vector_gen(v1, v2, lat_size, tot_atoms):
    
    proto_x = np.arange(0, v1[0]* lat_size, v1[0])
    proto_y = np.arange(0, v2[1]* lat_size, v2[1])

    xm, ym = np.meshgrid(proto_x, proto_y)
    
    try:
        xm = xm + np.arange(0, v2[0] * lat_size, v2[0])[:,None]
    
    finally:
        points = np.column_stack([xm.flatten(), ym.flatten()]) 
        
        dist = distance_matrix(points, points) + np.identity(tot_atoms)

        return dist


# function for calculating the dipole-dipole relation matrix
@njit()
def generate_dip_relation(tot_atoms, euc):
    
    # term 1 dissapears due to perpendicularity, division is simplified to -3 power
    relation = np.power(euc, -3) - np.identity(len(euc))
    relation = np.negative(relation)
        
    return relation 
    

#finding the extremes of the dipole-relation eigenvalues
def calc_alpha(dip_relation):
    
    relation_eig = la.eigvalsh(dip_relation)
    
    alpha = 1 / relation_eig[0]

    return alpha

# =============================================================================
# add logging here, or change print commands to update instead of create new ones
# tqdm module for loading bar
# =============================================================================

#master function, containing excecution order and print commands
def run_sim(v1, v2, lat_size):
    
    print("--------------------------------")
    print(" v2:", v2)
    print(" Lattice resolution: {0} by {0}".format(lat_size))

    tot_atoms = (lat_size ** 2)
    
    dist = vector_gen(v1, v2, lat_size, tot_atoms)
    
    dip_relation = generate_dip_relation(tot_atoms, dist)

    extreme_a = calc_alpha(dip_relation)
    
    print(" Extreme Alphas:", extreme_a)
    extreme_a = extreme_a / v2[1]
    return extreme_a


if __name__ == "__main__":
    
    results = []
    
    results1 = []
    results2 = []
    
    for i in range(18):
        
        i2 = round(0.1 + 0.05 * i, 3)
        
        v2[1] = i2
        alpha = run_sim(v1, v2, lat_size[0])
        #alpha2 = run_sim(v1, v2, lat_size[1])
        
        #alpha = linregress(x, y).intercept
        
        results.append([i2, alpha])
        results1.append([i2, alpha])
        
        v2[1] = 1/i2
        alpha = run_sim(v1, v2, lat_size[0])
        results2.append([i2, alpha])
        
        results.append([1/i2, alpha])
        
#%%
results = np.array(results)
results1 = np.array(results1)
results2 = np.array(results2)


fig = plt.figure(dpi = 300)

plt.scatter(results[:,0], results[:,1])
plt.title("x = 0, nc values")
#plt.plot(results[:,0], results[:,1])

fig = plt.figure(dpi = 300)
ax1 = fig.add_subplot(111)
ax1.scatter(results1[:,0], results1[:,1], label = "y<1")
ax1.scatter(results2[:,0], results2[:,1], label = "y>1, (1/y)")
plt.legend(loc='lower left')
plt.title("x = 0, nc values")
plt.show()