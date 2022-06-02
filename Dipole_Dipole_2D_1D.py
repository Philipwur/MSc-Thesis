#%%
# -*- coding: utf-8 -*-
import time
import math
#import logging soon!

import numpy as np
import scipy.linalg as la
from scipy.spatial import distance_matrix

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
v1 = (1, 0)
v2 = (0, 1)

#lattice resolution (how many atoms in x and y directions)
lat_size = 29


#functions defined here

# generates entire lattice and then slants it by v2
def vector_gen(v1, v2, latsize, tot_atoms):
    
    proto_x = np.linspace(0, v1[0]*latsize, latsize)
    proto_y = np.linspace(0, v2[1]*latsize, latsize)
    
    xm, ym = np.meshgrid(proto_x, proto_y)

    try:
        xm = xm + np.linspace(0, v2[0]*latsize, latsize)[:,None]
    
    finally:
        points = np.column_stack([xm.ravel(), ym.ravel()]) 
        
        dist = distance_matrix(points, points) + np.identity(tot_atoms)
    
        return points, dist


# function for calculating the dipole-dipole relation matrix
def generate_dip_relation(tot_atoms, euc):
    
    # term 1 dissapears due to perpendicularity, division is simplified to -3 power
    relation = np.power(euc, -3) - np.identity(len(euc))
    relation = np.negative(relation)
        
    return relation 
    

#finding the extremes of the dipole-relation eigenvalues
def calc_alpha(dip_relation):
    
    #ev and evd were found to be most appropriate, ev working better at low N and evd at high N
    relation_eig = la.eigh(dip_relation,
                           eigvals_only = True,
                           overwrite_a = True,
                           overwrite_b = True,
                           check_finite = False,
                           driver = "evd"
                           )
    
    extreme_a = [1/relation_eig[0], 1/relation_eig[-1]]

    return np.round(extreme_a, 5)

# =============================================================================
# add logging here
# =============================================================================

#master function, containing excecution order and print commands
def run_sim(v1, v2, lat_size):
    
    # print("--------------------------------")
    # print(" Variables:")
    # print(" v1:", v1)
    # print(" v2:", v2)
    # print(" Lattice resolution: {0} by {0}".format(lat_size))
    
    start_time = time.perf_counter()
    
    tot_atoms = (lat_size ** 2)
    
    points, dist = vector_gen(v1, v2, lat_size, tot_atoms)
    
    time_dist = time.perf_counter()
    # print("\n", "Distance Matrix Created (s):", np.round(time_dist - start_time, 3))
    
    dip_relation = generate_dip_relation(tot_atoms, dist)
    
    time_dip = time.perf_counter()
    # print(" Dipole Matrix Created (s):", np.round(time_dip - time_dist, 3))
    
    extreme_a = calc_alpha(dip_relation)
    
    end_time = time.perf_counter()
    runtime = np.round(end_time - start_time, 3)
    # print(" Crit Alphas Calculated (s):", np.round(end_time - time_dip, 3))
    # print("\n", "Total Runtime (s):", runtime)
    # print(" Total Runtime (m):", np.round((end_time - start_time)/60, 3))
    # print(" Extreme Alphas:", extreme_a)
    # print("--------------------------------")
    
    return extreme_a


if __name__ == "__main__":
    
    alpha = run_sim(v1, v2, lat_size)