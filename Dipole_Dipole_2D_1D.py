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

#lattice spacing
b_dash = 1

#angle between planes
theta = math.pi / 2

#lattice resolution (how many atoms in x and y directions)
lat_size = 100


#functions defined here

#initialises entire rows of the lattice at once
def single_row(b_dash, theta, j, lat_size):
    
    x_term = round((b_dash * np.cos(theta) * j) % 1, 5)
    y_term = j * (round(b_dash * np.sin(theta), 5))
    
    x_row = np.arange(x_term, lat_size * 1, 1)
    y_row = np.repeat(y_term, lat_size)
    
    return np.column_stack((x_row, y_row))


#Generates the lattice by appending entire layers of the lattice at once, then calculates distance matrix
def generate_lattice(lat_size, b_dash, theta, tot_atoms):
    
    points = np.empty([0, 2])
    
    #appending x row along y axis
    for j in range(lat_size):
        points = np.append(points, 
                           single_row(b_dash, theta, j, lat_size), 
                           axis = 0
                           )
    
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
def run_sim(b_dash, theta, lat_size):
    
    print("--------------------------------")
    print(" Variables:")
    print(" b':", b_dash)
    print(" θ:", round(theta, 4))
    print(" Lattice resolution: {0} by {0}".format(lat_size))
    
    start_time = time.perf_counter()
    
    tot_atoms = (lat_size ** 2)
    
    points, dist = generate_lattice(lat_size, b_dash, theta, tot_atoms)
    
    time_dist = time.perf_counter()
    print("\n", "Distance Matrix Created (s):", round(time_dist - start_time, 3))
    
    dip_relation = generate_dip_relation(tot_atoms, dist)
    
    time_dip = time.perf_counter()
    print(" Dipole Matrix Created (s):", round(time_dip - time_dist, 3))
    
    extreme_a = calc_alpha(dip_relation)
    
    end_time = time.perf_counter()
    runtime = round(end_time - start_time, 3)
    print(" Crit Alphas Calculated (s):", round(end_time - time_dip, 3))
    print("\n", "Total Runtime (s):", runtime)
    print(" Total Runtime (m):", round((end_time - start_time)/60, 3))
    print(" Extreme Alphas:", extreme_a)
    print("--------------------------------")
    
    return extreme_a


if __name__ == "__main__":
    
    alpha = run_sim(b_dash, theta, lat_size)