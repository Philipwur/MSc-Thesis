# -*- coding: utf-8 -*-
import time
import math
import numpy as np
import scipy.linalg as la
from scipy.spatial import distance_matrix 

#import my_functions as fun

"""
#test values verified rahul on 2d-1d case (square lattice)
Nx, Ny = 500, 500
a, b = 1, 1
-ac = -0.110836246
+ac = 0.377945157

fixed dipolarisable entities system

DONE
- squashed 3D bugs
- made 3D system 2D (unsure about kronecker delta)
- made new point generation system using theta
- rewrote genertation of points to use faster numpy systems
- rewrote dip-dip relation function, lowering 100x100 runtime from 7min to 25secs

WIP
- try to save more time 
- create regression to N = inf
- start literature review - get to dipole-dipole matrix from scratch and look into alphac and Nc
- thinking about the boundary conditions of the coordinate system
  will have to be a boundary functions
  e.g. 1, pi/2 = 2, pi/3 in coordinates generated 
"""

#variables defined here

#lattice spacing
b_dash = 1

#angle between planes
theta = math.pi/2

#lattice resolution (how many atoms in x and y directions)
lat_size = 77


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
        points = np.append(points, single_row(b_dash, theta, j, lat_size),axis=0)
    
    dist = distance_matrix(points, points) + np.identity(tot_atoms)
    
    return points, dist


# function for calculating the dipole-dipole relation matrix
def generate_dip_relation(tot_atoms, euc):
    
    relation = np.power(euc, -3) - np.identity(len(euc))
    relation = np.negative(relation)
        
    return relation 
    

#finding the extremes of the dipole-relation eigenvalues
def calc_alpha(dip_relation, solver):
    
    if solver in ["evr", "evx"]:
    
        relation_eig = la.eigh(dip_relation,
                               eigvals_only = True,
                               overwrite_a = True,
                               overwrite_b = True,
                               check_finite = False,
                               subset_by_index = ([0, 0]),
                               driver = solver
                               )
    else:
        
        relation_eig = la.eigh(dip_relation,
                               eigvals_only = True,
                               overwrite_a = True,
                               overwrite_b = True,
                               check_finite = False,
                               driver = solver
                               )
        
    extreme_a = 1/relation_eig[0]

    return np.round(extreme_a, 5)


#master function, containing excecution order and print commands
def run_sim(b_dash, theta, lat_size, solver):
    
    print("--------------------------------")
    print(" Variables:")
    print(" b':", b_dash)
    print(" Theta:", round(math.degrees(theta), 4))
    print(" Lattice resolution:", lat_size)
    print(" Solver:", solver)
    
    start_time = time.time()
    
    tot_atoms = (lat_size ** 2)
    
    points, dist = generate_lattice(lat_size, b_dash, theta, tot_atoms)
    
    time_dist = time.time()
    print("\n", "Distance Matrix Created (s):", round(time_dist - start_time, 3))
    
    dip_relation = generate_dip_relation(tot_atoms, dist)
    
    time_dip = time.time()
    print(" Dipole Matrix Created (s):", round(time_dip - time_dist, 3))
    
    extreme_a = calc_alpha(dip_relation, solver)
    
    end_time = time.time()
    runtime = round(end_time - start_time, 3)
    print(" Crit Alphas Calculated (s):", round(end_time - time_dip, 3))
    print("\n", "Total Runtime (s):", runtime)
    print(" Total Runtime (m):", round((end_time - start_time)/60, 3))
    print(" Extreme Alphas:", extreme_a)
    print("--------------------------------")
    
    return extreme_a, runtime


if __name__ == "__main__":
    
    alpha, runtime = run_sim(b_dash, theta, lat_size)
    
    
    
    
#%%


