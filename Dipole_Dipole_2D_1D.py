#%%
# -*- coding: utf-8 -*-
import time

import numpy as np
import numpy.linalg as la

from numba import njit
from scipy.spatial import distance_matrix


#variables defined here

#lattice vectors
v1 = (1, 0)
v2 = (0, 0.1)

#lattice resolution (how many atoms in x and y directions)
lat_size = 21



#functions defined here

# generates entire lattice and then slants it by v2
def generate_lattice(N, v2):
    
    points = np.empty((0, 2))
    
    #appending x row along y axis
    for j in range(N):
        
        row = [(v2[0] * j + i, v2[1] * j) for i in range(N)]
        
        points = np.append(points, 
                           row, 
                           axis = 0)
        
    dist = distance_matrix(points, points) + np.identity((N ** 2))

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


#master function, containing excecution order and print commands
def run_sim(v1, v2, lat_size):
    
    #print("--------------------------------")
    print("v2:", v2)
    #print("Lattice resolution: {0} by {0}".format(lat_size))
    
    start_time = time.perf_counter()
    
    tot_atoms = (lat_size ** 2)
    
    dist = generate_lattice(lat_size, v2)
    
    dip_relation = generate_dip_relation(tot_atoms, dist)
        
    extreme_a = calc_alpha(dip_relation)
    
    end_time = time.perf_counter()
    runtime = np.round(end_time - start_time, 5)
    #print("Total Runtime (s):", runtime)
    print("Extreme Alphas:", extreme_a)
    
    
    return extreme_a, runtime


if __name__ == "__main__":
    
    alpha, runtime = run_sim(v1, v2, lat_size)