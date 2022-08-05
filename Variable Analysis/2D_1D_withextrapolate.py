#%%
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
from scipy.stats import linregress

from numba import njit
from scipy.spatial import distance_matrix


#lattice vectors
v1 = (1, 0)
v2 = (0.5, 0.4)

#lattice resolution (how many atoms in x and y directions)
lat_size = [27, 28]


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
def generate_dip_relation(euc):
    
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
    
    tot_atoms = (lat_size ** 2)
    dist = vector_gen(v1, v2, lat_size, tot_atoms)
    dip_relation = generate_dip_relation(dist)
    extreme_a = calc_alpha(dip_relation)
    
    print(extreme_a)
    
    return extreme_a


if __name__ == "__main__":
    
    alpha1 = run_sim(v1, v2, lat_size[0])
    alpha2 = run_sim(v1, v2, lat_size[1])
    
    y = [alpha1, alpha2]
    x = np.divide(1, lat_size)
    
    print("alpha:", linregress(x, y).intercept)