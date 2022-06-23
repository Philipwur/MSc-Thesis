#%%
import sys

import time
import numpy as np

import numpy.linalg as la

from numba import njit


@njit()
def dipole_dipole(a, lat_res):
    
    tot_atoms = (lat_res ** 3)
    relation = np.zeros((3 * tot_atoms, 3 * tot_atoms))
    
    #Assigning the coordinates of the SC atoms (present in all lattices)
    points = np.array([[i * a, j * a, k * a] 
                             for k in range(lat_res) 
                             for j in range(lat_res) 
                             for i in range(lat_res)]).astype(np.float64)
    
    for i in range(0, 3 * tot_atoms):
        
        x1 = i % 3
        x2 = i // 3
        
        p1 = points[x2]
        p2 = points[x2][x1]
        
        for j in range(0, i):
            
            y1 = j % 3
            y2 = j // 3
            
            kron = (((j - i) % 3 == 0) and (j != i))
            
            term1 = ((p2 - points[y2][x1]) *
                     (p1[y1] - points[y2][y1]) *
                     3)
            
            euc = la.norm(p1 - points[y2]) if x2 != y2 else 1 
            
            relation[i][j] = (term1 - kron * euc * euc) / (euc ** 5)
    
    return relation

a = 2
lat_res = 5

relation = dipole_dipole(a, lat_res)
    
