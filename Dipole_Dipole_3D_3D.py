#%%
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:56:52 2022

@author: Philipwur
"""

import time

import numpy as np
import numpy.linalg as la
from numba import njit

#hyperparameters

lat_type = "SC"  #"FCC" and "BCC", anything other than FCC or BCC is assumed to be SC
lat_res = 10 #lattice resolution


@njit()
def dipole_dipole(lat_type, lat_res):
    
    #Assigning the coordinates of the SC atoms (present in all lattices)
    #this can be sped up but doesnt take much time in the grand scheme of things
    points = np.array([[i, j, k] 
                       for k in range(lat_res) 
                       for j in range(lat_res) 
                       for i in range(lat_res)]).astype(np.float64)

    if lat_type == "FCC":

        extra_points = np.array([[i + 0.5, j + 0.5, k] 
                                for k in range(lat_res) 
                                for j in range(lat_res - 1) 
                                for i in range(lat_res - 1)]).astype(np.float64)

        points = np.concatenate((points, extra_points))

        extra_points = np.array([[i, j + 0.5, k + 0.5] 
                            for k in range(lat_res - 1) 
                            for j in range(lat_res - 1) 
                            for i in range(lat_res)]).astype(np.float64)

        points = np.concatenate((points, extra_points))

        extra_points = np.array([[i + 0.5, j, k + 0.5] 
                            for k in range(lat_res - 1) 
                            for j in range(lat_res) 
                            for i in range(lat_res - 1)]).astype(np.float64)

        points = np.concatenate((points, extra_points))

        extra_points = None

    if lat_type == "BCC":

        extra_points = np.array([[i + 0.5, j + 0.5, k + 0.5] 
                                for k in range(lat_res - 1) 
                                for j in range(lat_res - 1) 
                                for i in range(lat_res - 1)]).astype(np.float64)

        points = np.concatenate((points, extra_points))

        extra_points = None    
    
    tot_atoms = len(points)

    print("atoms:",  len(points))
    
    #preallocation
    relation = np.zeros((3 * tot_atoms, 3 * tot_atoms))
    
    #calulating the dipole-dipole relation without any stored arrays for kron or euc to save RAM
    for i in range(0, 3 * tot_atoms):
        
        x1 = i % 3
        x2 = i // 3
        
        p1 = points[x2]
        p2 = points[x2][x1]
        
        #only calculates lower triangular for the symmetric hermetic matrix
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

#finds alpha from the lower tril of the symmetric matrix
def find_alpha(relation):
    
    relation_eig = la.eigvalsh(relation, UPLO = "L")
    
    alpha = 1 / relation_eig[0]
    
    return alpha


def main(lat_type, lat_res):
    
    relation = dipole_dipole(lat_type, lat_res)
    
    #print("array size:", relation.data.nbytes/(1024*1024*1024))
    
    alpha = find_alpha(relation)
    
    return alpha


if __name__ == "__main__":
    
    
    _ = main(2, 5) #warmup function to get the main function compiled
    
    del _
    
    start = time.perf_counter()
    
    alpha = main(lat_type, lat_res) #actual high resolution simulation
    
    end = time.perf_counter()
    
    #stats
    print("type:", lat_type)
    print("alpha:", alpha)
    print("time - h:", (end - start)/(60*60))
    print("time - m:", (end - start)/(60))
    print("time - s:", (end - start))
    
#%%

