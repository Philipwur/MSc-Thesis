# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:56:52 2022

@author: Philipwur
"""

#Imports
import time
import numpy as np
import numpy.linalg as la
import scipy.linalg as sa #sa seems to be more reliable for large matrices, and la seems to be faster for smaller matrices
from numba import njit

#hyperparameters

lat_type = "FCC"  #"SC, "FCC" and "BCC" are the lattice options
lat_res = 10 #lattice resolution (N)


@njit()
def dipole_dipole(lat_type, lat_res):
    
    #Assigning the coordinates of the SC atoms
    
    if lat_type == "SC":
        
        points = np.array([[i, j, k] 
                        for k in range(lat_res) 
                        for j in range(lat_res) 
                        for i in range(lat_res)]).astype(np.float64)

    #Assigning the coordinates of the FCC atoms
        
    if lat_type == "FCC":
        
        prim_mult = 4 ** (1/3)
        
        prim_vec = 0.5 * prim_mult * np.array([[0, 1, 1],[1, 0, 1],[1, 1, 0]])
        
        points = np.zeros((lat_res ** 3, 3))
        
        count = 0
        
        for i in range(lat_res):
            for j in range(lat_res):
                for k in range(lat_res):
                    
                    points[count, :] = (i * prim_vec[0] 
                                        + j * prim_vec[1] 
                                        + k * prim_vec[2]
                                        )
                    count += 1
    #Assigning the coordinates of the FCC atoms
    
    if lat_type == "BCC":

        prim_mult = 2 ** (1/3)
        
        prim_vec = 0.5 * prim_mult * np.array([[-1, 1, 1],[1, -1, 1],[1, 1, -1]])
        
        points = np.zeros((lat_res ** 3, 3))
        
        count = 0
        
        for i in range(lat_res):
            for j in range(lat_res):
                for k in range(lat_res):
                    
                    points[count, :] = (i * prim_vec[0] 
                                        + j * prim_vec[1] 
                                        + k * prim_vec[2])
                    count += 1
                    
    #Finding the total amount of atoms in the lattice
    tot_atoms = len(points)
    
    #preallocation of dipole-dipole matrix
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
            
            kron = (((j - i) % 3 == 0) and (j != i)) #Kronecker-Delta Condition
            
            term1 = ((p2 - points[y2][x1]) *
                     (p1[y1] - points[y2][y1]) *
                     3) #term1 is one of the two components on the numerator of the CMM 
            
            euc = la.norm(p1 - points[y2]) if x2 != y2 else 1 #Magnitude of the vector between points
            
            relation[i][j] = (term1 - kron * euc * euc) / (euc ** 5) 
    
    return relation

#finds alpha from the lower tril of the symmetric matrix, since the matrix is symmetric the upper can be inferred
def find_alpha(relation):
    
    relation_eig = sa.eigh(relation,
                           lower = True,
                           overwrite_a = True,
                           check_finite = False,
                           eigvals_only= True)
    
    alpha = [1 / relation_eig[0], 1 / relation_eig[-1]] #taking the extremes of the eigenvalues
    
    return alpha


def main(lat_type, lat_res):
    
    relation = dipole_dipole(lat_type, lat_res)
    
    #print("array size:", relation.data.nbytes/(1024*1024*1024)) #if memory size of the total array needs to be known
    
    alpha = find_alpha(relation)
    
    return alpha


if __name__ == "__main__":
    
    _ = main("SC", 3) #warmup function to get the main function njit compiled
    
    del _
    
    print("# atoms: {}".format(lat_res ** 3))
    print("type:", lat_type)
    
    start = time.perf_counter()

    alpha = main(lat_type, lat_res) #actual high resolution simulation
    
    end = time.perf_counter()

    print("alpha:", alpha)
    print("time - m:", (end - start)/(60))
    print("time - s:", (end - start))
