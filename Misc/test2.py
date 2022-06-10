#%%
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 13:53:38 2022

@author: Philipwur
"""

import time
from matplotlib.font_manager import json_load
import numpy as np


import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sa
from scipy.spatial import distance_matrix 

import numba as nb
from numba import njit, prange
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


#%% no sparse matrix

a = 2
lat_res = 20
start = time.perf_counter()

@njit(parallel = True)
def dipole_dipole(a, lat_res):
        
    tot_atoms = (lat_res ** 3)
    relation = np.zeros((3 * tot_atoms, 3 * tot_atoms))
    
    #Assigning the coordinates of the SC atoms (present in all lattices)
    points = np.array([[i * a, j * a, k * a] 
                             for k in range(lat_res) 
                             for j in range(lat_res) 
                             for i in range(lat_res)]).astype(np.float_)
    
    for i in prange(0, 3 * tot_atoms):
        
        x1 = i % 3
        x2 = i // 3
        
        p1 = points[x2]
        p2 = points[x2][x1]
        
        for j in prange(0, i):
            
            y1 = j % 3
            y2 = j // 3
            
            kron = (((j - i) % 3 == 0) and (j != i))
            
            term1 = ((p2 - points[y2][x1]) *
                     (p1[y1] - points[y2][y1]) *
                     3)
            
            euc = la.norm(p1 - points[y2]) if x2 != y2 else 1 
            
            relation[i, j] = (term1 - kron * euc * euc) / (euc ** 5)
    
    return relation

relation = dipole_dipole(a, lat_res)
    
relation = relation + relation.T

@njit
def get_alpha(relation):
    
    alphac = 1/(min(la.eigvalsh(relation)))
    
    return alphac

alphac = get_alpha(relation)

print("alpha:", alphac)
end = time.perf_counter()
print("time - h:", (end - start)/(60*60))
print("time - m:", (end - start)/(60))
print("time - s:", (end - start))
print("array size:", relation.data.nbytes/(1024*1024*1024))

#%% sparse matrix

a, b, c = 2, 2, 2
lat_res = 10

start = time.perf_counter()
#calculating total amount of atoms for the SC
tot_atoms = (lat_res ** 3)

relation = sp.lil_matrix(np.zeros((3, 3)))
relation.resize(3 * tot_atoms, 3 * tot_atoms)

#Assigning the coordinates of the SC atoms (present in all lattices)
points = np.array([[i * a, j * b, k * c] 
                   for k in range(lat_res) 
                   for j in range(lat_res) 
                   for i in range(lat_res)])

@njit(nogil = True)
def col_prep(i, points):
    
    x1 = i % 3
    x2 = i // 3
        
    p1 = points[x2]
    p2 = points[x2][x1]
    
    return x1, x2, p1, p2

@njit(nogil = True)
def point_prep(i, j, points, p1, p2, x1, x2):
    
    y1 = j % 3
    y2 = j // 3
    
    kron = (((j - i) % 3 == 0) and (j != i))
    
    term1 = ((p2 - points[y2][x1]) *
             (p1[y1] - points[y2][y1]) *
             3)
     
    euc = la.norm(p1 - points[y2]) if x2 != y2 else 1 
    
    dip_dip = (term1 - kron * euc * euc) / (euc ** 5)
    
    return dip_dip


for i in tqdm(range(0, 3 * tot_atoms)):
    
    x1, x2, p1, p2 = col_prep(i, points)
    
    for j in range(0, i):
        
        relation[i, j] = point_prep(i, 
                                    j,
                                    points.astype(float), 
                                    p1.astype(float), 
                                    p2, x1, x2)

print("\n")
#print(relation.data.nbytes/(1024*1024*1024))
relation = relation + relation.transpose()
print(relation.data.nbytes/(1024*1024*1024))

    
alphac = 1/(min(sa.eigsh(relation, return_eigenvectors=(False))))

end = time.perf_counter()
print("alpha:", alphac)
print("time - h:", (end - start)/(60*60))
print("time - m:", (end - start)/(60))
print("time - s:", (end - start))
print("array size:", relation.data.nbytes/(1024*1024*1024))

#%% sparse matrix v2

a, b, c = 2, 2, 2
lat_res = 10

start = time.perf_counter()
#calculating total amount of atoms for the SC
tot_atoms = (lat_res ** 3)

relation = sp.lil_matrix(np.zeros((3, 3)))
relation.resize(3 * tot_atoms, 3 * tot_atoms)

#Assigning the coordinates of the SC atoms (present in all lattices)
points = np.array([[i * a, j * b, k * c] 
                   for k in range(lat_res) 
                   for j in range(lat_res) 
                   for i in range(lat_res)])

@njit(nogil = True)
def dipole_dipole(tot_atoms, points, i, j):
    
     x1 = i % 3
     x2 = i // 3
         
     p1 = points[x2]
     p2 = points[x2][x1]
     
     y1 = j % 3
     y2 = j // 3
     
     kron = (((j - i) % 3 == 0) and (j != i))
     
     term1 = ((p2 - points[y2][x1]) *
              (p1[y1] - points[y2][y1]) *
              3)
      
     euc = la.norm(p1 - points[y2]) if x2 != y2 else 1
     
     return (term1 - kron * euc * euc) / (euc ** 5)
    

for i in tqdm(range(0, 3 * tot_atoms)):
    
    for j in range(0, i):
        
        relation[i, j] = dipole_dipole(tot_atoms, points.astype(float), i, j)
    

print("\n")
#print(relation.data.nbytes/(1024*1024*1024))
relation = relation + relation.transpose()
print(relation.data.nbytes/(1024*1024*1024))

alphac = 1/(min(sa.eigsh(relation, return_eigenvectors=(False))))

end = time.perf_counter()
print("alpha:", alphac)
print("time - h:", (end - start)/(60*60))
print("time - m:", (end - start)/(60))
print("time - s:", (end - start))
print("array size:", relation.data.nbytes/(1024*1024*1024))

#%% sparse matrix v3 (slightly more mem use)

a, b, c = 2, 2, 2
lat_res = 10

start = time.perf_counter()
#calculating total amount of atoms for the SC
tot_atoms = (lat_res ** 3)


#relation = np.zeros((3 * tot_atoms, 3 * tot_atoms))

relation = sp.lil_matrix(np.zeros((3, 3)))
relation.resize(3 * tot_atoms, 3 * tot_atoms)

#Assigning the coordinates of the SC atoms (present in all lattices)
points = np.array([[i * a, j * b, k * c] 
                   for k in range(lat_res) 
                   for j in range(lat_res) 
                   for i in range(lat_res)])

@njit
def prep_row(i, points, tot_atoms):
    
    x1 = i % 3
    x2 = i // 3
        
    p1 = points[x2]
    p2 = points[x2][x1]

    y1 = np.repeat(np.arange(0, tot_atoms, 1), 3)[0:i:1]

    kron = np.zeros((1,i))
    kron[:,x1::3] = 1

    return x1, x2, p1, p2, y1, kron

@njit
def create_row(i, p1, p2, p3, p4, y1, y2, x2, kron, points):

    term1 = (p2 - p3) * (p1[y2] - p4) * 3

    euc = np.zeros(i)

    for j in range(i):
        
        if x2 != y1[j]:
            euc[j] = la.norm(p1 - points[y1][j])
        else:
            euc[j] = 1

    dip_dip = (term1 - kron * euc ** 2) / (euc ** 5)
    
    return dip_dip


for i in tqdm(range(1, 3 * tot_atoms)):
    
    x1, x2, p1, p2, y1, kron = prep_row(i, points, tot_atoms)
        
    y2 = np.tile([0, 1, 2], tot_atoms)[0:i:1]

    p3 = points[y1, x1]
    p4 = points[y1, y2]

    relation[i,:i] = create_row(i, p1.astype(float), p2, p3, p4, y1, y2, x2, kron, points)
        
    

relation = relation + relation.transpose()

alphac = 1/(min(sa.eigsh(relation, return_eigenvectors=(False))))

end = time.perf_counter()
print("\n","alpha:", alphac)
print("time - h:", (end - start)/(60*60))
print("time - m:", (end - start)/(60))
print("time - s:", (end - start))
print("array size:", relation.data.nbytes/(1024*1024*1024))