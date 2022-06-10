# -*- coding: utf-8 -*-
#%%
import time
import math
import sys
import numpy as np

import scipy.sparse as sp
import scipy.linalg as la
from scipy.spatial import distance_matrix 
#import itertools
from tqdm import tqdm

import dask.array as da

#%% slowest

a, b, c = 2, 2, 2
Nx, Ny, Nz = 5, 5, 5

#calculating total amount of atoms for the SC
tot_atoms = (Nx * Ny * Nz)
relation = np.zeros((3*tot_atoms, 3*tot_atoms))
test = np.zeros((3*tot_atoms, 3*tot_atoms))

#Assigning the coordinates of the SC atoms (present in all lattices)
points = np.array([[i * a, j * b, k * c] 
                   for k in range(Nz) 
                   for j in range(Ny) 
                   for i in range(Nx)])

#Calculating Eucledian distances between each point
euc = distance_matrix(points, points) + np.identity(tot_atoms)
start = time.perf_counter()
#Calculating dipole-dipole relation matrix
for i in tqdm(range(0, 3 * tot_atoms)):

    count = 0
    for j in range(0, 3 * tot_atoms):
        
        i1, j1 = i + 1, j + 1
        
        #kroneckar delta condition test
        if i != j and abs(i1 - j1) % 3 == 0:
            kronec_delta = 1
        else: 
            kronec_delta = 0
        
        x = ((i1 % 3 == 0) * 3 +
             (i1 % 3 != 0) * (i1 % 3)
             ) - 1
        
        y = ((j1 % 3 == 0) * 3 +
             (j1 % 3 != 0) * (j1 % 3)
             ) - 1
        
        term1 = ((points[math.floor(i/3)][x] - points[count][x]) *
                 (points[math.floor(i/3)][y] - points[count][y]) *
                 3)
        
        term2 = kronec_delta * euc[math.floor(i/3)][math.floor(j/3)] ** 2
        
        test[i][j] = euc[math.floor(i/3)][math.floor(j/3)]
        
        relation[i][j] = (np.divide(np.subtract(term1, term2), 
                                        (euc[math.floor(i/3)][math.floor(j/3)]) ** 5))
    
        if np.mod(j1, 3) == 0:
            count += 1

end = time.perf_counter()

print(end - start)
relation_eig = la.eigvalsh(relation)
extreme_eig = [min(relation_eig), max(relation_eig)]

#transforming this into the alpha values
#the minimum alpha is the objective function
extreme_a = [1/extreme_eig[0], 1/extreme_eig[1]]

fig_o_merit = extreme_a[0]

print(fig_o_merit)


#%% middle ground

a, b, c = 2, 2, 2
lat_res = 5

tot_atoms = (lat_res ** 3)


points = np.array([[i * a, j * b, k * c] 
                   for k in range(lat_res) 
                   for j in range(lat_res) 
                   for i in range(lat_res)])

euc = distance_matrix(points, points) + np.identity(tot_atoms)

term1 = np.zeros((3*tot_atoms, 3*tot_atoms))
test = np.zeros((3*tot_atoms, 3*tot_atoms))
start = time.perf_counter()

for i in tqdm(range(0, 3 * tot_atoms)):
    
    x1 = i % 3
    x2 = i // 3
    
    for j in range(0, 3 * tot_atoms):
        
        y1 = j % 3
        y2 = j // 3
        
        test [i, j] = points[x2][y1] - points[y2][y1]
        #- points[y2][y1]
        term1[i, j] = ((points[x2][x1] - points[y2][x1]) *
                       (points[x2][y1] - points[y2][y1]) *
                       3)
        
        
# creation of the kronekar delta matrix
x = np.arange(3 * tot_atoms)
kron = (abs(np.meshgrid(x, x)[0] - np.meshgrid(x, x)[1]) % 3 == 0).astype("b")
kron = kron - np.identity(len(kron), dtype = 'b')

# repearing euclidian matrix 3 times in both dimensions
# any way of doing this without storing it in memory, ie just pointing?
# look into np.where for this
new_euc = np.repeat(np.repeat(euc, [3], axis = 0), 3, axis = 1)
    
# matrix-wise calculation of term 2 and term 3
term2 = np.power(np.multiply(new_euc, kron), 2)
term3 = np.power(new_euc, 5)


relation = np.divide((term1 - term2), term3)

relation_eig = la.eigvalsh(relation)
extreme_eig = [min(relation_eig), max(relation_eig)]
extreme_a = [1/extreme_eig[0], 1/extreme_eig[1]]

fig_o_merit = extreme_a[0]

end = time.perf_counter()
print(end - start)
print(fig_o_merit)

#%% optimised for speed

a, b, c = 2, 2, 2
lat_res = 15

start = time.perf_counter()

tot_atoms = (lat_res ** 3)
relation = np.zeros((3*tot_atoms, 3*tot_atoms))

points = np.array([[i * a, j * b, k * c] 
                   for k in range(lat_res) 
                   for j in range(lat_res) 
                   for i in range(lat_res)])

euc = distance_matrix(points, points) + np.identity(tot_atoms)


# function to show only kronecker delta points, alternative method would be to 
# use and displace eye functions, although this may require iteration
def term1(tot_atoms, points):
    
    t1 = np.repeat(np.arange(0, tot_atoms, 1), 3).astype(np.int16)
    t2 = np.tile([0, 1, 2], tot_atoms).astype(np.int8)
    
    p4 = np.tile(t2, (3 * tot_atoms, 1))
    p3 = np.tile(t1, (3 * tot_atoms, 1))
    p1 = p3.T
    p2 = p4.T
    
    term1 = (3
             * (points[p1, p2] - points[p3, p2])
             * (points[p1, p4] - points[p3, p4]))

    return term1

def term2_3(x, euc):
    
    # creation of the kronekar delta matrix
    x = np.arange(3 * x)
    kron = (abs(np.meshgrid(x, x)[0] - np.meshgrid(x, x)[1]) % 3 == 0)
    kron = kron.astype(int) - np.identity(len(kron))

    # repearing euclidian matrix 3 times in both dimensions
    # any way of doing this without storing it in memory, ie just pointing?
    # look into np.where for this
    new_euc = np.repeat(np.repeat(euc, [3], axis = 0), 3, axis = 1)
        
    # matrix-wise calculation of term 2 and term 3
    term2 = np.power(np.multiply(new_euc, kron), 2)
    term3 = np.power(new_euc, 5)
    
    return kron, term2, term3

term1 = term1(tot_atoms, points)
kron, term2, term3 = term2_3(tot_atoms, euc)

relation = np.divide((term1 - term2), term3)

relation_eig = la.eigvalsh(relation)
extreme_eig = [min(relation_eig), max(relation_eig)]
extreme_a = [1/extreme_eig[0], 1/extreme_eig[1]]
fig_o_merit = extreme_a[0]

end = time.perf_counter()
print(end - start)
print(fig_o_merit)


#%%

a, b, c = 2, 2, 2
lat_res = 10

tot_atoms = (lat_res ** 3)

relation = da.zeros((3*tot_atoms, 3*tot_atoms), chunks = (tot_atoms, tot_atoms))
relation
#%%

start = time.perf_counter()
points = np.array([[i * a, j * b, k * c] 
                   for k in range(lat_res) 
                   for j in range(lat_res) 
                   for i in range(lat_res)])

euc = distance_matrix(points, points) + np.identity(tot_atoms)


# function to show only kronecker delta points, alternative method would be to 
# use and displace eye functions, although this may require iteration
def term1(tot_atoms, points):
    
    t1 = np.repeat(np.arange(0, tot_atoms, 1), 3).astype(np.int16)
    t2 = np.tile([0, 1, 2], tot_atoms).astype(np.int8)
    
    p4 = np.tile(t2, (3 * tot_atoms, 1))
    p3 = np.tile(t1, (3 * tot_atoms, 1))
    p1 = p3.T
    p2 = p4.T
    
    term1 = (3
             * (points[p1, p2] - points[p3, p2])
             * (points[p1, p4] - points[p3, p4]))

    return term1

def term2_3(x, euc):
    
    # creation of the kronekar delta matrix
    x = np.arange(3 * x)
    kron = (abs(np.meshgrid(x, x)[0] - np.meshgrid(x, x)[1]) % 3 == 0)
    kron = kron.astype(int) - np.identity(len(kron))

    # repearing euclidian matrix 3 times in both dimensions
    # any way of doing this without storing it in memory, ie just pointing?
    # look into np.where for this
    new_euc = np.repeat(np.repeat(euc, [3], axis = 0), 3, axis = 1)
        
    # matrix-wise calculation of term 2 and term 3
    term2 = np.power(np.multiply(new_euc, kron), 2)
    term3 = np.power(new_euc, 5)
    
    return term2, term3

term1 = term1(tot_atoms, points)
term2, term3 = term2_3(tot_atoms, euc)

relation = np.divide((term1 - term2), term3)

end = time.perf_counter()
print(end - start)

relation_eig = la.eigvalsh(relation)
extreme_eig = [min(relation_eig), max(relation_eig)]
extreme_a = [1/extreme_eig[0], 1/extreme_eig[1]]
fig_o_merit = extreme_a[0]


#%% memory efficient

"""
steps
1. optimise runtime 
2. look at sparse matrixes/memory optimisation
3. look at storing the data on drive

"""
a, b, c = 2, 2, 2
Nx, Ny, Nz = 10, 10, 10

#calculating total amount of atoms for the SC
tot_atoms = (Nx * Ny * Nz)
relation = np.zeros((3*tot_atoms, 3*tot_atoms))

#Assigning the coordinates of the SC atoms (present in all lattices)
points = np.array([[i * a, j * b, k * c] 
                   for k in range(Nz) 
                   for j in range(Ny) 
                   for i in range(Nx)])

#Calculating Eucledian distances between each point
euc = distance_matrix(points, points) + np.identity(tot_atoms)
start = time.perf_counter()

#creating array of kronekar delta
x = np.arange(3 * tot_atoms)
kron = (abs(np.meshgrid(x, x)[0] - np.meshgrid(x, x)[1]) % 3 == 0).astype("bool")

"""
kron = np.fill_diagonal(kron, False)

for i in tqdm(range(0, 3 * tot_atoms)):

    x1 = i % 3
    x2 = i // 3
    
    for j in range(0, 3 * tot_atoms):
        
        y1 = j % 3
        y2 = j // 3
        
        term1 = ((points[x2][x1] - points[y2][x1]) *
                 (points[x2][y1] - points[y2][y1]) *
                 3)
        
        term2 = kron[i][j] * euc[x2][y2] ** 2
        term3 = euc[x2][y2] ** 5
        
        relation[i][j] = (np.divide((term1 - term2), term3))
    

end = time.perf_counter()

print(end - start)
relation_eig = la.eigvalsh(relation)
extreme_eig = [min(relation_eig), max(relation_eig)]

#transforming this into the alpha values
#the minimum alpha is the objective function
extreme_a = [1/extreme_eig[0], 1/extreme_eig[1]]

fig_o_merit = extreme_a[0]

print(fig_o_merit)
"""