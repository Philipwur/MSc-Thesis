#%%
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 13:53:38 2022

@author: Philipwur
"""

import os

os.environ["MKL_NUM_THREADS"] = "8"

import time
import numpy as np

import numpy.linalg as la

import scipy.sparse as sp
import scipy.linalg as sa

from numba import njit

import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go

#%% no sparse matrix

lat_type = "SC"  #choice between SC, FCC and BCC, anything other than FCC or BCC is assumed to be SC
lat_res = 10 #lattice resolution


@njit()
def dipole_dipole(lat_type, lat_res):
    
    tot_atoms = (lat_res ** 3)
    
    #preallocation
    relation = np.zeros((3 * tot_atoms, 3 * tot_atoms))
    
    #Assigning the coordinates of the SC atoms (present in all lattices)
    #this can be sped up but doesnt take much time in the grand scheme of things
    points = np.array([[i, j, k] 
                       for k in range(lat_res) 
                       for j in range(lat_res) 
                       for i in range(lat_res)]).astype(np.float64)
    
    if lat_type == "FCC":
        extra_points = points[1:,1:,1:] + 0.5
    
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


def main(a, lat_res):
    
    relation = dipole_dipole(a, lat_res)
    
    #print("array size:", relation.data.nbytes/(1024*1024*1024))
    
    alpha = find_alpha(relation)
    
    return alpha


if __name__ == "__main__":
    
    
    _ = main(2, 5) #warmup function to get the main function compiled
    
    del _
    
    start = time.perf_counter()
    
    alpha = main(a, lat_res) #actual high resolution simulation
    
    end = time.perf_counter()
    
    #stats
    print("alpha:", alpha)
    print("time - h:", (end - start)/(60*60))
    print("time - m:", (end - start)/(60))
    print("time - s:", (end - start))


#%%

lat_type = "FCC"  #"FCC" and "BCC", anything other than FCC or BCC is assumed to be SC
lat_res = 3 #lattice resolution
    


#Assigning the coordinates of the SC atoms (present in all lattices)
#this can be sped up but doesnt take much time in the grand scheme of things
points = np.array([[i, j, k] 
                    for k in range(lat_res) 
                    for j in range(lat_res) 
                    for i in range(lat_res)]).astype(np.float64)

tot_atoms = len(points)

#preallocation
relation = np.zeros((3 * tot_atoms, 3 * tot_atoms))

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
    
    del extra_points
    
    new_size = len(points) * 3
    
    np.resize(relation, [new_size, new_size])
    
    del new_size
    
if lat_type == "BCC":

    extra_points = np.array([[i + 0.5, j + 0.5, k + 0.5] 
                             for k in range(lat_res - 1) 
                             for j in range(lat_res - 1) 
                             for i in range(lat_res - 1)]).astype(np.float64)
    
    points = np.concatenate((points, extra_points))
    
    del extra_points
    
    new_size = len(points) * 3
    
    np.resize(relation, [new_size, new_size])
    
    del new_size
    
    
plotly.offline.init_notebook_mode()

trace = go.Scatter3d(
    x=points[:,0],  
    y=points[:,1],
    z=points[:,2],
    mode='markers',
    marker={
        'size': 10,
        'opacity': 0.8,
    }
)

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)

# Render the plot.
plotly.offline.iplot(plot_figure)

#%% compare numpy, numpy + numba and scipy times


def alpha1(relation):
    
    relation_eig = la.eigvalsh(relation, UPLO = "L")
    
    alpha = 1 / relation_eig[0]
    
    print(alpha)
    
    return alpha


@njit()
def alpha2(relation):
    
    #relation = relation + relation.T
    
    relation_eig = la.eigvalsh(relation)
    
    alpha = 1 / relation_eig[0]
    print(alpha)
    return alpha


def alpha3(relation):
    
    relation_eig = sa.eigh(relation,
                           eigvals_only = True,
                           overwrite_a = True,
                           overwrite_b = True,
                           check_finite = False,
                           driver = "evd"
                           )
    
    alpha = 1/relation_eig[0]
    print(alpha)

    return alpha


test_relation = dipole_dipole(2, 15)

tic = time.perf_counter()
_ = alpha1(test_relation)
toc = time.perf_counter()

print("alpha1:",toc-tic)

tic = time.perf_counter()
_ = alpha2(test_relation)
toc = time.perf_counter()

print("alpha2:",toc-tic)

tic = time.perf_counter()
_ = alpha3(test_relation)
toc = time.perf_counter()

print("alpha3:",toc-tic)

#%% sparse matrix

#try copying above structure but with object mode for final appending action

a, b, c = 2, 2, 2
lat_res = 10

start = time.perf_counter()
#calculating total amount of atoms for the SC
tot_atoms = (lat_res ** 3)


relation = np.zeros((3 * tot_atoms, 3 * tot_atoms))
#relation = sp.lil_matrix(np.zeros((3, 3)))
#relation.resize(3 * tot_atoms, 3 * tot_atoms)

#Assigning the coordinates of the SC atoms (present in all lattices)
points = np.array([[i * a, j * b, k * c] 
                   for k in range(lat_res) 
                   for j in range(lat_res) 
                   for i in range(lat_res)])

@njit()
def col_prep(i, points):
    
    x1 = i % 3
    x2 = i // 3
        
    p1 = points[x2]
    p2 = points[x2][x1]
    
    return x1, x2, p1, p2

@njit()
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

relation = np.empty((3 * tot_atoms, 3 * tot_atoms))

#relation = sp.lil_matrix(np.zeros((3, 3)))
#relation.resize(3 * tot_atoms, 3 * tot_atoms)

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


relation = np.zeros((3 * tot_atoms, 3 * tot_atoms))

#relation = sp.lil_matrix(np.zeros((3, 3)))
#relation.resize(3 * tot_atoms, 3 * tot_atoms)

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


