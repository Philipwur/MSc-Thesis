#%%
import math
import numpy as np
import numpy.linalg as la
from scipy.spatial import distance_matrix
import time

start_time = time.perf_counter()
#import my_functions as fun

"""
develop code 1D-2D for lattice for 1D polarizability (square lattice) (term 1 wouldnt be there, mostly term)
relation matrix will be NxN rather than 3N*3N
then go to hexagonal structure

needs a significant rewrite
look into the numpy function memmap for the generation of large datapoints 
or sparse arrays
"""

a, b, c = 2, 2, 2
Nx, Ny, Nz = 3, 3, 3

#a, b, c = fun.spacing_choice()
#Nx, Ny, Nz = fun.resolution_choice()

#calculating total amount of atoms for the SC
tot_atoms = (Nx * Ny * Nz)


#setting up zero arrays 
relation = np.zeros((3*tot_atoms, 3*tot_atoms))
E0 = np.zeros((3 * tot_atoms, 1)) #electric field array
p = np.zeros((3 * tot_atoms, 1)) #dipole moment list

#Assigning the coordinates of the SC atoms (present in all lattices)
points = np.array([[i * a, j * b, k * c] 
                   for k in range(Nz) 
                   for j in range(Ny) 
                   for i in range(Nx)])

t1 = time.perf_counter()
print("points created:", round(t1-start_time,5))
#Calculating Eucledian distances between each point
euc = distance_matrix(points, points) + np.identity(tot_atoms)
t2 = time.perf_counter()
print("distance matrix:", round(t2-t1,5))

#Calculating dipole-dipole relation matrix
for i in range(0, 3 * tot_atoms):
    
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
    
        relation[i][j] = (np.divide(np.subtract(term1, term2), 
                                        (euc[math.floor(i/3)][math.floor(j/3)]) ** 5))
    
        if np.mod(j1, 3) == 0:
            count += 1

t3 = time.perf_counter()
print("dipole matrix:", round(t3-t2,5))
#finding the extremes of the dipole-relation eigenvalues
relation_eig = la.eigvalsh(relation)
extreme_eig = [min(relation_eig), max(relation_eig)]

#transforming this into the alpha values
#the minimum alpha is the objective function
extreme_a = [1/extreme_eig[0], 1/extreme_eig[1]]

fig_o_merit = extreme_a[0]
t4 = time.perf_counter()
print("eigenvalue:", round(t4-t3,5))
end_time = time.perf_counter()
print("total:", round(end_time-start_time, 5))

print("alpha:", fig_o_merit)