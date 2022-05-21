# -*- coding: utf-8 -*-
import time
import math
import numpy as np
import scipy.linalg as la
from scipy.spatial import distance_matrix 

#import my_functions as fun

"""
my testing area - idek what i put here lmao
"""


#%%

#variables defined here

#lattice spacing
b_dash = 1

#angle between planes
theta = math.pi/2

#lattice resolution
Nx, Ny = 27, 27


#functions defined here

#defines where the coords of the first point on each plane is and then adds i to coords
def coords(i, j, b_dash, theta):
    
    x = round(((b_dash * np.cos(theta) * j) % 1), 5) + i
    y = j * (round(b_dash * np.sin(theta), 5))

    return x, y


#Generates an array with lattice points and an array with the distances between them
def generate_lattice(Nx, Ny, b_dash, theta, tot_atoms):
    
    points = np.empty((tot_atoms, 2))
    count = 0
    
    points = np.array([[]])
    
    for j in range(Ny):
        for i in range(Nx):
            points[count] = coords(i,j, b_dash, theta)
            count += 1
    
    dist = distance_matrix(points, points) + np.identity(tot_atoms)
    
    return (points, dist)



def generate_lattice2(sz_x, sz_y):
  """Generates a lattice with sz_x vertices along x and sz_y vertices along y
  direction Each of these vertices is step_size distance apart. Origin is at
  (0,0).  """
  
  x, y = np.meshgrid(np.arange(sz_x), np.arange(sz_y))
  x = np.reshape(x, [-1,1]) 
  y = np.reshape(y, [-1,1])
  nodes = np.concatenate((x,y), axis=1)
  return nodes 

nodes = generate_lattice2((5),(5))

#%%

import numpy
a = numpy.array([(1, 2), (2, 3)])
b = numpy.array([(5,5), (6,6)])
c = numpy.concatenate((a, b), axis = 0)

print(c)

#%%

#lattice spacing
b_dash = 1

#angle between planes
theta = math.pi/2

#lattice resolution
Nx, Ny = 100, 100

tot_atoms = (Nx * Ny)

def single_row(b_dash, theta, j, Nx):
    
    term1 = round((b_dash * np.cos(theta) * j) % 1, 5)
    term2 = j * (round(b_dash * np.sin(theta), 5))
    
    row1 = np.arange(term1, Nx * 1, 1)
    row2 = np.repeat(term2, Nx)
    
    return np.column_stack((row1, row2))


def generate_lattice2(Nx, Ny, b_dash, theta, tot_atoms):
    
    points = np.empty([0, 2])
    
    for j in range(Ny):
        points = np.append(points, single_row(b_dash, theta, j, Nx),axis=0)
    
    dist = distance_matrix(points, points) + np.identity(tot_atoms)
    
    return points, dist


#print(row)
points, dist = generate_lattice2(Nx, Ny, b_dash, theta, tot_atoms)