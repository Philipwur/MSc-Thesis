#%%
"""
Showing off how pog the the coordinate generation is
"""

import math
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#lattice spacing between planes
b_dash = 1

#angle between planes
theta = math.pi/2.5

#lattice resolution
Nx, Ny = 17, 17

tot_atoms = Nx * Ny

#%% mod generation

#initialises entire rows of the lattice at once
def single_row(b_dash, theta, j, Nx):
    
    x_term = round((b_dash * np.cos(theta) * j) % 1, 5)
    y_term = j * (round(b_dash * np.sin(theta), 5))
    
    x_row = np.arange(x_term, Nx * 1, 1)
    y_row = np.repeat(y_term, Nx)
    
    return np.column_stack((x_row, y_row))


#Generates the lattice by appending entire layers of the lattice at once, then calculates distance matrix
def generate_lattice(Nx, Ny, b_dash, theta, tot_atoms):
    
    points = np.empty([0, 2])
    
    for j in range(Ny):
        points = np.append(points, single_row(b_dash, theta, j, Nx),axis=0)
       
    return points

points = generate_lattice(Nx, Ny, b_dash, theta, tot_atoms)

#%% slanted generation

#initialises entire rows of the lattice at once
def single_row(b_dash, theta, j, lat_size):
    
    x_term = round((b_dash * np.cos(theta) * j), 5)
    y_term = j * (round(b_dash * np.sin(theta), 5))
    
    row = [(x_term + i, y_term) for i in range(lat_size)]
    
    return row

#Generates the lattice by appending entire layers of the lattice at once, then calculates distance matrix
def generate_lattice(lat_size, b_dash, theta, tot_atoms):
    
    points = np.empty([0, 2])
    
    #appending x row along y axis
    for j in range(lat_size):
        points = np.append(points, 
                           single_row(b_dash, theta, j, lat_size), 
                           axis = 0
                           )

    return points

points = generate_lattice(Nx, b_dash, theta, tot_atoms)

#%% plotting coordinates for inspection

plt.figure(dpi = 300)
g1 = sns.scatterplot(x = points[:,0],
                     y = points[:,1],
                     s = 10)
plt.suptitle("Full Lattice", 
             y= 1)

plt.title((r"b' : {},   $\theta$ : {}°,   N = {}".format(b_dash, 
                                                         math.degrees(theta), 
                                                         Nx)), 
          fontsize = 12)

plt.xlabel("x")
plt.ylabel("y")

plt.figure(dpi = 300)
g2 = sns.scatterplot(x = points[:,0], 
                     y = points[:,1])

g2.set(xlim = (-0.5, 3.5), 
       ylim = (-0.5, 5.5))

plt.suptitle("Zoomed in Lattice", 
          y = 1)

plt.title((r"b' : {},   $\theta$ : {}°,   N = {}".format(b_dash, 
                                                         math.degrees(theta), 
                                                         Nx)), 
          fontsize = 12)

plt.xlabel("x")
plt.ylabel("y")

plt.show