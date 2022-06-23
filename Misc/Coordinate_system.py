#%%
"""
Showing off how pog the the coordinate generation is
3 types of generation include modulo (box), slant, and vector slant
"""

import math
import numpy as np
from timeit import default_timer as timer

import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import seaborn as sns

import plotly
import plotly.graph_objs as go

#lattice spacing between planes
b_dash = 1

#angle between planes
theta = math.pi/3

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

start = timer()
points = generate_lattice(Nx, Ny, b_dash, theta, tot_atoms)
end = timer()
print(end- start)


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

start = timer()
points = generate_lattice(Nx, b_dash, theta, tot_atoms)
end = timer()
print(end- start)
#%% vector slant

v1 = (1, 0)
v2 = (1.5, 1)

def vector_gen(v1, v2, Nx):
    proto_x = np.arange(0, v1[0]*Nx, v1[0])
    proto_y = np.arange(0, v2[1]*Nx, v2[1])

    xm, ym = np.meshgrid(proto_x, proto_y)
    
    try:
        xm = xm + np.arange(0, v2[0]*Nx, v2[0])[:,None]
    
    finally:
        points = np.column_stack([xm.flatten(), ym.flatten()]) 
    
    return points


points = vector_gen(v1, v2, Nx)
euc = distance_matrix(points, points)
print(np.average(euc))


#%% 3D 3D pointgen

lat_type = "SC"  #"FCC" and "BCC", anything other than FCC or BCC is assumed to be SC
lat_res = 2 #lattice resolution
    


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