#%%
import sys
sys.path.append("..")

import math
import numpy as np
from pandas import DataFrame as df
from scipy.stats import linregress

#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import colors
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter

import Dipole_Dipole_2D_1D as fun 

#hyperparameters

steps = 64

start_x = 0.2
end_x = 0.8
start_y = 0.25
end_y = 1.2


v1 = (1, 0)
res = [19, 21]

#%%
def objective_function(v1, v2):
    
    x = (1/res[0], 1/res[1])
    y = (fun.run_sim(v1, v2, res[0])[0], fun.run_sim(v1, v2, res[1])[0])

    alpha = linregress(x, y).intercept
    
    # objective density function
    nc = np.divide((abs(alpha) ** (2/3)), (v2[1]))
    return nc

x = np.arange(start_x, end_x, (end_x-start_x)/steps)
y = np.arange(start_y, end_y, (end_y-start_y)/steps)
xx, yy = np.round(np.meshgrid(x, y), 5)

z = np.empty((len(xx), len(xx)))

for i in range(len(xx)):
    for j in range(len(xx)):
        v2 = (xx[i][j], yy[i][j])
        z[i][j] = objective_function(v1, v2)

#%%

#simple version (no extrapolation)

def objective_function(v1, v2):

    alpha = fun.run_sim(v1, v2, 24)[0]
    nc = np.divide(alpha, (v2[1]))
    
    return nc

x = np.arange(start_x, end_x, (end_x-start_x)/steps)
y = np.arange(start_y, end_y, (end_y-start_y)/steps)
xx, yy = np.round(np.meshgrid(x, y), 5)

z = np.empty((len(xx), len(xx)))

for i in range(len(xx)):
    for j in range(len(xx)):
        v2 = (xx[i][j], yy[i][j])
        z[i][j] = objective_function(v1, v2)
        
#%% ac graphing

'''
fig = plt.figure(dpi = 300)
ax = fig.gca(projection="3d")
surf = ax.plot_surface(xx, yy, z, cmap = "viridis")
ax.zaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel(r"$\alpha_{c-}^{'}$")
ax.zaxis.labelpad = 5
cbar = fig.colorbar(surf, pad = 0.15)
plt.suptitle(r"$\alpha_{c-}^{'}$ with x and y vector components", y = 1)
plt.title("Resolution = 64",  fontsize = 10, x = 0.67, y = 1.1)
ax.view_init(-130, 45)
plt.show()
'''


fig2, ax = plt.subplots(dpi = 300)
clev = np.arange(z.min(), z.max(),.001)
cp = ax.contourf(xx, yy, z, clev, extend = 'both')
cbar2 = fig2.colorbar(cp)
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.suptitle(r"$\alpha_{c-}^{'}$ with x and y vector components", y = 1.00)
plt.title("Resolution = 32",  fontsize = 10, x = 0.60)
cbar2.ax.get_yaxis().labelpad = 10
cbar2.ax.set_ylabel(r"$\alpha_{c-}^{'}$")
plt.show()

#%% Nc graphing

'''
fig = plt.figure(dpi = 300)
ax = fig.gca(projection="3d")
surf = ax.plot_surface(xx, yy, z, cmap = "viridis")
ax.zaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel(r"$N_{c-}^{'}$")
ax.zaxis.labelpad = 5
cbar = fig.colorbar(surf, pad = 0.15)
plt.suptitle(r"$N_{c-}^{'}$ with x and y vector components", y = 1)
plt.title("Resolution = 64",  fontsize = 10, x = 0.67, y = 1.1)
ax.view_init(-130, 45)
plt.show()
'''

fig2, ax = plt.subplots(dpi = 300)
clev = np.arange(z.min(), z.max(),.004)
cp = ax.contourf(xx, yy, z, clev, extend = 'both', cmap = "viridis")
cbar2 = fig2.colorbar(cp)
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.suptitle(r"$N_{c-}^{'}$ with x and y vector components", y = 1.00)
plt.title("Resolution = 64",  fontsize = 10, x = 0.60)
cbar2.ax.get_yaxis().labelpad = 10
cbar2.ax.set_ylabel(r"$N_{c-}^{'}$")
plt.show()

#%%
steps = 64
x1 = np.arange(1, 10, 9/64)
y1 = np.arange(math.pi/2, math.pi, math.pi/128)
xx1, yy1 = np.round(np.meshgrid(x, y), 5)
z1 = xx1 * np.sin(yy1)

fig3, ax = plt.subplots(dpi = 300)
cp = ax.contourf(xx1, np.degrees(yy1), z1, vmin=1)
cbar3 = fig3.colorbar(cp)
ax.set_xlabel("b'")
ax.set_ylabel(r"$\theta$")
plt.suptitle(r"x values with b' and $\theta$", y = 1.00)
plt.title("Resolution = 64",  fontsize = 10, x = 0.60)
cbar3.ax.get_yaxis().labelpad = 10
cbar3.ax.set_ylabel(r"x")
plt.show()

#%% for showing where the boundary lies between what can be used and not 

steps = 64
x1 = np.arange(1, 10, 9/64)
y1 = np.arange(math.pi/2, math.pi, math.pi/128)
xx1, yy1 = np.round(np.meshgrid(x, y), 5)
z1 = xx1 * np.sin(yy1)
fig, ax2 = plt.subplots(dpi = 300)
cmap = colors.ListedColormap(['r','g','b'])
bounds = [0, 0.4,0.6, 1.1]
norm = colors.BoundaryNorm(bounds, cmap.N)

ax2.imshow(data, interpolation = 'none', cmap=cmap, norm=norm)
ax2.set_title('imshow')
                           
#%%

points = np.column_stack([xx.flatten(), yy.flatten(), z.flatten()])
points2 = df(points, index = None)
points2 = points2.loc[points2[0] == 0.55]
points2[3] = 1/points2[2]


fig = plt.figure(dpi = 300)
plt.plot(points2[1], points2[2])
plt.show()