#%%

import scipy.optimize as opt
import Dipole_Dipole_2D_1D as fun
import scipy.stats as stat
import matplotlib.pyplot as plt
import numpy as np


#%%

#initialise reults array
results = []

#the objective function, finds 2 points and then takes linear regression to y-axis
#it then appends the coordinates, result and optimiser iteration to results array
def OF(v2):
    
    res = [19, 21]
    v1 = (1, 0)
    
    x = (1 / res[0], 
         1 / res[1])
    
    y = (fun.run_sim(v1, v2, res[0])[0], 
         fun.run_sim(v1, v2, res[1])[0])

    alpha = stat.linregress(x, y).intercept
    
    nc = -1 *  np.divide((abs(alpha) ** (2/3)), (v2[1]))
    
    results.append((v2[0], v2[1], nc, len(results)))
    
    return nc

#bounds and initial guess
bnds = ((0, 1), (0.1, 1.5))
x0 = (0.25, 0.5)




res = opt.differential_evolution(OF,
                                 bounds = bnds,
                                 popsize = 15,
                                 polish = True,
                                 init = 'latinhypercube',
                                 )


'''
definitions of the optimisers

res = opt.minimize(OF, x0, method = "Nelder-Mead", bounds = bnds, 
                   options = {'initial_simplex': [[0, 0.1],[1, 0.1],[0.5, 1]]})


res = opt.differential_evolution(OF,
                                 bounds = bnds,
                                 popsize = 15,
                                 polish = True,
                                 init = 'latinhypercube',
                                 )

'''

res

#%%

#plotting
results = np.array(results)
plt.figure(dpi = 300)
plt.scatter(x = results[:,0], y = results[:,1], c = results[:,3], cmap = "viridis")
plt.suptitle("Differential Evolution Points Evaluated")
plt.title("Optimum found: ((0.4924355, 0.87035905), OF: -0.23143850298970164)", fontsize = 8)
plt.xlabel(r"$v2_{x}^{'}$")
plt.ylabel(r"$v2_{y}^{'}$")
plt.colorbar(label = "Iteration of Point")


# %%
