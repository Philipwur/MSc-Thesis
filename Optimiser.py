#%%

import scipy.optimize as opt
import Dipole_Dipole_2D_1D as fun
import scipy.stats as stat
import matplotlib.pyplot as plt

import numpy as np

from skopt import gp_minimize


#%%
results = []

def OF(v2):
    
    res = [19, 21]
    
    v1 = (1, 0)
    
    x = (1 / res[0], 
         1 / res[1])
    
    y = (fun.run_sim(v1, v2, res[0])[0], 
         fun.run_sim(v1, v2, res[1])[0])

    alpha = stat.linregress(x, y).intercept
    
    nc = -1 *  np.divide((abs(alpha) ** (2/3)), (v2[1]))
    
    results.append((v2[0], v2[1], nc))
    
    return nc

bnds = ((0, 1), (0.1, 1.5))
x0 = (0.25, 0.5)


res = opt.minimize(OF, x0, method = "SLSQP", bounds = bnds)



'''
res = opt.minimize(OF, x0, method = "Nelder-Mead", bounds = bnds)
res = opt.minimize(OF, x0, method = "SLSQP", bounds = bnds)


res = gp_minimize(OF, bnds)

res = opt.differential_evolution(OF,
               bounds = bnds,
               #iters = 4,
               #minimizer_kwargs = {},
               #options = {'minimize_every_iter': True}
               )

res = opt.shgo(OF,
               bounds = bnds,
               #iters = 4,
               #minimizer_kwargs = {},
               #options = {'minimize_every_iter': True}
               )
'''

res

#%%
results = np.array(results)
plt.figure(dpi = 300)
plt.scatter(x = results[:,0], y = results[:,1])
plt.suptitle("SLSQP")
plt.title("Optimum found: ((0.5, 0.87), -0.23)", fontsize = 8)

