# -*- coding: utf-8 -*-
# for testing the runtime and extrapolation in the 2D dipole-dipole function

#%%

import pandas as pd
import math
import numpy as np
import scipy.stats as stat

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

import Dipole_Dipole_2D_1D_wSolver as sim

#Hyperparameters
#==============================================================================
b_dash = 2
divisor = 4
theta = math.pi/divisor

start = 5 
end = 125
step = 2

#select whether you want to compare runtimes of solvers
test_solvers = False 
# =============================================================================

#test runs of Dipole-Dipole 2D-1D script with 1 or multiple solvers
def generate_data(b_dash, theta, test_solvers, start, end, step):
       
       count = 0
       
       if test_solvers:
              
              solvers = ["ev", "evr", "evd", "evx"]
              
              for i in range(start, end, step):
                     data = np.zeros((int((end-start)/step), 6))
                     data[count][0] = i
                     
                     alpha, runtime = sim.run_sim(b_dash, theta, i, solvers[0])
                     
                     data[count][1] = alpha
                     data[count][2] = runtime
                     
                     for j in range(1, 4):
                            alpha, runtime = sim.run_sim(b_dash, theta, i, solvers[j])
                            data[count][j+2] = runtime
                     
                     count += 1
                     
              df = pd.DataFrame(data = data)

              df.columns = ["Resolution", "Alpha", 
                            "Runtime(s) ev", "Runtime(s) evr", 
                            "Runtime(s) evd", "Runtime(s) evx"]
              
       else:
              for i in range(start, end, step):
                     data = np.zeros((int((end-start)/step), 3))
                     data[count][0] = i
                     
                     alpha, runtime = sim.run_sim(b_dash, theta, i, "evd")
                     data[count][1] = alpha
                     data[count][2] = runtime
                     
                     count += 1
                     
              df = pd.DataFrame(data = data)

              df.columns = ["Resolution", "Alpha", "Runtime(s) evd"]
                     

       df["Resolution_inv"] = np.power(df["Resolution"], -1)

       #saving pandas dataframe to csv 
       df.to_csv("results/{}_pi_div_{}.csv".format(b_dash, divisor))

# tests whether data exists, and if not, generates new data 
def get_data(b_dash, theta, test_solvers, start, end, step):
       
       datafile = "{}_pi_div_{}.csv".format(b_dash, divisor)
       
       try:
              df = pd.read_csv("results/{}".format(datafile), index_col=(0))

       except:
              df = generate_data(b_dash, theta, test_solvers, start, end, step)
              
       return df
       
if __name__ == "__main__":
       df = get_data(b_dash, theta, test_solvers, start, end, step)

#%%
#plotting all results 

plt.figure(dpi = 300)
sns.set_theme()
sns.set_style("white")

g = sns.lineplot(x = df["Resolution"], 
                 y = df["Alpha"], 
                 color = "k", 
                 linewidth = 1.5)

plt.ylabel(r"$\alpha_{c-}^{'}$")

g = sns.lineplot(x = df["Resolution"], 
                 y = df["Runtime(s) ev"], 
                 color = "r", 
                 linewidth = 1,
                 ax = plt.twinx())

g = sns.lineplot(x = df["Resolution"], 
                 y = df["Runtime(s) evr"], 
                 color = "g",
                 linestyle = "--", 
                 linewidth = 1)

g = sns.lineplot(x = df["Resolution"], 
                 y = df["Runtime(s) evd"],
                 color = "b", 
                 linewidth = 1)

# =============================================================================
# g = sns.lineplot(x = df["Resolution"], 
#                  y = df["Runtime(s) evx"], 
#                  color = "y", 
#                  linewidth = 1) 
# =============================================================================

plt.ylabel("Runtime(s)")

plt.suptitle(r"Resolution's effect on Runtime (s) and $\alpha_{c-}^{'}$", 
             y = 1)

plt.title(r"b' : {},   $\theta$ : {}°".format(b_dash, 
                                              round(math.degrees(theta), 3)), 
          fontsize = 10 )

g.legend(handles=[Line2D([], [], marker='_', color="k", 
                         label=r"$\alpha_{c-}^{'}$"), 
                  Line2D([], [], marker='_', color="r", 
                         label='Runtime ev'),
                  Line2D([], [], linestyle = '--', color="g", 
                         label='Runtime evr'),
                  Line2D([], [], marker='_', color="b", 
                         label='Runtime evd')],
                  #Line2D([], [], marker='_', color="y", label='Runtime evx')],
         bbox_to_anchor=(0,0,1,0.2), loc="lower right",
         mode="expand", borderaxespad=-6, ncol=5)

plt.show()

#%%

#plotting alphas over 1/N

plt.figure(dpi = 300)
sns.set_theme()
g2 = sns.scatterplot(x = df["Resolution_inv"],
                     y = df["Alpha"],
                     marker = "o",
                     s = 20)

plt.suptitle(r"Inverse Lattice Resolution's Relationship with $\alpha_{c-}^{'}$",
             y = 1.00)
plt.title((r"b' : {},   $\theta$ : {}°,   N $\in$ {{ 5, 7, $\cdots$, 123 }}"
           .format(b_dash, math.degrees(theta))), fontsize = 9)
plt.xlabel(r"Inverse Lattice Resolution ( $\frac{1}{N}$ )")
plt.ylabel(r"$\alpha_{c-}^{'}$")

plt.show()

#%%

#same as before but now with regressions added to the mix

plt.figure()
fig, ax = plt.subplots(figsize = (6,6), dpi = 300)
#sns.set_theme()
sns.set_style("white")

g3 = sns.regplot(x = df["Resolution_inv"],
                 y = df["Alpha"],
                 fit_reg = True,
                 ci = None,
                 ax = ax,
                 marker = "o",
                 scatter_kws = {"s": 15},
                 line_kws = {"linewidth": 0,
                             "linestyle": "--"},
                 truncate = False)

g3 = sns.regplot(x = df.loc[11:12,"Resolution_inv"],
                 y = df.loc[11:12,"Alpha"],
                 fit_reg = True,
                 ci = None,
                 scatter = False,
                 ax = ax,
                 line_kws = {"linewidth": 1,
                             "linestyle": "--"},
                 truncate = False)

g3 = sns.regplot(x = df.Resolution_inv.iloc[-3:-1],
                 y = df.Alpha.iloc[-3:-1],
                 fit_reg = True,
                 ci = None,
                 scatter = False,
                 ax = ax,
                 line_kws = {"linewidth": 1,
                             "linestyle": "-"},
                 truncate = False)

plt.suptitle(r"Inverse Lattice Resolution's Relationship with $\alpha_{c-}^{'}$",
             y = 1.00)
plt.title((r"b' : {},   $\theta$ : {}°,   N $\in$ {{ 5, 7, 9, $\cdots$, 123 }}"
           .format(b_dash, round(math.degrees(theta), 3))), fontsize = 12)
plt.xlabel(r"Inverse Lattice Resolution ( $N^{-1}$ )")
plt.ylabel(r"$\alpha_{c-}^{'}$")

g3.legend(handles=[#Line2D([], [], linestyle = '--', color="b", label='Entire Regression'),
                   Line2D([], [], linestyle = '--', color="tab:orange", label ='Term 11 & 12 Regression'),
                   Line2D([], [], linestyle ='-', color="g", label='Last Term Regression')],
          bbox_to_anchor=(0,0,1,0.2), loc="lower right", 
          mode="expand", borderaxespad=-6, ncol=5)

plt.show()

#%%

#calculating the error between extrapolations

print(df.loc[10:11,["Resolution_inv", "Alpha", "Resolution"]])
result_aprox = stat.linregress(x = df.loc[11:12, "Resolution_inv"],
                               y = df.loc[11:12, "Alpha"])

result_fin = stat.linregress(x = df.Resolution_inv.iloc[-3:-1],
                             y = df.Alpha.iloc[-3:-1])

extrap_error = ((result_fin[1] - result_aprox[1])/result_fin[1])*100

print(round(extrap_error, 3), "%")

print(result_fin[1])
