# -*- coding: utf-8 -*-

# for testing the runtime and extrapolation in the 2D dipole-dipole function
# as a function of N

#%%
import sys
from typing_extensions import runtime
sys.path.append("..")

import pandas as pd
import math
import numpy as np
import scipy.stats as stat

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

import Dipole_Dipole_2D_1D as sim

#Hyperparameters
#==============================================================================
v2 = (0.5, 1)

#range of N you wish to test
start = 5 
end = 150
step = 2

# =============================================================================

# tests whether data exists, and if not, generates new data 
def get_data(v2, start, end, step):
       
       datafile = "{}_{}_end{}.csv".format(v2[0], v2[1], end)
       
       try:
              df = pd.read_csv("results/{}".format(datafile), index_col=(0))
              print("df found!")
              
       except:
              df = generate_data(v2, start, end, step, datafile)
              
       return df


#test runs of Dipole-Dipole 2D-1D script with 1 or multiple solvers
def generate_data(v2, start, end, step, datafile):

       data = np.zeros((int(np.ceil((end-start)/step)), 3))
       
       for count, i in enumerate(range(start, end, step)):
              
              data[count][0] = i
              
              data[count][1], data[count][2] = sim.run_sim((1, 0), v2, i)
              
       df = pd.DataFrame(data = data)

       df.columns = ["Resolution", "Alpha", "Runtime(s)"]
                     
       df["Resolution_inv"] = np.power(df["Resolution"], -1)

       #error calculations
       
       best_extrap = stat.linregress(x = df.Resolution_inv.iloc[len(df) - 2 : len(df)],
                                     y = df.Alpha.iloc[len(df) - 2 : len(df)])[1]

       extrapolations = np.zeros((len(df), 3))

       for i in range(1, len(df)):
              
              extrap = stat.linregress(x = df.Resolution_inv.iloc[i-1:i+1],
                                   y = df.Alpha.iloc[i-1:i+1])[1]
              
              extrapolations[i][0] = extrap
              extrapolations[i][1] = best_extrap - extrap
              extrapolations[i][2] = ((best_extrap - extrap)/(best_extrap)) * 100
       
       df["Extrapolation"] = extrapolations[:,0]
       df["Error"] = extrapolations[:,1]
       df["Error %"] = extrapolations[:,2]
       
       #saving pandas dataframe to csv 
       
       df.to_csv("results/{}".format(datafile))
       
       return df


       
if __name__ == "__main__":
       
       df = get_data(v2, start, end, step)

#%%
#plotting all results 

plt.figure(dpi = 300)
sns.set_theme()
sns.set_style("white")

g = sns.lineplot(x = df["Resolution"], 
                 y = df["Alpha"], 
                 color = "k", 
                 linewidth = 1.5)
plt.xlabel("Resolution N")
plt.ylabel(r"$\alpha_{c-}^{'}$")

g = sns.lineplot(x = df["Resolution"], 
                 y = df["Runtime(s)"],
                 color = "b", 
                 linewidth = 1,
                 ax = plt.twinx())

plt.ylabel("Runtime(s)")

plt.suptitle(r"Runtime (s) and $\alpha_{c-}^{'}$ as N increases", 
             y = 1)

plt.title((r"v2 : {},  N $\in$ {{ 5, 7, $\cdots$, 149 }}"
           .format(v2)), fontsize = 9)

g.legend(handles=[Line2D([], [], marker='_', color="k", label=r"$\alpha_{c-}^{'}$"), 
                  Line2D([], [], marker='_', color="b", label='Runtime')])


plt.show()

#%%

#plotting alphas over 1/N

plt.figure(dpi = 300)
#sns.set_style("white")
g2 = sns.scatterplot(x = df["Resolution_inv"],
                     y = df["Alpha"],
                     marker = "o",
                     s = 20)

g3 = sns.regplot(x = df.loc[7:8,"Resolution_inv"],
                 y = df.loc[7:8,"Alpha"],
                 fit_reg = True,
                 ci = None,
                 scatter = False,
                 #ax = ax,
                 line_kws = {"linewidth": 1,
                             "linestyle": "--"},
                 truncate = False)

"""
g3 = sns.regplot(x = df.Resolution_inv.iloc[-3:-1],
                 y = df.Alpha.iloc[-3:-1],
                 fit_reg = True,
                 ci = None,
                 scatter = False,
                 #ax = ax,
                 line_kws = {"linewidth": 1,
                             "linestyle": "-"},
                 truncate = False)
"""

g3.legend(handles=[Line2D([], [], marker = "o", color = "w", markerfacecolor='b',
                          label = r"$\alpha_{c-}^{'}$ at $\frac{1}{N}$"),
                   Line2D([], [], linestyle = '--', color = "b", 
                          label='Regression with N = (19, 21)')])
                   #Line2D([], [], linestyle = '-', color="tab:orange", 
                    #      label ='Regression with N = (147, 149)')])
          
plt.xlabel(r"Inverse Lattice Resolution ( $\frac{1}{N}$ )")
plt.ylabel(r"$\alpha_{c-}^{'}$")
plt.suptitle(r"Inverse Lattice Resolution's Relationship with $\alpha_{c-}^{'}$",
             y = 1.00)

plt.title((r"v2 : {},  N $\in$ {{ 5, 7, $\cdots$, 149 }}"
           .format(v2,)), fontsize = 9)
plt.show()
