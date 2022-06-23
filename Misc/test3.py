#%%
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:56:52 2022

@author: Philipwur
"""

import time

import numpy as np
import numpy.linalg as la
import numba
from numba import njit, jit

#hyperparameters

lat_type = "FCC"  #"SC", "FCC" and "BCC"
lat_res = 3 #lattice resolution


@njit()
def dipole_dipole(lat_type, lat_res):
    
    #Assigning the coordinates of the SC atoms (present in all lattices)
    #this can be sped up but doesnt take much time in the grand scheme of things
    if lat_type == "SC":
        
        points = np.array([[i, j, k] 
                        for k in range(lat_res) 
                        for j in range(lat_res) 
                        for i in range(lat_res)]).astype(np.float64)

    if lat_type == "FCC":
        
        prim_mult = 2 ** (1/3)
        
        prim_vec = 0.5 * prim_mult * np.array([[0, 1, 1],[1, 0, 1],[1, 1, 0]])
        
        points = np.zeros((lat_res ** 3, 3))
        
        count = 0
        
        for i in range(lat_res):
            for j in range(lat_res):
                for k in range(lat_res):
                    
                    points[count, :] = i * prim_vec[0] + j * prim_vec[1] + k * prim_vec[2]
                    count += 1

    if lat_type == "BCC":

        prim_mult = 4 ** (1/3)
        
        prim_vec = 0.5 * prim_mult * np.array([[-1, 1, 1],[1, -1, 1],[1, 1, -1]])
        
        points = np.zeros((lat_res ** 3, 3))
        
        count = 0
        
        for i in range(lat_res):
            for j in range(lat_res):
                for k in range(lat_res):
                    
                    points[count, :] = i * prim_vec[0] + j * prim_vec[1] + k * prim_vec[2]
                    count += 1
    
    return points



points = dipole_dipole(lat_type, lat_res)
