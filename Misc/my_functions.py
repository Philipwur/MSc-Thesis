# -*- coding: utf-8 -*-
"""
@author: Philipwur

these are useful functions for the main program, 
that dont necessarily belong in the primary script
"""

#getting lattice parameters (working with reduced dimensions (any unit will work as long as ratio is right))
def spacing_choice():
    
    while True:
        print("Are interspacial gaps between atoms the same or different?")
        choice = str(input("(same/different):").lower())
        
        if choice not in ["same", "different"]:
            print("")
            print("Wrong input, please try again")
            print("")
            continue
            
        if choice == "same":
               
            a = float(input("atomic spacing:"))
               
            return a, a, a
           
        elif choice == "different":
           
            a = float(input("x-axis spaing:"))
            b = float(input("y-axis spaing:"))
            c = float(input("z-axis spaing:"))
           
            return a, b, c


#getting lattice resolution
#this could be a try loop
def resolution_choice():

    while True:
        print("Is the lattice resolution the same in all directions?")
        choice = str(input(("(same/different):")).lower())
    
        if choice not in ["same", "different"]:
            print("")
            print("Wrong input, please try again")
            print("")
            continue
        
        if choice == "same":
            
            Nx = int(input("# of atoms along each axis:"))
            
            return Nx, Nx, Nx
                
        elif choice == "different":
            
            Nx = int(input("# of atoms along x-axis:"))
            Ny = int(input("# of atoms along y-axis:"))
            Nz = int(input("# of atoms along z-axis:"))
            
            return Nx, Ny, Nz
        

#choosing polarisability
def polarisability_choice(extreme_a):
    
    while True:
    
        print("Enter the Polarisability")
        alpha = input("({} to {}):".format(round(extreme_a[0], 4), 
                                           round(extreme_a[1], 4)))
    
        alpha = float(alpha) * extreme_a[1]
    
        if alpha < extreme_a[0] or alpha > extreme_a[1]:
            print("Not within acceptable ranges")
            print("")
            continue
    
        return alpha
    

#choosing the values of the electric field
def field_choice(E0, tot_atoms):
    
    for i in range(2):
        
        while i == 0:
            print("Enter the Electric Field Values")
            
        E0[i][1] = input("({}):".format(i+1))
        
        for j in range(tot_atoms-2):
            E0[i+3*j][1] = E0[i][1]