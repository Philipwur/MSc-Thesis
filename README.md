# MSc Thesis 

Philip Wurzner @ QMUL  
Supervisor: Flynn Castles  
Co-Supervisor: Rahul Dutta

Program for simulating the Claussius-Mossoti model of spontaneous 
polarisability for a 2D metamaterial with a negative susceptibility. Only the
perpendicular field is considered (2D-1D). Model is a  fixed dipolarisable entities system. 

## Code Segments

Below are some descriptions of all the code
1. **Variable Analysis**  
    Contains file for testing variables, runtimes, lattice geometries, the effect the size of lattice has on the results and contour graphing results from the model. Descriptions of the codes can be found within the code itself

2. **Dipole_Dipole_2D_1D**  
    The model which genreates a 2D lattice of points, only accounting for the field it generates perpendicularly and calulates the maximum electric susceptibility values the lattice can take on without spontanously polarising.

3. **Dipole_Dipole_3D_3D**  
The 3D version of the script which relates the lattice geometry of the lattice to the maximum polarisability each meta-atom can take on. 

4. **Optimiser**  
The Script which optimises the lattice geometry of the 2D-1D lattice, to create the optimal "di-electric" levitation strength n_c

5. **requirements**  
The python modules required to run each file in this deposit  
note - pipreqs used to make requirements file

 6. **thesis_#########**  
 The final version of the masters' thesis written on the 2D-1D + optimiser 
