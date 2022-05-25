# MSc Thesis 

Philip Wurzner @ QMUL  
Supervisor: Flynn Castles  

Program for simulating the Claussius-Mossoti model of spontaneous 
polarisability for a 2D metamaterial with a negative susceptibility. Only the
perpendicular field is considered (2D-1D). Model is a  fixed dipolarisable entities system. 

## Code Segments

Below are some descriptions of all the code
1. **Variable Analysis**  
Contains file for testing variables, runtimes and graphing results from the 
model.
2. **Misc**  
Contains miscellaneous testing and demo files. See folder for more info.
3. **3D_3D Model**  
Contains a Python translation of Rahul's original MATLAB script for the 3D case. Currently unoptimised.
4. **Dipole_Dipole_2D_1D**  
The model which genreates a lattice of points, and calulates the maximum 
electric susceptibility values the lattice can take on without spontanously 
polarising.

## To-Do

### Working on now

- Think about the boundary conditions of the coordinate system
  - e.g. equivalent lattices generated with different b' and thetas
  - periodicty garuanteed at 2pi 
  - look up irreducable balloon zone ("irreducable parameter space")
  - would be a lot easier to use lattice vectors for this purpose
  - b'*cos(theta)<=1
- Finish inital literature review - get to dipole-dipole matrix from scratch

### Later

- change linalg into numpy variant and add numba to all functions (should 
significantly improve runtimes)
  + compare Scipy distance matrix to np broadcasting method with numba
  + see week1 of deep learning module for info on broadcasting
  + link: https://bit.ly/3kpsAdO
- alternative to numba is to call a fortran/MATLAB function
- start thinking about Nc calculation from alpha
- see 1/N^gamma variant of variable analysis
- look into CUDA for optimisation procedure
- make 2D model work with array inputs (for multi-threading)
- create optimisation framework (pick optimiser)


### Extras

- Look into Julia
- change print commands to logging
- add points and interceptions to legend in regression graphs
- add resolution/subtitles in surface/contour plot for final report
- look into animating the 3D surface plot
- look into enumerate and object oriented code (classes)
- think about animating the regression relationship for the thesis presentation
- see how results of a strip of lattice compare to squares (might be able to save computational time)
- see if you can optimise the runtime for the 3D model


## Done

W0
- translated script
- squashed 3D bugs
- made 3D system 2D
- implemented new variables (b', theta and lattice_resolution) 
   + show coordinate system py file to confirm its working as intended)
- rewrote dip-dip relation function, lowering 100x100 runtime from 7min to 25secs
- did variable analysis, and chose best eigenvalue solver
- did regression to n=inf, must make a decision on suitable error to runtime  

W1
- started github/folder setup
- Try Rahul's way of generating points without the modulo and compare 1/N
  - Exact same results (should be able to leverage this for some time savings
    and array OF acceptance) 
- See if regression results hold up for different lattice parameters
  - Regression results hold up, but error range varies (0.666-0.4%)
- First surface plots (both Nc and ac) (resolution of 64)
 
## Notes 

just random stuff for me, not rigourously checked

- first draft is no big worry
- polarisability of matrix is alpha
- critical value is used for maximum susceptibilities without spontaneous polarisatbility 
  
  atoms polarise eachother without the need of an electric field at sufficiently close distances
  this happens with small fluctuations, which reinforce themselves into a polarised state
  negative polarisation means that they polarise opposite of the field applied to them. 
  we are creating a meta atom structure whereby each atom polarises eachother 
  in opposite directions from eachother rather than the positive polarisability alignment.
  
  we are assuming we are free to choose alpha of the meta atoms
  we want our material to operate with a range of alpha values, 
  whereby it doesnt self polarise. well documented for positive alphas, 
  when it exceeds positive critical value material becomes ferroelectric
  
  point of not self polarising is to allo
  w a point charge to levitate as a resullt
  of a force from a 2D plane. 
  
- Next Meeting Friday the 27th meeting at 1pm