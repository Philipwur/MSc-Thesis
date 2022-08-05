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

- Continued literature review
  + Figure of merit Nc (understand where the permitivity term comes from etc)
  + ashcroft solid state physics page 95
- start making optimisation algo
- add to-do page rather than putting it in readme

- start writeup of literature review (page and a half )
  - Meta materials
  - Electric susceptibility and negative susceptibility
    - negative static susceptibility (mention flynn's paper here)
    - what it could do 
  - Ferroelectricity (spontaneous polarisability) (briefly mention as an analogous process)
    - we only model the transition not the polarised state
  - Claussius mossotti model (allen)
  - 2D array for modelling for the levitation problem, what is the best arrangement of this    array
  - 2D bravais lattices review



- next meet next week 1/07/20 @ 10AM (not the week after)



### Later

- start thinking about Nc calculation from alpha
- see 1/N^gamma variant of variable analysis
- look into CUDA for optimisation procedure
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
 
W2
- implemented the vector lattice generation method
- tested speed of vector lattice generation method (is about 225% faster with lattice vectors)
- tested irreducible parameter space using lattice vectors

W3
- optimised runtime of 3D-3D
- ran benchmarks
- calculating high resolution sims
- thought about periodicity
  - there is no reason for there to be perfect periodicity because the key element in calculating the alpha is the euclidian distance between points. When you distort the lattice to that extent, the average distance between the points will increase, which in turn will lower the alpha. The critical density calculation is not the problem. 
- read up on eigenvalues 
- do average distance calculation for x = 0.5 and x = 1.5 etc

W4
- send flynn a version of the paragraph (why our size was limit)
- Rewrote FCC and BCC point gen in 3D-3D
- got results for FCC and BCC 27
  - howcome results are so similar is the new generation style, 
  but different in the old one?
- explored symmetry, see explanation in figures

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
  
- lack of peridoicity is a result of the error in extrapolation and the border 
  of the lattice (distance increasing on average as x increases more than one)
  - this effect vanishes the more the lattice resolution increases 

- 8 pages
