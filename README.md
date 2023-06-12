## BEMsim3D
Open-source code base of the SIGGRAPH 2023 paper, "A Full-Wave Reference Simulator for Computing Surface Reflectance." For details on the underlying simulation method and acceleration technique, please refer to the paper.

Our simulation code supports individual simulations on provided surface samples; it also supports grouped simulations on subregions of given surface samples, which can be used along with our "beam steering" technique for generating BRDFs corresponding to densely sampled directions.

### Code Base Overview
Our simulation code implements the 3D boundary element method (BEM) in a surface scattering context, and acceleration is achieved using the Adaptive Integral Method (AIM). Our code is written in C++ and CUDA C++, and different modules of the simulation are implemented in varied C++ classes. 

#### Users do not need to read through or understand the code in the following class files:

— Estimate

— Grid

— Incidence

— Kernels

— MVProd (including MVProd0, MVProd1, MVProd2, MVProd3, MVProd4)

— Scattering

— Singular (including Singular0, Singular12, Singular34)

— Solver

#### Users are strongly not recommended to modify any of the code in the above class files.

### Individual Simulations
The driver code of the simulation is written in the file bem3d.cpp. 

#### Users are responsible for providing input data of the correct formats.

Three input files are expected for simulations on each surface sample. Users choose a short name for each surface sample they wish to simulate, and create a folder of this name in the data/ directory. Users should then put three input files into this created folder.

##### zvals.txt

##### wvl.txt

##### wi.txt

#### Users also need to understand the command line arguments specified at the beginning of the main function. 

For individual simulations, the following command-line arguments are relevant:

-c: selects the simulated wavelength from the provided wvl.txt file. If there are C rows (corresponding to C wavelengths) in wvl.txt, this input argument should be an integer between 0 and C-1, inclusive. If the user does not specify this argument, simulations will be done for all the wavelengths specified in wvl.txt.

-d: selects the simulated incident direction from the provided wi.txt file. If there are D rows (corresponding to D incident directions) in wi.txt, this input argument should be an integer between 0 and D-1, inclusive. If the user does not specify this argument, simulations will be done for all the incident directions specified in wi.txt.

-e: The index of refraction (IOR) of the medium where the light is incident from. Usually chosen as 1.0, but it can be any real number indicating any dielectric medium.

-o: An integer that represents the resolution of the output image containing BRDF values. All the figures in our paper were made with this argument set to 1024 (1024 * 1024 images).

-w: The primary waist of the incident Gaussian beams used for simulations (see Section 3 of the paper for the term primary waist), in microns. Usually, for a simulated surface of X um * Y um, this argument can be chosen as ~min{X, Y} / 2.5.

-x: The length of the simulated surface, along the x direction, in microns.

-y: The length of the simulated surface, along the y direction, in microns.

-z: The name of the simulated surface. All the expected input files, wi.txt, wvl.txt, zvals.txt, as well as all the output data, exist in the folder with the provided name, under the data/ directory.
