## BEMsim3D
Open-source code base of the SIGGRAPH 2023 paper, "A Full-Wave Reference Simulator for Computing Surface Reflectance." For details on the underlying simulation method and acceleration technique, please refer to the paper.

Our simulation code supports individual simulations on provided surface samples; it also supports grouped simulations on subregions of given surface samples, which can be used along with our "beam steering" technique for generating BRDFs corresponding to densely sampled directions.

### Building the Code Base
The code base has been tested on Unix systems. The CUDA Toolkit and nvcc compiler are required for building this code base. The [FFTW3](https://www.fftw.org) library is also required, and please install the single-precision version of FFTW and enable compilation and installation of the FFTW threads library (with Unix systems, these require adding the --enable-float and --enable-threads flags to configure).

With the dependencies installed, building the code base using the provided Makefile should be as simple as running:
```
make
```
You may see some warning messages from building a few of the modules, and these warning messages are safe to disregard.

#### One Potential Issue with Building
One issue we encountered when testing the code on some different machines occurred with building some modules implemented in CUDA C++.

### Code Base Overview
Our simulation code implements the 3D boundary element method (BEM) in a surface scattering context, and acceleration is achieved using the Adaptive Integral Method (AIM). Our code is written in C++ and CUDA C++, and different modules of the simulation are implemented in varied C++ classes. 

#### Users do not need to understand and are strongly not recommended to modify the code in the following class files:

— Estimate

— Grid

— Incidence

— Kernels

— MVProd (including MVProd0, MVProd1, MVProd2, MVProd3, MVProd4)

— Scattering

— Singular (including Singular0, Singular12, Singular34)

— Solver

### Tutorial and Support
For instructions on running individual simulations, please see [this tutorial](https://github.com/blaire9989/BEMsim3D/blob/main/tutorial_individual.md).

For instructions on grouped simulations with beam steering, please see [this longer tutorial](dummy.com).

For any questions on the code base, please contact the author at yy735@cornell.edu.
