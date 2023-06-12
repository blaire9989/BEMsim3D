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

#### Users are strongly not recommended to modify any code in the above class files.

For instructions on running individual simulations, please see [this tutorial](https://github.com/blaire9989/BEMsim3D/blob/main/tutorial_individual.md).
