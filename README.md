## BEMsim3D
Open-source code base of the SIGGRAPH 2023 paper, "A Full-Wave Reference Simulator for Computing Surface Reflectance." For details on the underlying simulation method and acceleration technique, please refer to the paper.

Our simulation code supports individual simulations on provided surface samples; it also supports grouped simulations on subregions of given surface samples, which can be used along with our "beam steering" technique for generating BRDFs corresponding to densely sampled directions.

### Code Base Overview
Our simulation code implements the 3D boundary element method (BEM) in a surface scattering context, and acceleration is achieved using the Adaptive Integral Method (AIM). Our code is written in $\texttt{C++}$ and $\texttt{CUDA C++}$, and different modules of the simulation are implemented in varied $\texttt{C++}$ classes. The only 3 files that users may need to read or understand are $\texttt{bem3d.cpp}$, $\texttt{steerBasicDirs.cpp}$, and $\texttt{steerBRDFs.cpp}$. 

#### Users do not need to understand and are strongly not recommended to modify the code in any other file.

### Building the Code Base
The code base has been tested on $\texttt{Unix}$ systems. The $\texttt{CUDA}$ Toolkit and $\texttt{nvcc}$ compiler are required for building this code base. The [FFTW3](https://www.fftw.org) library is also required, and please install the single-precision version of $\texttt{FFTW}$ and enable compilation and installation of the $\texttt{FFTW}$ threads library (with $\texttt{Unix}$ systems, these require adding the $\texttt{--enable-float}$ and $\texttt{--enable-threads}$ flags to $\texttt{configure}$).

With the dependencies installed, building the code base using the provided $\texttt{Makefile}$ should be as simple as running:
```
make
```
You may see some warning messages from building a few of the modules, and these warning messages are safe to disregard.

#### One Potential Issue with Building
One issue we encountered when testing the code on some different machines occurred with building some modules implemented in $\texttt{CUDA C++}$. If you encounter an error message that looks like:
```
/usr/include/c++/11/bits/std_function.h:530:146: error: parameter packs not expanded with '...'
```
please refer to [this post](https://github.com/NVIDIA/nccl/issues/650).

### Tutorial and Support
For instructions on running individual simulations, please see [this tutorial](https://github.com/blaire9989/BEMsim3D/blob/main/tutorial_individual.md).

For instructions on grouped simulations with beam steering, please see [this additional tutorial](https://github.com/blaire9989/BEMsim3D/blob/main/tutorial_steering.md).

For any questions on the code base, please contact the author at yy735@cornell.edu.

Full-wave simulations are in general slow and require intense computations. The following suggestions / facts may be helpful:

(a) It is recommended to run these simulations on large-scale platforms with at least 4 GPUs available. The average simulation time estimations reported in our paper correspond to simulations done with 4 Nvidia RTX 3090 GPUs.

(b) It is recommended to allocate at least 32 GB of RAM when running simulation jobs.

(c) Running simulations corresponding to a collection of wavelengths and incident directions may take many hours, even days.

(d) When running simulations using shared resources (e.g. Linux clusters), using job scheduling (e.g. with [SLURM](https://nesi.github.io/hpc_training/lessons/pan/job-scheduler/)) may be preferred over running in interactive mode, since heavy jobs might be killed before completion.
