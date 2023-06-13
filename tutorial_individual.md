### Individual Simulations
The driver code of the simulation is written in the file bem3d.cpp. 

#### Input Data
Three input files are expected for simulations on each surface sample. Users choose a short name for each surface sample they wish to simulate, and create a folder of this name in the $\texttt{data}$ directory. Users should then put three input files into this created folder.

##### zvals.txt
A $\texttt{.txt}$ file that contains height field data that specify the surface heights. Each surface is assumed to be square and discretized into $N \times N$ quadrilateral basis elements. Height values at the corners of all the basis elements need to be provided, meaning that $\texttt{zvals.txt}$ should contain a matrix with size $(N + 1) \times (N + 1)$. See Fig. 3 of our paper for an illustration.

As we will discuss in the command line arguments section, users are also expected to provide the size of the simulated surface, in microns. Our code will infer the size of each quadrilateral basis element. Importantly, our code assumes that each basis element is square and that all the basis elements have the same size when projected onto the $xy$ plane. This means that if the specified size of the surface is $L \mu m \times L \mu m$, and the matrix contained in $\texttt{zvals.txt}$ has size $(N + 1) \times (N + 1)$, then the basis element size is then given by $d = L / N$. Please see our examples below for more clarifications.

##### wvl.txt
A $\texttt{.txt}$ files that contains some rows of data, where each row specifies a wavelength to be simulated. Each row contains 3 values, where the first value is the wavelength in microns, generally between 0.4 and 0.7. The second and third values are the real and imaginary parts of the surface material's index of refraction (IOR) at the provided wavelength. For materials such as metals, these values vary can significantly with wavelength.

##### wi.txt
A $\texttt{.txt}$ files that contains some rows of data, where each row specifies an incident direction to be simulated. Each row contains 2 values, where the first value is the zenith ($\theta$) angle of the incident direction, and the second is the azimuth ($\phi$) angle of the incident direction. Fpr normal incidence, both angles are 0. For grazing incident directions, the first value should be close to $\pi / 2$.

##### Formatting
All the data values in the aforementioned $\texttt{.txt}$ files are assumed to be delimited by commas(,) in each row. We usually construct the input data matrices in $\texttt{MATLAB}$ and use the $\texttt{writematrix}$ function in $\texttt{MATLAB}$ to write the $\texttt{.txt}$ files.

##### Additional Tips
If you intend to simulate a surface with size $L \mu m \times L \mu m$, and your prepared height field matrix has size $(N + 1) \times (N + 1)$, we have two suggestions on the choice of $N$ (your simulation resolution):

(a) The basis element size, $d = L / N$ is recommended to be at most 0.04, to ensure accuracy.

(b) If the number $N / 1.6$ happens to be an integer and this integer is a product of small primes (2, 3, 5, 7), the simulation speed will be in general more optimal (due to underlying FFT routines).

#### Command Line Arguments 
For individual simulations, the following command-line arguments are relevant:

$\texttt{-c}$: Selects the simulated wavelength from the provided $\texttt{wvl.txt}$ file. If there are $C$ rows (corresponding to $C$ wavelengths) in $\texttt{wvl.txt}$, this input argument should be an integer between 0 and $C-1$, inclusive. If the user does not specify this argument, simulations will be done for all the wavelengths specified in $\texttt{wvl.txt}$.

$\texttt{-d}$: Selects the simulated incident direction from the provided $\texttt{wi.txt}$ file. If there are $D$ rows (corresponding to $D$ incident directions) in $\texttt{wi.txt}$, this input argument should be an integer between 0 and $D-1$, inclusive. If the user does not specify this argument, simulations will be done for all the incident directions specified in $\texttt{wi.txt}$.

$\texttt{-e}$: The index of refraction (IOR) of the medium where the light is incident from. Usually chosen as 1.0, but it can be any real number indicating any dielectric medium.

$\texttt{-l}$: The side length of the simulated surface, along the $x$ and $y$ directions, in microns. Note that since surface samples are assumed to be square, we only have this one size parameter.

$\texttt{-o}$: An integer that represents the resolution of the output image containing BRDF values. All the figures in our paper were made with this argument set to 1024 ($1024 \times 1024$ images).

$\texttt{-w}$: The primary waist of the incident Gaussian beams used for simulations (see Section 3 of the paper for the term primary waist), in microns. Usually, for a simulated surface of $L \mu m \times L \mu m$, this argument can be chosen as $L / 2.5$. The incident Gaussian beam is approximately focused at the center of the surface patch.

$\texttt{-z}$: The name of the simulated surface. All the expected input files, $\texttt{wi.txt}$, $\texttt{wvl.txt}$, $\texttt{zvals.txt}$, as well as all the output data, exist in the folder with the provided name, under the $\texttt{data}$ directory.

#### Examples
We provided a simple, small-scale example that can be used for initial testing. The required input data are provided in $\texttt{data/test}$ folder. This surface sample is intended to be $12.5 \mu m \times 12.5 \mu m$, and the matrix provided in $\texttt{zvals.txt}$ is $401 \times 401$.

After building the code base, running
```
./bem3d -c 0 -d 0 -e 1.0 -l 12.5 -o 1024 -w 2.5 -z test
```
should generate one binary file. This binary file, as well as all the output files with similarly formatted names, can be opened in $\texttt{MATLAB}$ using
```
id = fopen('BRDF_wvl0_wi0.binary');
brdf00 = fread(id, [1024 1024], 'float');
```
The $1024 \times 1024$ image should look like the following:

Additionally, we also provided 6 folders that correspond to our featured $24 \mu m \times 24 \mu m$ surface samples. The surface heights, as well as the 25 wavelengths and 5 incident directions we considered are provided in the $\texttt{zvals.txt}$, $\texttt{wvl.txt}$ and $\texttt{wi.txt}$ files. For these surfaces, our intended basis element length is $d = 0.025 \mu m$, and so the height data are provided as $961 \times 961$ matrices in $\texttt{zvals.txt}$. As an example, running the command
```
./bem3d -d 0 -e 1.0 -l 24.0 -o 1024 -w 5.5 -z isotropicSmooth
```
should generate 25 binary files that contain spectral BRDF data, which can be converted into RGB to make the first BRDF plot in our paper's teaser image (it requires having 4 GPUs and takes around 2 hours):
