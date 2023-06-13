### Individual Simulations
The driver code of the simulation is written in the file bem3d.cpp. 

#### Input Data

Three input files are expected for simulations on each surface sample. Users choose a short name for each surface sample they wish to simulate, and create a folder of this name in the $\texttt{data}$ directory. Users should then put three input files into this created folder.

##### zvals.txt

A .txt file that contains height field data that specify the surface heights. Each surface is assumed to be square and discretized into $N \times N$ quadrilateral basis elements. Height values at the corners of all the basis elements need to be provided, meaning that zvals.txt should contain a matrix with size $(N + 1) \times (N + 1)$. See Fig. 3 of our paper for an illustration.

As we will discuss in the command line arguments section, users are also expected to provide the size of the simulated surface, in microns. Our code will infer the size of each quadrilateral basis element. Importantly, our code assumes that each basis element is square and that all the basis elements have the same size when projected onto the $xy$ plane. This means that if the specified size of the surface is $L \mu m \times L \mu m$, and the matrix contained in zvals.txt has size $(N + 1) \times (N + 1)$, then the basis element size is then given by $d = L / N$. Please see our examples below for more clarifications.

##### wvl.txt

A .txt files that contains some rows of data, where each row specifies a wavelength to be simulated. Each row contains 3 values, where the first value is the wavelength in microns, generally between 0.4 and 0.7. The second and third values are the real and imaginary parts of the surface material's index of refraction (IOR) at the provided wavelength. For materials such as metals, these values vary can significantly with wavelength.

##### wi.txt

A .txt files that contains some rows of data, where each row specifies an incident direction to be simulated. Each row contains 2 values, where the first value is the zenith ($\theta$) angle of the incident direction, and the second is the azimuth ($\phi$) angle of the incident direction. Fpr normal incidence, both angles are 0. For grazing incident directions, the first value should be close to $\pi / 2$.

##### Formatting

All the data values in the aforementioned .txt files are assumed to be delimited by commas(,) in each row. We usually construct the input data matrices in MATLAB and use the writematrix function in MATLAB to write the .txt files.

#### Command Line Arguments 

For individual simulations, the following command-line arguments are relevant:

-c: selects the simulated wavelength from the provided wvl.txt file. If there are $C$ rows (corresponding to $C$ wavelengths) in wvl.txt, this input argument should be an integer between 0 and $C-1$, inclusive. If the user does not specify this argument, simulations will be done for all the wavelengths specified in wvl.txt.

-d: selects the simulated incident direction from the provided wi.txt file. If there are $D$ rows (corresponding to $D$ incident directions) in wi.txt, this input argument should be an integer between 0 and $D-1$, inclusive. If the user does not specify this argument, simulations will be done for all the incident directions specified in wi.txt.

-e: The index of refraction (IOR) of the medium where the light is incident from. Usually chosen as 1.0, but it can be any real number indicating any dielectric medium.

-l: The side length of the simulated surface, along the $x$ and $y$ directions, in microns. Note that since surface samples are assumed to be square, we only have this one size parameter.

-o: An integer that represents the resolution of the output image containing BRDF values. All the figures in our paper were made with this argument set to 1024 ($1024 \times 1024$ images).

-w: The primary waist of the incident Gaussian beams used for simulations (see Section 3 of the paper for the term primary waist), in microns. Usually, for a simulated surface of $L \mu m \times L \mu m$, this argument can be chosen as $L / 2.5$. The incident Gaussian beam is approximately focused at the center of the surface patch.

-z: The name of the simulated surface. All the expected input files, wi.txt, wvl.txt, zvals.txt, as well as all the output data, exist in the folder with the provided name, under the data/ directory.

#### Examples

We now provide some example scripts that can set different collections of simulations running. In the data/ directory, we have included 6 folders that corresponding to our featured $24 \mu m \times 24 \mu m$ surface samples. The surface heights, as well as the 25 wavelengths and 5 incident directions we considered are specified in the zvals.txt, wvl.txt and wi.txt files in each subfolder.

Our intended basis element length was $d = 0.025 \mu m$, so the height data were provided as $961 \times 961$ matrices in zvals.txt. We now use the brushedRough surface as an example.

```
./bem3d -c 0 -d 0 -e 1.0 -l 24.0 -o 1024 -w 5.5 -z brushedRough
```
simulates scattering from the surface using the first provided wavelength (0.4 $\mu m$) and the first provided incident direction (normal).

```
./bem3d -d 4 -e 1.0 -l 24.0 -o 1024 -w 5.5 -z brushedRough
```
simulates scattering from the surface using the 25 wavelengths and the last incident direction provided.

```
./bem3d -c 24 -e 1.0 -l 24.0 -o 1024 -w 5.5 -z brushedRough
```
simulates scattering from the surface using the longest wavelength provided and the 5 incident directions.

```
./bem3d -e 1.0 -l 24.0 -o 1024 -w 5.5 -z brushedRough
```
simulates scattering from the surface using all 25 wavelengths and all 5 incident directions.

#### Outputs

For individual simulations, output BRDFs will be written into binary files under the names BRDF_wvlA_wiB.binary, where A and B are integers. These binary files can be opened in MATLAB, using the following example code:

```
id = fopen('BRDF_wvl0_wi0.binary');
brdf00 = fread(id, [1024 1024], 'float');
```
Note that our outputs are single-precision floating point numbers.
