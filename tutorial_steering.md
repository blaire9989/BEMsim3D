### Grouped Simulations
The driver code for grouped simulations is also written in the file $\texttt{bem3d.cpp}$. In addition, we provided two code files, $\texttt{steerBasicDirs.cpp$ and $\texttt{steerBRDFs.cpp}$, as example code that demonstrates beam steering.

#### Input Data
As with individual simulations, users choose a short name for the surface sample they wish to simulate, and create a folder of this name in the $\texttt{data}$ directory. 

##### .txt Files
Even though simulations are performed on subregions of the surface, we still ask users to provide the file $\texttt{zvals.txt}$ that stores the height values for the $\textit{entire}$ surface area, as this is important for integrating together the results from subregion simulations. The files $\texttt{wi.txt}$ and $\texttt{wvl.txt}$ also need to be provided in the created folder.

##### Patch Subfolders
Our provided two folders of data as examples for beam steering, and in each folder, there are many subfolders under the names $\texttt{patchXY}$, where $X$ and $Y$ are integers. These subfolders correspond to subregions of the simulated surface, and in our examples, the surface area is divided into $9 \times 9 = 81$ subregions (partially overlapping). Each of the $\texttt{patchXY}$ subfolders needs to contain its own $\texttt{zvals.txt} file, which contains the subregion height data. Note that the data matrix contained in these subfolder $\texttt{zvals.txt}$ files are submatrices of the data in $\texttt{zvals.txt}$ in the umbrella data folder.

Users need to create all the subfolders as well as the height data file in each subfolder.

#### Preprocessing
The code in $\texttt{steerBasicDirs.cpp}$ preprocesses the input file $\texttt{wi.txt}$, since with beam steering, different incident directions queried in $\texttt{wi.txt}$ may correspond to the same set of subregion simulations using the same basic incident direction. Please see Section 5 of our paper for the term basic incident direction.

Two command line arguments are required for running the code in $\texttt{steerBasicDirs.cpp}$:

-h: The name of the $\texttt{.binary}$ file that contains information on available incident directions. These binary files exist in the $\texttt{include/hexagons} folder, where we have provided an example file as well as a $\texttt{MATLAB}$ script containing code and instructions for generating more of these binary files.

-z: The name of the simulated surface that points the code to the folder that to read input data and write output files.

See a later section for an example run of this preprocessing code.

#### Simulating
To invoke grouped simulations, users need to run the code in $\texttt{bem3d.cpp}$ while providing a few more command line arguments than for individual simulations. Importantly, for each given wavelength and basic incident direction, simulations are done on different subregions of the surface in loops, and intermediate results are written in corresponding subfolders. To avoid flooding the disk, these intermediate results are immediately processed after all the subregion simulations are completed. Simulations with the next wavelength or incident direction will overwrite the intermediate files. For this reason, only one group of simulations should be done before each postprocessing step, so users must explicitly provide the wavelength and incident direction with the $\texttt{-c}$ and $\texttt{-d}$ arguments.

-a: The $x$ index of the subregion simulation. If the entire surface area is divided into $M \times M$ subregions, this argument should be an integer between 0 and $M-1$, inclusive.

-b: The $y$ index of the subregion simulation. Analogous to the previous argument.

-c: As for individual simulations, selects the simulated wavelength from the $\texttt{wvl.txt}$ file. For grouped simulations, users MUST provide this argument. If the user did not provide this argument, simulations would be run using the first wavelength provided.

-d: Selects the simulated incident direction from the required basic incident directions. The preprocessing code prints out a message indicating the total number of required incident directions. If there are $D$ required directions, this argument should be an integer between 0 and $D-1$. Users MUST provide this argument. If the user did not provide this argument, simulations would be run using the first basic incident direction.

-e: Same as for individual simulations.

-l: Same as for individual simulations, and this is the size of the simulated surface subregion.

-m: The total number of subregion simulations along the $x$ direction.

-n: The total number of subregion simulations along the $y$ direction.

-o: Same as for individual simulations.

-w: Same as for individual simulation, and this is the beam waist used in the subregion simulation.

-z: The name of the simulated surface.

#### Postprocessing
After a group of subregion simulations are completed, users can run the code in $\texttt{steerBRDFs.cpp}$ to process the subregion results and synthesize BRDFs. The command line arguments are as follows:

-c: Index of the simulated wavelength. This should be the same as the $\texttt{-c}$ argument provided in each subregion simulation.

-d: Index of the simulated basic incident direction. This should be the same as the $\texttt{-d}$ argument provided in each subregion simulation.

-e: Same as the $\texttt{-e}$ argument provided in each subregion simulation.

-m: Same as the $\texttt{-m}$ argument provided in each subregion simulation.

-n: Same as the $\texttt{-n}$ argument provided in each subregion simulation.

-o: Same as the $\texttt{-o}$ argument provided in each subregion simulation.

-w: Same as the $\texttt{-w}$ argument provided in each subregion simulation.

-x: The length of the entire surface area, along the $x$ direction, in microns.

-y: The length of the entire surface area, along the $y$ direction, in microns. Note that the entire surface area does not need to be square, since we can combine together different numbers of subregions along the two directions.

-z: The name of the simulated surface.

Running the code will generate the BRDF lobes for the entire, large surface area, for all the queried incident directions provided in $\texttt{wi.txt}$ that correspond to the currently simulated basic incident direction.

#### Examples
In the $\texttt{data}$ directory, we have included 2 folders that correspond to two featured $32 \mu m \times 32 \mu m$ surface samples. Each surface sample is divided into $9 \times 9 = 81$ partially overlapping subregions, and each subregion is $12 \mu m \times 12 \mu m$ (the shift between adjacent subregions is $2.5 \mu m$). For both examples, all the queried incident directions correspond to the same basic incident direction. We now use the $\texttt{steerBumpy}$ surface as an example.

Running the code
```
./steerBasicDirs -h hexagon_2.5um -z steerBumpy
```
should print out a message indicating there is only one basic incident direction needed.

We then run subregion simulations in loops, and postprocess results for each wavelength once the corresponding group of simulations completes:
```
for k in {0..24..1}
do
  for i in {0..8..1}
  do
    for j in {0..8..1}
    do
      ./bem3d -a ${i} -b ${j} -c ${k} -d 0 -e 1.0 -l 12.0 -m 9 -n 9 -o 1024 -w 2.5 -z steerBumpy
    done
  done
  ./steerBRDFs -c ${k} -d 0 -e 1.0 -m 9 -n 9 -o 1024 -w 2.5 -x 32.0 -y 32.0 -z steerBumpy
done
```

#### Output
Output binary files have the same formatted names as for individual simulations, while the indexing with respect to incident directions correspond to the original $\texttt{wi.txt}$ file provided by the user, not the basic incident directions, since one BRDF lobe is computed for each queried direction.
