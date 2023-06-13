all: bem3d steerBasicDirs steerBRDFs

bem3d: Estimate.o Grid.o Incidence.o Kernels.o MVProd.o MVProd0.o MVProd1.o MVProd2.o MVProd3.o MVProd4.o Scattering.o Singular.o Singular0.o Singular12.o Singular34.o Solver.o bem3d.cpp
	nvcc -I include bem3d.cpp Estimate.o Grid.o Incidence.o Kernels.o MVProd.o MVProd0.o MVProd1.o MVProd2.o MVProd3.o MVProd4.o Scattering.o Singular.o Singular0.o Singular12.o Singular34.o Solver.o -lcudart -lcufft -lcusparse -lfftw3f -lfftw3f_threads -lm -O3 -o bem3d
	
Estimate.o: Estimate.h Estimate.cpp
	g++ -std=c++11 -c -I include Estimate.cpp -O3 -o Estimate.o
	
Grid.o: Grid.h Grid.cpp
	g++ -std=c++11 -c -I include Grid.cpp -O3 -o Grid.o
	
Incidence.o: Incidence.h Incidence.cpp
	g++ -std=c++11 -c -I include Incidence.cpp -O3 -o Incidence.o
	
Kernels.o: Kernels.h Kernels.cu
	nvcc -c Kernels.cu -O3 -o Kernels.o -lcudart
	
MVProd.o: MVProd.h MVProd.cpp
	g++ -std=c++11 -c -I include MVProd.cpp -O3 -o MVProd.o
	
MVProd0.o: MVProd0.h MVProd0.cpp
	g++ -std=c++11 -c -I include MVProd0.cpp -O3 -o MVProd0.o
	
MVProd1.o: MVProd1.h MVProd1.cu
	nvcc -c -I include MVProd1.cu -O3 -o MVProd1.o -lcudart -lcufft -lcusparse --expt-relaxed-constexpr
	
MVProd2.o: MVProd2.h MVProd2.cu
	nvcc -c -I include MVProd2.cu -O3 -o MVProd2.o -lcudart -lcufft -lcusparse --expt-relaxed-constexpr

MVProd3.o: MVProd3.h MVProd3.cu
	nvcc -c -I include MVProd3.cu -O3 -o MVProd3.o -lcudart -lcufft -lcusparse --expt-relaxed-constexpr

MVProd4.o: MVProd4.h MVProd4.cu
	nvcc -c -I include MVProd4.cu -O3 -o MVProd4.o -lcudart -lcufft -lcusparse --expt-relaxed-constexpr

Scattering.o: Scattering.h Scattering.cpp
	g++ -std=c++11 -c -I include Scattering.cpp -O3 -o Scattering.o

Singular.o: Singular.h Singular.cpp
	g++ -std=c++11 -c -I include Singular.cpp -O3 -o Singular.o
	
Singular0.o: Singular0.h Singular0.cpp
	g++ -std=c++11 -c -I include Singular0.cpp -O3 -o Singular0.o
	
Singular12.o: Singular12.h Singular12.cu
	nvcc -c -I include Singular12.cu -O3 -o Singular12.o --expt-relaxed-constexpr

Singular34.o: Singular34.h Singular34.cu
	nvcc -c -I include Singular34.cu -O3 -o Singular34.o --expt-relaxed-constexpr

Solver.o: Solver.h Solver.cpp
	nvcc -c -I include Solver.cpp -O3 -o Solver.o
	
steerBasicDirs: steerBasicDirs.cpp
	g++ -std=c++11 -I include steerBasicDirs.cpp -O3 -o steerBasicDirs
	
steerBRDFs: steerBRDFs.cpp
	g++ -std=c++11 -I include steerBRDFs.cpp -lpthread -O3 -o steerBRDFs
	
clean:
	rm -f bem3d steerBasicDirs steerBRDFs
	rm -f *.o
