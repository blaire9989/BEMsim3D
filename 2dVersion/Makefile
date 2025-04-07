all: main

main: main.cpp HankelTable.o Incidence.o MatrixBuild.o Scattering.o Solver.o
	nvcc -I include main.cpp HankelTable.o Incidence.o MatrixBuild.o Scattering.o Solver.o -O3 -o main -lpthread -lcublas

HankelTable.o: HankelTable.h HankelTable.cpp
	g++ -std=c++11 -c -I include HankelTable.cpp -O3 -o HankelTable.o
	
Incidence.o: Incidence.h Incidence.cpp
	g++ -std=c++11 -c -I include Incidence.cpp -O3 -o Incidence.o

MatrixBuild.o: MatrixBuild.h MatrixBuild.cpp
	g++ -std=c++11 -c -I include MatrixBuild.cpp -O3 -o MatrixBuild.o
	
Scattering.o: Scattering.h Scattering.cpp
	g++ -std=c++11 -c -I include Scattering.cpp -O3 -o Scattering.o

Solver.o: Solver.h Solver.cu
	nvcc -c -I include Solver.cu -O3 -o Solver.o --expt-relaxed-constexpr

clean:
	rm -f main
	rm -f *.o