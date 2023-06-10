#pragma once
#include <fftw3.h>
#include "constants.h"
#include "Estimate.h"
#include "Grid.h"

class Scattering {
    public:
        int Nx, Ny, hori_num, vert_num, num_pts, numX, numY, numZ, M1, M2, M3, h_interp, n_interp;
        double normalize, space_range;
        bool downSample;
        fcomp c1, c2, c3, c4;
        Grid* grid;
        VectorXf filter, r_range;
        MatrixXcf JMxyz, fft_block, interp_data, Ex, Ey, Ez, Hx, Hy, Hz;
        MatrixXf brdf;
        fftwf_complex* fft_data;
        fftwf_plan plan;
        
        Scattering(Estimate* est, Grid* grid);
        void computeBRDF(double eta, double lambda, VectorXcf x, int outres, float irra);
        void computeFields(double eta, double lambda, VectorXcf x, int outres, double xshift, double yshift);
        void pointSources(VectorXcf x);
        void computeComponent(int component);
        void interpolate(int num1, int num2, float r1, float r2, float r3);
        void cleanAll();
};