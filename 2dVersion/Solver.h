#pragma once
#include "cublas_v2.h"
#include "constants.h"
#include "MatrixBuild.h"

class Solver {
    public:
        int N;
        MatrixBuild* mat;
        VectorXcf x;
        cuComplex alpha, beta, *d_x, *d_y, *d_Z;
        cublasHandle_t handle;
        
        Solver(MatrixBuild* mat);
        void transferMatrix(int polarization);
        VectorXcf multiply(VectorXcf h_x);
        int csMINRES(VectorXcf b, int maxit, float rtol, bool show = false);
        VectorXcf symOrtho(fcomp a, fcomp b);
        void cleanAll();
};