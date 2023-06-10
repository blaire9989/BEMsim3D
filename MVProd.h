#pragma once
#include "constants.h"
#include "Estimate.h"
#include "Grid.h"
#include "Singular.h"

class MVProd {
    public:
        int Nx, Ny, hori_num, vert_num, totalX, totalY, totalZ, N, num_pts, hori_row, hori_col, vert_row, vert_col;
        float d, dx, dy, dz;
        Estimate* est;
        Singular* singular;
        Grid* grid;
    
        MVProd(Estimate* est, Singular* singular, Grid* grid);
        virtual void setParameters(double eta1, dcomp eta2, double lambda);
        virtual VectorXcf multiply(VectorXcf x);
        virtual void cleanAll();
};