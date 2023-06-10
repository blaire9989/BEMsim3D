#pragma once
#include "constants.h"
#include "parallel.h"
#include "Estimate.h"

class Grid {
    public:
        int Nx, Ny, hori_num, vert_num, numX, numY, numZ, totalX, totalY, totalZ, num_directions, num_pts, order;
        double d, dx, dy, dz, regularization;
        dcomp k1, k2;
        VectorXd xvals, yvals, xrange, yrange, zrange, pvals, wvals;
        MatrixXd zvals, directions;
        MatrixXi hori_f, hori_b, vert_f, vert_b;
        MatrixXcf hori_x, hori_z, hori_d, vert_y, vert_z, vert_d, hori_X, hori_Z, hori_D, vert_Y, vert_Z, vert_D;
        
        Grid(Estimate* est);
        void computeIndices();
        void computeCoefficients(double eta1, dcomp eta2, double lambda);
        void computeHori(bool isDielectric);
        MatrixXcd horiLHS(int m, dcomp k);
        MatrixXcd horiRHS(int m, dcomp k);
        void computeVert(bool isDielectric);
        MatrixXcd vertLHS(int m, dcomp k);
        MatrixXcd vertRHS(int m, dcomp k);
};