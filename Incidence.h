#pragma once
#include "constants.h"
#include "parallel.h"
#include "Estimate.h"

class Incidence {
    public:
        int Nx, Ny, order, hori_num, vert_num;
        double d, eta, k, eps, omega, w0, w1, zR0, zR1, irradiance1, irradiance2;
        dcomp scale_factor;
        Matrix3d R1, R2;
        VectorXd xvals, yvals, pvals, wvals;
        MatrixXd zvals;
        VectorXcf b;
        
        Incidence(Estimate* est);
        void setParameters(double eta, double lambda, double w, double theta_i, double phi_i);
        void computePower();
        void computeVector(int polarization);
        MatrixXcd gaussian(double x0, double y0, double z0, dcomp scale, int polarization);
};