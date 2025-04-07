#pragma once
#include "constants.h"
#include "parallel.h"

class Incidence {
    public:
        int N, order;
        double air, eps, omega, k, phi, yc, w0;
        float irra_TE, irra_TM;
        VectorXd pvals, wvals, center;
        VectorXcd phase;
        MatrixXi geo_info, mat_info;
        MatrixXd xyvals;
        VectorXcf v_TE, v_TM;
        
        Incidence(MatrixXd xyvals, MatrixXi geo_info, MatrixXi mat_info);
        void computeIncidence(double lambda, double phi, double w0, double xc, double yc, int nBeams);
        void computeVectors();
        void computeIrradiance();
        MatrixXcd plane(double x0, double y0);
        MatrixXcd gaussian(double x0, double y0);
};