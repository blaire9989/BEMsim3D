#pragma once
#include "constants.h"

class Estimate {
    public:
        double d;
        int Nx, Ny;
        VectorXd xrange, yrange, zrange;
        MatrixXd zvals;
        MatrixXi A[4], B[4];
        
        Estimate(double x_um, double y_um, MatrixXd z, bool useProvided, double shift);
        VectorXd gridXY(double size_um, int size_int, int& trim);
        VectorXd gridZ();
        void computeNearIndices(int thres);
        void memGB(double& mem11, double& mem12, double& mem21, double& mem22, double& mem34);
};