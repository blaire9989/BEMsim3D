#pragma once
#include "constants.h"
#include "parallel.h"
#include "HankelTable.h"
#include "Incidence.h"

class Scattering {
    public:
        int order;
        MatrixXd xyvals;
        MatrixXi geo_info, mat_info;
        VectorXd pvals, wvals;
        MatrixXf far;
        MatrixXcf near_TE, near_TM;
        HankelTable* hankel;
        Incidence* inci;
        
        Scattering(MatrixXd xyvals, MatrixXi geo_info, MatrixXi mat_info, HankelTable* hankel, Incidence* inci);
        void computeFarField(double lambda, VectorXcf x_TE, VectorXcf x_TM, int resolution);
        MatrixXcf farFieldValues(double phi, double k, int polarization, VectorXd X, VectorXd Y, VectorXcf J0, VectorXcf M0);
        void computeNearField(double lambda, double ior, VectorXcf x_TE, VectorXcf x_TM, int x_res, double x_min, double x_max, int y_res, double y_min, double y_max, bool add_incident);
        int locatePoint(double x0, double y0);
        Vector2cf nearFieldEHz(Vector2d p0, double k, int polarization, VectorXd X, VectorXd Y, VectorXcf J0, VectorXcf M0);
};
