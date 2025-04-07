#pragma once
#include "constants.h"

class HankelTable {
    public:
        double dielectric_min, dielectric_max;
        VectorXd dielectric_key;
        MatrixXd dielectric;
    
        HankelTable(double lambda);
        dcomp lookUp(dcomp k, double dist, int type);
        dcomp lookUpDielectric(double value, int type);
};