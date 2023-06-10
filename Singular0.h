#pragma once
#include "Singular.h"

class Singular0: public Singular {
    public:
        Singular0(Estimate* est);
        void computeInvariants();
        void selfEE();
        void neighborEE();
        MatrixXf computeSelfEE(float z1, float z2, float z3, float& div);
        MatrixXf computeNeighborEE(float z1, float z2, float z3, float z4, float z5, float& div);
};