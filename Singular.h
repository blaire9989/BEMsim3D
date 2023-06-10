#pragma once
#include "constants.h"
#include "parallel.h"
#include "Estimate.h"

class Singular {
    public:
        int Nx, Ny, hori_num, vert_num, order1, order2, order3;
        float d;
        MatrixXf zvals;
        VectorXf p1, w1, p2, w2, p3, w3;
        Estimate* est;
        MatrixXf LS1, LS2, LH1, LH2, LV1, LV2, KH0, KV0, quarter;
    
        Singular(Estimate* est);
        virtual void computeInvariants();
        void neighborEM();
        MatrixXf computeNeighborEM(float z1, float z2, float z3, float z4, float z5);
        void computeQuarter(int dev);
        void computeHH();
        void computeHV(int dev);
        void computeVV();
};