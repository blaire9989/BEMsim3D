#pragma once
#include "Kernels.h"
#include "Singular.h"

class Singular34: public Singular {
    public:
        int devNumber[4];
        float *d_zvals[4], *d_self[4], *d_hori[4], *d_vert[4], *d_p1[4], *d_w1[4], *d_p2[4], *d_w2[4];
    
        Singular34(Estimate* est, int ind0, int ind1, int ind2, int ind3);
        void computeInvariants();
        void selfEE();
        void neighborEE();
};