#pragma once
#include "Kernels.h"
#include "Singular.h"

class Singular12: public Singular {
    public:
        float *d_zvals, *d_self, *d_hori, *d_vert, *d_p1, *d_w1, *d_p2, *d_w2;
    
        Singular12(Estimate* est, int ind);
        void computeInvariants();
        void selfEE();
        void neighborEE();
};